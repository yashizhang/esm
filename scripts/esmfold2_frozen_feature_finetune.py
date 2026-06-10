#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from esm.training.esmfold2_frozen_features.collate import (  # noqa: E402
    collate_esmfold2_frozen_features,
)
from esm.training.esmfold2_frozen_features.config import (  # noqa: E402
    ESMFold2FrozenFeatureConfig,
)
from esm.training.esmfold2_frozen_features.dataset import (  # noqa: E402
    ESMFold2FrozenFeatureDataset,
    write_sample_manifest,
)
from esm.training.esmfold2_frozen_features.distributed import (  # noqa: E402
    DistributedEnv,
    barrier_if_distributed,
    destroy_process_group_if_initialized,
    detect_distributed_env,
    distributed_requested,
    max_metrics,
    normalize_distributed_env_for_init,
    reduce_metrics,
    resolve_dist_backend,
    resolve_precision,
    sum_metrics,
    uses_grad_scaler,
)
from esm.training.esmfold2_frozen_features.logging_utils import (  # noqa: E402
    JsonlLogger,
    format_tqdm_postfix,
    maybe_tqdm,
)
from esm.training.esmfold2_frozen_features.losses import (  # noqa: E402
    LOSS_KEYS,
    compute_esmfold2_loss,
)
from esm.training.esmfold2_frozen_features.modeling import (  # noqa: E402
    assert_no_esmc_parameters,
    build_optimizer,
    learning_rate_for_step,
    load_esmfold2_for_training,
    set_optimizer_lr,
)


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
    return moved


def _attach_training_forward(model: nn.Module) -> nn.Module:
    def forward(
        self: nn.Module,
        batch: dict[str, Any],
        lm_hidden_states: torch.Tensor,
        stage: int,
        seed: int | None = None,
        num_recycles_for_test: int | None = None,
        confidence_num_sampling_steps: int | None = None,
        config: ESMFold2FrozenFeatureConfig | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.forward_train_from_precomputed_lm(  # type: ignore[attr-defined]
            batch=batch,
            lm_hidden_states=lm_hidden_states,
            stage=stage,
            seed=seed,
            num_recycles_for_test=num_recycles_for_test,
            confidence_num_sampling_steps=confidence_num_sampling_steps,
            config=config,
        )

    model.forward = MethodType(forward, model)  # type: ignore[method-assign]
    return model


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _call_training_forward(
    model: nn.Module,
    *,
    batch: dict[str, Any],
    lm_hidden_states: torch.Tensor,
    stage: int,
    seed: int,
    num_recycles_for_test: int | None,
    confidence_num_sampling_steps: int | None,
    config: ESMFold2FrozenFeatureConfig,
) -> dict[str, torch.Tensor]:
    return model(
        batch,
        lm_hidden_states,
        stage,
        seed,
        num_recycles_for_test,
        confidence_num_sampling_steps,
        config,
    )


def create_train_dataloader(
    dataset: Dataset[Any],
    args: argparse.Namespace,
    dist_env: DistributedEnv,
    *,
    use_ddp: bool,
) -> tuple[DataLoader[Any], DistributedSampler[Any] | None]:
    sampler: DistributedSampler[Any] | None = None
    if use_ddp:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist_env.world_size,
            rank=dist_env.rank,
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        collate_fn=collate_esmfold2_frozen_features,
    )
    return loader, sampler


def _git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except Exception:
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": bool(status.strip())}


def _gpu_names() -> list[str]:
    if not torch.cuda.is_available():
        return []
    return [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]


def build_run_config(
    args: argparse.Namespace,
    *,
    dist_env: DistributedEnv,
    device: torch.device,
    backend: str | None,
    resolved_precision: str,
    config: ESMFold2FrozenFeatureConfig,
) -> dict[str, Any]:
    effective_global_batch_size = (
        args.batch_size * args.gradient_accumulation_steps * dist_env.world_size
    )
    result: dict[str, Any] = {
        "script_args": _json_safe(vars(args)),
        "stage": args.stage,
        "precision": resolved_precision,
        "device": str(device),
        "backend": backend,
        "world_size": dist_env.world_size,
        "rank_count": dist_env.world_size,
        "nproc_per_node": dist_env.local_world_size,
        "per_device_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_global_batch_size": effective_global_batch_size,
        "manifest_path": args.manifest,
        "cache_index_path": args.cache_index,
        "model_checkpoint_path": args.model_checkpoint,
        "output_dir": args.output_dir,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_names": _gpu_names(),
        "esmfold2_frozen_feature_config": asdict(config),
    }
    result.update(_git_metadata())
    return result


def _auto_wandb_run_name(args: argparse.Namespace, dist_env: DistributedEnv) -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"esmfold2-stage{args.stage}-ws{dist_env.world_size}-{timestamp}"


def init_wandb_run(
    args: argparse.Namespace,
    *,
    dist_env: DistributedEnv,
    run_config: dict[str, Any],
) -> Any | None:
    if args.wandb_mode == "disabled" or not dist_env.is_main:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "Weights & Biases logging was requested, but wandb is not installed. "
            "Install it with `pip install wandb`, or pass --wandb-mode disabled."
        ) from exc

    os.environ["WANDB_MODE"] = args.wandb_mode
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or _auto_wandb_run_name(args, dist_env),
        group=args.wandb_group,
        tags=args.wandb_tags,
        mode=args.wandb_mode,
        config=run_config,
    )


def _batch_counts(batch: dict[str, Any]) -> dict[str, float]:
    token_mask = batch.get("token_attention_mask")
    atom_mask = batch.get("atom_attention_mask")
    batch_size = 0
    if isinstance(token_mask, torch.Tensor):
        batch_size = int(token_mask.shape[0])
    elif isinstance(batch.get("example_id"), list):
        batch_size = len(batch["example_id"])
    return {
        "samples": float(batch_size),
        "tokens": float(token_mask.bool().sum().item()) if isinstance(token_mask, torch.Tensor) else 0.0,
        "atoms": float(atom_mask.bool().sum().item()) if isinstance(atom_mask, torch.Tensor) else 0.0,
    }


def _grad_norm(model: nn.Module) -> float:
    total = 0.0
    for parameter in _unwrap_model(model).parameters():
        if parameter.grad is None:
            continue
        norm = parameter.grad.detach().float().norm(2)
        total += float(norm.item() ** 2)
    return total**0.5


def _optimizer_weight_decay(optimizer: torch.optim.Optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("weight_decay", 0.0))


def build_training_metric_payload(
    loss_metrics: dict[str, float],
    *,
    lr: float,
    weight_decay: float,
    grad_norm: float | None,
    grad_scale: float | None,
    data_metrics: dict[str, float],
    perf_metrics: dict[str, float],
    dist_env: DistributedEnv,
    global_step: int,
    epoch: int,
    effective_global_batch_size: int,
) -> dict[str, float | int]:
    payload: dict[str, float | int] = {}
    for key in LOSS_KEYS:
        if key in loss_metrics:
            payload[f"train/{key}"] = float(loss_metrics[key])
    payload["opt/lr"] = float(lr)
    payload["opt/weight_decay"] = float(weight_decay)
    if grad_norm is not None:
        payload["opt/grad_norm"] = float(grad_norm)
    if grad_scale is not None:
        payload["opt/grad_scale"] = float(grad_scale)

    payload.update({key: float(value) for key, value in data_metrics.items()})
    payload["data/effective_global_batch_size"] = float(effective_global_batch_size)
    payload.update({key: float(value) for key, value in perf_metrics.items()})
    payload["dist/world_size"] = dist_env.world_size
    payload["dist/rank0_local_rank"] = dist_env.local_rank if dist_env.is_main else 0
    payload["dist/global_step"] = global_step
    payload["dist/epoch"] = epoch
    return payload


def _jsonl_record(payload: dict[str, float | int], global_step: int) -> dict[str, float | int]:
    record = dict(payload)
    record["step"] = global_step
    if "opt/lr" in payload:
        record["lr"] = payload["opt/lr"]
    for key in LOSS_KEYS:
        prefixed = f"train/{key}"
        if prefixed in payload:
            record[key] = payload[prefixed]
    return record


def _log_wandb(
    wandb_run: Any | None,
    payload: dict[str, float | int],
    *,
    global_step: int,
    every_n_steps: int,
) -> None:
    if wandb_run is None:
        return
    if every_n_steps <= 0 or global_step % every_n_steps != 0:
        return
    wandb_run.log(payload, step=global_step)


def _write_loss_breakdown(
    output_dir: Path,
    payload: dict[str, float | int],
    *,
    global_step: int,
    enabled: bool,
) -> None:
    if not enabled:
        return
    (output_dir / f"loss_breakdown_step_{global_step}.json").write_text(
        json.dumps(_jsonl_record(payload, global_step), indent=2, sort_keys=True) + "\n"
    )


def _make_grad_scaler(enabled: bool) -> torch.amp.GradScaler:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)  # type: ignore[return-value]


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    scaler: torch.amp.GradScaler | None,
    stage: int,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
    config: ESMFold2FrozenFeatureConfig,
    optimizer_param_names: list[str],
    resolved_precision: str,
    is_main: bool,
) -> bool:
    if not is_main:
        return False
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] = {
        "model_state_dict": _unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None,
        "stage": stage,
        "epoch": epoch,
        "global_step": global_step,
        "max_steps": args.max_steps,
        "args": _json_safe(vars(args)),
        "config": asdict(config),
        "resolved_precision": resolved_precision,
        "optimizer_param_names": optimizer_param_names,
        "rng_state": _rng_state(),
    }
    if scaler is not None and scaler.is_enabled():
        checkpoint["grad_scaler_state_dict"] = scaler.state_dict()
    torch.save(checkpoint, path)
    return True


def _restore_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    _unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if (
        scaler is not None
        and scaler.is_enabled()
        and checkpoint.get("grad_scaler_state_dict") is not None
    ):
        scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
    return int(checkpoint.get("global_step", 0)), int(checkpoint.get("epoch", 0))


def _prepare_rank0_mmcif_downloads(
    *,
    manifest: str,
    cache_index: str | None,
    is_main: bool,
) -> None:
    if not is_main:
        return
    dataset = ESMFold2FrozenFeatureDataset(
        manifest=manifest,
        cache_index=cache_index,
        require_cache=True,
        download=True,
    )
    for row in dataset.rows:
        dataset._mmcif_path(row)  # noqa: SLF001 - rank-0 warmup prevents download races.


def _startup_summary(
    args: argparse.Namespace,
    *,
    dist_env: DistributedEnv,
    resolved_precision: str,
) -> dict[str, Any]:
    return {
        "output_dir": args.output_dir,
        "stage": args.stage,
        "precision": resolved_precision,
        "world_size": dist_env.world_size,
        "global_batch_size": args.batch_size
        * args.gradient_accumulation_steps
        * dist_env.world_size,
        "manifest": args.manifest,
        "cache_index": args.cache_index,
        "wandb": {
            "mode": args.wandb_mode,
            "project": args.wandb_project,
            "run_name": args.wandb_run_name,
        },
    }


def _initialize_distributed_if_needed(
    args: argparse.Namespace,
    dist_env: DistributedEnv,
    *,
    use_ddp: bool,
    device: torch.device,
) -> str | None:
    if not use_ddp:
        return None
    normalize_distributed_env_for_init(dist_env)
    backend = resolve_dist_backend(args.dist_backend, device.type)
    if backend == "nccl":
        if device.type != "cuda":
            raise SystemExit("NCCL distributed training requires a CUDA device.")
        torch.cuda.set_device(device)
    if dist_env.master_addr is None and os.environ.get("MASTER_ADDR") is None:
        raise SystemExit(
            "MASTER_ADDR is required for distributed initialization. "
            "torchrun sets this automatically; Slurm srun launches should set it."
        )
    init_kwargs: dict[str, Any] = {"backend": backend, "init_method": "env://"}
    if backend == "nccl":
        init_kwargs["device_id"] = device
    try:
        torch.distributed.init_process_group(**init_kwargs)
    except TypeError:
        init_kwargs.pop("device_id", None)
        torch.distributed.init_process_group(**init_kwargs)
    return backend


def _select_device(args: argparse.Namespace, dist_env: DistributedEnv, *, use_ddp: bool) -> torch.device:
    if use_ddp and torch.cuda.is_available() and args.device.startswith("cuda"):
        return torch.device("cuda", dist_env.local_rank)
    return torch.device(args.device)


def _finish_wandb(wandb_run: Any | None) -> None:
    if wandb_run is not None:
        wandb_run.finish()


def run_training(args: argparse.Namespace) -> None:
    config = ESMFold2FrozenFeatureConfig()
    dist_env = detect_distributed_env()
    use_ddp = distributed_requested(args.distributed, dist_env)
    device = _select_device(args, dist_env, use_ddp=use_ddp)
    backend: str | None = None
    wandb_run: Any | None = None

    try:
        backend = _initialize_distributed_if_needed(
            args, dist_env, use_ddp=use_ddp, device=device
        )
        resolved_precision = resolve_precision(args.precision, device_type=device.type)
        scaler_enabled = uses_grad_scaler(resolved_precision, device.type)

        if dist_env.is_main:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            print(
                json.dumps(
                    _startup_summary(
                        args,
                        dist_env=dist_env,
                        resolved_precision=resolved_precision,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )

        if use_ddp:
            _prepare_rank0_mmcif_downloads(
                manifest=args.manifest,
                cache_index=args.cache_index,
                is_main=dist_env.is_main,
            )
            barrier_if_distributed()

        model = load_esmfold2_for_training(
            args.model_checkpoint, device, resolved_precision
        )
        _attach_training_forward(model)
        optimizer, optimizer_param_names = build_optimizer(model, args.stage, config)
        assert_no_esmc_parameters(optimizer_param_names)
        scaler = _make_grad_scaler(scaler_enabled)

        start_global_step = 0
        start_epoch = 0
        if args.resume is not None:
            start_global_step, start_epoch = _restore_checkpoint(
                args.resume, model, optimizer, scaler, device
            )
            barrier_if_distributed()

        if use_ddp:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[dist_env.local_rank] if device.type == "cuda" else None,
                output_device=dist_env.local_rank if device.type == "cuda" else None,
                find_unused_parameters=args.ddp_find_unused_parameters,
            )

        dataset = ESMFold2FrozenFeatureDataset(
            manifest=args.manifest,
            cache_index=args.cache_index,
            require_cache=True,
            download=dist_env.is_main or not use_ddp,
        )
        loader, sampler = create_train_dataloader(
            dataset, args, dist_env, use_ddp=use_ddp
        )
        run_config = build_run_config(
            args,
            dist_env=dist_env,
            device=device,
            backend=backend,
            resolved_precision=resolved_precision,
            config=config,
        )
        wandb_run = init_wandb_run(args, dist_env=dist_env, run_config=run_config)

        output_dir = Path(args.output_dir)
        log_mode = "a" if args.resume else "w"
        effective_global_batch_size = (
            args.batch_size * args.gradient_accumulation_steps * dist_env.world_size
        )
        global_step = start_global_step
        epoch = start_epoch
        accumulation_index = 0
        loss_sums: dict[str, float] = {}
        local_samples = 0.0
        local_tokens = 0.0
        local_atoms = 0.0
        step_start_time = time.perf_counter()

        model.train()
        optimizer.zero_grad(set_to_none=True)
        progress = maybe_tqdm(
            None,
            enabled=dist_env.is_main,
            total=args.max_steps,
            initial=min(global_step, args.max_steps),
            desc=f"stage {args.stage}",
            dynamic_ncols=True,
        )
        with JsonlLogger(
            output_dir / "train_log.jsonl", enabled=dist_env.is_main, mode=log_mode
        ) as jsonl:
            while global_step < args.max_steps:
                if sampler is not None:
                    sampler.set_epoch(epoch)
                for batch in loader:
                    if accumulation_index == 0:
                        step_start_time = time.perf_counter()
                        loss_sums = {}
                        local_samples = 0.0
                        local_tokens = 0.0
                        local_atoms = 0.0
                        lr = learning_rate_for_step(global_step, args.stage, config)
                        set_optimizer_lr(optimizer, lr)
                        optimizer.zero_grad(set_to_none=True)
                        if device.type == "cuda":
                            torch.cuda.reset_peak_memory_stats(device)

                    batch = _move_batch(batch, device)
                    counts = _batch_counts(batch)
                    local_samples += counts["samples"]
                    local_tokens += counts["tokens"]
                    local_atoms += counts["atoms"]
                    lm_hidden_states = batch["lm_hidden_states"]
                    final_accumulation = (
                        accumulation_index + 1
                    ) >= args.gradient_accumulation_steps
                    sync_context = (
                        model.no_sync()
                        if use_ddp and not final_accumulation
                        else nullcontext()
                    )
                    with sync_context:
                        outputs = _call_training_forward(
                            model,
                            batch=batch,
                            lm_hidden_states=lm_hidden_states,
                            stage=args.stage,
                            seed=args.seed
                            + global_step * args.gradient_accumulation_steps
                            + accumulation_index,
                            num_recycles_for_test=args.num_recycles_for_test,
                            confidence_num_sampling_steps=args.confidence_sampling_steps_for_test,
                            config=config,
                        )
                        losses = compute_esmfold2_loss(
                            outputs, batch, args.stage, config
                        )
                        loss_for_backward = (
                            losses["loss"] / args.gradient_accumulation_steps
                        )
                        if scaler.is_enabled():
                            scaler.scale(loss_for_backward).backward()
                        else:
                            loss_for_backward.backward()

                    for key, value in losses.items():
                        loss_sums[key] = loss_sums.get(key, 0.0) + float(
                            value.detach().float().cpu()
                        )

                    accumulation_index += 1
                    if accumulation_index < args.gradient_accumulation_steps:
                        continue

                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    local_grad_norm = _grad_norm(model)
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                        grad_scale = float(scaler.get_scale())
                    else:
                        optimizer.step()
                        grad_scale = None

                    global_step += 1
                    accumulation_microsteps = max(1, accumulation_index)
                    accumulation_index = 0

                    local_loss_avg = {
                        key: value / accumulation_microsteps
                        for key, value in loss_sums.items()
                    }
                    reduced_losses = reduce_metrics(local_loss_avg, average=True)
                    reduced_grad = reduce_metrics(
                        {"opt/grad_norm": local_grad_norm}, average=True
                    )
                    reduced_grad_scale = (
                        reduce_metrics({"opt/grad_scale": grad_scale}, average=True)[
                            "opt/grad_scale"
                        ]
                        if grad_scale is not None
                        else None
                    )
                    reduced_data_per_rank = reduce_metrics(
                        {
                            "data/batch_size_per_rank": local_samples
                            / accumulation_microsteps,
                            "data/tokens_per_rank": local_tokens,
                            "data/atoms_per_rank": local_atoms,
                        },
                        average=True,
                    )
                    global_data = sum_metrics(
                        {
                            "data/tokens_global": local_tokens,
                            "data/atoms_global": local_atoms,
                            "samples_global": local_samples,
                        }
                    )
                    step_time = time.perf_counter() - step_start_time
                    reduced_perf_time = max_metrics(
                        {"perf/step_time_sec": step_time}
                    )
                    step_time_global = max(
                        1e-12, reduced_perf_time["perf/step_time_sec"]
                    )
                    gpu_memory = {
                        "perf/gpu_max_allocated_gb": 0.0,
                        "perf/gpu_max_reserved_gb": 0.0,
                    }
                    if device.type == "cuda":
                        gpu_memory = {
                            "perf/gpu_max_allocated_gb": torch.cuda.max_memory_allocated(
                                device
                            )
                            / 1e9,
                            "perf/gpu_max_reserved_gb": torch.cuda.max_memory_reserved(
                                device
                            )
                            / 1e9,
                        }
                    reduced_gpu_memory = max_metrics(gpu_memory)
                    samples_global = global_data.pop("samples_global")
                    perf_metrics = {
                        "perf/step_time_sec": step_time_global,
                        "perf/samples_per_sec_global": samples_global
                        / step_time_global,
                        "perf/tokens_per_sec_global": global_data[
                            "data/tokens_global"
                        ]
                        / step_time_global,
                        "perf/atoms_per_sec_global": global_data[
                            "data/atoms_global"
                        ]
                        / step_time_global,
                    }
                    perf_metrics.update(reduced_gpu_memory)
                    data_metrics = dict(reduced_data_per_rank)
                    data_metrics.update(global_data)
                    payload = build_training_metric_payload(
                        reduced_losses,
                        lr=lr,
                        weight_decay=_optimizer_weight_decay(optimizer),
                        grad_norm=reduced_grad["opt/grad_norm"],
                        grad_scale=reduced_grad_scale,
                        data_metrics=data_metrics,
                        perf_metrics=perf_metrics,
                        dist_env=dist_env,
                        global_step=global_step,
                        epoch=epoch,
                        effective_global_batch_size=effective_global_batch_size,
                    )

                    if dist_env.is_main:
                        progress.update(1)
                        progress.set_postfix(format_tqdm_postfix(payload))
                        if (
                            args.log_every_n_steps > 0
                            and global_step % args.log_every_n_steps == 0
                        ):
                            jsonl.write(_jsonl_record(payload, global_step))
                            _write_loss_breakdown(
                                output_dir,
                                payload,
                                global_step=global_step,
                                enabled=True,
                            )
                    _log_wandb(
                        wandb_run,
                        payload,
                        global_step=global_step,
                        every_n_steps=args.wandb_log_every_n_steps,
                    )

                    if (
                        args.save_every_n_steps > 0
                        and global_step % args.save_every_n_steps == 0
                    ):
                        save_checkpoint(
                            output_dir / f"checkpoint_step_{global_step}.pt",
                            model,
                            optimizer,
                            scaler=scaler,
                            stage=args.stage,
                            epoch=epoch,
                            global_step=global_step,
                            args=args,
                            config=config,
                            optimizer_param_names=optimizer_param_names,
                            resolved_precision=resolved_precision,
                            is_main=dist_env.is_main,
                        )
                        barrier_if_distributed()

                    if global_step >= args.max_steps:
                        break
                epoch += 1

        progress.close()
        save_checkpoint(
            output_dir / "checkpoint_last.pt",
            model,
            optimizer,
            scaler=scaler,
            stage=args.stage,
            epoch=epoch,
            global_step=global_step,
            args=args,
            config=config,
            optimizer_param_names=optimizer_param_names,
            resolved_precision=resolved_precision,
            is_main=dist_env.is_main,
        )
        barrier_if_distributed()
    finally:
        _finish_wandb(wandb_run)
        destroy_process_group_if_initialized()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune ESMFold2 from precomputed frozen ESMC hidden states."
    )
    parser.add_argument("--write-sample-manifest", default=None)
    parser.add_argument("--manifest", default="data/sample_nanobody10/manifest.csv")
    parser.add_argument("--cache-index", default="data/sample_nanobody10/cache_index.jsonl")
    parser.add_argument("--model-checkpoint", default=None)
    parser.add_argument("--output-dir", default="runs/esmfold2_nanobody_smoke")
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=["auto", "fp32", "float32", "fp16", "bf16", "bfloat16"],
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-recycles-for-test", type=int, default=None)
    parser.add_argument("--confidence-sampling-steps-for-test", type=int, default=None)

    parser.add_argument(
        "--wandb-mode",
        choices=["disabled", "offline", "online"],
        default=os.environ.get("WANDB_MODE", "disabled"),
    )
    parser.add_argument(
        "--wandb-project", default="esmfold2-nanobody-finetune"
    )
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    parser.add_argument("--wandb-log-every-n-steps", type=int, default=1)
    parser.add_argument("--log-every-n-steps", type=int, default=1)

    parser.add_argument(
        "--distributed", choices=["auto", "none", "ddp"], default="auto"
    )
    parser.add_argument(
        "--dist-backend", choices=["nccl", "gloo", "auto"], default="auto"
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--ddp-find-unused-parameters", type=_str_to_bool, default=False
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", type=_str_to_bool, default=True)
    parser.add_argument("--persistent-workers", type=_str_to_bool, default=True)
    parser.add_argument("--save-every-n-steps", type=int, default=0)
    parser.add_argument("--resume", default=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.gradient_accumulation_steps < 1:
        raise SystemExit("--gradient-accumulation-steps must be >= 1")
    if args.write_sample_manifest:
        write_sample_manifest(args.write_sample_manifest)
        return
    if args.model_checkpoint is None:
        raise SystemExit("--model-checkpoint is required unless --write-sample-manifest is used")
    run_training(args)


if __name__ == "__main__":
    main()
