from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class DistributedEnv:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    local_world_size: int | None = None
    source: str = "single"
    master_addr: str | None = None
    master_port: str | None = None

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1


def _env_int(environ: Mapping[str, str], key: str, default: int | None = None) -> int:
    value = environ.get(key)
    if value is None:
        if default is None:
            raise KeyError(key)
        return default
    return int(value)


def _parse_slurm_local_world_size(value: str | None) -> int | None:
    if not value:
        return None
    match = re.match(r"(\d+)", value)
    return int(match.group(1)) if match else None


def detect_distributed_env(
    environ: Mapping[str, str] | None = None,
) -> DistributedEnv:
    env = os.environ if environ is None else environ
    if "RANK" in env and "WORLD_SIZE" in env:
        return DistributedEnv(
            rank=_env_int(env, "RANK"),
            local_rank=_env_int(env, "LOCAL_RANK", 0),
            world_size=_env_int(env, "WORLD_SIZE"),
            local_world_size=_env_int(env, "LOCAL_WORLD_SIZE", 0) or None,
            source="torchrun",
            master_addr=env.get("MASTER_ADDR"),
            master_port=env.get("MASTER_PORT"),
        )
    if "SLURM_PROCID" in env and "SLURM_NTASKS" in env:
        return DistributedEnv(
            rank=_env_int(env, "SLURM_PROCID"),
            local_rank=_env_int(env, "SLURM_LOCALID", 0),
            world_size=_env_int(env, "SLURM_NTASKS"),
            local_world_size=_parse_slurm_local_world_size(
                env.get("SLURM_NTASKS_PER_NODE")
            ),
            source="slurm",
            master_addr=env.get("MASTER_ADDR"),
            master_port=env.get("MASTER_PORT"),
        )
    return DistributedEnv()


def distributed_requested(mode: str, env: DistributedEnv) -> bool:
    if mode == "none":
        return False
    if mode == "ddp":
        return env.world_size > 1
    if mode == "auto":
        return env.world_size > 1
    raise ValueError(f"unknown distributed mode: {mode}")


def normalize_distributed_env_for_init(env: DistributedEnv) -> None:
    os.environ.setdefault("RANK", str(env.rank))
    os.environ.setdefault("LOCAL_RANK", str(env.local_rank))
    os.environ.setdefault("WORLD_SIZE", str(env.world_size))
    os.environ.setdefault("MASTER_PORT", env.master_port or "29500")


def resolve_dist_backend(requested: str, device_type: str) -> str:
    if requested == "auto":
        return "nccl" if device_type == "cuda" else "gloo"
    if requested not in {"nccl", "gloo"}:
        raise ValueError(f"unknown distributed backend: {requested}")
    return requested


def _normalize_precision_name(precision: str) -> str:
    aliases = {
        "float32": "fp32",
        "bfloat16": "bf16",
    }
    return aliases.get(precision, precision)


def resolve_precision(precision: str, device_type: str | None = None) -> str:
    requested = _normalize_precision_name(precision)
    if requested not in {"auto", "fp32", "fp16", "bf16"}:
        raise ValueError(f"unknown precision: {precision}")

    if device_type is None:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    if device_type != "cuda":
        return "fp32" if requested == "auto" else requested

    if requested == "auto":
        major, _ = torch.cuda.get_device_capability()
        return "bf16" if major >= 8 else "fp16"
    if requested == "bf16":
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            raise ValueError(
                "bf16 was requested, but this CUDA device has capability "
                f"{major}.x; use fp16 on V100-class GPUs."
            )
    return requested


def autocast_dtype_for_precision(precision: str) -> torch.dtype | None:
    resolved = _normalize_precision_name(precision)
    if resolved == "fp16":
        return torch.float16
    if resolved == "bf16":
        return torch.bfloat16
    return None


def uses_grad_scaler(precision: str, device_type: str) -> bool:
    return _normalize_precision_name(precision) == "fp16" and device_type == "cuda"


def _metric_device(metrics: Mapping[str, torch.Tensor | float | int]) -> torch.device:
    for value in metrics.values():
        if isinstance(value, torch.Tensor):
            return value.device
    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_backend() == "nccl"
    ):
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _as_float(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu())
    return float(value)


def _reduce_metrics_with_op(
    metrics: Mapping[str, torch.Tensor | float | int],
    op: torch.distributed.ReduceOp,
    *,
    average: bool = False,
) -> dict[str, float]:
    if not metrics:
        return {}
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return {key: _as_float(value) for key, value in metrics.items()}

    keys = list(metrics)
    device = _metric_device(metrics)
    values = torch.tensor(
        [_as_float(metrics[key]) for key in keys], dtype=torch.float64, device=device
    )
    torch.distributed.all_reduce(values, op=op)
    if average:
        values /= torch.distributed.get_world_size()
    return {key: float(value.cpu()) for key, value in zip(keys, values)}


def reduce_metrics(
    metrics: Mapping[str, torch.Tensor | float | int], average: bool = True
) -> dict[str, float]:
    return _reduce_metrics_with_op(
        metrics, torch.distributed.ReduceOp.SUM, average=average
    )


def sum_metrics(
    metrics: Mapping[str, torch.Tensor | float | int],
) -> dict[str, float]:
    return _reduce_metrics_with_op(
        metrics, torch.distributed.ReduceOp.SUM, average=False
    )


def max_metrics(
    metrics: Mapping[str, torch.Tensor | float | int],
) -> dict[str, float]:
    return _reduce_metrics_with_op(
        metrics, torch.distributed.ReduceOp.MAX, average=False
    )


def barrier_if_distributed() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def destroy_process_group_if_initialized() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
