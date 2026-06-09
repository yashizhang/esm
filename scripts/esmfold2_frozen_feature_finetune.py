#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from itertools import cycle
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAMPLE_MANIFEST_TEXT = """example_id,pdb_id,mmcif_url,label_asym_id,auth_asym_id,chain_match_policy,expected_entity_name_regex,task,split
nanobody_001,1MEL,https://files.rcsb.org/download/1MEL.cif,A,A,label_asym_id_first_then_auth_asym_id,"(?i)(nanobody|vhh|single-domain|vh single-domain|antibody vhh)",monomer_vhh_train,train
nanobody_002,1MEL,https://files.rcsb.org/download/1MEL.cif,B,B,label_asym_id_first_then_auth_asym_id,"(?i)(nanobody|vhh|single-domain|vh single-domain|antibody vhh)",monomer_vhh_train,train
nanobody_003,1I3V,https://files.rcsb.org/download/1I3V.cif,A,A,label_asym_id_first_then_auth_asym_id,"(?i)(nanobody|vhh|single-domain|vh single-domain|antibody vhh)",monomer_vhh_train,train
nanobody_004,1I3V,https://files.rcsb.org/download/1I3V.cif,B,B,label_asym_id_first_then_auth_asym_id,"(?i)(nanobody|vhh|single-domain|vh single-domain|antibody vhh)",monomer_vhh_train,train
nanobody_005,6JB9,https://files.rcsb.org/download/6JB9.cif,A,A,label_asym_id_first_then_auth_asym_id,"(?i)(nanobody|vhh|single-domain|vh single-domain|antibody vhh)",monomer_vhh_train,train
nanobody_006,5U64,https://files.rcsb.org/download/5U64.cif,A,B,label_asym_id_first_then_auth_asym_id,"(?i)(nanobody|vhh|single-domain|vh single-domain|antibody vhh)",monomer_vhh_train,train
nanobody_007,4KRN,https://files.rcsb.org/download/4KRN.cif,A,A,label_asym_id_first_then_auth_asym_id,"(?i)(nanobody|vhh|single-domain|vh single-domain|antibody vhh)",monomer_vhh_train,train
nanobody_008,4KRL,https://files.rcsb.org/download/4KRL.cif,A,B,label_asym_id_first_then_auth_asym_id,"(?i)(nanobody|vhh|single-domain|vh single-domain|antibody vhh)",monomer_vhh_train,train
nanobody_009,4KRO,https://files.rcsb.org/download/4KRO.cif,B,B,label_asym_id_first_then_auth_asym_id,"(?i)(nanobody|vhh|single-domain|vh single-domain|antibody vhh)",monomer_vhh_train,train
nanobody_010,4KRP,https://files.rcsb.org/download/4KRP.cif,B,B,label_asym_id_first_then_auth_asym_id,"(?i)(nanobody|vhh|single-domain|vh single-domain|antibody vhh)",monomer_vhh_train,train
"""


def _jsonable_loss_dict(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in losses.items()}


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def run_training(args: argparse.Namespace) -> None:
    global torch
    import torch
    from torch.utils.data import DataLoader

    from esm.training.esmfold2_frozen_features.collate import (
        collate_esmfold2_frozen_features,
    )
    from esm.training.esmfold2_frozen_features.config import ESMFold2FrozenFeatureConfig
    from esm.training.esmfold2_frozen_features.dataset import ESMFold2FrozenFeatureDataset
    from esm.training.esmfold2_frozen_features.losses import compute_esmfold2_loss
    from esm.training.esmfold2_frozen_features.modeling import (
        assert_no_esmc_parameters,
        build_optimizer,
        learning_rate_for_step,
        load_esmfold2_for_training,
        set_optimizer_lr,
    )

    device = torch.device(args.device)
    config = ESMFold2FrozenFeatureConfig()
    model = load_esmfold2_for_training(args.model_checkpoint, device, args.precision)
    optimizer, optimizer_param_names = build_optimizer(model, args.stage, config)
    assert_no_esmc_parameters(optimizer_param_names)

    dataset = ESMFold2FrozenFeatureDataset(
        manifest=args.manifest,
        cache_index=args.cache_index,
        require_cache=True,
        download=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_esmfold2_frozen_features,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"

    model.train()
    data_iter = cycle(loader)
    with log_path.open("w") as log_handle:
        for step in range(args.max_steps):
            lr = learning_rate_for_step(step, args.stage, config)
            set_optimizer_lr(optimizer, lr)
            batch = _move_batch(next(data_iter), device)
            lm_hidden_states = batch["lm_hidden_states"]

            optimizer.zero_grad(set_to_none=True)
            outputs = model.forward_train_from_precomputed_lm(
                batch=batch,
                lm_hidden_states=lm_hidden_states,
                stage=args.stage,
                seed=args.seed + step,
                num_recycles_for_test=args.num_recycles_for_test,
                confidence_num_sampling_steps=args.confidence_sampling_steps_for_test,
                config=config,
            )
            losses = compute_esmfold2_loss(outputs, batch, args.stage, config)
            losses["loss"].backward()
            optimizer.step()

            loss_record = _jsonable_loss_dict(losses)
            loss_record.update({"step": step, "lr": lr})
            log_handle.write(json.dumps(loss_record) + "\n")
            log_handle.flush()
            (output_dir / f"loss_breakdown_step_{step}.json").write_text(
                json.dumps(loss_record, indent=2) + "\n"
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "stage": args.stage,
            "max_steps": args.max_steps,
            "optimizer_param_names": optimizer_param_names,
        },
        output_dir / "checkpoint_last.pt",
    )


def write_sample_manifest(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SAMPLE_MANIFEST_TEXT)
    return path


def main() -> None:
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
    parser.add_argument("--precision", choices=["bf16", "bfloat16", "fp32", "float32"], default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-recycles-for-test", type=int, default=None)
    parser.add_argument("--confidence-sampling-steps-for-test", type=int, default=None)
    args = parser.parse_args()

    if args.write_sample_manifest:
        write_sample_manifest(args.write_sample_manifest)
        return
    if args.model_checkpoint is None:
        raise SystemExit("--model-checkpoint is required unless --write-sample-manifest is used")
    run_training(args)


if __name__ == "__main__":
    main()
