#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from esm.models.esmc import ESMC
from esm.training.esmfold2_frozen_features.dataset import (
    download_mmcif,
    load_selected_chain,
    read_manifest,
)


def _dtype_from_arg(name: str) -> torch.dtype:
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    if name in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(f"unsupported dtype {name!r}")


def compute_hidden_states(model: ESMC, sequence: str, dtype: torch.dtype) -> torch.Tensor:
    tokens = model._tokenize([sequence])
    with torch.no_grad():
        initial = model.embed(tokens)
        output = model(sequence_tokens=tokens)
    if output.hidden_states is None:
        raise RuntimeError("ESMC did not return hidden states")
    hidden = output.hidden_states.permute(1, 2, 0, 3)
    initial = initial.unsqueeze(2)
    hidden = torch.cat([initial, hidden], dim=2)
    hidden = hidden[:, 1:-1]
    return hidden[0].to(dtype=dtype).cpu()


def _index_path_string(hidden_path: Path, cache_index: Path) -> str:
    hidden_path = hidden_path.resolve()
    cache_dir = cache_index.parent.resolve()
    try:
        return str(hidden_path.relative_to(cache_dir))
    except ValueError:
        return os.path.relpath(hidden_path, cache_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute all-layer ESMC hidden states for ESMFold2 frozen-feature fine-tuning."
    )
    parser.add_argument("--manifest", default="data/sample_nanobody10/manifest.csv")
    parser.add_argument("--out-dir", default="data/sample_nanobody10/esmc_cache")
    parser.add_argument("--model", default="esmc_6b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--cache-index", default=None)
    parser.add_argument("--mmcif-dir", default=None)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_index = Path(args.cache_index) if args.cache_index else out_dir.parent / "cache_index.jsonl"
    mmcif_dir = Path(args.mmcif_dir) if args.mmcif_dir else manifest.parent / "mmcif"
    dtype = _dtype_from_arg(args.dtype)
    device = torch.device(args.device)

    model = ESMC.from_pretrained(args.model, device=device, use_flash_attn=False).eval()
    model = model.to(dtype=dtype if device.type != "cpu" else torch.float32)

    rows = read_manifest(manifest)
    sequence_to_path: dict[str, Path] = {}
    sequence_to_shape: dict[str, list[int]] = {}
    records = []
    for row in rows:
        mmcif_path = download_mmcif(row, mmcif_dir)
        chain, _, _ = load_selected_chain(row, mmcif_path, keep_source=False)
        sequence = chain.sequence
        if sequence not in sequence_to_path:
            hidden = compute_hidden_states(model, sequence, dtype=dtype)
            hidden_path = out_dir / f"{row.example_id}_{row.pdb_id}_{chain.chain_id}.pt"
            torch.save(hidden, hidden_path)
            sequence_to_path[sequence] = hidden_path
            sequence_to_shape[sequence] = list(hidden.shape)
        hidden_path = sequence_to_path[sequence]
        shape = sequence_to_shape[sequence]
        records.append(
            {
                "example_id": row.example_id,
                "sequence": sequence,
                "hidden_states_path": _index_path_string(hidden_path, cache_index),
                "shape": shape,
                "dtype": "bfloat16" if dtype is torch.bfloat16 else str(dtype).replace("torch.", ""),
                "model": args.model,
            }
        )

    cache_index.parent.mkdir(parents=True, exist_ok=True)
    with cache_index.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
