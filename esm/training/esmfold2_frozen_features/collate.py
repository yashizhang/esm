from __future__ import annotations

from typing import Any

import torch

TOKEN_KEYS_1D = {
    "token_index",
    "residue_index",
    "asym_id",
    "entity_id",
    "sym_id",
    "mol_type",
    "res_type",
    "input_ids",
    "token_attention_mask",
    "pocket_feature",
    "deletion_mean",
    "distogram_atom_idx",
}

TOKEN_KEYS_2D_LAST = {"frames_idx"}
PAIR_KEYS = {"token_bonds", "disto_cond", "disto_cond_mask"}
ATOM_KEYS_1D = {
    "ref_element",
    "ref_charge",
    "ref_space_uid",
    "atom_attention_mask",
    "atom_to_token",
    "is_resolved",
    "gt_atom_mask",
    "atom_loss_weight",
}
ATOM_KEYS_2D_LAST = {"ref_pos", "ref_atom_name_chars", "gt_atom_coords"}
MSA_KEYS = {"msa", "has_deletion", "deletion_value", "msa_attention_mask"}
SCALAR_TENSOR_KEYS = {"resolution"}


def _pad_1d(x: torch.Tensor, length: int, value: int | float | bool = 0) -> torch.Tensor:
    out = x.new_full((length,), value)
    out[: x.shape[0]] = x
    return out


def _pad_2d_last(x: torch.Tensor, length: int, value: int | float | bool = 0) -> torch.Tensor:
    out = x.new_full((length, x.shape[1]), value)
    out[: x.shape[0], :] = x
    return out


def _pad_pair(x: torch.Tensor, length: int) -> torch.Tensor:
    out = x.new_zeros((length, length, *x.shape[2:]))
    out[: x.shape[0], : x.shape[1], ...] = x
    return out


def _pad_msa(x: torch.Tensor, depth: int, length: int) -> torch.Tensor:
    out = x.new_zeros((depth, length, *x.shape[2:]))
    out[: x.shape[0], : x.shape[1], ...] = x
    return out


def _pad_lm_hidden_states(x: torch.Tensor, length: int) -> torch.Tensor:
    out = x.new_zeros((length, x.shape[1], x.shape[2]))
    out[: x.shape[0], :, :] = x
    return out


def collate_esmfold2_frozen_features(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty batch")

    max_tokens = max(int(sample["token_attention_mask"].shape[0]) for sample in samples)
    max_atoms = max(int(sample["atom_attention_mask"].shape[0]) for sample in samples)
    max_msa_depth = max(int(sample.get("msa", torch.empty(0, 0)).shape[0]) for sample in samples)

    batch: dict[str, Any] = {}
    keys = set().union(*(sample.keys() for sample in samples))
    for key in sorted(keys):
        values = [sample[key] for sample in samples if key in sample]
        if len(values) != len(samples):
            continue
        first = values[0]
        if key in TOKEN_KEYS_1D:
            pad_value = False if first.dtype == torch.bool else 0
            batch[key] = torch.stack([_pad_1d(v, max_tokens, pad_value) for v in values])
        elif key in TOKEN_KEYS_2D_LAST:
            batch[key] = torch.stack([_pad_2d_last(v, max_tokens, 0) for v in values])
        elif key in PAIR_KEYS:
            batch[key] = torch.stack([_pad_pair(v, max_tokens) for v in values])
        elif key in ATOM_KEYS_1D:
            pad_value = False if first.dtype == torch.bool else 0
            batch[key] = torch.stack([_pad_1d(v, max_atoms, pad_value) for v in values])
        elif key in ATOM_KEYS_2D_LAST:
            batch[key] = torch.stack([_pad_2d_last(v, max_atoms, 0) for v in values])
        elif key in MSA_KEYS:
            batch[key] = torch.stack([_pad_msa(v, max_msa_depth, max_tokens) for v in values])
        elif key == "lm_hidden_states":
            batch[key] = torch.stack([_pad_lm_hidden_states(v, max_tokens) for v in values])
        elif key == "gt_coords":
            squeezed = [v.squeeze(0) if v.ndim == 3 and v.shape[0] == 1 else v for v in values]
            batch[key] = torch.stack([_pad_2d_last(v, max_atoms, 0) for v in squeezed])
        elif key in SCALAR_TENSOR_KEYS and isinstance(first, torch.Tensor):
            batch[key] = torch.stack(values)
        elif isinstance(first, torch.Tensor):
            batch[key] = torch.stack(values)
        else:
            batch[key] = values
    return batch
