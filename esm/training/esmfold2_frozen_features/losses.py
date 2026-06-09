from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from esm.training.esmfold2_frozen_features.config import (
    ESMFold2FrozenFeatureConfig,
    get_stage_config,
)
from esm.utils.structure.protein_structure import compute_alignment_tensors


LOSS_KEYS = (
    "loss",
    "loss_struct",
    "loss_mse",
    "loss_slddt",
    "loss_bond",
    "loss_dist",
    "loss_conf",
    "loss_plddt",
    "loss_pde",
    "loss_resolved",
    "loss_pae",
)


def _zero(device: torch.device) -> Tensor:
    return torch.zeros((), device=device)


def _aligned_mobile(mobile: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    _, centroid_mobile, _, centroid_target, rotation_matrix, _ = compute_alignment_tensors(
        mobile=mobile.float(), target=target.float(), atom_exists_mask=mask.bool()
    )
    return torch.matmul(
        mobile.float() - centroid_mobile.detach(), rotation_matrix.detach()
    ) + centroid_target.detach()


def _masked_mean(value: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
    return (value * mask.to(value.dtype)).sum() / mask.to(value.dtype).sum().clamp(min=eps)


def _per_sample_masked_mean(value: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
    dims = tuple(range(1, value.ndim))
    mask_f = mask.to(value.dtype)
    while mask_f.ndim < value.ndim:
        mask_f = mask_f.unsqueeze(-1)
    return (value * mask_f).sum(dim=dims) / mask_f.sum(dim=dims).clamp(min=eps)


def diffusion_mse_loss(outputs: dict[str, Tensor], batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
    pred = outputs["x_denoised"].float()
    target = batch["gt_atom_coords"].to(pred.device).float()
    mask = (batch["gt_atom_mask"].to(pred.device).bool() & batch["atom_attention_mask"].to(pred.device).bool())
    weights = batch.get("atom_loss_weight")
    if weights is None:
        weights = torch.ones(mask.shape, device=pred.device, dtype=pred.dtype)
    else:
        weights = weights.to(pred.device, dtype=pred.dtype)
    aligned = _aligned_mobile(pred, target, mask)
    sq = (aligned - target).square().sum(dim=-1)
    weighted_mask = mask.to(pred.dtype) * weights
    per_sample = (sq * weighted_mask).sum(dim=-1) / weighted_mask.sum(dim=-1).clamp(min=1e-8)
    return per_sample.mean(), per_sample


def _atom_pair_cutoffs(batch: dict[str, Any], device: torch.device) -> Tensor:
    atom_to_token = batch["atom_to_token"].to(device).long()
    mol_type = batch["mol_type"].to(device).long()
    atom_mol = torch.gather(mol_type, 1, atom_to_token.clamp_min(0))
    is_na = (atom_mol == 1) | (atom_mol == 2)
    na_pair = is_na[:, :, None] & is_na[:, None, :]
    return torch.where(
        na_pair,
        torch.full(na_pair.shape, 30.0, device=device),
        torch.full(na_pair.shape, 15.0, device=device),
    )


def smooth_lddt_loss(
    outputs: dict[str, Tensor],
    batch: dict[str, Any],
    config: ESMFold2FrozenFeatureConfig,
) -> Tensor:
    pred = outputs["x_denoised"].float()
    target = batch["gt_atom_coords"].to(pred.device).float()
    mask = (batch["gt_atom_mask"].to(pred.device).bool() & batch["atom_attention_mask"].to(pred.device).bool())
    d_pred = torch.cdist(pred, pred, compute_mode="donot_use_mm_for_euclid_dist")
    d_target = torch.cdist(target, target, compute_mode="donot_use_mm_for_euclid_dist")
    pair_mask = mask[:, :, None] & mask[:, None, :]
    eye = torch.eye(mask.shape[1], device=pred.device, dtype=torch.bool).unsqueeze(0)
    cutoffs = _atom_pair_cutoffs(batch, pred.device)
    pair_mask = pair_mask & ~eye & (d_target < cutoffs)
    delta = (d_pred - d_target).abs()
    thresholds = torch.tensor([0.5, 1.0, 2.0, 4.0], device=pred.device, dtype=delta.dtype)
    score = torch.sigmoid(thresholds.view(1, 1, 1, 4) - delta.unsqueeze(-1)).mean(dim=-1)
    return 1.0 - _masked_mean(score, pair_mask)


def _bond_fields(bonds: Any, device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if isinstance(bonds, dict):
        atom_i = bonds["atom_i"].to(device).long()
        atom_j = bonds["atom_j"].to(device).long()
        target_length = bonds["target_length"].to(device).float()
        mask = bonds.get("mask")
        if mask is None:
            mask = torch.ones_like(target_length, dtype=torch.bool, device=device)
        else:
            mask = mask.to(device).bool()
        return atom_i, atom_j, target_length, mask

    if not isinstance(bonds, Tensor):
        raise TypeError("polymer_ligand_bonds must be a dict or tensor")
    bonds = bonds.to(device)
    if bonds.ndim != 3 or bonds.shape[-1] not in {3, 4}:
        raise ValueError(
            "polymer_ligand_bonds tensor must have shape [B, N, 3 or 4] with "
            "columns [atom_i, atom_j, target_length, optional_mask]"
        )
    atom_i = bonds[..., 0].long()
    atom_j = bonds[..., 1].long()
    target_length = bonds[..., 2].float()
    if bonds.shape[-1] == 4:
        mask = bonds[..., 3].bool()
    else:
        mask = torch.ones_like(target_length, dtype=torch.bool)
    return atom_i, atom_j, target_length, mask


def bond_loss(
    outputs: dict[str, Tensor], batch: dict[str, Any], device: torch.device
) -> tuple[Tensor, Tensor]:
    bsz = outputs["x_denoised"].shape[0]
    bonds = batch.get("polymer_ligand_bonds")
    if bonds is None:
        per_sample = torch.zeros(bsz, device=device)
        return _zero(device), per_sample
    atom_i, atom_j, target_length, mask = _bond_fields(bonds, device)
    if not mask.any():
        per_sample = torch.zeros(bsz, device=device)
        return _zero(device), per_sample

    pred = outputs["x_denoised"].float()
    atom_mask = batch["atom_attention_mask"].to(device).bool()
    if atom_i.shape != atom_j.shape or atom_i.shape != target_length.shape:
        raise ValueError("polymer_ligand_bonds atom_i, atom_j and target_length shapes must match")
    if atom_i.shape[0] != pred.shape[0]:
        raise ValueError("polymer_ligand_bonds batch dimension must match x_denoised")
    if atom_i.numel() == 0:
        per_sample = torch.zeros(bsz, device=device)
        return _zero(device), per_sample

    clamped_i = atom_i.clamp(min=0, max=pred.shape[1] - 1)
    clamped_j = atom_j.clamp(min=0, max=pred.shape[1] - 1)
    valid = (
        mask
        & (atom_i >= 0)
        & (atom_j >= 0)
        & (atom_i < pred.shape[1])
        & (atom_j < pred.shape[1])
        & torch.gather(atom_mask, 1, clamped_i)
        & torch.gather(atom_mask, 1, clamped_j)
    )
    coords_i = torch.gather(pred, 1, clamped_i.unsqueeze(-1).expand(-1, -1, 3))
    coords_j = torch.gather(pred, 1, clamped_j.unsqueeze(-1).expand(-1, -1, 3))
    pred_length = (coords_i - coords_j).norm(dim=-1)
    sq_error = (pred_length - target_length).square()
    valid_f = valid.to(sq_error.dtype)
    per_sample = (sq_error * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp(min=1e-8)
    active = valid.any(dim=1)
    if not active.any():
        return _zero(device), per_sample
    return per_sample[active].mean(), per_sample


def _gather_rep_atom(tensor: Tensor, indices: Tensor) -> Tensor:
    idx = indices.long().unsqueeze(-1).expand(-1, -1, tensor.shape[-1])
    return torch.gather(tensor, 1, idx)


def _gather_frame_atoms(tensor: Tensor, frames_idx: Tensor) -> Tensor:
    bsz, _, n_frame_atoms = frames_idx.shape
    idx = frames_idx.long().reshape(bsz, -1)
    gathered = torch.gather(tensor, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    return gathered.reshape(bsz, -1, n_frame_atoms, 3)


def _distance_bins(distances: Tensor, min_bin: float, max_bin: float, bins: int) -> Tensor:
    boundaries = torch.linspace(min_bin, max_bin, bins + 1, device=distances.device)
    return (torch.bucketize(distances, boundaries[:-1]) - 1).clamp(0, bins - 1).long()


def distogram_loss(outputs: dict[str, Tensor], batch: dict[str, Any], config: ESMFold2FrozenFeatureConfig) -> Tensor:
    logits = outputs.get("z_distogram_logits", outputs["distogram_logits"]).float()
    _require_last_dim("z_distogram", logits, config.distogram_bins)
    coords = batch["gt_atom_coords"].to(logits.device).float()
    atom_mask = batch["gt_atom_mask"].to(logits.device).bool()
    rep_idx = batch["distogram_atom_idx"].to(logits.device).long()
    token_mask = batch["token_attention_mask"].to(logits.device).bool()
    rep_coords = _gather_rep_atom(coords, rep_idx)
    rep_mask = torch.gather(atom_mask, 1, rep_idx) & token_mask
    distances = torch.cdist(rep_coords, rep_coords, compute_mode="donot_use_mm_for_euclid_dist")
    target = _distance_bins(
        distances, config.distogram_min_bin, config.distogram_max_bin, config.distogram_bins
    )
    pair_mask = rep_mask[:, :, None] & rep_mask[:, None, :]
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none")
    return _masked_mean(ce.reshape_as(target), pair_mask)


def _scalar_bins(values: Tensor, max_value: float, bins: int) -> Tensor:
    scaled = torch.floor(values.clamp(min=0, max=max_value) / max_value * bins)
    return scaled.clamp(0, bins - 1).long()


def _require_last_dim(name: str, logits: Tensor, expected: int) -> None:
    if logits.shape[-1] != expected:
        raise ValueError(
            f"{name} logits have {logits.shape[-1]} bins, expected {expected}"
        )


def _hard_lddt_per_atom(
    pred: Tensor, target: Tensor, mask: Tensor, cutoffs: Tensor | float = 15.0
) -> Tensor:
    d_pred = torch.cdist(pred, pred, compute_mode="donot_use_mm_for_euclid_dist")
    d_target = torch.cdist(target, target, compute_mode="donot_use_mm_for_euclid_dist")
    pair_mask = mask[:, :, None] & mask[:, None, :] & (d_target < cutoffs)
    eye = torch.eye(mask.shape[1], device=pred.device, dtype=torch.bool).unsqueeze(0)
    pair_mask = pair_mask & ~eye
    delta = (d_pred - d_target).abs()
    score = (
        (delta < 0.5).float()
        + (delta < 1.0).float()
        + (delta < 2.0).float()
        + (delta < 4.0).float()
    ) * 0.25
    denom = pair_mask.float().sum(dim=-1).clamp(min=1.0)
    return (score * pair_mask.float()).sum(dim=-1) / denom


def _frame_aligned_pair_error(
    pred: Tensor,
    target: Tensor,
    rep_idx: Tensor,
    rep_mask: Tensor,
    frames_idx: Tensor,
    atom_mask: Tensor,
    token_mask: Tensor,
) -> Tensor:
    rep_pred = _gather_rep_atom(pred, rep_idx)
    rep_target = _gather_rep_atom(target, rep_idx)
    pred_frames = _gather_frame_atoms(pred, frames_idx)
    target_frames = _gather_frame_atoms(target, frames_idx)
    frame_mask = torch.gather(atom_mask, 1, frames_idx.reshape(frames_idx.shape[0], -1))
    frame_mask = frame_mask.reshape(frames_idx.shape).all(dim=-1) & token_mask

    bsz, n_tokens = rep_idx.shape
    errors = torch.zeros(bsz, n_tokens, n_tokens, device=pred.device, dtype=pred.dtype)
    for b in range(bsz):
        for i in range(n_tokens):
            if not bool(frame_mask[b, i]):
                continue
            mobile = pred_frames[b, i]
            fixed = target_frames[b, i]
            mobile_centroid = mobile.mean(dim=0, keepdim=True)
            fixed_centroid = fixed.mean(dim=0, keepdim=True)
            mobile_centered = mobile - mobile_centroid
            fixed_centered = fixed - fixed_centroid
            u, _, v = torch.svd(mobile_centered.transpose(0, 1) @ fixed_centered)
            rotation = u @ v.transpose(0, 1)
            aligned = (rep_pred[b] - mobile_centroid) @ rotation + fixed_centroid
            errors[b, i] = (aligned - rep_target[b]).norm(dim=-1)
    return errors * (rep_mask[:, :, None] & rep_mask[:, None, :]).to(errors.dtype)


def _active_confidence_mask(batch: dict[str, Any], device: torch.device, cutoff: float) -> Tensor:
    resolution = batch.get("resolution")
    if resolution is None:
        return torch.zeros(batch["token_attention_mask"].shape[0], device=device, dtype=torch.bool)
    resolution = resolution.to(device)
    return torch.isfinite(resolution) & (resolution <= cutoff)


def confidence_losses(
    outputs: dict[str, Tensor],
    batch: dict[str, Any],
    config: ESMFold2FrozenFeatureConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    device = outputs["x_denoised"].device
    active = _active_confidence_mask(batch, device, config.confidence_resolution_cutoff)
    if not active.any():
        z = _zero(device)
        return z, z, z, z, z

    pred = outputs["x_pred"].detach().float()
    target = batch["gt_atom_coords"].to(device).float()
    atom_mask = (batch["gt_atom_mask"].to(device).bool() & batch["atom_attention_mask"].to(device).bool())
    token_mask = batch["token_attention_mask"].to(device).bool()
    rep_idx = batch["distogram_atom_idx"].to(device).long()
    rep_pred = _gather_rep_atom(pred, rep_idx)
    rep_target = _gather_rep_atom(target, rep_idx)
    rep_mask = torch.gather(atom_mask, 1, rep_idx) & token_mask

    losses: dict[str, Tensor] = {}
    if "x_plddt_logits" in outputs:
        plddt_logits = outputs["x_plddt_logits"].float()
        _require_last_dim("x_plddt", plddt_logits, config.plddt_bins)
        cutoffs = _atom_pair_cutoffs(batch, device)
        plddt_target = _scalar_bins(
            _hard_lddt_per_atom(pred, target, atom_mask, cutoffs),
            1.0,
            plddt_logits.shape[-1],
        )
        ce = F.cross_entropy(
            plddt_logits.reshape(-1, plddt_logits.shape[-1]),
            plddt_target.reshape(-1),
            reduction="none",
        ).reshape(plddt_target.shape)
        atom_active = atom_mask & active[:, None]
        losses["plddt"] = _masked_mean(ce, atom_active)
    else:
        losses["plddt"] = _zero(device)

    pair_mask = rep_mask[:, :, None] & rep_mask[:, None, :] & active[:, None, None]
    pred_dist = torch.cdist(rep_pred, rep_pred, compute_mode="donot_use_mm_for_euclid_dist")
    target_dist = torch.cdist(rep_target, rep_target, compute_mode="donot_use_mm_for_euclid_dist")
    distance_error = (pred_dist - target_dist).abs()

    if "z_pde_logits" in outputs:
        pde_logits = outputs["z_pde_logits"].float()
        _require_last_dim("z_pde", pde_logits, config.pde_bins)
        pde_target = _scalar_bins(distance_error, config.pde_max_bin, pde_logits.shape[-1])
        ce = F.cross_entropy(
            pde_logits.reshape(-1, pde_logits.shape[-1]), pde_target.reshape(-1), reduction="none"
        ).reshape(pde_target.shape)
        losses["pde"] = _masked_mean(ce, pair_mask)
    else:
        losses["pde"] = _zero(device)

    if "x_resolved_logits" in outputs:
        resolved_logits = outputs["x_resolved_logits"].float()
        _require_last_dim("x_resolved", resolved_logits, 2)
        resolved_target = atom_mask.long()
        ce = F.cross_entropy(
            resolved_logits.reshape(-1, resolved_logits.shape[-1]),
            resolved_target.reshape(-1),
            reduction="none",
        ).reshape(resolved_target.shape)
        losses["resolved"] = _masked_mean(ce, active[:, None].expand_as(atom_mask) & batch["atom_attention_mask"].to(device).bool())
    else:
        losses["resolved"] = _zero(device)

    if "z_pae_logits" in outputs:
        pae_logits = outputs["z_pae_logits"].float()
        _require_last_dim("z_pae", pae_logits, config.pae_bins)
        pae_error = _frame_aligned_pair_error(
            pred,
            target,
            rep_idx,
            rep_mask,
            batch["frames_idx"].to(device).long(),
            atom_mask,
            token_mask,
        )
        pae_target = _scalar_bins(pae_error, config.pae_max_bin, pae_logits.shape[-1])
        ce = F.cross_entropy(
            pae_logits.reshape(-1, pae_logits.shape[-1]), pae_target.reshape(-1), reduction="none"
        ).reshape(pae_target.shape)
        losses["pae"] = _masked_mean(ce, pair_mask)
    else:
        losses["pae"] = _zero(device)

    loss_conf = losses["plddt"] + losses["pde"] + losses["resolved"] + losses["pae"]
    return loss_conf, losses["plddt"], losses["pde"], losses["resolved"], losses["pae"]


def compute_esmfold2_loss(
    outputs: dict[str, Tensor],
    batch: dict[str, Any],
    stage: int,
    config: ESMFold2FrozenFeatureConfig | None = None,
) -> dict[str, Tensor]:
    config = ESMFold2FrozenFeatureConfig() if config is None else config
    stage_cfg = get_stage_config(stage)
    device = outputs["x_denoised"].device
    zero = _zero(device)

    loss_mse, mse_per_sample = diffusion_mse_loss(outputs, batch)
    sigma = outputs["sigma"].float()
    sigma_weight = (sigma.square() + config.sigma_data**2) / (
        (sigma * config.sigma_data).square().clamp(min=1e-8)
    )
    weighted_mse = (sigma_weight * mse_per_sample).mean()

    loss_slddt = smooth_lddt_loss(outputs, batch, config) if stage_cfg.smooth_lddt else zero
    if stage_cfg.bond_loss:
        loss_bond, bond_per_sample = bond_loss(outputs, batch, device)
    else:
        loss_bond = zero
        bond_per_sample = torch.zeros_like(mse_per_sample)
    loss_dist = distogram_loss(outputs, batch, config) if stage_cfg.distogram else zero
    if stage_cfg.confidence:
        loss_conf, loss_plddt, loss_pde, loss_resolved, loss_pae = confidence_losses(
            outputs, batch, config
        )
    else:
        loss_conf = loss_plddt = loss_pde = loss_resolved = loss_pae = zero

    weighted_bond = (sigma_weight * config.alpha_bond * bond_per_sample).mean()
    loss_struct = weighted_mse + (weighted_bond if stage_cfg.bond_loss else zero)
    if stage_cfg.smooth_lddt:
        loss_struct = loss_struct + config.alpha_slddt * loss_slddt

    loss = (
        config.alpha_struct * loss_struct
        + config.alpha_dist * loss_dist
        + config.alpha_conf * loss_conf
    )
    result = {
        "loss": loss,
        "loss_struct": loss_struct,
        "loss_mse": loss_mse,
        "loss_slddt": loss_slddt,
        "loss_bond": loss_bond,
        "loss_dist": loss_dist,
        "loss_conf": loss_conf,
        "loss_plddt": loss_plddt,
        "loss_pde": loss_pde,
        "loss_resolved": loss_resolved,
        "loss_pae": loss_pae,
    }
    return {key: result.get(key, zero) for key in LOSS_KEYS}
