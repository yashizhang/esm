from __future__ import annotations

import math
from contextlib import contextmanager, nullcontext
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from esm.training.esmfold2_frozen_features.config import (
    ESMFold2FrozenFeatureConfig,
    get_stage_config,
)

try:
    from transformers import ESMFold2Config
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    from transformers.models.esmfold2.modeling_esmfold2_common import (
        CHAR_VOCAB_SIZE,
        MAX_ATOMIC_NUMBER,
        NUM_RES_TYPES,
        _compute_intra_token_idx,
        _seed_context,
        gather_rep_atom_coords,
        gather_token_to_atom,
    )
except Exception:  # pragma: no cover - imported lazily by callers with transformers installed
    ESMFold2Config = None  # type: ignore[assignment]
    ESMFold2Model = None  # type: ignore[assignment]
    CHAR_VOCAB_SIZE = 64
    MAX_ATOMIC_NUMBER = 128
    NUM_RES_TYPES = 33
    gather_rep_atom_coords = None  # type: ignore[assignment]
    gather_token_to_atom = None  # type: ignore[assignment]
    _compute_intra_token_idx = None  # type: ignore[assignment]

    def _seed_context(seed: int | None, *, cuda: bool = True):  # type: ignore[no-redef]
        return nullcontext()


MODEL_FORWARD_KEYS = {
    "token_index",
    "residue_index",
    "asym_id",
    "sym_id",
    "entity_id",
    "mol_type",
    "res_type",
    "token_bonds",
    "token_attention_mask",
    "ref_pos",
    "ref_element",
    "ref_charge",
    "ref_atom_name_chars",
    "ref_space_uid",
    "atom_attention_mask",
    "atom_to_token",
    "distogram_atom_idx",
    "deletion_mean",
    "input_ids",
}


def model_forward_batch(batch: dict[str, Any], device: torch.device | str | None = None) -> dict[str, Tensor]:
    result: dict[str, Tensor] = {}
    for key in MODEL_FORWARD_KEYS:
        value = batch.get(key)
        if isinstance(value, Tensor):
            result[key] = value.to(device) if device is not None else value
    return result


def _sample_recycle_count(
    config: ESMFold2FrozenFeatureConfig,
    device: torch.device,
    seed: int | None,
    num_recycles_for_test: int | None,
) -> int:
    if num_recycles_for_test is not None:
        return max(1, int(num_recycles_for_test))
    with _seed_context(seed):
        value = torch.poisson(torch.tensor(config.recurrent_poisson_mean, device=device)).item()
    return int(max(config.recurrent_min_loops, min(config.recurrent_max_loops, value)))


def _one_hot_inputs(batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    tok_mask = batch["token_attention_mask"].bool()
    atm_mask = batch["atom_attention_mask"].bool()
    res_type = batch["res_type"]
    if res_type.dim() == 2:
        res_type_oh = F.one_hot(res_type.long(), num_classes=NUM_RES_TYPES).float()
        res_type_oh = res_type_oh * tok_mask.unsqueeze(-1).float()
    else:
        res_type_oh = res_type.float()

    ref_element_oh = F.one_hot(
        batch["ref_element"].long(), num_classes=MAX_ATOMIC_NUMBER
    ).float()
    ref_atom_name_chars_oh = F.one_hot(
        batch["ref_atom_name_chars"].long(), num_classes=CHAR_VOCAB_SIZE
    ).float()
    atom_mask_f = atm_mask.float()
    ref_element_oh = ref_element_oh * atom_mask_f.unsqueeze(-1)
    ref_atom_name_chars_oh = ref_atom_name_chars_oh * atom_mask_f.unsqueeze(-1).unsqueeze(-1)
    atom_to_token = batch["atom_to_token"].long() * atm_mask.long()
    return res_type_oh, ref_element_oh, ref_atom_name_chars_oh, atom_to_token, tok_mask


def _input_embeddings(
    model: nn.Module,
    batch: dict[str, Tensor],
    res_type_oh: Tensor,
    ref_element_oh: Tensor,
    ref_atom_name_chars_oh: Tensor,
    atom_to_token: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    deletion_mean = batch.get("deletion_mean")
    if deletion_mean is None:
        deletion_mean = torch.zeros_like(batch["residue_index"], dtype=torch.float32)

    profile = res_type_oh
    if getattr(model.config, "disable_msa_features", False):
        profile = torch.zeros_like(profile)
        deletion_mean = torch.zeros_like(deletion_mean)

    x_inputs = model.inputs_embedder(
        aatype=res_type_oh,
        profile=profile.float(),
        deletion_mean=deletion_mean.float(),
        ref_pos=batch["ref_pos"],
        atom_attention_mask=batch["atom_attention_mask"].bool(),
        ref_space_uid=batch["ref_space_uid"],
        ref_charge=batch["ref_charge"],
        ref_element=ref_element_oh,
        ref_atom_name_chars=ref_atom_name_chars_oh,
        atom_to_token=atom_to_token,
    )

    z_init = model.z_init_1(x_inputs).unsqueeze(2) + model.z_init_2(x_inputs).unsqueeze(1)
    relative_position_encoding = model.rel_pos(
        residue_index=batch["residue_index"],
        asym_id=batch["asym_id"],
        sym_id=batch["sym_id"],
        entity_id=batch["entity_id"],
        token_index=batch["token_index"],
    )
    token_bonds_encoding = model.token_bonds(batch["token_bonds"].float())
    z_init = z_init + relative_position_encoding + token_bonds_encoding
    return x_inputs, z_init, relative_position_encoding, token_bonds_encoding


def _run_release_recycles(
    model: nn.Module,
    z_init: Tensor,
    lm_z: Tensor | None,
    tok_mask: Tensor,
    total_loops: int,
    grad_loops: int,
) -> Tensor:
    pair_mask = tok_mask[:, :, None].float() * tok_mask[:, None, :].float()
    z = model._init_pair_state(z_init)
    a, b = model._discretized_dynamics()
    a = a.view(1, 1, 1, -1).to(device=z.device, dtype=z.dtype)
    b_mat = b.to(device=z.device, dtype=z.dtype)

    warmup = max(0, total_loops - grad_loops)
    if warmup:
        with torch.no_grad():
            z = model._run_one_loop(
                z=z,
                z_init=z_init,
                lm_z=lm_z,
                _msa_inputs=None,
                pair_mask=pair_mask,
                a=a,
                b_mat=b_mat,
                tok_mask=tok_mask,
                total_steps=warmup,
            )
        z = z.detach()

    tracked = max(1, min(total_loops, grad_loops))
    z = model._run_one_loop(
        z=z,
        z_init=z_init,
        lm_z=lm_z,
        _msa_inputs=None,
        pair_mask=pair_mask,
        a=a,
        b_mat=b_mat,
        tok_mask=tok_mask,
        total_steps=tracked,
    )
    z = model.parcae_readout(z)
    z = model.parcae_coda(z, pair_attention_mask=pair_mask)
    return z.float()


def _run_experimental_recycles(
    model: nn.Module,
    z_init: Tensor,
    tok_mask: Tensor,
    total_loops: int,
    grad_loops: int,
) -> Tensor:
    pair_mask = tok_mask[:, :, None].float() * tok_mask[:, None, :].float()
    z = torch.zeros_like(z_init)

    def run_steps(z_in: Tensor, n_steps: int) -> Tensor:
        z_cur = z_in
        for _ in range(n_steps):
            z_cur = z_init + model.pair_loop_proj(z_cur)
            z_cur = model.folding_trunk(z_cur, pair_attention_mask=pair_mask)
        return z_cur

    warmup = max(0, total_loops - grad_loops)
    if warmup:
        with torch.no_grad():
            z = run_steps(z, warmup)
        z = z.detach()
    z = run_steps(z, max(1, min(total_loops, grad_loops)))
    return z.float()


def _training_diffusion_sample(
    model: nn.Module,
    batch: dict[str, Tensor],
    z: Tensor,
    x_inputs: Tensor,
    relative_position_encoding: Tensor,
    ref_element_oh: Tensor,
    ref_atom_name_chars_oh: Tensor,
    atom_to_token: Tensor,
    config: ESMFold2FrozenFeatureConfig,
    seed: int | None,
) -> dict[str, Tensor | None]:
    gt = batch["gt_atom_coords"].float()
    atom_mask = batch["gt_atom_mask"].bool() & batch["atom_attention_mask"].bool()
    with _seed_context(seed):
        z_noise = torch.randn(gt.shape[0], device=gt.device)
        sigma = config.sigma_data * torch.exp(
            config.train_noise_log_mean + config.train_noise_log_std * z_noise
        )
        x_noisy = gt + sigma[:, None, None] * torch.randn_like(gt)
    x_noisy = torch.where(atom_mask.unsqueeze(-1), x_noisy, torch.zeros_like(x_noisy))

    diffusion_output = model.structure_head.diffusion_module(
        x_noisy=x_noisy,
        t_hat=sigma,
        ref_pos=batch["ref_pos"],
        ref_charge=batch["ref_charge"],
        ref_mask=batch["atom_attention_mask"].bool(),
        ref_element=ref_element_oh,
        ref_atom_name_chars=ref_atom_name_chars_oh,
        ref_space_uid=batch["ref_space_uid"],
        tok_idx=atom_to_token,
        s_inputs=x_inputs,
        s_trunk=None,
        z_trunk=z,
        relative_position_encoding=relative_position_encoding,
        asym_id=batch["asym_id"],
        residue_index=batch["residue_index"],
        entity_id=batch["entity_id"],
        token_index=batch["token_index"],
        sym_id=batch["sym_id"],
        token_attention_mask=batch["token_attention_mask"].bool(),
        num_diffusion_samples=1,
        return_token_repr=True,
        return_atom_repr=False,
        inference_cache=None,
    )
    diffusion_output["sigma"] = sigma
    diffusion_output["x_noisy"] = x_noisy
    return diffusion_output


def _confidence_coordinates(
    model: nn.Module,
    batch: dict[str, Tensor],
    z: Tensor,
    x_inputs: Tensor,
    relative_position_encoding: Tensor,
    ref_element_oh: Tensor,
    ref_atom_name_chars_oh: Tensor,
    atom_to_token: Tensor,
    fallback: Tensor,
    sampling_steps: int,
    seed: int | None,
) -> Tensor:
    if sampling_steps <= 0:
        return fallback.detach()
    with torch.no_grad(), _seed_context(seed):
        sampled = model.structure_head.sample(
            z_trunk=z.detach().float(),
            s_inputs=x_inputs.detach(),
            s_trunk=None,
            relative_position_encoding=relative_position_encoding.detach(),
            ref_pos=batch["ref_pos"],
            ref_charge=batch["ref_charge"],
            ref_mask=batch["atom_attention_mask"].bool(),
            ref_element=ref_element_oh,
            ref_atom_name_chars=ref_atom_name_chars_oh,
            ref_space_uid=batch["ref_space_uid"],
            tok_idx=atom_to_token,
            asym_id=batch["asym_id"],
            residue_index=batch["residue_index"],
            entity_id=batch["entity_id"],
            token_index=batch["token_index"],
            sym_id=batch["sym_id"],
            token_attention_mask=batch["token_attention_mask"].bool(),
            num_diffusion_samples=1,
            num_sampling_steps=sampling_steps,
            return_atom_repr=False,
            denoising_early_exit_rmsd=None,
        )
    x_pred = sampled["sample_atom_coords"]
    assert x_pred is not None
    return x_pred.detach()


@contextmanager
def _disable_checkpoint_for_folding_trunk(trunk: nn.Module):
    saved_forward = trunk.forward

    def forward_no_checkpoint(pair: Tensor, pair_attention_mask: Tensor | None = None) -> Tensor:
        orig_dtype = pair.dtype
        for block in trunk.blocks:  # type: ignore[attr-defined]
            pair = block(pair, pair_attention_mask=pair_attention_mask)
        if pair.dtype != orig_dtype:
            pair = pair.to(orig_dtype)
        return pair

    trunk.forward = forward_no_checkpoint  # type: ignore[method-assign]
    try:
        yield
    finally:
        trunk.forward = saved_forward  # type: ignore[method-assign]


def _safe_confidence_head_forward(
    confidence_head: nn.Module,
    *,
    s_inputs: Tensor,
    z: Tensor,
    x_pred: Tensor,
    distogram_atom_idx: Tensor,
    token_attention_mask: Tensor,
    atom_to_token: Tensor,
    atom_attention_mask: Tensor,
    asym_id: Tensor,
    mol_type: Tensor,
    num_diffusion_samples: int,
    relative_position_encoding: Tensor | None,
    token_bonds_encoding: Tensor | None,
    trunk_z_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    if gather_rep_atom_coords is None or gather_token_to_atom is None or _compute_intra_token_idx is None:
        raise ImportError("Biohub ESMFold2 confidence helpers are required")

    head_param = next(confidence_head.parameters(), None)
    head_dtype = head_param.dtype if head_param is not None else s_inputs.dtype
    s_inputs = s_inputs.to(dtype=head_dtype)
    z = z.to(dtype=head_dtype)
    x_pred = x_pred.float()
    if relative_position_encoding is not None:
        relative_position_encoding = relative_position_encoding.to(dtype=head_dtype)
    if token_bonds_encoding is not None:
        token_bonds_encoding = token_bonds_encoding.to(dtype=head_dtype)

    s_inputs_normed = confidence_head.s_inputs_norm(s_inputs)
    z_trunk = confidence_head.z_norm(z)
    if trunk_z_mask is not None:
        z_trunk = z_trunk * trunk_z_mask.to(device=z_trunk.device, dtype=z_trunk.dtype)
    z_base = z_trunk
    if relative_position_encoding is not None:
        z_base = z_base + relative_position_encoding
    if token_bonds_encoding is not None:
        z_base = z_base + token_bonds_encoding
    z_base = z_base + confidence_head.s_to_z(s_inputs_normed).unsqueeze(2)
    z_base = z_base + confidence_head.s_to_z_transpose(s_inputs_normed).unsqueeze(1)
    z_base = z_base + confidence_head.s_to_z_prod_out(
        confidence_head.s_to_z_prod_in1(s_inputs_normed)[:, :, None, :]
        * confidence_head.s_to_z_prod_in2(s_inputs_normed)[:, None, :, :]
    )

    pair = confidence_head._repeat_batch(z_base, num_diffusion_samples)
    x_pred_flat = confidence_head._flatten_sample_axis(x_pred)
    atom_to_token_m = confidence_head._repeat_batch(atom_to_token, num_diffusion_samples)
    atom_mask_m = confidence_head._repeat_batch(atom_attention_mask, num_diffusion_samples)
    rep_idx_m = confidence_head._repeat_batch(distogram_atom_idx, num_diffusion_samples).long()
    mask = confidence_head._repeat_batch(token_attention_mask, num_diffusion_samples)

    rep_coords = gather_rep_atom_coords(x_pred_flat, rep_idx_m)
    rep_distances = torch.cdist(
        rep_coords, rep_coords, compute_mode="donot_use_mm_for_euclid_dist"
    )
    distogram_bins = (rep_distances.unsqueeze(-1) > confidence_head.boundaries).sum(dim=-1).long()
    pair = pair + confidence_head.dist_bin_pairwise_embed(distogram_bins)

    pair_mask = mask[:, :, None].float() * mask[:, None, :].float()
    pair_delta = confidence_head.folding_trunk(pair, pair_attention_mask=pair_mask)
    pair = pair + pair_delta.to(pair.dtype)
    single = confidence_head.row_attention_pooling(pair, mask)

    s_at_atoms = gather_token_to_atom(single, atom_to_token_m)
    s_at_atoms_ln = confidence_head.plddt_ln(s_at_atoms)
    intra_idx = _compute_intra_token_idx(atom_to_token_m)
    intra_idx = intra_idx.clamp(max=confidence_head.plddt_weight.shape[0] - 1)
    w_plddt = confidence_head.plddt_weight[intra_idx]
    plddt_logits = torch.einsum("...c,...cb->...b", s_at_atoms_ln, w_plddt)

    pae_input = confidence_head.pae_ln(pair) if hasattr(confidence_head, "pae_ln") else pair
    pae_logits = confidence_head.pae_head(pae_input)
    output: dict[str, Tensor] = {
        "plddt_logits": plddt_logits,
        "pae_logits": pae_logits,
    }
    if hasattr(confidence_head, "pde_head"):
        pde_input = confidence_head.pde_ln(pair) if hasattr(confidence_head, "pde_ln") else pair
        output["pde_logits"] = confidence_head.pde_head(pde_input)
    if hasattr(confidence_head, "resolved_weight"):
        s_at_atoms_res = confidence_head.resolved_ln(s_at_atoms)
        w_res = confidence_head.resolved_weight[intra_idx]
        output["resolved_logits"] = torch.einsum("...c,...cb->...b", s_at_atoms_res, w_res)
    return output


def forward_train_from_precomputed_lm(
    self: nn.Module,
    batch: dict[str, Any],
    lm_hidden_states: Tensor,
    stage: int,
    seed: int | None = None,
    num_recycles_for_test: int | None = None,
    confidence_num_sampling_steps: int | None = None,
    config: ESMFold2FrozenFeatureConfig | None = None,
) -> dict[str, Tensor]:
    if lm_hidden_states is None:
        raise ValueError("lm_hidden_states must be provided from a precomputed cache")
    config = ESMFold2FrozenFeatureConfig() if config is None else config
    features = model_forward_batch(batch, device=lm_hidden_states.device)
    train_batch = dict(features)
    for key, value in batch.items():
        if isinstance(value, Tensor):
            train_batch[key] = value.to(lm_hidden_states.device)
    lm_hidden_states = lm_hidden_states.detach()
    if lm_hidden_states.ndim != 4:
        raise ValueError(
            f"lm_hidden_states must have shape [B, L, N_layers+1, D], got "
            f"{tuple(lm_hidden_states.shape)}"
        )
    if lm_hidden_states.shape[:2] != features["token_attention_mask"].shape:
        raise ValueError(
            "lm_hidden_states batch/token dimensions do not match token_attention_mask"
        )
    if getattr(self, "_esmc", None) is not None:
        raise RuntimeError("ESMC must not be loaded on the ESMFold2 model during frozen-feature training")
    lm_param = next(self.language_model.parameters(), None)
    if lm_param is not None:
        lm_hidden_states = lm_hidden_states.to(dtype=lm_param.dtype)

    stage_cfg = get_stage_config(stage)
    if stage_cfg.stage == 3:
        configure_stage_trainability(self, stage=3)

    total_loops = _sample_recycle_count(
        config, lm_hidden_states.device, seed=seed, num_recycles_for_test=num_recycles_for_test
    )
    grad_loops = config.recurrent_grad_loops

    res_type_oh, ref_element_oh, ref_atom_name_chars_oh, atom_to_token, tok_mask = _one_hot_inputs(features)
    use_amp = features["ref_pos"].device.type == "cuda"
    autocast_ctx = torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16)
    with autocast_ctx:
        x_inputs, z_init, relative_position_encoding, token_bonds_encoding = _input_embeddings(
            self, features, res_type_oh, ref_element_oh, ref_atom_name_chars_oh, atom_to_token
        )
        if hasattr(self, "_run_one_loop") and hasattr(self, "parcae_coda"):
            lm_z = self.language_model(lm_hidden_states)
            z = _run_release_recycles(self, z_init, lm_z, tok_mask, total_loops, grad_loops)
        else:
            lm_z = self.language_model(lm_hidden_states, lm_dropout=0.25)
            z = _run_experimental_recycles(
                self, z_init + lm_z.to(z_init.dtype), tok_mask, total_loops, grad_loops
            )
        distogram_logits = self.distogram_head(z + z.transpose(-2, -3))

    diffusion_output = _training_diffusion_sample(
        self,
        train_batch,
        z.float(),
        x_inputs,
        relative_position_encoding,
        ref_element_oh,
        ref_atom_name_chars_oh,
        atom_to_token,
        config,
        seed,
    )
    x_denoised = diffusion_output["x_denoised"]
    assert isinstance(x_denoised, Tensor)
    conf_steps = (
        config.confidence_mini_rollout_steps
        if confidence_num_sampling_steps is None
        else int(confidence_num_sampling_steps)
    )
    x_pred = _confidence_coordinates(
        self,
        train_batch,
        z.float(),
        x_inputs,
        relative_position_encoding,
        ref_element_oh,
        ref_atom_name_chars_oh,
        atom_to_token,
        fallback=x_denoised,
        sampling_steps=conf_steps,
        seed=seed,
    )

    output: dict[str, Tensor] = {
        "distogram_logits": distogram_logits,
        "z_distogram_logits": distogram_logits,
        "x_denoised": x_denoised,
        "x_pred": x_pred,
        "sample_atom_coords": x_pred,
        "sigma": diffusion_output["sigma"],  # type: ignore[dict-item]
        "x_noisy": diffusion_output["x_noisy"],  # type: ignore[dict-item]
        "z": z,
        "x_inputs": x_inputs,
        "relative_position_encoding": relative_position_encoding,
        "token_bonds_encoding": token_bonds_encoding,
        "num_recycles": torch.tensor(total_loops, device=x_denoised.device),
        "atom_pad_mask": features["atom_attention_mask"],
        "residue_index": features["residue_index"],
        "entity_id": features["entity_id"],
    }

    confidence_head = getattr(self, "confidence_head", None)
    if confidence_head is not None:
        confidence_z = z.detach().float()
        trunk_z_mask = None
        if self.training and config.confidence_pair_drop_prob > 0:
            with _seed_context(seed):
                trunk_z_mask = torch.rand(
                    confidence_z.shape[0],
                    1,
                    1,
                    1,
                    device=confidence_z.device,
                ) >= config.confidence_pair_drop_prob
        trunk = getattr(confidence_head, "folding_trunk", None)
        checkpoint_context = (
            _disable_checkpoint_for_folding_trunk(trunk)
            if trunk is not None
            else nullcontext()
        )
        confidence_autocast_ctx = torch.amp.autocast(
            "cuda", enabled=confidence_z.device.type == "cuda", dtype=torch.bfloat16
        )
        with checkpoint_context, confidence_autocast_ctx:
            confidence_output = _safe_confidence_head_forward(
                confidence_head,
                s_inputs=x_inputs.detach(),
                z=confidence_z,
                x_pred=x_pred.detach(),
                distogram_atom_idx=features["distogram_atom_idx"],
                token_attention_mask=features["token_attention_mask"].bool(),
                atom_to_token=atom_to_token,
                atom_attention_mask=features["atom_attention_mask"].bool(),
                asym_id=features["asym_id"],
                mol_type=features["mol_type"],
                num_diffusion_samples=1,
                relative_position_encoding=relative_position_encoding.detach(),
                token_bonds_encoding=token_bonds_encoding.detach(),
                trunk_z_mask=trunk_z_mask,
            )
        output.update(confidence_output)
        if "plddt_logits" in confidence_output:
            output["x_plddt_logits"] = confidence_output["plddt_logits"]
        if "resolved_logits" in confidence_output:
            output["x_resolved_logits"] = confidence_output["resolved_logits"]
        if "pde_logits" in confidence_output:
            output["z_pde_logits"] = confidence_output["pde_logits"]
        if "pae_logits" in confidence_output:
            output["z_pae_logits"] = confidence_output["pae_logits"]
    return output


def ensure_forward_train_from_precomputed_lm(model: nn.Module) -> nn.Module:
    model.forward_train_from_precomputed_lm = MethodType(  # type: ignore[attr-defined]
        forward_train_from_precomputed_lm, model
    )
    return model


def configure_stage_trainability(model: nn.Module, stage: int) -> None:
    if getattr(model, "_esmc", None) is not None:
        raise RuntimeError("ESMC must not be loaded during frozen-feature training")
    for _, param in model.named_parameters():
        param.requires_grad_(True)
    if int(stage) != 3:
        return

    for _, param in model.named_parameters():
        param.requires_grad_(False)
    for module_name in ("structure_head", "confidence_head"):
        module = getattr(model, module_name, None)
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad_(True)


def named_trainable_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def assert_no_esmc_parameters(param_names: list[str]) -> None:
    bad = [name for name in param_names if "esmc" in name.lower()]
    if bad:
        raise AssertionError(f"ESMC parameters are present in the optimizer: {bad[:5]}")


def build_optimizer(
    model: nn.Module,
    stage: int,
    config: ESMFold2FrozenFeatureConfig | None = None,
) -> tuple[torch.optim.Optimizer, list[str]]:
    config = ESMFold2FrozenFeatureConfig() if config is None else config
    configure_stage_trainability(model, stage)
    named_params = named_trainable_parameters(model)
    param_names = [name for name, _ in named_params]
    assert_no_esmc_parameters(param_names)
    stage_cfg = get_stage_config(stage)
    lr = stage_cfg.fixed_lr if stage_cfg.fixed_lr is not None else config.max_lr
    optimizer = torch.optim.AdamW(
        [param for _, param in named_params],
        lr=lr,
        betas=(config.optimizer_beta1, config.optimizer_beta2),
        eps=config.optimizer_eps,
        weight_decay=config.weight_decay,
    )
    return optimizer, param_names


def learning_rate_for_step(
    step: int, stage: int, config: ESMFold2FrozenFeatureConfig | None = None
) -> float:
    config = ESMFold2FrozenFeatureConfig() if config is None else config
    stage_cfg = get_stage_config(stage)
    if stage_cfg.fixed_lr is not None:
        return stage_cfg.fixed_lr
    warmup_scale = min(1.0, float(step + 1) / float(config.warmup_steps))
    decay_count = (step + 1) // config.step_decay_steps
    return config.max_lr * warmup_scale * (config.step_decay_factor**decay_count)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def tiny_esmfold2_model() -> nn.Module:
    if ESMFold2Config is None or ESMFold2Model is None:
        raise ImportError("Biohub transformers ESMFold2 classes are required")
    tiny_d_inputs = 75
    cfg = ESMFold2Config(
        type="release",
        d_single=16,
        d_pair=16,
        lm_d_model=8,
        lm_num_layers=1,
        folding_trunk={"n_layers": 1, "n_heads": 1, "dropout": 0.0},
        lm_encoder={"enabled": True, "n_layers": 1, "lm_dropout": 0.25, "per_loop_lm_dropout": True},
        parcae={
            "enabled": True,
            "poisson_mean": 3.0,
            "min_steps": 1,
            "max_steps": 6,
            "coda_n_layers": 1,
        },
        inputs={
            "d_inputs": tiny_d_inputs,
            "atom_encoder": {
                "d_atom": 16,
                "d_token": 16,
                "n_blocks": 1,
                "n_heads": 1,
                "swa_window_size": 64,
                "expansion_ratio": 2,
                "spatial_rope_base_frequency": 20.0,
                "n_spatial_rope_pairs_per_axis": 1,
                "n_uid_rope_pairs": 2,
                "uid_rope_base_frequency": 10000.0,
            },
        },
        structure_head={
            "distogram_bins": 64,
            "train_noise_log_mean": -1.2,
            "train_noise_log_std": 1.5,
            "diffusion_module": {
                "sigma_data": 16.0,
                "c_atom": 16,
                "c_token": 16,
                "c_z": 16,
                "c_s_inputs": tiny_d_inputs,
                "fourier_dim": 16,
                "relpos_r_max": 32,
                "relpos_s_max": 2,
                "atom_num_blocks": 1,
                "atom_num_heads": 1,
                "token_num_blocks": 1,
                "token_num_heads": 1,
                "transition_multiplier": 2,
            },
        },
        confidence_head={
            "enabled": True,
            "num_plddt_bins": 50,
            "num_pde_bins": 64,
            "num_pae_bins": 64,
            "min_dist": 3.25,
            "max_dist": 50.75,
            "distogram_bins": 39,
            "folding_trunk": {"n_layers": 1, "n_heads": 1, "dropout": 0.0},
        },
        msa_encoder={"enabled": False},
    )
    model = ESMFold2Model(cfg)
    model._esmc = None
    return ensure_forward_train_from_precomputed_lm(model)


def load_esmfold2_for_training(model_checkpoint: str, device: torch.device, precision: str) -> nn.Module:
    if model_checkpoint in {"tiny-random", "random-tiny"}:
        model = tiny_esmfold2_model()
    else:
        if ESMFold2Model is None:
            raise ImportError("Biohub transformers ESMFold2Model is required")
        model = ESMFold2Model.from_pretrained(model_checkpoint, load_esmc=False)
        model = ensure_forward_train_from_precomputed_lm(model)
    model = model.to(device)
    if device.type == "cuda" and precision in {"bf16", "bfloat16"}:
        model = model.to(torch.bfloat16)
        if hasattr(model, "structure_head"):
            model.structure_head.float()
    else:
        model = model.float()
    if getattr(model, "_esmc", None) is not None:
        raise RuntimeError("loaded ESMFold2 model contains ESMC despite load_esmc=False")
    return model
