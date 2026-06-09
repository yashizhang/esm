import pytest

import torch

from esm.training.esmfold2_frozen_features.losses import (
    LOSS_KEYS,
    compute_esmfold2_loss,
    diffusion_mse_loss,
)
from esm.training.esmfold2_frozen_features.modeling import build_optimizer


def test_loss_keys_and_backward(tiny_model, synthetic_batch):
    build_optimizer(tiny_model, stage=1)
    lm_hidden_states = synthetic_batch["lm_hidden_states"].detach().clone().requires_grad_(True)
    outputs = tiny_model.forward_train_from_precomputed_lm(
        batch=synthetic_batch,
        lm_hidden_states=lm_hidden_states,
        stage=1,
        seed=0,
        num_recycles_for_test=1,
        confidence_num_sampling_steps=0,
    )
    losses = compute_esmfold2_loss(outputs, synthetic_batch, stage=1)
    assert set(LOSS_KEYS).issubset(losses)
    losses["loss"].backward()
    assert lm_hidden_states.grad is None
    assert any(
        param.grad is not None and param.grad.abs().sum() > 0
        for name, param in tiny_model.named_parameters()
        if param.requires_grad and "esmc" not in name.lower()
    )


def test_loss_rejects_unexpected_confidence_bins(tiny_model, synthetic_batch):
    outputs = tiny_model.forward_train_from_precomputed_lm(
        batch=synthetic_batch,
        lm_hidden_states=synthetic_batch["lm_hidden_states"],
        stage=1,
        seed=0,
        num_recycles_for_test=1,
        confidence_num_sampling_steps=0,
    )
    outputs["z_pde_logits"] = outputs["z_pde_logits"][..., :10]
    with pytest.raises(ValueError, match="z_pde logits"):
        compute_esmfold2_loss(outputs, synthetic_batch, stage=1)


def test_diffusion_mse_loss_backpropagates_through_aligned_prediction():
    pred = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.3, 0.1, 0.0], [0.0, 0.8, 0.2], [1.0, 1.0, 0.4]]],
        requires_grad=True,
    )
    batch = {
        "gt_atom_coords": torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]]
        ),
        "gt_atom_mask": torch.tensor([[True, True, True, True]]),
        "atom_attention_mask": torch.tensor([[True, True, True, True]]),
        "atom_loss_weight": torch.ones(1, 4),
    }
    loss, _ = diffusion_mse_loss({"x_denoised": pred}, batch)
    loss.backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum() > 0


def test_confidence_loss_stop_gradient_routes_only_confidence_head(tiny_model, synthetic_batch):
    outputs = tiny_model.forward_train_from_precomputed_lm(
        batch=synthetic_batch,
        lm_hidden_states=synthetic_batch["lm_hidden_states"],
        stage=1,
        seed=0,
        num_recycles_for_test=1,
        confidence_num_sampling_steps=0,
    )
    losses = compute_esmfold2_loss(outputs, synthetic_batch, stage=1)
    tiny_model.zero_grad(set_to_none=True)
    losses["loss_conf"].backward()

    confidence_grads = [
        param.grad
        for name, param in tiny_model.named_parameters()
        if name.startswith("confidence_head.") and param.requires_grad
    ]
    non_confidence_grads = [
        (name, param.grad)
        for name, param in tiny_model.named_parameters()
        if not name.startswith("confidence_head.") and param.requires_grad
    ]
    assert any(grad is not None and grad.abs().sum() > 0 for grad in confidence_grads)
    assert all(
        grad is None or grad.abs().sum() == 0 for _, grad in non_confidence_grads
    )


def test_distogram_loss_does_not_train_diffusion_or_confidence_heads(
    tiny_model, synthetic_batch
):
    outputs = tiny_model.forward_train_from_precomputed_lm(
        batch=synthetic_batch,
        lm_hidden_states=synthetic_batch["lm_hidden_states"],
        stage=2,
        seed=0,
        num_recycles_for_test=1,
        confidence_num_sampling_steps=0,
    )
    losses = compute_esmfold2_loss(outputs, synthetic_batch, stage=2)
    tiny_model.zero_grad(set_to_none=True)
    losses["loss_dist"].backward()

    excluded_head_grads = [
        (name, param.grad)
        for name, param in tiny_model.named_parameters()
        if (
            name.startswith("structure_head.")
            or name.startswith("confidence_head.")
        )
        and param.requires_grad
    ]
    assert losses["loss_dist"] > 0
    assert all(
        grad is None or grad.abs().sum() == 0 for _, grad in excluded_head_grads
    )
