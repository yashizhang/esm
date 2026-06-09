import torch

from esm.training.esmfold2_frozen_features.config import ESMFold2FrozenFeatureConfig
from esm.training.esmfold2_frozen_features.losses import (
    bond_loss,
    compute_esmfold2_loss,
    diffusion_mse_loss,
)


def test_polymer_ligand_bond_loss_uses_target_lengths():
    coords = torch.tensor(
        [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        requires_grad=True,
    )
    batch = {
        "atom_attention_mask": torch.tensor([[True, True, False]]),
        "polymer_ligand_bonds": {
            "atom_i": torch.tensor([[0]]),
            "atom_j": torch.tensor([[1]]),
            "target_length": torch.tensor([[1.0]]),
            "mask": torch.tensor([[True]]),
        },
    }
    loss, per_sample = bond_loss({"x_denoised": coords}, batch, coords.device)
    assert torch.isclose(loss, torch.tensor(1.0))
    assert torch.allclose(per_sample, torch.tensor([1.0]))
    loss.backward()
    assert coords.grad is not None
    assert coords.grad.abs().sum() > 0


def test_stage2_struct_loss_scales_bond_inside_sigma_weight(tiny_model, synthetic_batch):
    outputs = tiny_model.forward_train_from_precomputed_lm(
        batch=synthetic_batch,
        lm_hidden_states=synthetic_batch["lm_hidden_states"],
        stage=2,
        seed=0,
        num_recycles_for_test=1,
        confidence_num_sampling_steps=0,
    )
    outputs["sigma"] = torch.tensor([2.0], device=outputs["x_denoised"].device)
    synthetic_batch["polymer_ligand_bonds"] = {
        "atom_i": torch.tensor([[0]], device=outputs["x_denoised"].device),
        "atom_j": torch.tensor([[1]], device=outputs["x_denoised"].device),
        "target_length": torch.tensor([[0.0]], device=outputs["x_denoised"].device),
        "mask": torch.tensor([[True]], device=outputs["x_denoised"].device),
    }

    config = ESMFold2FrozenFeatureConfig()
    losses = compute_esmfold2_loss(outputs, synthetic_batch, stage=2, config=config)
    _, mse_per_sample = diffusion_mse_loss(outputs, synthetic_batch)
    _, bond_per_sample = bond_loss(outputs, synthetic_batch, outputs["x_denoised"].device)
    sigma_weight = (outputs["sigma"].square() + config.sigma_data**2) / (
        outputs["sigma"] * config.sigma_data
    ).square()
    expected = (sigma_weight * (mse_per_sample + bond_per_sample)).mean()
    assert torch.allclose(losses["loss_struct"], expected)
