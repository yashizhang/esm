import inspect

import pytest
import torch
import torch.nn as nn

from esm.training.esmfold2_frozen_features.modeling import (
    _safe_confidence_head_forward,
    forward_train_from_precomputed_lm,
)


def test_training_forward_is_not_inference_decorated():
    assert not hasattr(forward_train_from_precomputed_lm, "__wrapped__")
    source = inspect.getsource(forward_train_from_precomputed_lm)
    assert "torch.inference_mode" not in source
    assert "torch.no_grad" not in source


def test_training_forward_rejects_loaded_esmc(tiny_model, synthetic_batch):
    tiny_model._esmc = nn.Linear(1, 1)
    with pytest.raises(RuntimeError, match="ESMC must not be loaded"):
        tiny_model.forward_train_from_precomputed_lm(
            batch=synthetic_batch,
            lm_hidden_states=synthetic_batch["lm_hidden_states"],
            stage=1,
            seed=0,
            num_recycles_for_test=1,
            confidence_num_sampling_steps=0,
        )


def test_training_forward_accepts_bfloat16_cached_lm_on_float_model(
    tiny_model, synthetic_batch
):
    lm_hidden_states = synthetic_batch["lm_hidden_states"].to(torch.bfloat16)
    outputs = tiny_model.forward_train_from_precomputed_lm(
        batch=synthetic_batch,
        lm_hidden_states=lm_hidden_states,
        stage=1,
        seed=0,
        num_recycles_for_test=1,
        confidence_num_sampling_steps=0,
    )
    assert outputs["z"].dtype == next(tiny_model.language_model.parameters()).dtype


def test_confidence_pair_dropout_masks_normalized_trunk_bias(tiny_model, synthetic_batch):
    outputs = tiny_model.forward_train_from_precomputed_lm(
        batch=synthetic_batch,
        lm_hidden_states=synthetic_batch["lm_hidden_states"],
        stage=1,
        seed=0,
        num_recycles_for_test=1,
        confidence_num_sampling_steps=0,
    )
    confidence_head = tiny_model.confidence_head
    atom_to_token = synthetic_batch["atom_to_token"].long()
    trunk_z_mask = torch.zeros(
        outputs["z"].shape[0], 1, 1, 1, device=outputs["z"].device, dtype=torch.bool
    )

    with torch.no_grad():
        confidence_head.z_norm.bias.zero_()
    baseline = _safe_confidence_head_forward(
        confidence_head,
        s_inputs=outputs["x_inputs"].detach(),
        z=outputs["z"].detach(),
        x_pred=outputs["x_pred"].detach(),
        distogram_atom_idx=synthetic_batch["distogram_atom_idx"],
        token_attention_mask=synthetic_batch["token_attention_mask"].bool(),
        atom_to_token=atom_to_token,
        atom_attention_mask=synthetic_batch["atom_attention_mask"].bool(),
        asym_id=synthetic_batch["asym_id"],
        mol_type=synthetic_batch["mol_type"],
        num_diffusion_samples=1,
        relative_position_encoding=outputs["relative_position_encoding"].detach(),
        token_bonds_encoding=outputs["token_bonds_encoding"].detach(),
        trunk_z_mask=trunk_z_mask,
    )

    with torch.no_grad():
        confidence_head.z_norm.bias.fill_(5.0)
    masked = _safe_confidence_head_forward(
        confidence_head,
        s_inputs=outputs["x_inputs"].detach(),
        z=outputs["z"].detach(),
        x_pred=outputs["x_pred"].detach(),
        distogram_atom_idx=synthetic_batch["distogram_atom_idx"],
        token_attention_mask=synthetic_batch["token_attention_mask"].bool(),
        atom_to_token=atom_to_token,
        atom_attention_mask=synthetic_batch["atom_attention_mask"].bool(),
        asym_id=synthetic_batch["asym_id"],
        mol_type=synthetic_batch["mol_type"],
        num_diffusion_samples=1,
        relative_position_encoding=outputs["relative_position_encoding"].detach(),
        token_bonds_encoding=outputs["token_bonds_encoding"].detach(),
        trunk_z_mask=trunk_z_mask,
    )
    assert torch.allclose(masked["plddt_logits"], baseline["plddt_logits"])
    assert torch.allclose(masked["pae_logits"], baseline["pae_logits"])


def test_confidence_head_accepts_bfloat16_inputs_with_float_weights(
    tiny_model, synthetic_batch
):
    outputs = tiny_model.forward_train_from_precomputed_lm(
        batch=synthetic_batch,
        lm_hidden_states=synthetic_batch["lm_hidden_states"],
        stage=1,
        seed=0,
        num_recycles_for_test=1,
        confidence_num_sampling_steps=0,
    )
    tiny_model.confidence_head.float()
    confidence_output = _safe_confidence_head_forward(
        tiny_model.confidence_head,
        s_inputs=outputs["x_inputs"].detach().to(torch.bfloat16),
        z=outputs["z"].detach().to(torch.bfloat16),
        x_pred=outputs["x_pred"].detach().to(torch.bfloat16),
        distogram_atom_idx=synthetic_batch["distogram_atom_idx"],
        token_attention_mask=synthetic_batch["token_attention_mask"].bool(),
        atom_to_token=synthetic_batch["atom_to_token"].long(),
        atom_attention_mask=synthetic_batch["atom_attention_mask"].bool(),
        asym_id=synthetic_batch["asym_id"],
        mol_type=synthetic_batch["mol_type"],
        num_diffusion_samples=1,
        relative_position_encoding=outputs["relative_position_encoding"]
        .detach()
        .to(torch.bfloat16),
        token_bonds_encoding=outputs["token_bonds_encoding"].detach().to(torch.bfloat16),
    )
    assert confidence_output["plddt_logits"].dtype == torch.float32
