from esm.training.esmfold2_frozen_features.losses import compute_esmfold2_loss
from esm.training.esmfold2_frozen_features.modeling import build_optimizer


def test_stage_switches(tiny_model, synthetic_batch):
    build_optimizer(tiny_model, stage=1)
    outputs = tiny_model.forward_train_from_precomputed_lm(
        batch=synthetic_batch,
        lm_hidden_states=synthetic_batch["lm_hidden_states"],
        stage=1,
        seed=0,
        num_recycles_for_test=1,
        confidence_num_sampling_steps=0,
    )
    stage1 = compute_esmfold2_loss(outputs, synthetic_batch, stage=1)
    stage2 = compute_esmfold2_loss(outputs, synthetic_batch, stage=2)
    stage3 = compute_esmfold2_loss(outputs, synthetic_batch, stage=3)
    assert stage1["loss_slddt"] > 0
    assert stage1["loss_bond"] == 0
    assert stage2["loss_slddt"] == 0
    assert stage2["loss_dist"] > 0
    assert stage3["loss_dist"] == 0

    _, param_names = build_optimizer(tiny_model, stage=3)
    assert any(name.startswith("structure_head.") for name in param_names)
    assert any(name.startswith("confidence_head.") for name in param_names)
    assert not any(name.startswith("folding_trunk.") for name in param_names)
