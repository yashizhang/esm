import argparse

import torch

from esm.training.esmfold2_frozen_features.config import ESMFold2FrozenFeatureConfig


def _args(tmp_path):
    return argparse.Namespace(
        max_steps=3,
        output_dir=str(tmp_path),
        stage=1,
        batch_size=1,
        gradient_accumulation_steps=1,
        manifest="manifest.csv",
        cache_index="cache_index.jsonl",
        model_checkpoint="tiny-random",
    )


def test_checkpoint_write_is_rank0_only(tmp_path, finetune_script):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = ESMFold2FrozenFeatureConfig()

    nonzero_path = tmp_path / "rank1" / "checkpoint_last.pt"
    wrote = finetune_script.save_checkpoint(
        nonzero_path,
        model,
        optimizer,
        scaler=None,
        stage=1,
        epoch=0,
        global_step=2,
        args=_args(tmp_path),
        config=config,
        optimizer_param_names=["weight", "bias"],
        resolved_precision="fp32",
        is_main=False,
    )
    assert not wrote
    assert not nonzero_path.exists()

    rank0_path = tmp_path / "rank0" / "checkpoint_last.pt"
    wrote = finetune_script.save_checkpoint(
        rank0_path,
        model,
        optimizer,
        scaler=None,
        stage=1,
        epoch=0,
        global_step=2,
        args=_args(tmp_path),
        config=config,
        optimizer_param_names=["weight", "bias"],
        resolved_precision="fp32",
        is_main=True,
    )
    assert wrote
    checkpoint = torch.load(rank0_path, map_location="cpu", weights_only=False)
    assert checkpoint["global_step"] == 2
    assert checkpoint["stage"] == 1
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
