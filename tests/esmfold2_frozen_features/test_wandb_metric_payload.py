import argparse
import sys
import types

import pytest

from esm.training.esmfold2_frozen_features.distributed import DistributedEnv
from esm.training.esmfold2_frozen_features.losses import LOSS_KEYS


class FakeWandbRun:
    def __init__(self) -> None:
        self.logs = []
        self.finished = False

    def log(self, payload, step=None):
        self.logs.append((dict(payload), step))

    def finish(self):
        self.finished = True


def test_wandb_payload_logs_all_loss_keys_and_optimizer_step(finetune_script):
    dist_env = DistributedEnv(rank=0, local_rank=0, world_size=4)
    loss_metrics = {key: float(index) for index, key in enumerate(LOSS_KEYS)}
    payload = finetune_script.build_training_metric_payload(
        loss_metrics,
        lr=1.8e-4,
        weight_decay=1e-4,
        grad_norm=2.0,
        grad_scale=128.0,
        data_metrics={
            "data/batch_size_per_rank": 1.0,
            "data/tokens_per_rank": 10.0,
            "data/atoms_per_rank": 100.0,
            "data/tokens_global": 40.0,
            "data/atoms_global": 400.0,
        },
        perf_metrics={
            "perf/step_time_sec": 0.5,
            "perf/samples_per_sec_global": 8.0,
            "perf/tokens_per_sec_global": 80.0,
            "perf/atoms_per_sec_global": 800.0,
            "perf/gpu_max_allocated_gb": 1.0,
            "perf/gpu_max_reserved_gb": 2.0,
        },
        dist_env=dist_env,
        global_step=7,
        epoch=2,
        effective_global_batch_size=4,
    )

    for key in LOSS_KEYS:
        assert payload[f"train/{key}"] == loss_metrics[key]
    assert payload["opt/lr"] == 1.8e-4
    assert payload["opt/grad_scale"] == 128.0
    assert payload["dist/global_step"] == 7

    run = FakeWandbRun()
    finetune_script._log_wandb(run, payload, global_step=7, every_n_steps=1)
    assert run.logs == [(payload, 7)]


@pytest.mark.parametrize("wandb_mode", ["offline", "online"])
def test_rank0_wandb_init_uses_fake_module(monkeypatch, finetune_script, wandb_mode):
    fake_run = FakeWandbRun()
    captured = {}

    def fake_init(**kwargs):
        captured.update(kwargs)
        return fake_run

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(init=fake_init))
    args = argparse.Namespace(
        wandb_mode=wandb_mode,
        wandb_project="project",
        wandb_entity="entity",
        wandb_run_name="run-name",
        wandb_group="group",
        wandb_tags=["smoke"],
        stage=1,
    )
    run = finetune_script.init_wandb_run(
        args,
        dist_env=DistributedEnv(rank=0, local_rank=0, world_size=1),
        run_config={"world_size": 1},
    )
    assert run is fake_run
    assert captured["mode"] == wandb_mode
    assert captured["project"] == "project"
    assert captured["name"] == "run-name"
    assert captured["config"] == {"world_size": 1}
