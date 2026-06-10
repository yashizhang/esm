import argparse
import sys
import types

from esm.training.esmfold2_frozen_features.distributed import DistributedEnv
from esm.training.esmfold2_frozen_features.logging_utils import JsonlLogger, maybe_tqdm


def test_nonzero_rank_disables_tqdm_output(monkeypatch, capsys):
    monkeypatch.setenv("RANK", "1")
    bar = maybe_tqdm(range(2), enabled=True, total=2)
    for _ in bar:
        bar.update(1)
    bar.close()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_jsonl_writes_rank0_only(monkeypatch, tmp_path):
    path = tmp_path / "train_log.jsonl"
    monkeypatch.setenv("RANK", "1")
    with JsonlLogger(path, enabled=True) as logger:
        assert not logger.write({"step": 1})
    assert not path.exists()

    monkeypatch.setenv("RANK", "0")
    with JsonlLogger(path, enabled=True) as logger:
        assert logger.write({"step": 1})
    assert path.read_text() == '{"step": 1}\n'


def test_nonzero_rank_does_not_initialize_wandb(monkeypatch, finetune_script):
    called = {"init": 0}

    def fake_init(**_):
        called["init"] += 1
        raise AssertionError("wandb.init should not run on nonzero ranks")

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(init=fake_init))
    args = argparse.Namespace(
        wandb_mode="offline",
        wandb_project="project",
        wandb_entity=None,
        wandb_run_name=None,
        wandb_group=None,
        wandb_tags=[],
        stage=1,
    )
    run = finetune_script.init_wandb_run(
        args,
        dist_env=DistributedEnv(rank=1, local_rank=1, world_size=4),
        run_config={},
    )
    assert run is None
    assert called["init"] == 0
