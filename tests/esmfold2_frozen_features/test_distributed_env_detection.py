from esm.training.esmfold2_frozen_features.distributed import (
    detect_distributed_env,
    distributed_requested,
)


def test_torchrun_style_env_var_parsing():
    env = detect_distributed_env(
        {
            "RANK": "3",
            "LOCAL_RANK": "1",
            "WORLD_SIZE": "8",
            "LOCAL_WORLD_SIZE": "4",
            "MASTER_ADDR": "host0",
            "MASTER_PORT": "12345",
        }
    )
    assert env.source == "torchrun"
    assert env.rank == 3
    assert env.local_rank == 1
    assert env.world_size == 8
    assert env.local_world_size == 4
    assert env.master_addr == "host0"
    assert env.master_port == "12345"
    assert distributed_requested("auto", env)


def test_slurm_style_env_var_parsing():
    env = detect_distributed_env(
        {
            "SLURM_PROCID": "9",
            "SLURM_LOCALID": "1",
            "SLURM_NTASKS": "16",
            "SLURM_NTASKS_PER_NODE": "4(x4)",
            "MASTER_ADDR": "node-a",
        }
    )
    assert env.source == "slurm"
    assert env.rank == 9
    assert env.local_rank == 1
    assert env.world_size == 16
    assert env.local_world_size == 4
    assert env.master_addr == "node-a"
    assert distributed_requested("ddp", env)


def test_single_process_fallback():
    env = detect_distributed_env({})
    assert env.source == "single"
    assert env.rank == 0
    assert env.local_rank == 0
    assert env.world_size == 1
    assert not distributed_requested("auto", env)
    assert not distributed_requested("none", env)
