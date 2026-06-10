import argparse

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from esm.training.esmfold2_frozen_features.distributed import DistributedEnv


class DummyDataset(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return {"x": torch.tensor(index)}


def _args():
    return argparse.Namespace(
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        seed=13,
    )


def test_distributed_sampler_used_when_world_size_gt_one(finetune_script):
    dataset = DummyDataset()
    loader, sampler = finetune_script.create_train_dataloader(
        dataset,
        _args(),
        DistributedEnv(rank=1, local_rank=1, world_size=4),
        use_ddp=True,
    )
    assert isinstance(sampler, DistributedSampler)
    assert loader.sampler is sampler
    assert sampler.num_replicas == 4
    assert sampler.rank == 1


def test_single_process_loader_behavior_unchanged(finetune_script):
    dataset = DummyDataset()
    loader, sampler = finetune_script.create_train_dataloader(
        dataset,
        _args(),
        DistributedEnv(rank=0, local_rank=0, world_size=1),
        use_ddp=False,
    )
    assert sampler is None
    assert not isinstance(loader.sampler, DistributedSampler)
