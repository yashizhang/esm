import json

import pytest
import torch

from esm.training.esmfold2_frozen_features.dataset import ESMFold2FrozenFeatureDataset


def test_precomputed_cache_shapes(tmp_path):
    base_dataset = ESMFold2FrozenFeatureDataset(
        "data/sample_nanobody10/manifest.csv", require_cache=False
    )
    item = base_dataset[0]
    hidden = torch.randn(len(item["sequence"]), 2, 8, dtype=torch.float32)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    hidden_path = cache_dir / "nanobody_001.pt"
    torch.save(hidden, hidden_path)
    index_path = tmp_path / "cache_index.jsonl"
    record = {
        "example_id": "nanobody_001",
        "sequence": item["sequence"],
        "hidden_states_path": str(hidden_path),
        "shape": list(hidden.shape),
        "dtype": "float32",
        "model": "test",
    }
    index_path.write_text(json.dumps(record) + "\n")

    dataset = ESMFold2FrozenFeatureDataset(
        "data/sample_nanobody10/manifest.csv", cache_index=index_path
    )
    cached = dataset[0]
    assert list(cached["lm_hidden_states"].shape) == [len(item["sequence"]), 2, 8]
    assert cached["lm_hidden_states"].shape[0] == len(cached["sequence"])


def test_cache_metadata_mismatch_fails(tmp_path):
    base_dataset = ESMFold2FrozenFeatureDataset(
        "data/sample_nanobody10/manifest.csv", require_cache=False
    )
    item = base_dataset[0]
    hidden = torch.randn(len(item["sequence"]), 2, 8, dtype=torch.float32)
    hidden_path = tmp_path / "nanobody_001.pt"
    torch.save(hidden, hidden_path)
    index_path = tmp_path / "cache_index.jsonl"
    record = {
        "example_id": "nanobody_001",
        "sequence": item["sequence"],
        "hidden_states_path": str(hidden_path),
        "shape": [len(item["sequence"]), 3, 8],
        "dtype": "float32",
        "model": "test",
    }
    index_path.write_text(json.dumps(record) + "\n")

    dataset = ESMFold2FrozenFeatureDataset(
        "data/sample_nanobody10/manifest.csv", cache_index=index_path
    )
    with pytest.raises(ValueError, match="cache shape metadata"):
        dataset[0]
