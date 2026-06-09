from esm.training.esmfold2_frozen_features.dataset import ESMFold2FrozenFeatureDataset


def test_dataset_parses_mmcif_chain():
    dataset = ESMFold2FrozenFeatureDataset(
        "data/sample_nanobody10/manifest.csv", require_cache=False
    )
    item = dataset[0]
    assert 80 < len(item["sequence"]) < 180
    assert item["gt_atom_coords"].shape[-1] == 3
    assert item["gt_atom_mask"].any()
    assert item["atom_attention_mask"].any()
    assert "SINGLE-DOMAIN" in item["entity_description"].upper()
