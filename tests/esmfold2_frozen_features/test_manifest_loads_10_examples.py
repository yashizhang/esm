from pathlib import Path

from esm.training.esmfold2_frozen_features.dataset import SAMPLE_MANIFEST_TEXT, read_manifest


def test_manifest_loads_10_examples():
    manifest_path = Path("data/sample_nanobody10/manifest.csv")
    assert manifest_path.read_text() == SAMPLE_MANIFEST_TEXT

    rows = read_manifest(manifest_path)
    assert len(rows) == 10
    for row in rows:
        assert row.pdb_id
        assert row.label_asym_id
        assert row.auth_asym_id
        assert row.mmcif_url.startswith("https://")
        assert row.expected_entity_name_regex
        assert row.chain_match_policy == "label_asym_id_first_then_auth_asym_id"
