from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from esm.models.esmfold2.prepare_input import prepare_esmfold2_input
from esm.models.esmfold2.types import ProteinInput, StructurePredictionInput
from esm.training.esmfold2_frozen_features.collate import collate_esmfold2_frozen_features
from esm.training.esmfold2_frozen_features.modeling import tiny_esmfold2_model


@pytest.fixture()
def synthetic_batch():
    sequence = "ACDEFGHIK"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features, _ = prepare_esmfold2_input(
            StructurePredictionInput(
                sequences=[ProteinInput(id="A", sequence=sequence, msa=None)]
            )
        )
    n_atoms = features["atom_attention_mask"].shape[0]
    atom_mask = features["atom_attention_mask"].bool()
    generator = torch.Generator().manual_seed(7)
    features["gt_atom_coords"] = torch.randn(n_atoms, 3, generator=generator)
    features["gt_atom_mask"] = atom_mask.clone()
    features["atom_loss_weight"] = atom_mask.float()
    features["lm_hidden_states"] = torch.randn(
        features["token_attention_mask"].shape[0], 2, 8, generator=generator
    )
    features["resolution"] = torch.tensor(2.0)
    features["example_id"] = "synthetic"
    features["sequence"] = sequence
    return collate_esmfold2_frozen_features([features])


@pytest.fixture()
def tiny_model():
    return tiny_esmfold2_model()


@pytest.fixture()
def finetune_script():
    path = ROOT / "scripts" / "esmfold2_frozen_feature_finetune.py"
    spec = importlib.util.spec_from_file_location(
        "esmfold2_frozen_feature_finetune", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
