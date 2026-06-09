import pytest
import torch.nn as nn

from esm.training.esmfold2_frozen_features.modeling import build_optimizer


def test_no_esmc_in_training_optimizer(tiny_model):
    _, param_names = build_optimizer(tiny_model, stage=1)
    assert param_names
    assert not any("esmc" in name.lower() for name in param_names)


def test_optimizer_rejects_loaded_esmc(tiny_model):
    tiny_model._esmc = nn.Linear(1, 1)
    with pytest.raises(RuntimeError, match="ESMC must not be loaded"):
        build_optimizer(tiny_model, stage=1)
