import pytest

import torch

from esm.training.esmfold2_frozen_features.distributed import (
    resolve_precision,
    uses_grad_scaler,
)


def test_precision_auto_selects_fp16_on_v100(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 0))
    assert resolve_precision("auto", device_type="cuda") == "fp16"
    assert uses_grad_scaler("fp16", "cuda")


def test_precision_auto_selects_bf16_on_ampere_or_newer(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))
    assert resolve_precision("auto", device_type="cuda") == "bf16"


def test_explicit_bf16_rejected_on_v100(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 0))
    with pytest.raises(ValueError, match="use fp16 on V100"):
        resolve_precision("bf16", device_type="cuda")


def test_precision_auto_cpu_is_fp32():
    assert resolve_precision("auto", device_type="cpu") == "fp32"
