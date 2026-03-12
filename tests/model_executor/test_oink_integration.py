# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import types

import pytest
import torch


def _load_oink_ops_module():
    # Import the module normally (vllm is installed as an editable package in CI).
    from vllm import _oink_ops

    return _oink_ops


def test_oink_availability_checks(monkeypatch: pytest.MonkeyPatch):
    _oink_ops = _load_oink_ops_module()

    # Ensure the ops namespace exists and is mutable for tests.
    monkeypatch.setattr(
        torch.ops,
        "oink",
        types.SimpleNamespace(rmsnorm=lambda x, w, eps: x),
        raising=False,
    )

    # Case 1: CUDA not available.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _oink_ops.is_oink_available_for_device(0) is False

    # Case 2: CUDA available but < SM100.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda idx: (9, 0))
    assert _oink_ops.is_oink_available_for_device(0) is False

    # Case 3: CUDA available and SM100, rmsnorm op registered.
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda idx: (10, 0))
    assert _oink_ops.is_oink_available_for_device(0) is True

    # fused op presence probe
    assert _oink_ops.has_fused_add_rms_norm() is False
    monkeypatch.setattr(
        torch.ops,
        "oink",
        types.SimpleNamespace(
            rmsnorm=lambda x, w, eps: x,
            fused_add_rms_norm=lambda x, residual, w, eps: None,
        ),
        raising=False,
    )
    assert _oink_ops.has_fused_add_rms_norm() is True


def test_can_view_as_2d_stride_guard():
    # Import the helper from the layernorm module.
    from vllm.model_executor.layers.layernorm import _can_view_as_2d

    x = torch.zeros((2, 3, 4))
    assert _can_view_as_2d(x) is True

    # Size-1 dims should be ignored by the viewability check.
    # Create a tensor where stride(0) != stride(1) * size(1) due to padding,
    # but view(-1, H) is still valid because dim 1 has size 1.
    base = torch.zeros((2, 10, 4))
    x_singleton = base[:, :1, :]
    x_singleton.view(-1, x_singleton.shape[-1])
    assert _can_view_as_2d(x_singleton) is True

    # Middle-dimension stride break: view(-1, hidden) should be invalid.
    x2 = x[:, ::2, :]
    with pytest.raises(RuntimeError):
        x2.view(-1, x2.shape[-1])
    assert _can_view_as_2d(x2) is False
