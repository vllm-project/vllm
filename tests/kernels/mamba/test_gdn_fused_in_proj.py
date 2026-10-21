# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``QwenGatedDeltaNetAttention`` folds ``in_proj_ba`` into the qkvz GEMM.

After loading, the ba rows are appended to ``in_proj_qkvz``'s weight so the
CUDA forward runs one input GEMM per layer instead of two.
"""

from __future__ import annotations

import types

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)

HIDDEN = 64
N_QKVZ = 96
N_BA = 6  # 2 * num_v_heads / tp, not a multiple of 8


class _Linear(nn.Module):
    def __init__(self, n: int, k: int, quant_method=None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n, k), requires_grad=False)
        self.bias = None
        self.quant_method = quant_method or UnquantizedLinearMethod()
        self.calls = 0

    def forward(self, x: torch.Tensor):
        self.calls += 1
        return x @ self.weight.t(), None


def _layer(*, lora=False, disable_tp_ba=False, ba_quant_method=None):
    layer = types.SimpleNamespace(
        in_proj_qkvz=_Linear(N_QKVZ, HIDDEN),
        in_proj_ba=_Linear(N_BA, HIDDEN, ba_quant_method),
        in_proj_fused_split=None,
        _lora_enabled=lora,
        disable_tp_for_ba_proj=disable_tp_ba,
    )
    for name in ("_can_fuse_in_proj", "process_weights_after_loading", "project_input"):
        method = getattr(QwenGatedDeltaNetAttention, name)
        setattr(layer, name, types.MethodType(method, layer))
    return layer


@pytest.fixture
def on_cuda(monkeypatch):
    platform = types.SimpleNamespace(is_cuda=lambda: True)
    monkeypatch.setattr(qwen_gdn_linear_attn, "current_platform", platform)


@torch.inference_mode()
def test_fused_projection_matches_separate(default_vllm_config, on_cuda) -> None:
    layer = _layer()
    x = torch.randn(5, HIDDEN)
    ref_qkvz = x @ layer.in_proj_qkvz.weight.t()
    ref_ba = x @ layer.in_proj_ba.weight.t()

    layer.process_weights_after_loading(torch.float32)

    assert layer.in_proj_fused_split == (N_QKVZ, N_BA)
    fused = layer.in_proj_qkvz.weight
    assert fused.shape == (112, HIDDEN)  # 96 + 6 rows padded to a multiple of 16
    assert torch.count_nonzero(fused[N_QKVZ + N_BA :]) == 0
    # in_proj_ba keeps a view of its rows, so weight loading still lands there
    ba_w = layer.in_proj_ba.weight
    assert ba_w.shape == (N_BA, HIDDEN)
    assert ba_w.data_ptr() == fused.data_ptr() + N_QKVZ * HIDDEN * fused.element_size()

    qkvz, ba = layer.project_input(x)
    assert layer.in_proj_qkvz.calls == 1
    assert layer.in_proj_ba.calls == 0
    torch.testing.assert_close(qkvz, ref_qkvz)
    torch.testing.assert_close(ba, ref_ba)

    # A later write through in_proj_ba (e.g. a weight reload) reaches the GEMM.
    ba_w.data.zero_()
    _, ba = layer.project_input(x)
    assert torch.count_nonzero(ba) == 0

    # Idempotent.
    layer.process_weights_after_loading(torch.float32)
    assert layer.in_proj_qkvz.weight.shape == (112, HIDDEN)


class _OtherMethod(UnquantizedLinearMethod):
    pass


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(lora=True),
        dict(disable_tp_ba=True),
        dict(ba_quant_method="other"),
    ],
)
@torch.inference_mode()
def test_not_fused(default_vllm_config, on_cuda, kwargs) -> None:
    if kwargs.get("ba_quant_method") == "other":
        kwargs["ba_quant_method"] = _OtherMethod()
    layer = _layer(**kwargs)
    layer.process_weights_after_loading(torch.float32)
    assert layer.in_proj_fused_split is None
    assert layer.in_proj_qkvz.weight.shape == (N_QKVZ, HIDDEN)
    layer.project_input(torch.randn(2, HIDDEN))
    assert layer.in_proj_qkvz.calls == layer.in_proj_ba.calls == 1


@torch.inference_mode()
def test_not_fused_off_cuda(default_vllm_config, monkeypatch) -> None:
    platform = types.SimpleNamespace(is_cuda=lambda: False)
    monkeypatch.setattr(qwen_gdn_linear_attn, "current_platform", platform)
    layer = _layer()
    layer.process_weights_after_loading(torch.float32)
    assert layer.in_proj_fused_split is None
