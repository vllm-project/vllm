# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``QwenGatedDeltaNetAttention`` folds ``in_proj_ba`` into the qkvz GEMM.

After loading, the qkvz and ba rows are stacked in one buffer so the CUDA
forward runs one input GEMM per layer instead of two.
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
from vllm.model_executor.layers.utils import default_unquantized_gemm
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

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


class _GDN(nn.Module):
    """The input-projection part of ``QwenGatedDeltaNetAttention``."""

    _can_fuse_in_proj = QwenGatedDeltaNetAttention._can_fuse_in_proj
    process_weights_after_loading = (
        QwenGatedDeltaNetAttention.process_weights_after_loading
    )
    project_input = QwenGatedDeltaNetAttention.project_input

    def __init__(self, *, lora=False, disable_tp_ba=False, ba_quant_method=None):
        super().__init__()
        self.in_proj_qkvz = _Linear(N_QKVZ, HIDDEN)
        self.in_proj_ba = _Linear(N_BA, HIDDEN, ba_quant_method)
        self.in_proj_fused_split = None
        self._ba_stream = None
        self._lora_enabled = lora
        self.disable_tp_for_ba_proj = disable_tp_ba
        self.enable_fused_gdn_spec_decode = False
        self._in_proj_fused_gemm = default_unquantized_gemm


@pytest.fixture
def on_cuda(monkeypatch):
    platform = types.SimpleNamespace(is_cuda=lambda: True)
    monkeypatch.setattr(qwen_gdn_linear_attn, "current_platform", platform)


@torch.inference_mode()
def test_fused_projection_matches_separate(default_vllm_config, on_cuda) -> None:
    layer = _GDN()
    x = torch.randn(5, HIDDEN)
    ref_qkvz = x @ layer.in_proj_qkvz.weight.t()
    ref_ba = x @ layer.in_proj_ba.weight.t()

    layer.process_weights_after_loading(torch.float32)

    assert layer.in_proj_fused_split == (N_QKVZ, N_BA)
    fused = layer.in_proj_fused_weight
    assert fused.shape == (112, HIDDEN)  # 96 + 6 rows padded to a multiple of 16
    assert torch.count_nonzero(fused[N_QKVZ + N_BA :]) == 0
    assert "in_proj_fused_weight" not in layer.state_dict()
    # Both weights keep their shapes as views of their rows, so weight loading
    # still lands there.
    qkvz_w, ba_w = layer.in_proj_qkvz.weight, layer.in_proj_ba.weight
    assert qkvz_w.shape == (N_QKVZ, HIDDEN)
    assert qkvz_w.data_ptr() == fused.data_ptr()
    assert ba_w.shape == (N_BA, HIDDEN)
    assert ba_w.data_ptr() == fused.data_ptr() + N_QKVZ * HIDDEN * fused.element_size()

    qkvz, ba = layer.project_input(x)
    assert layer.in_proj_qkvz.calls == layer.in_proj_ba.calls == 0
    torch.testing.assert_close(qkvz, ref_qkvz)
    torch.testing.assert_close(ba, ref_ba)

    # A later write through in_proj_ba (e.g. a weight reload) reaches the GEMM.
    ba_w.data.zero_()
    _, ba = layer.project_input(x)
    assert torch.count_nonzero(ba) == 0

    # Idempotent.
    layer.process_weights_after_loading(torch.float32)
    assert layer.in_proj_fused_weight is fused
    assert layer.in_proj_qkvz.weight.shape == (N_QKVZ, HIDDEN)


@torch.inference_mode()
def test_fused_projection_survives_layerwise_reload(
    default_vllm_config, on_cuda, monkeypatch
) -> None:
    # The unquantized method does no post-load processing on CUDA.
    monkeypatch.setattr(
        UnquantizedLinearMethod, "process_weights_after_loading", lambda *_: None
    )
    layer = _GDN()
    model = nn.Sequential(layer)
    record_metadata_for_reloading(model)  # at model construction, before fusion
    layer.process_weights_after_loading(torch.float32)
    storage = layer.in_proj_ba.weight.untyped_storage().data_ptr()

    new_qkvz = torch.randn(N_QKVZ, HIDDEN)
    new_ba = torch.randn(N_BA, HIDDEN)
    initialize_layerwise_reload(model)
    for proj, value in ((layer.in_proj_qkvz, new_qkvz), (layer.in_proj_ba, new_ba)):
        loader = getattr(proj.weight, "weight_loader", default_weight_loader)
        loader(proj.weight, value)
    finalize_layerwise_reload(model, model_config=None)

    # The reload wrote into the storage the forward (and its CUDA graphs) reads.
    fused = layer.in_proj_fused_weight
    assert "in_proj_fused_weight" in layer._non_persistent_buffers_set
    assert fused.untyped_storage().data_ptr() == storage
    assert layer.in_proj_qkvz.weight.data_ptr() == fused.data_ptr()
    torch.testing.assert_close(fused[:N_QKVZ], new_qkvz)
    torch.testing.assert_close(fused[N_QKVZ : N_QKVZ + N_BA], new_ba)
    assert torch.count_nonzero(fused[N_QKVZ + N_BA :]) == 0
    x = torch.randn(3, HIDDEN)
    qkvz, ba = layer.project_input(x)
    torch.testing.assert_close(qkvz, x @ new_qkvz.t())
    torch.testing.assert_close(ba, x @ new_ba.t())


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
    layer = _GDN(**kwargs)
    layer.process_weights_after_loading(torch.float32)
    assert layer.in_proj_fused_split is None
    assert layer.in_proj_qkvz.weight.shape == (N_QKVZ, HIDDEN)
    layer.project_input(torch.randn(2, HIDDEN))
    assert layer.in_proj_qkvz.calls == layer.in_proj_ba.calls == 1


@torch.inference_mode()
def test_not_fused_off_cuda(default_vllm_config, monkeypatch) -> None:
    platform = types.SimpleNamespace(is_cuda=lambda: False)
    monkeypatch.setattr(qwen_gdn_linear_attn, "current_platform", platform)
    layer = _GDN()
    layer.process_weights_after_loading(torch.float32)
    assert layer.in_proj_fused_split is None
