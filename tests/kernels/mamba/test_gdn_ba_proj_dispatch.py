# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eligibility rules for routing the GDN b/a projection to FlashInfer."""

import pytest
import torch

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn

MAX_TOKENS = 4


class _Layer:
    """Stand-in for in_proj_ba; a quantized layer has `qweight`, no `weight`."""

    def __init__(self, quant_method, *, weight=None, bias=None, qweight=None):
        self.quant_method = quant_method
        if weight is not None:
            self.weight = weight
        if bias is not None:
            self.bias = bias
        if qweight is not None:
            self.qweight = qweight


def _unquantized(**kwargs):
    weight = kwargs.pop("weight", torch.zeros(96, 5120, dtype=torch.bfloat16))
    return _Layer(UnquantizedLinearMethod(), weight=weight, **kwargs)


def _quantized():
    return _Layer(object(), qweight=torch.zeros(96, 640, dtype=torch.int32))


@pytest.fixture(autouse=True)
def _backend_supported(monkeypatch):
    monkeypatch.setattr(gdn, "is_flashinfer_bf16_gemm_supported", lambda _: True)
    monkeypatch.setattr(
        gdn.current_platform, "is_device_capability_family", lambda c: c == 120
    )
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")


def test_eligible_when_unquantized_and_supported():
    assert gdn._ba_gemv_eligible(_unquantized(), MAX_TOKENS, lora_enabled=False)


@pytest.mark.parametrize("max_tokens", [0, -1])
def test_disabled_by_default(max_tokens):
    assert not gdn._ba_gemv_eligible(_unquantized(), max_tokens, lora_enabled=False)


@pytest.mark.parametrize("max_tokens", [0, MAX_TOKENS])
def test_quantized_layer_declines_without_touching_weight(max_tokens):
    """Reading `weight` before the quantization check raised AttributeError."""
    assert not gdn._ba_gemv_eligible(_quantized(), max_tokens, lora_enabled=False)


def test_lora_declines():
    """LoRA wraps the module after construction; bypassing it drops the adapter."""
    assert not gdn._ba_gemv_eligible(_unquantized(), MAX_TOKENS, lora_enabled=True)


def test_batch_invariant_declines(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    assert not gdn._ba_gemv_eligible(_unquantized(), MAX_TOKENS, lora_enabled=False)


def test_unsupported_backend_declines(monkeypatch):
    monkeypatch.setattr(gdn, "is_flashinfer_bf16_gemm_supported", lambda _: False)
    assert not gdn._ba_gemv_eligible(_unquantized(), MAX_TOKENS, lora_enabled=False)


def test_non_sm12x_declines(monkeypatch):
    """On sm_10x cuBLAS already picks a good kernel; flashinfer is 0.93-1.13x."""
    monkeypatch.setattr(
        gdn.current_platform, "is_device_capability_family", lambda c: c == 100
    )
    assert not gdn._ba_gemv_eligible(_unquantized(), MAX_TOKENS, lora_enabled=False)


def test_non_2d_weight_declines():
    assert not gdn._ba_gemv_eligible(
        _unquantized(weight=torch.zeros(2, 96, 5120, dtype=torch.bfloat16)),
        MAX_TOKENS,
        lora_enabled=False,
    )


def test_bias_declines():
    bias = torch.zeros(96, dtype=torch.bfloat16)
    layer = _unquantized(bias=bias)
    assert not gdn._ba_gemv_eligible(layer, MAX_TOKENS, lora_enabled=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA tensors")
def test_shape_guard_bounds_projection_width():
    """Past the bound cuBLAS is faster, so wide N must fall back."""
    x = torch.zeros(4, 5120, dtype=torch.bfloat16, device="cuda")
    narrow = torch.zeros(gdn._BA_PROJ_MAX_N, 5120, dtype=torch.bfloat16, device="cuda")
    wide = torch.zeros(
        gdn._BA_PROJ_MAX_N + 16, 5120, dtype=torch.bfloat16, device="cuda"
    )
    assert gdn._ba_proj_flashinfer_ok(x, narrow)
    assert not gdn._ba_proj_flashinfer_ok(x, wide)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA tensors")
def test_shape_guard_enforces_flashinfer_m_bound():
    """mm_bf16 accepts 1 <= m <= 32; oversized token counts fall back."""
    weight = torch.zeros(96, 5120, dtype=torch.bfloat16, device="cuda")
    over = torch.zeros(
        gdn._BA_PROJ_MAX_TOKENS + 1, 5120, dtype=torch.bfloat16, device="cuda"
    )
    assert not gdn._ba_proj_flashinfer_ok(over, weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA tensors")
def test_single_token_falls_back():
    """CuBLAS has a GEMV path at one token; flashinfer is 0.5-0.9x there on
    every shipped model shape except Qwen3.8-27B."""
    weight = torch.zeros(96, 5120, dtype=torch.bfloat16, device="cuda")
    one = torch.zeros(1, 5120, dtype=torch.bfloat16, device="cuda")
    two = torch.zeros(2, 5120, dtype=torch.bfloat16, device="cuda")
    assert not gdn._ba_proj_flashinfer_ok(one, weight)
    assert gdn._ba_proj_flashinfer_ok(two, weight)


# (N, K) of the b/a projection for every shipped Qwen GDN model:
# N = 2 * linear_num_value_heads, K = hidden_size.
MODEL_SHAPES = [
    pytest.param(32, 1024, id="Qwen3.5-0.8B"),
    pytest.param(64, 2048, id="Qwen3-Next-80B-A3B"),
    pytest.param(64, 2560, id="Qwen3.5-4B"),
    pytest.param(96, 2560, id="Qwen3.8-Flash-Next"),
    pytest.param(96, 5120, id="Qwen3.8-27B"),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA tensors")
@pytest.mark.parametrize("n,k", MODEL_SHAPES)
@pytest.mark.parametrize("m", [2, 4])
def test_routed_path_matches_linear(n, k, m):
    """The routed call must agree with the layer it bypasses.

    Guards the operand layout: mm_bf16 takes b already transposed, so a
    mistake here stays silent -- every eligibility test still passes.
    """
    torch.manual_seed(0)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02
    assert gdn._ba_proj_flashinfer_ok(x, w)
    out = gdn.flashinfer_bf16_mm(x, w.t(), None, False, gdn._BA_PROJ_BACKEND)
    ref = torch.nn.functional.linear(x, w)
    torch.testing.assert_close(out, ref, rtol=1.6e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA tensors")
@pytest.mark.parametrize(
    "m,expected", [(2, "flashinfer"), (4, "flashinfer"), (8, "layer")]
)
def test_op_dispatches_on_token_count(monkeypatch, m, expected):
    """Above the threshold cuBLAS is faster, so the op defers to the layer."""
    seen: list[str] = []

    class _Layer:
        def in_proj_ba(self, x):
            seen.append("layer")
            return torch.zeros(x.size(0), 96, dtype=x.dtype, device=x.device), None

    class _Ctx:
        no_compile_layers = {"L": _Layer()}

    def _fake_mm(a, b, bias, pdl, backend):
        seen.append("flashinfer")
        return torch.zeros(a.size(0), b.size(1), dtype=a.dtype, device=a.device)

    monkeypatch.setattr(gdn, "get_forward_context", lambda: _Ctx())
    monkeypatch.setattr(gdn, "_resolve_layer_name", lambda name: "L")
    monkeypatch.setattr(gdn, "flashinfer_bf16_mm", _fake_mm)

    w = torch.zeros(96, 5120, dtype=torch.bfloat16, device="cuda")
    x = torch.zeros(m, 5120, dtype=torch.bfloat16, device="cuda")
    gdn.gdn_ba_proj(x, w, "L", MAX_TOKENS)
    assert seen == [expected]
