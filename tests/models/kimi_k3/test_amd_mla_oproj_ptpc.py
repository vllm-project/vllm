# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for fused MLA sigmoid-mul + per-token FP8 ``o_proj``.

The producer must match the torch reference and wrap
``kFp8DynamicTokenSym`` so PTPC ``o_proj`` skips in-kernel quant.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
    kFp8DynamicTokenSym,
)
from vllm.models.kimi_k3.amd.mla import KimiK3MultiHeadLatentAttentionWrapper
from vllm.models.kimi_k3.amd.ops.sigmoid_mul_fp8_per_token import (
    _sigmoid_mul_fp8_torch,
    maybe_fused_mla_oproj_ptpc,
    o_proj_is_ptpc_fp8,
    sigmoid_mul_fp8_per_token,
    wrap_ptpc_activation,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

# Kimi-K3 MLA o_proj at TP8: 12 local heads * 128 v_head_dim.
K_TP8 = 1536
DTYPE = torch.bfloat16


def _cuda_available() -> bool:
    return torch.cuda.is_available()


pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="MLA PTPC o_proj fusion is ROCm-only",
)


class _GProj:
    def __init__(self, gate: torch.Tensor) -> None:
        self.gate = gate

    def __call__(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
        return (self.gate,)


class _OProj:
    def __init__(self, *, ptpc: bool = False) -> None:
        if ptpc:
            self.input_quant_key = kFp8DynamicTokenSym
        self.seen: object | None = None

    def __call__(self, x: object) -> tuple[torch.Tensor]:
        self.seen = x
        if isinstance(x, QuantizedActivation):
            return (torch.ones(x.data.shape[0], 4),)
        assert isinstance(x, torch.Tensor)
        return (x,)


def test_o_proj_is_ptpc_fp8_reads_input_quant_key():
    layer = SimpleNamespace()
    assert o_proj_is_ptpc_fp8(layer) is False
    layer.input_quant_key = kFp8DynamicTokenSym
    assert o_proj_is_ptpc_fp8(layer) is True
    layer.input_quant_key = object()
    assert o_proj_is_ptpc_fp8(layer) is False


def test_maybe_fused_declines_without_ptpc_key():
    x = torch.zeros(2, 4)
    gate = torch.zeros(2, 4)
    assert maybe_fused_mla_oproj_ptpc(x, gate, SimpleNamespace()) is None


def test_maybe_fused_declines_mismatched_shapes():
    o_proj = SimpleNamespace(input_quant_key=kFp8DynamicTokenSym)
    x = torch.zeros(2, 4)
    gate = torch.zeros(2, 8)
    assert maybe_fused_mla_oproj_ptpc(x, gate, o_proj) is None
    assert maybe_fused_mla_oproj_ptpc(torch.zeros(2, 4, 4), x, o_proj) is None


def test_wrap_ptpc_activation_uses_token_sym_key():
    data = torch.zeros(3, 8)
    scale = torch.ones(3, 1)
    qa = wrap_ptpc_activation(data, scale, torch.bfloat16, torch.Size([3, 8]))
    assert isinstance(qa, QuantizedActivation)
    assert qa.quant_key == kFp8DynamicTokenSym
    assert qa.orig_shape == torch.Size([3, 8])


def test_gated_o_proj_falls_through_without_ptpc():
    wrapper = object.__new__(KimiK3MultiHeadLatentAttentionWrapper)
    attn = torch.ones(2, 4) * 2
    gate = torch.zeros(2, 4)
    hidden = torch.ones(2, 8)
    wrapper.g_proj = _GProj(gate)
    wrapper.o_proj = _OProj(ptpc=False)
    out = KimiK3MultiHeadLatentAttentionWrapper._gated_o_proj(wrapper, attn, hidden)
    # sigmoid(0) = 0.5, 2 * 0.5 = 1, o_proj is identity.
    torch.testing.assert_close(out, torch.ones(2, 4))
    assert isinstance(wrapper.o_proj.seen, torch.Tensor)


def test_gated_o_proj_ptpc_passes_quantized_activation(monkeypatch: pytest.MonkeyPatch):
    wrapper = object.__new__(KimiK3MultiHeadLatentAttentionWrapper)
    attn = torch.randn(2, 8)
    hidden = torch.randn(2, 4)
    gate = torch.randn(2, 8)
    qa = wrap_ptpc_activation(
        torch.zeros(2, 8),
        torch.ones(2, 1),
        torch.bfloat16,
        torch.Size([2, 8]),
    )

    monkeypatch.setattr(
        "vllm.models.kimi_k3.amd.mla.maybe_fused_mla_oproj_ptpc",
        lambda *_args, **_kwargs: qa,
    )
    wrapper.g_proj = _GProj(gate)
    wrapper.o_proj = _OProj(ptpc=True)
    KimiK3MultiHeadLatentAttentionWrapper._gated_o_proj(wrapper, attn, hidden)
    assert wrapper.o_proj.seen is qa


@pytest.mark.skipif(not _cuda_available(), reason="CUDA/HIP required")
@pytest.mark.parametrize("num_tokens", [0, 1, 4, 14])
@pytest.mark.parametrize("k", [128, K_TP8])
def test_sigmoid_mul_fp8_matches_torch_reference(num_tokens: int, k: int) -> None:
    torch.manual_seed(0)
    device = "cuda"
    x = torch.randn(num_tokens, k, device=device, dtype=DTYPE)
    gate = torch.randn(num_tokens, k, device=device, dtype=DTYPE)
    quant_dtype = current_platform.fp8_dtype()
    got_q, got_s = sigmoid_mul_fp8_per_token(x, gate, quant_dtype)
    ref_q, ref_s = _sigmoid_mul_fp8_torch(x, gate, quant_dtype)
    assert got_q.shape == (num_tokens, k)
    assert got_s.shape == (num_tokens, 1)
    assert got_q.dtype == quant_dtype
    assert got_s.dtype == torch.float32
    if num_tokens == 0:
        return
    torch.testing.assert_close(got_s, ref_s, rtol=1e-4, atol=1e-4)
    # Compare dequantized values; fp8 codes can differ by 1 ULP on amax ties.
    got = got_q.float() * got_s
    ref = ref_q.float() * ref_s
    torch.testing.assert_close(got, ref, rtol=1e-2, atol=1e-2)
    fp8_max = float(get_fp8_min_max()[1])
    assert float(got.abs().amax()) <= fp8_max * float(got_s.amax()) + 1e-3


@pytest.mark.skipif(
    not _cuda_available() or not HAS_TRITON, reason="Triton GPU required"
)
def test_maybe_fused_wraps_ptpc_activation() -> None:
    torch.manual_seed(1)
    x = torch.randn(3, K_TP8, device="cuda", dtype=DTYPE)
    gate = torch.randn(3, K_TP8, device="cuda", dtype=DTYPE)
    o_proj = SimpleNamespace(input_quant_key=kFp8DynamicTokenSym)
    qa = maybe_fused_mla_oproj_ptpc(x, gate, o_proj)
    assert qa is not None
    assert qa.quant_key == kFp8DynamicTokenSym
    assert qa.data.shape == x.shape
    assert qa.scale.shape == (x.shape[0], 1)
    assert qa.orig_dtype == DTYPE
    assert qa.orig_shape == x.shape
