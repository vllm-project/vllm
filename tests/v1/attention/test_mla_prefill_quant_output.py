# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MLA prefill backend fused-quant-output support.

Covers two things:
  * `MLAPrefillBackend.supports_quant_output`, the capability gate that decides
    whether the prefill kernel writes quantized output directly (FA4 native
    fused FP8, see flash-attention#135) instead of the post-quant path.
  * The numerical equivalence of that fused FP8 write versus the bf16-attention
    + standalone static-FP8-quant path it replaces (GPU-only, SM100/SM110).
"""

from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic64Sym,
    kFp8Dynamic128Sym,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend
from vllm.v1.attention.backends.mla.prefill.flash_attn import (
    FlashAttnPrefillBackend,
)

_FA_MODULE = "vllm.v1.attention.backends.mla.prefill.flash_attn"


class _DummyPrefillBackend(MLAPrefillBackend):
    """Concrete backend that does NOT override supports_quant_output."""

    @staticmethod
    def get_name() -> str:
        return "DUMMY"

    def run_prefill_new_tokens(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def run_prefill_context_chunk(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


@pytest.mark.parametrize(
    "quant_key", [kFp8StaticTensorSym, kFp8Dynamic128Sym, kNvfp4Dynamic, None]
)
def test_base_backend_never_supports_quant_output(quant_key):
    """The base default opts every backend out unless it overrides."""
    backend = object.__new__(_DummyPrefillBackend)
    assert backend.supports_quant_output(quant_key) is False


def _make_fa_backend(version: int | None, is_vllm_fa: bool):
    """Build a FlashAttnPrefillBackend without running its heavy __init__."""
    backend = object.__new__(FlashAttnPrefillBackend)
    backend.vllm_flash_attn_version = version
    backend._is_vllm_fa = is_vllm_fa
    return backend


@pytest.mark.parametrize(
    ("version", "is_vllm_fa", "dc_major", "quant_key", "expected"),
    [
        # FA4 + vLLM-FA + Blackwell SM100/SM110 + static/per-group FP8/NVFP4 -> fused.
        (4, True, 10, kFp8StaticTensorSym, True),
        (4, True, 11, kFp8StaticTensorSym, True),
        (4, True, 10, kFp8Dynamic128Sym, True),
        (4, True, 11, kFp8Dynamic64Sym, True),
        (4, True, 10, kNvfp4Dynamic, True),
        (4, True, 11, kNvfp4Dynamic, True),
        # Wrong compute capability (SM90 / SM120) -> not supported (#135).
        (4, True, 9, kFp8StaticTensorSym, False),
        (4, True, 12, kFp8StaticTensorSym, False),
        (4, True, 9, kFp8Dynamic128Sym, False),
        (4, True, 9, kNvfp4Dynamic, False),
        # Not FA4.
        (3, True, 10, kFp8StaticTensorSym, False),
        (2, True, 10, kFp8StaticTensorSym, False),
        (None, True, 10, kFp8StaticTensorSym, False),
        # Upstream (ROCm) flash-attn, not vLLM-FA.
        (4, False, 10, kFp8StaticTensorSym, False),
    ],
)
def test_flash_attn_supports_quant_output(
    version, is_vllm_fa, dc_major, quant_key, expected
):
    backend = _make_fa_backend(version, is_vllm_fa)
    with patch(f"{_FA_MODULE}.current_platform") as plat:
        plat.get_device_capability.return_value = DeviceCapability(
            major=dc_major, minor=0
        )
        assert backend.supports_quant_output(quant_key) is expected


def test_flash_attn_supports_quant_output_unknown_device():
    """A None device capability (e.g. capability probe failed) is safe."""
    backend = _make_fa_backend(version=4, is_vllm_fa=True)
    with patch(f"{_FA_MODULE}.current_platform") as plat:
        plat.get_device_capability.return_value = None
        assert backend.supports_quant_output(kFp8StaticTensorSym) is False


def test_flash_attn_prefill_backend_signature_accepts_fused_kwargs():
    """run_prefill_new_tokens must accept out/output_scale so the direct
    (non-**kwargs) call in forward_mha type- and runtime-checks."""
    import inspect

    params = inspect.signature(
        FlashAttnPrefillBackend.run_prefill_new_tokens
    ).parameters
    assert "out" in params
    assert "output_scale" in params
    # The base contract must expose them too (Liskov / direct call site).
    base_params = inspect.signature(MLAPrefillBackend.run_prefill_new_tokens).parameters
    assert "out" in base_params
    assert "output_scale" in base_params


def test_mla_impl_forward_mha_accepts_output_scale():
    """The abstract MLA impl forward_mha must carry output_scale so every
    override (and the unconditional forward_impl call) stays compatible."""
    import inspect

    from vllm.v1.attention.backend import MLAAttentionImpl

    params = inspect.signature(MLAAttentionImpl.forward_mha).parameters
    assert "output_scale" in params
    assert params["output_scale"].default is None


def _fused_fp8_skip_reason() -> str | None:
    """FA4 fused FP8 output needs a real Blackwell SM100/SM110 GPU."""
    if not torch.cuda.is_available():
        return "requires CUDA"
    major = torch.cuda.get_device_capability()[0]
    if major not in (10, 11):
        return f"FA4 fused FP8 output requires SM100/SM110, got SM{major}x"
    return None


_FUSED_FP8_SKIP = _fused_fp8_skip_reason()


@pytest.mark.skipif(_FUSED_FP8_SKIP is not None, reason=_FUSED_FP8_SKIP or "")
def test_fa4_fused_fp8_output_matches_post_quant(default_vllm_config):
    """FA4's fused FP8 write (output_scale, flash-attention#135) must match the
    bf16-attention + standalone static-FP8-quant path it replaces, since
    production uses the same output_scale for both."""
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
    from vllm.platforms import current_platform
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    torch.manual_seed(0)
    device = torch.device("cuda")
    fp8_dtype = current_platform.fp8_dtype()

    # MLA prefill head dims (post kv_b_proj): q/k = qk_nope(128)+qk_rope(64),
    # v = v_head_dim(128); DeepSeek-V2-Lite has 16 query heads.
    num_heads, qk_head_dim, v_head_dim, seqlen = 16, 192, 128, 512
    cu_seqlens = torch.tensor([0, seqlen], dtype=torch.int32, device=device)
    q = torch.randn(seqlen, num_heads, qk_head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(seqlen, num_heads, qk_head_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(seqlen, num_heads, v_head_dim, dtype=torch.bfloat16, device=device)

    fa_kwargs = dict(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        causal=True,
        fa_version=4,
    )

    # Reference: bf16 attention, then standalone static per-tensor FP8 quant.
    out_bf16 = flash_attn_varlen_func(q=q, k=k, v=v, **fa_kwargs)
    out_2d = out_bf16.reshape(seqlen, num_heads * v_head_dim)
    # Scale the amax near e4m3 max so the check uses the representable range.
    finfo = torch.finfo(fp8_dtype)
    scale = (out_2d.abs().max() / finfo.max).to(torch.float32).reshape(1)
    quant_op = QuantFP8(static=True, group_shape=GroupShape.PER_TENSOR)
    ref_fp8, _ = quant_op(out_2d, scale)

    # Feature: FA4 writes e4m3 into the (tokens, heads*dim) buffer directly.
    fused_fp8 = torch.empty(
        seqlen, num_heads * v_head_dim, dtype=fp8_dtype, device=device
    )
    flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        out=fused_fp8.view(seqlen, num_heads, v_head_dim),
        output_scale=scale,
        **fa_kwargs,
    )

    # Non-degenerate (catches a no-op / all-zero write).
    assert torch.isfinite(fused_fp8.float()).all()
    assert fused_fp8.float().abs().any()

    # e4m3 has 3 mantissa bits, so allow ~1 mantissa step of rounding slack.
    ref = ref_fp8.float() * scale
    got = fused_fp8.float() * scale
    torch.testing.assert_close(got, ref, rtol=0.125, atol=float(scale) * 2)

    # ...and most elements land in the exact same fp8 bucket.
    exact = (fused_fp8.view(torch.uint8) == ref_fp8.view(torch.uint8)).float().mean()
    assert exact > 0.9, f"only {exact:.1%} of fused FP8 outputs matched the baseline"
