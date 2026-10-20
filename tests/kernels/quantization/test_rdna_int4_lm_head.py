# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the load-time int4 lm_head (KernelConfig.lm_head_quantization).

Run `pytest tests/kernels/quantization/test_rdna_int4_lm_head.py`.
"""

import pytest
import torch

from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.online import int4_lm_head
from vllm.model_executor.layers.quantization.online.int4_lm_head import (
    GROUP_SIZE,
    Int4LMHeadMethod,
    pack_zero_points,
    quantize_int4_rows,
    quantize_lm_head_int4,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.platforms import current_platform

# Not a multiple of the vocab padding, so the head has zero-filled rows.
VOCAB, HIDDEN = 4000, 2048


def _on_gfx1x() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx1x

    return on_gfx1x()


requires_gfx1x = pytest.mark.skipif(
    not _on_gfx1x(), reason="The RDNA hybrid W4A16 op is gfx11/gfx12 only"
)


def _int4_config() -> VllmConfig:
    return VllmConfig(
        kernel_config=KernelConfig(lm_head_quantization="int4_per_group_32")
    )


def _dequantize(codes, scales, zero_points, group_size=GROUP_SIZE):
    scale = scales.float().repeat_interleave(group_size, dim=1)
    zero_point = zero_points.float().repeat_interleave(group_size, dim=1)
    return (codes.float() - zero_point) * scale


def _unshuffle(w_q: torch.Tensor) -> torch.Tensor:
    """Invert the ExLlama shuffle: int8 [N, K // 2] -> uint4 codes [N, K]."""
    words = w_q.view(torch.int32)
    shifts = torch.tensor([0, 16, 4, 20, 8, 24, 12, 28], dtype=torch.int32)
    codes = (words.unsqueeze(-1) >> shifts.to(words.device)) & 0xF
    return codes.reshape(words.shape[0], -1).to(torch.uint8)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_quantize_int4_rows_error_bound(dtype):
    torch.manual_seed(0)
    weight = (torch.randn(64, 256) * 0.02).to(dtype)
    weight[-8:] = 0  # vocab padding rows
    codes, scales, zero_points = quantize_int4_rows(weight)

    assert codes.shape == (64, 256) and codes.dtype == torch.uint8
    assert scales.shape == (64, 256 // GROUP_SIZE) and scales.dtype == dtype
    assert int(codes.max()) <= 15 and int(zero_points.max()) <= 15

    dequant = _dequantize(codes, scales, zero_points)
    error = dequant - weight.float()
    # Rounding the zero point can clip a group's endpoint by up to half a step.
    step = scales.float().repeat_interleave(GROUP_SIZE, dim=1)
    assert (error.abs() <= step).all()
    assert (error.norm() / weight.float().norm()).item() < 0.1
    assert torch.equal(dequant[-8:], torch.zeros(8, 256))


def test_pack_zero_points_roundtrip():
    zero_points = torch.randint(0, 16, (64, 8), dtype=torch.uint8)
    words = pack_zero_points(zero_points)
    assert words.shape == (8, 8) and words.dtype == torch.int32
    shifts = (torch.arange(64) % 8 * 4).view(64, 1).to(torch.int32)
    unpacked = (words.repeat_interleave(8, dim=0) >> shifts) & 0xF
    assert torch.equal(unpacked.to(torch.uint8), zero_points)


def test_quantize_lm_head_int4_layout():
    torch.manual_seed(0)
    weight = (torch.randn(96, 128) * 0.02).to(torch.bfloat16)
    w_q, w_s, w_zp = quantize_lm_head_int4(weight, chunk_rows=40)
    codes, scales, zero_points = quantize_int4_rows(weight)

    assert w_q.dtype == torch.int8 and w_q.shape == (96, 64)
    assert torch.equal(_unshuffle(w_q), codes)
    assert torch.equal(w_s, scales)
    assert torch.equal(w_zp, pack_zero_points(zero_points))


def test_lm_head_keeps_its_method_by_default():
    with set_current_vllm_config(VllmConfig()):
        head = ParallelLMHead(VOCAB, HIDDEN, disable_tp=True)
    assert isinstance(head.quant_method, UnquantizedEmbeddingMethod)


def test_lm_head_quantization_selects_int4_method(monkeypatch):
    monkeypatch.setattr(int4_lm_head, "_on_gfx1x", lambda: True)
    with set_current_vllm_config(_int4_config()):
        head = ParallelLMHead(VOCAB, HIDDEN, disable_tp=True)
        embed = VocabParallelEmbedding(VOCAB, HIDDEN, disable_tp=True)
    assert isinstance(head.quant_method, Int4LMHeadMethod)
    # Only the lm_head is quantized.
    assert isinstance(embed.quant_method, UnquantizedEmbeddingMethod)


def test_lm_head_quantization_rejects_other_gpus(monkeypatch):
    monkeypatch.setattr(int4_lm_head, "_on_gfx1x", lambda: False)
    with (
        set_current_vllm_config(_int4_config()),
        pytest.raises(ValueError, match="gfx11 or gfx12"),
    ):
        ParallelLMHead(VOCAB, HIDDEN, disable_tp=True)


def test_lm_head_quantization_rejects_quantized_head(monkeypatch):
    monkeypatch.setattr(int4_lm_head, "_on_gfx1x", lambda: True)
    config = _int4_config()
    # A checkpoint that excludes the lm_head hands out UnquantizedLinearMethod.
    assert isinstance(
        Int4LMHeadMethod.from_config(UnquantizedLinearMethod(), config, "lm_head"),
        Int4LMHeadMethod,
    )
    with pytest.raises(ValueError, match="unquantized lm_head"):
        Int4LMHeadMethod.from_config(Int4LMHeadMethod(), config, "lm_head")


def _loaded_head(dtype: torch.dtype, tie: bool = False):
    with set_current_vllm_config(_int4_config()):
        head = ParallelLMHead(VOCAB, HIDDEN, disable_tp=True)
        embed = VocabParallelEmbedding(VOCAB, HIDDEN, disable_tp=True)
    gen = torch.Generator().manual_seed(0)
    for layer in (head, embed):
        layer.weight.data.copy_(torch.randn(layer.weight.shape, generator=gen) * 0.02)
    if tie:
        head = head.tie_weights(embed)
    return head.to("cuda", dtype), embed.to("cuda", dtype)


@requires_gfx1x
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [1, 4, 5, 6, 32])
def test_int4_lm_head_matches_reference(dtype, num_tokens):
    head, _ = _loaded_head(dtype)
    weight = head.weight.detach().clone()
    codes, scales, zero_points = quantize_int4_rows(weight)
    head.quant_method.process_weights_after_loading(head)

    assert head.weight.dtype == torch.int8
    assert head.weight.shape == (weight.shape[0], HIDDEN // 2)

    torch.manual_seed(1)
    x = torch.randn(num_tokens, HIDDEN, device="cuda", dtype=dtype)
    out = head.quant_method.apply(head, x).float()
    assert out.shape == (num_tokens, weight.shape[0])

    ref = x.float() @ _dequantize(codes, scales, zero_points).t()
    assert ((out - ref).norm() / ref.norm()).item() < 1e-2

    full = x.float() @ weight.float().t()
    assert ((out - full).norm() / full.norm()).item() < 0.12


@requires_gfx1x
def test_tied_int4_lm_head_keeps_embedding():
    head, embed = _loaded_head(torch.bfloat16, tie=True)
    embed_weight = embed.weight.detach().clone()
    head.quant_method.process_weights_after_loading(head)

    assert head.weight.dtype == torch.int8
    assert embed.weight.dtype == torch.bfloat16
    assert torch.equal(embed.weight, embed_weight)
