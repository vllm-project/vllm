# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the load-time int8 lm_head (KernelConfig.lm_head_quantization).

Run `pytest tests/kernels/quantization/test_rdna_int8_lm_head.py`.
"""

import pytest
import torch
from torch.nn import Module, Parameter

from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.online import int8_lm_head
from vllm.model_executor.layers.quantization.online.int8_lm_head import (
    Int8LMHeadMethod,
    quantize_lm_head_int8,
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
    not _on_gfx1x(), reason="The RDNA W8A16 op is gfx11/gfx12 only"
)


def _int8_config() -> VllmConfig:
    return VllmConfig(
        kernel_config=KernelConfig(lm_head_quantization="int8_per_channel")
    )


def _dequantize(codes, scales):
    return codes.float() * scales.float()[:, None]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_quantize_lm_head_int8_error_bound(dtype):
    torch.manual_seed(0)
    weight = (torch.randn(64, 256) * 0.02).to(dtype)
    weight[-8:] = 0  # vocab padding rows
    codes, scales = quantize_lm_head_int8(weight)

    assert codes.shape == (64, 256) and codes.dtype == torch.int8
    assert scales.shape == (64,) and scales.dtype == dtype
    assert int(codes.min()) >= -127 and int(codes.max()) <= 127

    dequant = _dequantize(codes, scales)
    error = (dequant - weight.float()).abs()
    # Rounding the scale to the model dtype can push a row's amax just past
    # 127 steps, but by less than half a step, so the clamp stays within it.
    assert (error <= scales.float()[:, None] / 2).all()
    assert torch.equal(dequant[-8:], torch.zeros(8, 256))


def test_quantize_lm_head_int8_chunks_match():
    torch.manual_seed(0)
    weight = (torch.randn(96, 128) * 0.02).to(torch.bfloat16)
    codes, scales = quantize_lm_head_int8(weight, chunk_rows=40)
    ref_codes, ref_scales = quantize_lm_head_int8(weight, chunk_rows=96)

    assert torch.equal(codes, ref_codes)
    assert torch.equal(scales, ref_scales)


def test_lm_head_keeps_its_method_by_default():
    with set_current_vllm_config(VllmConfig()):
        head = ParallelLMHead(VOCAB, HIDDEN, disable_tp=True)
    assert isinstance(head.quant_method, UnquantizedEmbeddingMethod)


def test_lm_head_quantization_selects_int8_method(monkeypatch):
    monkeypatch.setattr(int8_lm_head, "_on_gfx1x", lambda: True)
    with set_current_vllm_config(_int8_config()):
        head = ParallelLMHead(VOCAB, HIDDEN, disable_tp=True)
        embed = VocabParallelEmbedding(VOCAB, HIDDEN, disable_tp=True)
    assert isinstance(head.quant_method, Int8LMHeadMethod)
    # Only the lm_head is quantized.
    assert isinstance(embed.quant_method, UnquantizedEmbeddingMethod)


def test_lm_head_quantization_rejects_other_gpus(monkeypatch):
    monkeypatch.setattr(int8_lm_head, "_on_gfx1x", lambda: False)
    with (
        set_current_vllm_config(_int8_config()),
        pytest.raises(ValueError, match="gfx11 or gfx12"),
    ):
        ParallelLMHead(VOCAB, HIDDEN, disable_tp=True)


def test_lm_head_quantization_rejects_quantized_head(monkeypatch):
    monkeypatch.setattr(int8_lm_head, "_on_gfx1x", lambda: True)
    config = _int8_config()
    # A checkpoint that excludes the lm_head hands out UnquantizedLinearMethod.
    assert isinstance(
        Int8LMHeadMethod.from_config(UnquantizedLinearMethod(), config, "lm_head"),
        Int8LMHeadMethod,
    )
    with pytest.raises(ValueError, match="unquantized lm_head"):
        Int8LMHeadMethod.from_config(Int8LMHeadMethod(), config, "lm_head")


@pytest.mark.parametrize(
    "shape,dtype,match",
    [
        ((64, 2056), torch.bfloat16, "divisible by"),
        ((66, 2048), torch.bfloat16, "divisible by"),
        ((64, 2048), torch.float32, "fp16/bf16"),
    ],
    ids=["ragged_hidden", "ragged_vocab", "fp32"],
)
def test_int8_lm_head_rejects_unsupported_weights(shape, dtype, match):
    layer = Module()
    layer.weight = Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)
    with pytest.raises(ValueError, match=match):
        Int8LMHeadMethod().process_weights_after_loading(layer)


def _loaded_head(dtype: torch.dtype, tie: bool = False):
    with set_current_vllm_config(_int8_config()):
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
@pytest.mark.parametrize("num_tokens", [1, 5, 6, 32, 256])
def test_int8_lm_head_matches_reference(dtype, num_tokens):
    head, _ = _loaded_head(dtype)
    weight = head.weight.detach().clone()
    codes, scales = quantize_lm_head_int8(weight)
    head.quant_method.process_weights_after_loading(head)

    assert head.weight.dtype == torch.int8
    assert head.weight.shape == weight.shape

    torch.manual_seed(1)
    x = torch.randn(num_tokens, HIDDEN, device="cuda", dtype=dtype)
    out = head.quant_method.apply(head, x).float()
    assert out.shape == (num_tokens, weight.shape[0])

    ref = x.float() @ _dequantize(codes, scales).t()
    assert ((out - ref).norm() / ref.norm()).item() < 1e-2

    # Measured 8.4e-3 here, against 8.2e-2 for int4_per_group_32 (bound 0.12).
    full = x.float() @ weight.float().t()
    assert ((out - full).norm() / full.norm()).item() < 0.015


@requires_gfx1x
def test_tied_int8_lm_head_keeps_embedding():
    head, embed = _loaded_head(torch.bfloat16, tie=True)
    embed_weight = embed.weight.detach().clone()
    head.quant_method.process_weights_after_loading(head)

    assert head.weight.dtype == torch.int8
    assert embed.weight.dtype == torch.bfloat16
    assert torch.equal(embed.weight, embed_weight)
