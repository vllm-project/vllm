# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the load-time lm_head quantization (KernelConfig.lm_head_quantization).

Tests shared by every scheme are parametrized over ``SCHEMES``; the
quantizer tests of each scheme follow them.

Run `pytest tests/kernels/quantization/test_rdna_lm_head.py`.
"""

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.kernels.linear.mixed_precision import (
    rdna_hybrid_w4a16,
    rdna_w8a16,
)
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.online import rdna_lm_head
from vllm.model_executor.layers.quantization.online.int4_lm_head import (
    GROUP_SIZE,
    Int4LMHeadMethod,
    pack_zero_points,
    quantize_int4_rows,
    quantize_lm_head_int4,
)
from vllm.model_executor.layers.quantization.online.int8_lm_head import (
    Int8LMHeadMethod,
    quantize_lm_head_int8,
)
from vllm.model_executor.layers.quantization.online.rdna_lm_head import (
    RdnaLMHeadMethod,
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
    not _on_gfx1x(), reason="The RDNA lm_head ops are gfx11/gfx12 only"
)


def _dequantize_int4(codes, scales, zero_points, group_size=GROUP_SIZE):
    scale = scales.float().repeat_interleave(group_size, dim=1)
    zero_point = zero_points.float().repeat_interleave(group_size, dim=1)
    return (codes.float() - zero_point) * scale


def _dequantize_int8(codes, scales):
    return codes.float() * scales.float()[:, None]


@dataclass(frozen=True)
class Scheme:
    method: type[RdnaLMHeadMethod]
    reference: Callable[[torch.Tensor], torch.Tensor]
    """Full-precision weight -> the fp32 weight the kernel should compute with."""
    weight_dtype: torch.dtype
    weight_cols: Callable[[int], int]
    """Hidden size -> columns of the stored weight."""
    full_precision_tol: float
    """Bound on the relative error against the unquantized head."""

    @property
    def name(self) -> str:
        return self.method.scheme


SCHEMES = [
    pytest.param(
        Scheme(
            method=Int4LMHeadMethod,
            reference=lambda w: _dequantize_int4(*quantize_int4_rows(w)),
            weight_dtype=torch.int8,
            weight_cols=lambda k: k // 2,
            full_precision_tol=0.12,
        ),
        id="int4_per_group_32",
    ),
    pytest.param(
        Scheme(
            method=Int8LMHeadMethod,
            reference=lambda w: _dequantize_int8(*quantize_lm_head_int8(w)),
            weight_dtype=torch.int8,
            weight_cols=lambda k: k,
            # Measured 8.4e-3, against 8.2e-2 for int4_per_group_32.
            full_precision_tol=0.015,
        ),
        id="int8_per_channel",
    ),
]


def _config(scheme: Scheme) -> VllmConfig:
    return VllmConfig(kernel_config=KernelConfig(lm_head_quantization=scheme.name))


def test_lm_head_keeps_its_method_by_default():
    with set_current_vllm_config(VllmConfig()):
        head = ParallelLMHead(VOCAB, HIDDEN, disable_tp=True)
    assert isinstance(head.quant_method, UnquantizedEmbeddingMethod)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_lm_head_quantization_selects_method(scheme, monkeypatch):
    monkeypatch.setattr(rdna_lm_head, "_on_gfx1x", lambda: True)
    with set_current_vllm_config(_config(scheme)):
        head = ParallelLMHead(VOCAB, HIDDEN, disable_tp=True)
        embed = VocabParallelEmbedding(VOCAB, HIDDEN, disable_tp=True)
    assert type(head.quant_method) is scheme.method
    # Only the lm_head is quantized.
    assert isinstance(embed.quant_method, UnquantizedEmbeddingMethod)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_lm_head_quantization_rejects_other_gpus(scheme, monkeypatch):
    monkeypatch.setattr(rdna_lm_head, "_on_gfx1x", lambda: False)
    with (
        set_current_vllm_config(_config(scheme)),
        pytest.raises(ValueError, match="gfx11 or gfx12"),
    ):
        ParallelLMHead(VOCAB, HIDDEN, disable_tp=True)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_lm_head_quantization_rejects_quantized_head(scheme, monkeypatch):
    monkeypatch.setattr(rdna_lm_head, "_on_gfx1x", lambda: True)
    config = _config(scheme)
    # A checkpoint that excludes the lm_head hands out UnquantizedLinearMethod.
    assert isinstance(
        scheme.method.from_config(UnquantizedLinearMethod(), config, "lm_head"),
        scheme.method,
    )
    with pytest.raises(ValueError, match="unquantized lm_head"):
        scheme.method.from_config(scheme.method(), config, "lm_head")


@pytest.mark.parametrize("scheme", SCHEMES)
def test_lm_head_quantization_rejects_head_dtype_and_lora(scheme, monkeypatch):
    monkeypatch.setattr(rdna_lm_head, "_on_gfx1x", lambda: True)
    # from_config only reads these attributes of the VllmConfig.
    model_config = SimpleNamespace(dtype=torch.bfloat16, head_dtype=torch.float32)
    config = SimpleNamespace(model_config=model_config, lora_config=None)
    with pytest.raises(ValueError, match="head_dtype"):
        scheme.method.from_config(UnquantizedEmbeddingMethod(), config, "lm_head")

    model_config.head_dtype = None
    config.lora_config = object()
    with pytest.raises(ValueError, match="LoRA"):
        scheme.method.from_config(UnquantizedEmbeddingMethod(), config, "lm_head")


@pytest.mark.parametrize("scheme", SCHEMES)
def test_lm_head_quantization_rejects_bad_weights(scheme, monkeypatch):
    monkeypatch.setattr(rdna_lm_head, "_on_gfx1x", lambda: True)
    hidden = HIDDEN + scheme.method.k_align // 2
    with set_current_vllm_config(_config(scheme)):
        head = ParallelLMHead(
            VOCAB, hidden, disable_tp=True, params_dtype=torch.float32
        )
    with pytest.raises(ValueError, match="fp16/bf16"):
        head.quant_method.process_weights_after_loading(head)

    head = head.to(torch.bfloat16)
    with pytest.raises(ValueError, match=f"divisible by {scheme.method.k_align}"):
        head.quant_method.process_weights_after_loading(head)


def _loaded_head(
    scheme: Scheme, dtype: torch.dtype, tie: bool = False, device: str = "cuda"
):
    with set_current_vllm_config(_config(scheme)):
        head = ParallelLMHead(VOCAB, HIDDEN, disable_tp=True)
        embed = VocabParallelEmbedding(VOCAB, HIDDEN, disable_tp=True)
    gen = torch.Generator().manual_seed(0)
    for layer in (head, embed):
        layer.weight.data.copy_(torch.randn(layer.weight.shape, generator=gen) * 0.02)
    if tie:
        head = head.tie_weights(embed)
    return head.to(device, dtype), embed.to(device, dtype)


@requires_gfx1x
@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [1, 4, 5, 6, 32, 256])
def test_lm_head_matches_reference(scheme, dtype, num_tokens):
    head, _ = _loaded_head(scheme, dtype)
    weight = head.weight.detach().clone()
    reference = scheme.reference(weight)
    head.quant_method.process_weights_after_loading(head)

    assert head.weight.dtype == scheme.weight_dtype
    assert head.weight.shape == (weight.shape[0], scheme.weight_cols(HIDDEN))

    torch.manual_seed(1)
    x = torch.randn(num_tokens, HIDDEN, device="cuda", dtype=dtype)
    out = head.quant_method.apply(head, x).float()
    assert out.shape == (num_tokens, weight.shape[0])

    ref = x.float() @ reference.t()
    assert ((out - ref).norm() / ref.norm()).item() < 1e-2

    full = x.float() @ weight.float().t()
    assert ((out - full).norm() / full.norm()).item() < scheme.full_precision_tol


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("tie", [False, True], ids=["untied", "tied"])
def test_quantized_head_does_not_reuse_loaded_weight(scheme, tie, monkeypatch):
    monkeypatch.setattr(rdna_lm_head, "_on_gfx1x", lambda: True)
    # Without the gfx1151 row padding the quantized weight is contiguous, the
    # case where reusing the loaded storage would overwrite it.
    for module in (rdna_hybrid_w4a16, rdna_w8a16):
        monkeypatch.setattr(module, "_on_gfx1151", lambda: False)
    head, embed = _loaded_head(scheme, torch.bfloat16, tie=tie, device="cpu")
    loaded = head.weight
    embed_weight = embed.weight.detach().clone()
    head.quant_method.process_weights_after_loading(head)

    assert head.weight is not loaded
    assert head.weight.dtype == scheme.weight_dtype
    assert (
        head.weight.untyped_storage().data_ptr() != loaded.untyped_storage().data_ptr()
    )
    # A tied head shares the embedding's weight until it is quantized.
    assert embed.weight.dtype == torch.bfloat16
    assert torch.equal(embed.weight, embed_weight)


# int4_per_group_32 quantizer


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

    dequant = _dequantize_int4(codes, scales, zero_points)
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


# int8_per_channel quantizer


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_quantize_lm_head_int8_error_bound(dtype):
    torch.manual_seed(0)
    weight = (torch.randn(64, 256) * 0.02).to(dtype)
    weight[-8:] = 0  # vocab padding rows
    codes, scales = quantize_lm_head_int8(weight)

    assert codes.shape == (64, 256) and codes.dtype == torch.int8
    assert scales.shape == (64,) and scales.dtype == dtype
    assert int(codes.min()) >= -127 and int(codes.max()) <= 127

    dequant = _dequantize_int8(codes, scales)
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
