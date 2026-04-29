# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Round-trip tests for compressor → FP8 quant + KV cache insert → gather + dequant.

Two paths tested:
  A) DeepseekV4 Attention: head_dim=512 (448 FP8 nope + 64 bf16 rope), quant_block=64
  B) Indexer:       head_dim=128 (all FP8), quant_block=128

These serve as golden references for validating the future fused
compressor+quant+cache kernel.
"""

import math

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.v1.attention.ops.deepseek_v4_ops import (
    dequantize_and_gather_k_cache,
    quantize_and_insert_k_cache,
)


def _ue8m0_reference(x: torch.Tensor, block_size: int, fp8_max: float):
    """PyTorch reference for UE8M0 FP8 quantization (per-block, power-of-2 scale).

    Returns (x_fp8, scales) where x_fp8 is float8_e4m3fn and scales are float32.
    """
    assert x.dim() == 1
    n = x.numel()
    n_blocks = math.ceil(n / block_size)
    x_fp8 = torch.zeros(n, dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.zeros(n_blocks, dtype=torch.float32, device=x.device)

    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, n)
        block = x[start:end].float()
        amax = block.abs().max().clamp(min=1e-4)
        raw_scale = amax / fp8_max
        exponent = math.ceil(math.log2(raw_scale.item()))
        scale = 2.0**exponent
        scales[i] = scale
        quantized = (block / scale).clamp(-fp8_max, fp8_max)
        x_fp8[start:end] = quantized.to(torch.float8_e4m3fn)

    return x_fp8, scales


# ── Test A: DeepseekV4 Attention path ──────────────────────────────────────────────


@pytest.mark.parametrize("num_tokens", [1, 4, 8, 17])
@pytest.mark.parametrize("block_size", [16, 64])
def test_deepseek_v4_attention_quant_cache_roundtrip(num_tokens: int, block_size: int):
    """compressed_kv → quantize_and_insert_k_cache → dequantize_and_gather_k_cache
    → compare against original."""

    HEAD_DIM = 512
    NOPE_DIM = 448
    HEAD_BYTES = 584  # 448 fp8 + 128 bf16 + 8 uint8 scale
    FP8_MAX = 448.0
    QUANT_BLOCK = 64

    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    device = "cuda"

    # Random compressed_kv (simulates compressor output)
    compressed_kv = torch.randn(
        num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=device
    )

    # ── Quant + insert ──────────────────────────────────────────────────
    k_cache = torch.zeros(
        num_blocks, block_size, HEAD_BYTES, dtype=torch.uint8, device=device
    )
    k_cache_2d = k_cache.view(num_blocks, -1)

    # Sequential slot mapping: token i → slot i
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    quantize_and_insert_k_cache(
        compressed_kv, k_cache_2d, slot_mapping, block_size=block_size
    )

    # ── Gather + dequant ────────────────────────────────────────────────
    num_reqs = 1
    max_blocks_per_seq = num_blocks
    out = torch.zeros(
        num_reqs, num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    seq_lens = torch.tensor([num_tokens], dtype=torch.int32, device=device)
    # block_table: request 0 uses physical blocks 0, 1, ...
    block_table = torch.arange(
        max_blocks_per_seq, dtype=torch.int32, device=device
    ).unsqueeze(0)

    dequantize_and_gather_k_cache(
        out, k_cache, seq_lens, None, block_table, block_size, offset=0
    )

    recovered = out[0, :num_tokens]

    # ── NoPE portion (first 448): FP8 quantized, expect UE8M0 error ──
    nope_orig = compressed_kv[:, :NOPE_DIM].float()
    nope_recv = recovered[:, :NOPE_DIM].float()
    nope_diff = (nope_recv - nope_orig).abs()

    # Per-token check: FP8 e4m3 (3-bit mantissa) worst-case error is
    # half-ULP at the largest representable value.  At y ≈ 448 (max),
    # ULP = 2^(8-3) = 32, so error ≤ 16 * scale.
    for t in range(num_tokens):
        _, scales = _ue8m0_reference(
            compressed_kv[t, :NOPE_DIM].float(), QUANT_BLOCK, FP8_MAX
        )
        max_allowed = 16.0 * scales.max().item()
        token_diff = nope_diff[t].max().item()
        assert token_diff <= max_allowed, (
            f"Token {t} nope diff {token_diff} exceeds max_allowed "
            f"{max_allowed} (scale={scales.max().item()})"
        )

    # ── RoPE portion (last 64): stored as bf16, should be exact ─────
    rope_diff = (recovered[:, NOPE_DIM:] - compressed_kv[:, NOPE_DIM:]).abs()
    assert rope_diff.max().item() == 0.0, (
        f"RoPE portion should be exact but got max diff {rope_diff.max().item()}"
    )


# ── Test B: Indexer path ────────────────────────────────────────────────────


@pytest.mark.parametrize("num_tokens", [1, 4, 8, 17])
@pytest.mark.parametrize("block_size", [16, 64])
def test_indexer_quant_cache_roundtrip(num_tokens: int, block_size: int):
    """k → indexer_k_quant_and_cache → cp_gather_indexer_k_quant_cache
    → manual dequant → compare against original."""

    HEAD_DIM = 128
    QUANT_BLOCK_SIZE = 128
    # cache_stride = head_dim + (head_dim * 4 / quant_block_size) = 128 + 4 = 132
    CACHE_STRIDE = HEAD_DIM + HEAD_DIM * 4 // QUANT_BLOCK_SIZE

    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    device = "cuda"

    # Random K (simulates compressor output for indexer)
    k = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=device)

    # ── Quant + insert ──────────────────────────────────────────────────
    kv_cache = torch.zeros(
        num_blocks, block_size, CACHE_STRIDE, dtype=torch.uint8, device=device
    )
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    ops.indexer_k_quant_and_cache(k, kv_cache, slot_mapping, QUANT_BLOCK_SIZE, "ue8m0")

    # ── Gather ──────────────────────────────────────────────────────────
    max_blocks_per_seq = num_blocks
    block_table = torch.arange(
        max_blocks_per_seq, dtype=torch.int32, device=device
    ).unsqueeze(0)
    cu_seq_lens = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)

    # dst_k: [total_seq_len, head_dim] as uint8 (raw FP8 bytes)
    dst_k = torch.zeros(num_tokens, HEAD_DIM, dtype=torch.uint8, device=device)
    # dst_scale: [total_seq_len, head_dim/quant_block*4] as uint8 (raw float32 bytes)
    num_scale_bytes = HEAD_DIM * 4 // QUANT_BLOCK_SIZE  # 4
    dst_scale = torch.zeros(
        num_tokens, num_scale_bytes, dtype=torch.uint8, device=device
    )

    ops.cp_gather_indexer_k_quant_cache(
        kv_cache, dst_k, dst_scale, block_table, cu_seq_lens
    )

    # ── Manual dequant ──────────────────────────────────────────────────
    k_fp8 = dst_k.view(torch.float8_e4m3fn).float()  # [num_tokens, 128]
    scale = dst_scale.view(torch.float32)  # [num_tokens, 1]
    k_recovered = k_fp8 * scale  # [num_tokens, 128]

    # ── Compare ─────────────────────────────────────────────────────────
    diff = (k_recovered - k.float()).abs()
    k_abs = k.float().abs()

    for t in range(num_tokens):
        amax = k_abs[t].max().clamp(min=1e-4).item()
        # UE8M0: scale = 2^ceil(log2(amax / 448))
        exponent = math.ceil(math.log2(amax / 448.0))
        ue8m0_scale = 2.0**exponent
        # FP8 e4m3 (3-bit mantissa): worst-case error = 16 * scale
        max_allowed = 16.0 * ue8m0_scale
        token_diff = diff[t].max().item()
        assert token_diff <= max_allowed, (
            f"Token {t} diff {token_diff} exceeds max_allowed "
            f"{max_allowed} (scale={ue8m0_scale})"
        )


def test_indexer_gather_accepts_upper_bound_output():
    """Gather only exact cu_seq_lens even when dst is over-allocated."""

    head_dim = 128
    quant_block_size = 128
    cache_stride = head_dim + head_dim * 4 // quant_block_size
    valid_tokens = 9
    upper_bound_tokens = 13
    block_size = 16
    num_blocks = 2
    sentinel = 123
    device = "cuda"

    k = torch.randn(valid_tokens, head_dim, dtype=torch.bfloat16, device=device)
    kv_cache = torch.zeros(
        num_blocks, block_size, cache_stride, dtype=torch.uint8, device=device
    )
    slot_mapping = torch.arange(valid_tokens, dtype=torch.int64, device=device)
    ops.indexer_k_quant_and_cache(k, kv_cache, slot_mapping, quant_block_size, "ue8m0")

    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(
        0
    )
    cu_seq_lens = torch.tensor([0, valid_tokens], dtype=torch.int32, device=device)
    dst_k = torch.full(
        (upper_bound_tokens, head_dim), sentinel, dtype=torch.uint8, device=device
    )
    num_scale_bytes = head_dim * 4 // quant_block_size
    dst_scale = torch.full(
        (upper_bound_tokens, num_scale_bytes),
        sentinel,
        dtype=torch.uint8,
        device=device,
    )

    ops.cp_gather_indexer_k_quant_cache(
        kv_cache, dst_k, dst_scale, block_table, cu_seq_lens
    )
    torch.accelerator.synchronize()

    k_recovered = dst_k[:valid_tokens].view(torch.float8_e4m3fn).float() * dst_scale[
        :valid_tokens
    ].view(torch.float32)
    diff = (k_recovered - k.float()).abs()
    max_allowed = (16.0 * dst_scale[:valid_tokens].view(torch.float32).max()).item()
    assert diff.max().item() <= max_allowed
    assert torch.all(dst_k[valid_tokens:] == sentinel)
    assert torch.all(dst_scale[valid_tokens:] == sentinel)


# ── Test C: DeepseekV4 attention with values at different magnitudes ───────────


def test_deepseek_v4_quant_magnitude_range():
    """Test that quantization handles a range of magnitudes correctly."""

    HEAD_DIM = 512
    NOPE_DIM = 448
    HEAD_BYTES = 584
    block_size = 16
    num_tokens = 4
    num_blocks = 2
    device = "cuda"

    # Create inputs with varying magnitudes: small, medium, large
    compressed_kv = torch.zeros(
        num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    compressed_kv[0] = 0.001  # very small
    compressed_kv[1] = 1.0  # unit scale
    compressed_kv[2] = 100.0  # large
    compressed_kv[3] = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device=device)

    k_cache = torch.zeros(
        num_blocks, block_size, HEAD_BYTES, dtype=torch.uint8, device=device
    )
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    quantize_and_insert_k_cache(
        compressed_kv, k_cache.view(num_blocks, -1), slot_mapping, block_size
    )

    out = torch.zeros(1, num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=device)
    seq_lens = torch.tensor([num_tokens], dtype=torch.int32, device=device)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(
        0
    )

    dequantize_and_gather_k_cache(
        out, k_cache, seq_lens, None, block_table, block_size, offset=0
    )

    recovered = out[0, :num_tokens]

    # RoPE portion must be exact
    rope_diff = (recovered[:, NOPE_DIM:] - compressed_kv[:, NOPE_DIM:]).abs().max()
    assert rope_diff.item() == 0.0, f"RoPE diff {rope_diff.item()}"

    # NoPE: relative error should be reasonable
    for t in range(num_tokens):
        orig = compressed_kv[t, :NOPE_DIM].float()
        recv = recovered[t, :NOPE_DIM].float()
        abs_diff = (recv - orig).abs().max().item()
        magnitude = orig.abs().max().item()
        if magnitude > 0.01:
            rel_err = abs_diff / magnitude
            assert rel_err < 0.15, (
                f"Token {t}: rel_err={rel_err:.4f}, abs_diff={abs_diff:.6f}, "
                f"magnitude={magnitude:.4f}"
            )
