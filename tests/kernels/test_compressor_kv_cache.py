# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Round-trip tests for compressor → FP8 quant + KV cache insert → gather + dequant.

These tests cover:
  A) DeepseekV4 Attention: head_dim=512 (448 FP8 nope + 64 bf16 rope), quant_block=64
  B) Fused dequant+gather K cache
  C) Indexer:       head_dim=128 (all FP8), quant_block=128
  D) DeepseekV4 Attention magnitude range: correctness across small/large values
  E) Indexer fused Triton kernel: compress+norm+rope+quant+insert
  F) Indexer fused two-stage Triton kernel: head=512 cr>=128 (no-overlap)
"""

import math
from types import SimpleNamespace

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.models.deepseek_v4.common.ops import (
    dequantize_and_gather_k_cache,
    quantize_and_insert_k_cache,
)
from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    _fused_kv_compress_norm_rope_insert_indexer_attn,
    _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn,
    _launch_two_stage_sparse_attn_compressor,
)
from vllm.models.deepseek_v4.compressor import _get_c128_boundary
from vllm.platforms import current_platform

from .test_fused_indexer_q_rope_quant import quantize_to_mxfp4


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


@pytest.mark.parametrize(
    ("starts", "query_start_loc", "expected"),
    [
        ([0], [0, 127], False),
        ([0], [0, 128], True),
        ([127], [0, 1], True),
        ([128], [0, 127], False),
        ([1, 255], [0, 1, 2], True),
        (None, [0, 1], None),
    ],
)
def test_get_c128_boundary(starts, query_start_loc, expected):
    metadata = SimpleNamespace(
        _num_computed_tokens_cpu=None if starts is None else torch.tensor(starts),
        query_start_loc_cpu=torch.tensor(query_start_loc),
    )
    assert _get_c128_boundary(metadata) is expected


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


# ── Test B: Fused dequant+gather K cache ────────────────────────────────────


def _dequantize_and_gather_k_cache_reference(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    fp8_dim = 448
    bf16_dim = 64
    scale_dim = 8
    quant_block = 64
    token_data_size = fp8_dim + bf16_dim * 2

    for req_id in range(seq_lens.shape[0]):
        seq_len = seq_lens[req_id].item()
        gather_len = gather_lens[req_id].item() if gather_lens is not None else seq_len
        start_pos = seq_len - gather_len

        for i in range(gather_len):
            pos = start_pos + i
            pos_in_block = pos % block_size
            block_idx = block_table[req_id, pos // block_size].item()
            cache_block = k_cache[block_idx].view(-1)

            token_data_start = pos_in_block * token_data_size
            fp8_bytes = cache_block[token_data_start : token_data_start + fp8_dim]
            fp8_vals = fp8_bytes.view(torch.float8_e4m3fn).float()

            scale_start = block_size * token_data_size + pos_in_block * scale_dim
            encoded_scales = cache_block[scale_start : scale_start + scale_dim]
            scales = torch.exp2(encoded_scales[:7].float() - 127.0)
            dequant = fp8_vals * scales.repeat_interleave(quant_block)

            bf16_start = token_data_start + fp8_dim
            bf16_bytes = cache_block[bf16_start : bf16_start + bf16_dim * 2]
            bf16_tail = bf16_bytes.view(torch.bfloat16)

            out[req_id, offset + i, :fp8_dim] = dequant
            out[req_id, offset + i, fp8_dim:] = bf16_tail


@pytest.mark.parametrize(
    ("seq_lens_host", "gather_lens_host", "offset"),
    [
        ([9, 23, 7], None, 0),
        ([19, 8, 257], [6, 8, 129], 5),
    ],
)
def test_dequantize_and_gather_k_cache(
    seq_lens_host: list[int],
    gather_lens_host: list[int] | None,
    offset: int,
):
    block_size = 64
    head_dim = 512
    nope_dim = 448
    scale_dim = 8
    head_bytes = nope_dim + (head_dim - nope_dim) * 2 + scale_dim
    device = "cuda"
    num_reqs = len(seq_lens_host)
    num_tokens = sum(seq_lens_host)
    max_gather_len = max(gather_lens_host or seq_lens_host)
    max_blocks_per_seq = math.ceil(max(seq_lens_host) / block_size)
    num_blocks = sum(math.ceil(seq_len / block_size) for seq_len in seq_lens_host)

    compressed_kv = torch.randn(
        num_tokens, head_dim, dtype=torch.bfloat16, device=device
    )

    # Randomize physical pages so the test covers block-table translation.
    # Keep padded block-table entries invalid to catch accidental reads.
    physical_blocks = torch.randperm(num_blocks, device=device)
    block_table = torch.full(
        (num_reqs, max_blocks_per_seq), int(-1e6), dtype=torch.int32, device=device
    )
    start = 0
    for req_id, seq_len in enumerate(seq_lens_host):
        num_req_blocks = math.ceil(seq_len / block_size)
        req_blocks = physical_blocks[start : start + num_req_blocks]
        block_table[req_id, :num_req_blocks] = req_blocks
        start += num_req_blocks

    # Build slot_mapping for quantize_and_insert_k_cache.
    slot_mapping = torch.empty(num_tokens, dtype=torch.int64, device=device)
    start = 0
    for req_id, seq_len in enumerate(seq_lens_host):
        logical_pos = torch.arange(seq_len, dtype=torch.int64, device=device)
        block_idx = block_table[req_id, logical_pos // block_size].to(torch.int64)
        token_slots = block_idx * block_size + logical_pos % block_size
        slot_mapping[start : start + seq_len] = token_slots
        start += seq_len

    # Insert compressed K into the paged cache layout used by the gather op.
    k_cache = torch.empty(
        num_blocks, block_size, head_bytes, dtype=torch.uint8, device=device
    )
    k_cache_2d = k_cache.view(num_blocks, -1)
    quantize_and_insert_k_cache(compressed_kv, k_cache_2d, slot_mapping, block_size)

    out_shape = (num_reqs, offset + max_gather_len + 3, head_dim)
    ref_out = torch.empty(out_shape, dtype=torch.bfloat16, device=device)
    actual_out = torch.empty_like(ref_out)
    seq_lens = torch.tensor(seq_lens_host, dtype=torch.int32, device=device)
    gather_lens = (
        torch.tensor(gather_lens_host, dtype=torch.int32, device=device)
        if gather_lens_host is not None
        else None
    )

    # Compare production gather against a PyTorch reference for valid output rows.
    _dequantize_and_gather_k_cache_reference(
        ref_out, k_cache, seq_lens, gather_lens, block_table, block_size, offset
    )
    dequantize_and_gather_k_cache(
        actual_out, k_cache, seq_lens, gather_lens, block_table, block_size, offset
    )
    torch.accelerator.synchronize()

    # only check non-padded content
    for req_id, seq_len in enumerate(seq_lens_host):
        gather_len = (
            gather_lens_host[req_id] if gather_lens_host is not None else seq_len
        )
        actual = actual_out[req_id, offset : offset + gather_len]
        expected = ref_out[req_id, offset : offset + gather_len]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


# ── Test C: Indexer path ────────────────────────────────────────────────────


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


# ── Test D: DeepseekV4 attention with values at different magnitudes ───────────


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


# ── Test E: Indexer fused K-cache insert (Triton kernels) ────────────────────
#
# Both kernels share the same Triton signature; use_fp4 selects between them.
# Full pipeline: state-cache gather → softmax-weighted compress → RMSNorm →
#   GPT-J RoPE → quant (MXFP4 or FP8) → paged cache insert.


def _reference_kv_compress_norm_rope(
    state_cache: torch.Tensor,
    block_table: torch.Tensor,
    positions: torch.Tensor,
    rms_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    compress_ratio: int = 1,
    overlap: int = 0,
    use_fp4: bool = False,
    rms_eps: float = 1e-6,
    fp8_max: float = 448.0,
    return_full_cache: bool = False,
):
    """Compress → RMSNorm → GPT-J RoPE → quantize.

    Gathers (1+overlap)*compress_ratio state entries per output token, applies
    per-element softmax over the scores, and computes the weighted kv sum.
    Returns (quantized_values, scale) matching the kernel's output layout.
    """
    device = state_cache.device
    head_dim = rms_weight.shape[0]
    rope_dim = cos_sin_cache.shape[-1]
    state_block_size = state_cache.shape[1]
    state_width = state_cache.shape[-1] // 2
    nope_dim = head_dim - rope_dim
    total = (1 + overlap) * compress_ratio
    results = []
    for pos in positions.tolist():
        src = torch.arange(pos - total + 1, pos + 1, dtype=torch.int64, device=device)
        valid = src >= 0
        idx = src.clamp(min=0)
        pages = block_table[0, idx // state_block_size]
        offsets = idx % state_block_size
        raw = state_cache[pages, offsets].float()  # [total, state_dim]

        # Group 0 (tokens 0..cr-1):   kv[:H],   score[SW:SW+H]
        # Group 1 (tokens cr..2cr-1): kv[H:2H], score[SW+H:SW+2H]
        if overlap:
            sw = state_width
            g0_kv = raw[:compress_ratio, :head_dim]
            g1_kv = raw[compress_ratio:, head_dim : 2 * head_dim]
            g0_scores = raw[:compress_ratio, sw : sw + head_dim]
            g1_scores = raw[compress_ratio:, sw + head_dim : sw + 2 * head_dim]
            kv = torch.cat([g0_kv, g1_kv])
            scores = torch.cat([g0_scores, g1_scores])
        else:
            kv = raw[:, :head_dim]
            scores = raw[:, state_width : state_width + head_dim]

        scores[~valid] = float("-inf")
        kv[~valid] = 0.0
        weights = torch.softmax(scores, dim=0)
        compressed = (kv * weights).sum(dim=0)  # [H]
        var = (compressed * compressed).mean()
        normed = compressed * torch.rsqrt(var + rms_eps) * rms_weight.float()
        compressed_pos = (pos // compress_ratio) * compress_ratio
        cos, sin = cos_sin_cache[compressed_pos].float().chunk(2)
        nope, rope = normed.split([nope_dim, rope_dim])
        rope = torch.stack(
            [rope[0::2] * cos - rope[1::2] * sin, rope[1::2] * cos + rope[0::2] * sin],
            dim=-1,
        ).reshape(rope_dim)
        results.append(torch.cat([nope, rope]).to(state_cache.dtype))
    result = torch.stack(results)

    if return_full_cache:
        # Contiguous 512-wide bf16 row (nope unrotated + rope rotated), matching
        # the FlashInfer full-cache layout before any per-tensor fp8 quant. The
        # kernel rounds the fp32 result to bf16 once at the store.
        return result.to(torch.bfloat16)

    if use_fp4:
        return quantize_to_mxfp4(result)
    else:
        pairs = [
            _ue8m0_reference(result[t], head_dim, fp8_max) for t in range(len(result))
        ]
        quants, scales = zip(*pairs)
        return torch.stack(quants), torch.cat(scales)


@pytest.mark.parametrize("num_tokens", [1, 7, 32])
@pytest.mark.parametrize("kv_block_size", [16, 32])
@pytest.mark.parametrize("use_fp4", [False, True])
def test_fused_kv_insert_indexer(num_tokens: int, kv_block_size: int, use_fp4: bool):
    """Fused K compress+norm+rope+quant+insert for the indexer KV cache."""
    HEAD_DIM = 128
    ROPE_DIM = 64
    BLOCK_SIZE = 16
    RMS_EPS = 1e-6
    FP8_MAX = 448.0

    device = "cuda"
    torch.manual_seed(42)
    compress_ratio = 4

    if use_fp4:
        TOKEN_STRIDE = HEAD_DIM // 2  # packed nibbles: 64 bytes
        SCALE_DIM = HEAD_DIM // 32  # ue8m0 bytes: 4
        QUANT_BLOCK = 32
        kernel = _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn
    else:
        TOKEN_STRIDE = HEAD_DIM  # FP8 bytes: 128
        SCALE_DIM = 4  # 1 float32: 4 bytes
        QUANT_BLOCK = HEAD_DIM
        kernel = _fused_kv_compress_norm_rope_insert_indexer_attn

    # overlap=1 whenever compress_ratio==4, matching DeepseekCompressor logic.
    overlap = 1 if compress_ratio == 4 else 0
    coff = 1 + overlap  # multiplier for state_dim per entry

    num_pages = (compress_ratio * num_tokens - 1) // BLOCK_SIZE + 2
    state_cache = torch.randn(
        num_pages,
        BLOCK_SIZE,
        2 * coff * HEAD_DIM,  # kv_state + score_state, each coff*HEAD_DIM wide
        dtype=torch.bfloat16,
        device=device,
    )
    block_table = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0)
    token_to_req = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    positions = torch.arange(
        compress_ratio - 1,
        compress_ratio * num_tokens,
        compress_ratio,
        dtype=torch.int64,
        device=device,
    )
    rms_weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device=device)
    cos_sin_cache = torch.randn(compress_ratio * num_tokens, ROPE_DIM, device=device)

    kv_n_blocks = (num_tokens + kv_block_size - 1) // kv_block_size + 1
    kv_cache = torch.zeros(
        kv_n_blocks,
        kv_block_size * (TOKEN_STRIDE + SCALE_DIM),
        dtype=torch.uint8,
        device=device,
    )

    kernel[(num_tokens,)](
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        token_to_req,
        positions,
        slot_mapping,
        block_table,
        block_table.stride(0),
        BLOCK_SIZE,
        rms_weight,
        RMS_EPS,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        kv_cache,
        slot_mapping,
        kv_block_size,
        HEAD_SIZE=HEAD_DIM,
        TRITON_BLOCK_SIZE=HEAD_DIM,
        STATE_WIDTH=coff * HEAD_DIM,
        COMPRESS_RATIO=compress_ratio,
        OVERLAP=overlap,
        ROPE_HEAD_DIM=ROPE_DIM,
        FP8_MAX=FP8_MAX,
        QUANT_BLOCK=QUANT_BLOCK,
        TOKEN_STRIDE=TOKEN_STRIDE,
        SCALE_DIM=SCALE_DIM,
        KV_BLOCK_STRIDE=kv_cache.stride(0),
        num_warps=1,
    )

    k_quant, scale = _reference_kv_compress_norm_rope(
        state_cache,
        block_table,
        positions,
        rms_weight,
        cos_sin_cache,
        compress_ratio,
        overlap,
        use_fp4,
        rms_eps=RMS_EPS,
        fp8_max=FP8_MAX,
    )

    if use_fp4:
        for i in range(num_tokens):
            blk, pos = i // kv_block_size, i % kv_block_size
            val_off = pos * TOKEN_STRIDE
            fp4_actual = kv_cache[blk, val_off : val_off + TOKEN_STRIDE]
            assert torch.equal(k_quant[i], fp4_actual), (
                f"token {i}: packed nibbles differ, "
                f"{(k_quant[i] != fp4_actual).sum()} "
                f"/ {TOKEN_STRIDE}"
            )

            scale_off = kv_block_size * TOKEN_STRIDE + pos * SCALE_DIM
            scale_actual = kv_cache[blk, scale_off : scale_off + SCALE_DIM]
            assert torch.equal(scale_actual, scale[i]), (
                f"token {i}: ue8m0 {scale_actual.tolist()} != {scale[i].tolist()}"
            )

    else:
        k_quant = k_quant.view(torch.uint8)
        for i in range(num_tokens):
            blk, pos = i // kv_block_size, i % kv_block_size
            val_off = pos * TOKEN_STRIDE
            assert torch.equal(
                k_quant[i], kv_cache[blk, val_off : val_off + TOKEN_STRIDE]
            ), f"token {i}: FP8 bytes differ"

            scale_off = kv_block_size * TOKEN_STRIDE + pos * SCALE_DIM
            actual_scale = kv_cache[blk, scale_off : scale_off + SCALE_DIM].view(
                torch.float32
            )
            assert torch.equal(actual_scale, scale[i : i + 1]), (
                f"token {i}: scale {actual_scale.item()} != {scale[i].item()}"
            )


@pytest.mark.parametrize("compress_ratio", [4, 128])
@pytest.mark.parametrize("store_fp8", [False, True])
def test_cutedsl_full_cache_store(compress_ratio: int, store_fp8: bool):
    """CuTeDSL compressor full-cache (FlashInfer) store parity for head=512.

    Exercises the contiguous bf16 / per-tensor fp8 store branch of both the C4
    fused kernel and the C128 split kernel against the PyTorch reference.
    """
    cutedsl = pytest.importorskip("cutlass")  # noqa: F841
    from vllm.models.deepseek_v4.nvidia.ops.sparse_attn_compress_cutedsl import (
        fused_kv_compress_norm_rope_insert_sparse_attn_cutedsl,
        split_kv_compress_norm_rope_insert_sparse_attn_cutedsl,
    )

    HEAD_DIM = 512
    ROPE_DIM = 64
    RMS_EPS = 1e-6
    FP8_MAX = 448.0
    # C128 compress (Block8 kernel) requires state-cache block_size=8; C4 uses 16.
    BLOCK_SIZE = 8 if compress_ratio == 128 else 16
    KV_BLOCK_SIZE = 64
    device = "cuda"
    torch.manual_seed(7)

    overlap = 1 if compress_ratio == 4 else 0
    coff = 1 + overlap
    num_tokens = 8

    num_pages = (compress_ratio * num_tokens - 1) // BLOCK_SIZE + 2
    # The production CompressorStateCache is fp32.
    state_cache = torch.randn(
        num_pages, BLOCK_SIZE, 2 * coff * HEAD_DIM, dtype=torch.float32, device=device
    )
    block_table = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0)
    token_to_req = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    positions = torch.arange(
        compress_ratio - 1,
        compress_ratio * num_tokens,
        compress_ratio,
        dtype=torch.int64,
        device=device,
    )
    rms_weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device=device)
    cos_sin_cache = torch.randn(
        compress_ratio * num_tokens, ROPE_DIM, dtype=torch.float32, device=device
    )

    dtype = torch.float8_e4m3fn if store_fp8 else torch.bfloat16
    kv_n_blocks = (num_tokens + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE + 1
    k_cache = torch.zeros(
        kv_n_blocks, KV_BLOCK_SIZE, HEAD_DIM, dtype=dtype, device=device
    )
    fp8_scale = torch.tensor(
        [0.5 if store_fp8 else 1.0], dtype=torch.float32, device=device
    )

    if compress_ratio == 4:
        fused_kv_compress_norm_rope_insert_sparse_attn_cutedsl(
            state_cache,
            token_to_req,
            positions,
            slot_mapping,
            block_table,
            BLOCK_SIZE,
            rms_weight,
            RMS_EPS,
            cos_sin_cache,
            k_cache,
            slot_mapping,
            KV_BLOCK_SIZE,
            k_cache.stride(0),
            head_size=HEAD_DIM,
            state_width=coff * HEAD_DIM,
            rope_head_dim=ROPE_DIM,
            fp8_max=FP8_MAX,
            quant_block=64,
            token_stride=576,
            scale_dim=8,
            compress_ratio=compress_ratio,
            overlap=True,
            store_full_kv=True,
            store_full_fp8=store_fp8,
            fp8_scale=fp8_scale,
        )
    else:
        compressed_kv = torch.empty(
            (num_tokens, HEAD_DIM), dtype=torch.float32, device=device
        )
        split_kv_compress_norm_rope_insert_sparse_attn_cutedsl(
            state_cache,
            token_to_req,
            positions,
            slot_mapping,
            block_table,
            BLOCK_SIZE,
            compressed_kv,
            rms_weight,
            RMS_EPS,
            cos_sin_cache,
            k_cache,
            slot_mapping,
            KV_BLOCK_SIZE,
            k_cache.stride(0),
            head_size=HEAD_DIM,
            state_width=coff * HEAD_DIM,
            rope_head_dim=ROPE_DIM,
            fp8_max=FP8_MAX,
            quant_block=64,
            token_stride=576,
            scale_dim=8,
            compress_ratio=compress_ratio,
            overlap=bool(overlap),
            store_full_kv=True,
            store_full_fp8=store_fp8,
            fp8_scale=fp8_scale,
        )

    ref = _reference_kv_compress_norm_rope(
        state_cache,
        block_table,
        positions,
        rms_weight,
        cos_sin_cache,
        compress_ratio,
        overlap,
        rms_eps=RMS_EPS,
        return_full_cache=True,
    )  # [num_tokens, HEAD_DIM] bf16

    actual = torch.stack(
        [k_cache[i // KV_BLOCK_SIZE, i % KV_BLOCK_SIZE] for i in range(num_tokens)]
    )
    if store_fp8:
        ref_fp8 = torch.clamp(ref.float() / fp8_scale, -FP8_MAX, FP8_MAX).to(
            torch.float8_e4m3fn
        )
        torch.testing.assert_close(actual.float(), ref_fp8.float(), rtol=0.0, atol=0.3)
    else:
        torch.testing.assert_close(actual.float(), ref.float(), rtol=3e-2, atol=3e-2)


# ── Test F: DeepseekV4 Attention two-stage split compressor (Triton) ─────────
#
# Same full pipeline as Test E (state-cache gather -> softmax-weighted compress
# -> RMSNorm -> GPT-J RoPE -> quant -> paged insert), but for the head=512
# fp8_ds_mla layout via the two-stage split


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="two-stage split compressor is only enabled for ROCm at the moment",
)
@pytest.mark.parametrize("num_tokens", [1, 4, 8, 17])
@pytest.mark.parametrize("kv_block_size", [16, 64])
def test_fused_kv_insert_split(num_tokens: int, kv_block_size: int):
    """Two-stage split compress+norm+rope+quant+insert for the head=512 KV cache."""
    HEAD_DIM = 512
    NOPE_DIM = 448
    ROPE_DIM = 64
    HEAD_BYTES = 584  # 448 fp8 + 128 bf16 + 8 uint8 scale
    RMS_EPS = 1e-6
    FP8_MAX = 448.0
    QUANT_BLOCK = 64
    TOKEN_STRIDE = 576
    SCALE_DIM = 8
    STATE_BLOCK_SIZE = 8  # CompressorStateCache block_size for cr=128

    device = "cuda"
    torch.manual_seed(42)
    compress_ratio = 128
    overlap = 0  # no overlap for cr=128
    coff = 1 + overlap

    num_pages = (compress_ratio * num_tokens - 1) // STATE_BLOCK_SIZE + 2
    state_cache = torch.randn(
        num_pages,
        STATE_BLOCK_SIZE,
        2 * coff * HEAD_DIM,  # kv_state + score_state
        dtype=torch.float32,
        device=device,
    )
    block_table = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0)
    token_to_req = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    positions = torch.arange(
        compress_ratio - 1,
        compress_ratio * num_tokens,
        compress_ratio,
        dtype=torch.int64,
        device=device,
    )
    rms_weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device=device)
    cos_sin_cache = torch.randn(
        compress_ratio * num_tokens, ROPE_DIM, dtype=torch.float32, device=device
    )

    kv_n_blocks = (num_tokens + kv_block_size - 1) // kv_block_size + 1
    kv_cache = torch.zeros(
        kv_n_blocks, kv_block_size, HEAD_BYTES, dtype=torch.uint8, device=device
    )
    compress_scratch = torch.empty(
        num_tokens, HEAD_DIM, dtype=torch.float32, device=device
    )

    _launch_two_stage_sparse_attn_compressor(
        state_cache,
        token_to_req,
        positions,
        slot_mapping,
        block_table,
        STATE_BLOCK_SIZE,
        coff * HEAD_DIM,
        compress_ratio,
        cos_sin_cache,
        kv_cache,
        slot_mapping,
        rms_weight,
        RMS_EPS,
        QUANT_BLOCK,
        TOKEN_STRIDE,
        SCALE_DIM,
        HEAD_DIM,
        ROPE_DIM,
        num_tokens,
        compress_scratch,
    )

    # PyTorch reference: compress -> RMSNorm -> GPT-J RoPE (pre-quant bf16 row).
    ref = _reference_kv_compress_norm_rope(
        state_cache,
        block_table,
        positions,
        rms_weight,
        cos_sin_cache,
        compress_ratio,
        overlap,
        rms_eps=RMS_EPS,
        fp8_max=FP8_MAX,
        return_full_cache=True,
    )  # [num_tokens, HEAD_DIM] bf16

    # Dequant + gather the fp8_ds_mla cache back to bf16 (Test B op).
    out = torch.zeros(1, num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=device)
    seq_lens = torch.tensor([num_tokens], dtype=torch.int32, device=device)
    gather_block_table = torch.arange(
        kv_n_blocks, dtype=torch.int32, device=device
    ).unsqueeze(0)
    dequantize_and_gather_k_cache(
        out, kv_cache, seq_lens, None, gather_block_table, kv_block_size, offset=0
    )
    recovered = out[0, :num_tokens]

    # NoPE (first 448): FP8 quantized, expect UE8M0 error (same bound as Test A).
    nope_diff = (recovered[:, :NOPE_DIM].float() - ref[:, :NOPE_DIM].float()).abs()
    for t in range(num_tokens):
        _, scales = _ue8m0_reference(ref[t, :NOPE_DIM].float(), QUANT_BLOCK, FP8_MAX)
        max_allowed = 16.0 * scales.max().item()
        token_diff = nope_diff[t].max().item()
        assert token_diff <= max_allowed, (
            f"Token {t} nope diff {token_diff} exceeds max_allowed "
            f"{max_allowed} (scale={scales.max().item()})"
        )

    # RoPE (last 64): stored as bf16. The kernel recomputes the rotation, so it
    # is bf16-close to the reference rather than bit-exact (cf. test_cutedsl).
    torch.testing.assert_close(recovered[:, NOPE_DIM:], ref[:, NOPE_DIM:])
