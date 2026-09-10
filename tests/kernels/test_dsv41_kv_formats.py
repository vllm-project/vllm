# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 fp8 (528 B) and fp4 (288 B) paged KV formats: insert kernels
against torch ports of FlashMLA's reference quantizer, and the prefill
gather/dequant path."""

import pytest
import torch

from tests.kernels.attention.test_flashmla_fused_sparse import (
    make_cos_sin_cache,
    rope_gptj,
)
from tests.kernels.dsv41_kv_reference import (
    dequantize_v41_fp4,
    dequantize_v41_fp8,
    quantize_v41_fp4,
    quantize_v41_fp8,
)
from vllm.models.deepseek_v4_1.common.ops import dequantize_and_gather_k_cache
from vllm.models.deepseek_v4_1.common.ops.fused_compress_quant_cache import (
    rope_quant_insert,
)
from vllm.utils.math_utils import round_up

V41_FP8_BYTES, V41_FP8_ALIGN = 528, 512
V41_FP4_BYTES, V41_FP4_ALIGN = 288, 256


def _paged(num_tokens, block_size, bytes_per_token, align, device):
    """Zeroed ``[num_blocks, block_size, bytes]`` view with TMA-aligned pages."""
    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    page = round_up(block_size * bytes_per_token, align)
    cache = torch.zeros(num_blocks, page, dtype=torch.uint8, device=device)
    return cache[:, : block_size * bytes_per_token].view(
        num_blocks, block_size, bytes_per_token
    )


def _split_rows(cache, block_size, data_bytes, scale_bytes):
    """Per-token (data, scale) rows of a ``[data rows | scale rows]`` page."""
    flat = cache.reshape(cache.shape[0], -1)
    data = flat[:, : block_size * data_bytes].reshape(-1, data_bytes)
    scales = flat[:, block_size * data_bytes :].reshape(-1, scale_bytes)
    return data, scales


@pytest.mark.parametrize("num_tokens", [1, 17, 300])
@pytest.mark.parametrize("compress_ratio", [1, 2])
def test_v41_fp8_insert_matches_reference(num_tokens, compress_ratio):
    device = torch.device("cuda")
    block_size = 32
    cos_sin = make_cos_sin_cache(4096, device)
    positions = torch.arange(num_tokens, device=device) + 5
    latent = torch.randn(num_tokens, 512, device=device, dtype=torch.bfloat16)
    cache = _paged(num_tokens, block_size, V41_FP8_BYTES, V41_FP8_ALIGN, device)
    slots = torch.arange(num_tokens, dtype=torch.int64, device=device)
    rope_quant_insert(latent, positions, cos_sin, cache, slots, compress_ratio)
    values, scales = _split_rows(cache, block_size, 512, 16)
    written = (positions + 1) % compress_ratio == 0
    k_pos = positions // compress_ratio * compress_ratio
    roped = rope_gptj(latent, k_pos, cos_sin)
    ref_vals, ref_scales = quantize_v41_fp8(roped)
    assert torch.equal(scales[:num_tokens][written], ref_scales[written])
    mismatch = (values[:num_tokens][written] != ref_vals[written]).float().mean()
    assert mismatch < 1e-3, mismatch
    deq = dequantize_v41_fp8(values[:num_tokens][written], scales[:num_tokens][written])
    ref = roped[written].float()
    assert ((deq.float() - ref).abs() <= ref.abs() * 0.07 + 1e-3).all()


@pytest.mark.parametrize("num_tokens", [1, 17, 300])
@pytest.mark.parametrize("compress_ratio", [1, 2])
def test_v41_fp4_insert_matches_reference(num_tokens, compress_ratio):
    device = torch.device("cuda")
    block_size = 64
    cos_sin = make_cos_sin_cache(4096, device)
    positions = torch.arange(num_tokens, device=device) + 5
    latent = torch.randn(num_tokens, 512, device=device, dtype=torch.bfloat16)
    cache = _paged(num_tokens, block_size, V41_FP4_BYTES, V41_FP4_ALIGN, device)
    slots = torch.arange(num_tokens, dtype=torch.int64, device=device)
    rope_quant_insert(latent, positions, cos_sin, cache, slots, compress_ratio)
    packed, scales = _split_rows(cache, block_size, 256, 32)
    written = (positions + 1) % compress_ratio == 0
    k_pos = positions // compress_ratio * compress_ratio
    roped = rope_gptj(latent, k_pos, cos_sin)
    ref_packed, ref_scales = quantize_v41_fp4(roped)
    assert torch.equal(scales[:num_tokens][written], ref_scales[written])
    mismatch = (packed[:num_tokens][written] != ref_packed[written]).float().mean()
    assert mismatch < 1e-3, mismatch
    deq = dequantize_v41_fp4(packed[:num_tokens][written], scales[:num_tokens][written])
    ref = roped[written].float()
    # e2m1 spacing is at most 2 * scale, so the rounding error is <= scale.
    tol = _fp4_tile_scales(ref_scales[written])
    assert ((deq.float() - ref).abs() <= tol + 1e-3).all()


def _fp4_tile_scales(scale_bytes: torch.Tensor) -> torch.Tensor:
    """Per-element e4m3 tile scales ``[T, 512]`` from ``[T, 32]`` scale bytes."""
    scale = scale_bytes.view(torch.float8_e4m3fn).float()
    return scale.view(-1, 32, 1).expand(-1, -1, 16).reshape(-1, 512)


@pytest.mark.parametrize("bytes_per_token", [V41_FP8_BYTES, V41_FP4_BYTES])
def test_gather_dequant_new_formats(bytes_per_token):
    device = torch.device("cuda")
    block_size, max_len = 32, 192
    cos_sin = make_cos_sin_cache(4096, device)
    seq_lens = torch.tensor([192, 64, 1], device=device, dtype=torch.int32)
    starts = [0, 192, 256]
    total = 256 + 1
    latent = torch.randn(total, 512, device=device, dtype=torch.bfloat16)
    positions = torch.arange(total, device=device)
    align = V41_FP8_ALIGN if bytes_per_token == V41_FP8_BYTES else V41_FP4_ALIGN
    cache = _paged(total, block_size, bytes_per_token, align, device)
    slots = torch.arange(total, dtype=torch.int64, device=device)
    rope_quant_insert(latent, positions, cos_sin, cache, slots, 1)
    blocks_per_req = (max_len + block_size - 1) // block_size
    block_table = torch.zeros(3, blocks_per_req, dtype=torch.int32, device=device)
    for r in range(3):
        n = int(seq_lens[r])
        first = starts[r] // block_size
        nblk = (n + block_size - 1) // block_size
        block_table[r, :nblk] = torch.arange(first, first + nblk, device=device)
    out = torch.zeros(3, max_len, 512, device=device, dtype=torch.bfloat16)
    dequantize_and_gather_k_cache(
        out, cache, seq_lens, None, block_table, block_size, 0
    )
    roped = rope_gptj(latent, positions, cos_sin)
    _, fp4_scales = quantize_v41_fp4(roped)
    for r in range(3):
        n = int(seq_lens[r])
        rows = slice(starts[r], starts[r] + n)
        ref = roped[rows].float()
        got = out[r, :n].float()
        if bytes_per_token == V41_FP8_BYTES:
            tol = ref.abs() * 0.07 + 1e-3
        else:
            tol = _fp4_tile_scales(fp4_scales[rows]) + 1e-3
        assert ((got - ref).abs() <= tol).all(), r
