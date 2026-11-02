# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek-V4.1 MXFP4 compressed KV record (``mxfp4_ds_mla``, 272 B) on
ROCm gfx950: the insert kernel against a torch reference quantizer, and the
prefill gather and sparse decode against what the insert wrote."""

import math

import pytest
import torch

from vllm.platforms import current_platform

RECORD = 272
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _on_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import _ON_GFX950

    return bool(_ON_GFX950)


pytestmark = pytest.mark.skipif(
    not _on_gfx950(), reason="mxfp4_ds_mla is read only by the gfx950 decode"
)


def quantize_mxfp4_reference(rows: torch.Tensor) -> torch.Tensor:
    """``[N, 512]`` fp32 to ``[N, 272]`` uint8 records: scale
    ``ceil_pow2(amax / 6)`` coded in [1, 254] (127 for an all-zero group), then
    E2M1 round-half-to-even with the sign kept on a zero."""
    tiles = rows.float().reshape(-1, 16, 32)
    amax = tiles.abs().amax(-1)
    mantissa, exponent = torch.frexp(amax / 6.0)
    ceil_exp = torch.where(mantissa == 0.5, exponent - 1, exponent)
    code = (ceil_exp + 127).clamp(1, 254)
    code = torch.where(amax == 0, torch.full_like(code, 127), code)
    scaled = tiles * torch.exp2(-(code - 127).float())[..., None]
    a = scaled.abs()
    index = (
        (a > 0.25).int()
        + (a >= 0.75).int()
        + (a > 1.25).int()
        + (a >= 1.75).int()
        + (a > 2.5).int()
        + (a >= 3.5).int()
        + (a > 5.0).int()
    )
    nibbles = (index | ((scaled < 0).int() << 3)).reshape(rows.shape[0], 512)
    packed = (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).to(torch.uint8)
    return torch.cat([packed, code.to(torch.uint8)], dim=1)


def decode_mxfp4_reference(records: torch.Tensor) -> torch.Tensor:
    """``[N, 272]`` uint8 records to ``[N, 512]`` fp32."""
    mags = torch.tensor(E2M1, device=records.device)
    packed, codes = records[:, :256], records[:, 256:]
    nibbles = torch.stack([packed & 0xF, packed >> 4], dim=-1).reshape(-1, 512)
    values = mags[(nibbles & 7).long()] * torch.where(nibbles >= 8, -1.0, 1.0)
    scales = torch.exp2(codes.float() - 127.0)
    return (values.reshape(-1, 16, 32) * scales[..., None]).reshape(-1, 512)


def _paged_cache(num_blocks: int, block_size: int, fill: int = 0):
    page_bytes = math.ceil(block_size * RECORD / 512) * 512
    backing = torch.full(
        (num_blocks, page_bytes), fill, dtype=torch.uint8, device="cuda"
    )
    cache = backing.as_strided(
        (num_blocks, block_size, RECORD), (page_bytes, RECORD, 1)
    )
    return backing, cache


def _insert(latent, cos_sin, cache, slots, compress_ratio=1):
    from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import (
        rope_quant_insert,
    )

    positions = torch.arange(latent.shape[0], dtype=torch.int64, device="cuda")
    rope_quant_insert(latent, positions, cos_sin, cache, slots, compress_ratio)


@pytest.mark.parametrize("compress_ratio", [1, 2])
def test_insert_is_bit_exact_with_identity_rope(compress_ratio: int) -> None:
    """With cos = 1 and sin = 0 every dim passes through unrotated, so the
    record must match the reference quantizer byte for byte, including the
    power-of-two and all-zero groups and the rows it must skip."""
    torch.manual_seed(0)
    num_tokens, block_size = 24, 64
    latent = (torch.randn(num_tokens, 512, device="cuda") * 3).to(torch.bfloat16)
    latent[1, :32] = 6.0  # amax / 6 is a power of two exactly
    latent[2, 32:64] = 0.0  # all-zero group
    latent[4, :] = 2.0**-130  # below the smallest scale code
    cos_sin = torch.cat(
        [torch.ones(num_tokens, 32), torch.zeros(num_tokens, 32)], dim=1
    ).cuda()
    backing, cache = _paged_cache(1, block_size, fill=165)
    slots = torch.arange(num_tokens, dtype=torch.int64, device="cuda")
    slots[5] = -1
    _insert(latent, cos_sin, cache, slots, compress_ratio)

    expected = quantize_mxfp4_reference(latent.float())
    for t in range(num_tokens):
        row = cache[0, t]
        if slots[t] < 0 or (t + 1) % compress_ratio:
            assert (row == 165).all()
        else:
            torch.testing.assert_close(row, expected[t], rtol=0, atol=0)


def test_insert_rotates_the_rope_dims_before_quantizing() -> None:
    torch.manual_seed(1)
    num_tokens = 16
    latent = torch.randn(num_tokens, 512, device="cuda").to(torch.bfloat16)
    angles = torch.randn(num_tokens, 32, device="cuda")
    cos_sin = torch.cat([angles.cos(), angles.sin()], dim=1)
    _, cache = _paged_cache(1, 64)
    _insert(latent, cos_sin, cache, torch.arange(num_tokens, device="cuda"))

    rotated = latent.float().clone()
    even, odd = rotated[:, 448::2].clone(), rotated[:, 449::2].clone()
    c, s = angles.cos(), angles.sin()
    rotated[:, 448::2] = even * c - odd * s
    rotated[:, 449::2] = odd * c + even * s
    got = cache[0, :num_tokens]
    # The NoPE groups never see the rotation, so they are exact.
    expected = quantize_mxfp4_reference(rotated)
    torch.testing.assert_close(got[:, :224], expected[:, :224], rtol=0, atol=0)
    torch.testing.assert_close(got[:, 256:270], expected[:, 256:270], rtol=0, atol=0)
    # The RoPE groups may differ by an fp32 ulp before rounding: half an e2m1
    # step (at most one scale unit) bounds the decoded error.
    decoded = decode_mxfp4_reference(got)
    step = torch.exp2(got[:, 256:].float() - 127.0).repeat_interleave(32, dim=1)
    assert ((decoded - rotated).abs() <= step).all()


@pytest.mark.parametrize("num_tokens", [70, 129, 2051])
@pytest.mark.parametrize("partial", [False, True])
def test_gather_matches_insert(num_tokens: int, partial: bool) -> None:
    from vllm.models.deepseek_v41.common.ops import dequantize_and_gather_k_cache

    torch.manual_seed(2)
    block_size, num_reqs = 64, 4
    num_blocks = math.ceil(num_tokens / block_size)
    latent = torch.randn(num_tokens, 512, device="cuda").to(torch.bfloat16)
    angles = torch.randn(num_tokens, 32, device="cuda")
    cos_sin = torch.cat([angles.cos(), angles.sin()], dim=1)
    backing, cache = _paged_cache(num_blocks, block_size)
    _insert(latent, cos_sin, cache, torch.arange(num_tokens, device="cuda"))

    offset = 7 if partial else 0
    seq_lens = num_tokens - torch.arange(num_reqs, dtype=torch.int32, device="cuda")
    gather_lens = seq_lens - 3 if partial else None
    out = torch.full(
        (num_reqs, num_tokens + offset, 512), -123.0, dtype=torch.bfloat16
    ).cuda()
    block_table = torch.arange(num_blocks, dtype=torch.int32, device="cuda").repeat(
        num_reqs, 1
    )
    dequantize_and_gather_k_cache(
        out,
        cache,
        seq_lens=seq_lens,
        gather_lens=gather_lens,
        block_table=block_table,
        block_size=block_size,
        offset=offset,
    )
    expected = decode_mxfp4_reference(cache.reshape(-1, RECORD)[:num_tokens])
    start = 3 if partial else 0
    for req in range(num_reqs):
        length = num_tokens - req - start
        torch.testing.assert_close(
            out[req, offset : offset + length].float(),
            expected[start : start + length],
            rtol=0,
            atol=0,
        )
        assert (out[req, offset + length :] == -123).all()


@pytest.mark.parametrize("num_heads", [16, 64])
@pytest.mark.parametrize("num_tokens", [1, 8])
def test_decode_reads_what_the_insert_wrote(num_heads: int, num_tokens: int) -> None:
    """Write MXFP4 rows with the real insert kernel, then decode them as the
    compressed segment beside a 528 B SWA segment, against a torch softmax over
    the dequantized rows."""
    from vllm.models.deepseek_v41.common.ops import dequantize_and_gather_k_cache
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_sparse_attn_decode

    torch.manual_seed(3)
    block_size, num_slots, num_compressed, num_swa = 64, 512, 96, 64
    num_blocks = num_slots // block_size
    slots = torch.arange(num_slots, device="cuda")
    angles = torch.randn(num_slots, 32, device="cuda")
    cos_sin = torch.cat([angles.cos(), angles.sin()], dim=1)

    latent = (torch.randn(num_slots, 512, device="cuda") * 0.7).to(torch.bfloat16)
    _, compressed = _paged_cache(num_blocks, block_size)
    _insert(latent, cos_sin, compressed, slots)
    compressed_rows = decode_mxfp4_reference(compressed.reshape(-1, RECORD))

    # The SWA segment is the 528 B MXFP8 record from the same insert kernel,
    # read back through its gather.
    swa_page = math.ceil(block_size * 528 / 512) * 512
    swa = torch.zeros(num_blocks, swa_page, dtype=torch.uint8, device="cuda")
    swa = swa.as_strided((num_blocks, block_size, 528), (swa_page, 528, 1))
    swa_latent = (torch.randn(num_slots, 512, device="cuda") * 0.7).to(torch.bfloat16)
    _insert(swa_latent, cos_sin, swa, slots)
    swa_rows = torch.empty(1, num_slots, 512, dtype=torch.bfloat16, device="cuda")
    dequantize_and_gather_k_cache(
        swa_rows,
        swa,
        seq_lens=torch.tensor([num_slots], dtype=torch.int32, device="cuda"),
        gather_lens=None,
        block_table=torch.arange(num_blocks, dtype=torch.int32, device="cuda")[None],
        block_size=block_size,
        offset=0,
    )
    swa_rows = swa_rows[0].float()

    q = torch.randn(num_tokens, num_heads, 512, device="cuda", dtype=torch.bfloat16)
    sinks = torch.randn(num_heads, device="cuda")
    out = torch.empty_like(q)
    topk = torch.stack(
        [torch.randperm(num_slots, device="cuda")[:num_compressed] for _ in q]
    ).int()
    swa_idx = torch.stack(
        [torch.randperm(num_slots, device="cuda")[:num_swa] for _ in q]
    ).int()
    scale = 1.0 / math.sqrt(512)
    rocm_sparse_attn_decode(
        q=q,
        kv_cache=compressed,
        swa_k_cache=swa,
        swa_only=False,
        topk_indices=topk,
        topk_lens=torch.full((num_tokens,), num_compressed, device="cuda").int(),
        swa_indices=swa_idx,
        swa_lens=torch.full((num_tokens,), num_swa, device="cuda").int(),
        swa_ragged_indices=None,
        swa_ragged_indptr=None,
        topk_ragged_indices=None,
        topk_ragged_indptr=None,
        attn_sink=sinks,
        scale=scale,
        head_dim=512,
        nope_head_dim=448,
        rope_head_dim=64,
        output=out,
        extra_cache_nan_free=True,
    )
    for t in range(num_tokens):
        keys = torch.cat(
            [swa_rows[swa_idx[t].long()], compressed_rows[topk[t].long()]]
        ).double()
        logits = q[t].double() @ keys.T * scale
        weights = torch.cat([logits, sinks[:, None].double()], -1).softmax(-1)
        torch.testing.assert_close(
            out[t].double(), weights[:, :-1] @ keys, rtol=2e-2, atol=2e-2
        )
