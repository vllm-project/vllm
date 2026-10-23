# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 FlyDSL Project Contributors
"""Correctness test for the FlyDSL TurboQuant 4-bit KV decode kernel.

The kernel is validated against a pure-PyTorch full-precision attention
oracle rather than another kernel: the synthetic KV cache is built from
known centroid indices, K-norms and V-scales/zeros, so we can dequantize
the exact stored values and run fp32 softmax attention as ground truth.
Both sides therefore see identical quantized values; the only difference
is bf16-vs-fp32 accumulation, which is bounded by ``ATOL`` below.

Skipped unless running on ROCm/gfx950 with FlyDSL importable.
"""

import pytest
import torch

from vllm.platforms import current_platform

# Guard the ROCm-only import: `vllm.platforms.rocm` pulls in amdsmi and other
# ROCm-specific modules that are not importable on non-ROCm platforms, so it
# must not be imported at module load time on CUDA/CPU CI.
if not current_platform.is_rocm():
    pytest.skip(
        "TurboQuant FlyDSL decode only runs on ROCm gfx950.",
        allow_module_level=True,
    )

from vllm.platforms.rocm import on_gfx950  # noqa: E402

if not on_gfx950():
    pytest.skip(
        "TurboQuant FlyDSL decode only runs on ROCm gfx950.",
        allow_module_level=True,
    )

from vllm.v1.attention.ops.flydsl_turboquant_decode import (  # noqa: E402
    flydsl_turboquant_decode_attention,
    is_flydsl_available,
)

if not is_flydsl_available():
    pytest.skip("FlyDSL runtime is not available.", allow_module_level=True)

HEAD_SIZE = 128
N_CENTROIDS = 16
KEY_DATA_BYTES = HEAD_SIZE // 2  # 4-bit packed key nibbles
DATA_BYTES_PER_SLOT = HEAD_SIZE  # packed K (64) + packed V (64)
NUM_SOA_FIELDS = 3
SOA_K_NORM, SOA_V_SCALE, SOA_V_ZERO = 0, 1, 2

# bf16-vs-fp32 accumulation tolerance (FlyDSL test_pa.py default).
ATOL = 5e-3


def _build_cache(
    num_seqs,
    num_kv_heads,
    seq_len,
    qg,
    kv_block_size,
    seed=0xC0FFEE,
    device="cuda",
    lead_blocks=0,
):
    """Vectorized builder for a SoA-layout 4-bit KV cache + fp32 ground truth.

    Returns ``(centroids, q_bf16, kv_cache_4d, block_table, seq_lens,
    K_ref, V_ref)`` where ``K_ref``/``V_ref`` are the dequantized
    full-precision keys/values used by the reference attention.

    ``lead_blocks`` prepends that many unused blocks, which pushes the
    referenced blocks to a high byte offset without growing the data.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    hk = num_kv_heads
    hq = hk * qg

    centroids = (
        torch.randn(N_CENTROIDS, generator=g, dtype=torch.float32, device=device) * 0.5
    )
    q = (
        torch.randn(
            num_seqs, hq, HEAD_SIZE, generator=g, dtype=torch.float32, device=device
        )
        * 0.1
    )
    q_bf16 = q.to(torch.bfloat16)

    blocks_per_seq = (seq_len + kv_block_size - 1) // kv_block_size
    max_bps = blocks_per_seq + 4
    block_table = torch.zeros(num_seqs, max_bps, dtype=torch.int32, device=device)
    block_table[:, :blocks_per_seq] = (
        torch.arange(
            num_seqs * blocks_per_seq, dtype=torch.int32, device=device
        ).reshape(num_seqs, blocks_per_seq)
        + 1
        + lead_blocks
    )
    total_blocks = lead_blocks + num_seqs * blocks_per_seq + 8

    slot_size_aligned = (KEY_DATA_BYTES + 2) + (KEY_DATA_BYTES + 4)  # 134
    if slot_size_aligned % 2:
        slot_size_aligned += 1
    bytes_per_block = kv_block_size * hk * slot_size_aligned

    k_idx = torch.randint(
        0,
        N_CENTROIDS,
        (num_seqs, hk, seq_len, HEAD_SIZE),
        generator=g,
        dtype=torch.uint8,
        device=device,
    )
    v_idx = torch.randint(
        0,
        N_CENTROIDS,
        (num_seqs, hk, seq_len, HEAD_SIZE),
        generator=g,
        dtype=torch.uint8,
        device=device,
    )
    knorm = (
        torch.rand(num_seqs, hk, seq_len, generator=g, device=device) * 0.5 + 0.5
    ).to(torch.float16)
    vscale = (
        torch.rand(num_seqs, hk, seq_len, generator=g, device=device) * 0.05 + 0.01
    ).to(torch.float16)
    vzero = (
        (torch.rand(num_seqs, hk, seq_len, generator=g, device=device) - 0.5) * 0.1
    ).to(torch.float16)

    k_ref = centroids[k_idx.long()].float() * knorm.float().unsqueeze(-1)
    v_ref = v_idx.float() * vscale.float().unsqueeze(-1) + vzero.float().unsqueeze(-1)

    k_packed = k_idx[..., 0::2] | (k_idx[..., 1::2] << 4)
    v_packed = v_idx[..., 0::2] | (v_idx[..., 1::2] << 4)

    kv_cache = torch.zeros(
        total_blocks, bytes_per_block, dtype=torch.uint8, device=device
    )

    seq_ax = (
        torch.arange(num_seqs, device=device)
        .view(num_seqs, 1)
        .expand(num_seqs, seq_len)
    )
    tok_ax = (
        torch.arange(seq_len, device=device).view(1, seq_len).expand(num_seqs, seq_len)
    )
    blk_for_tok = block_table[seq_ax, tok_ax // kv_block_size]
    slot_for_tok = tok_ax % kv_block_size

    h_ax = torch.arange(hk, device=device)
    base_data = (
        slot_for_tok.unsqueeze(1) * hk * DATA_BYTES_PER_SLOT
        + h_ax.view(1, hk, 1) * DATA_BYTES_PER_SLOT
    )
    blk_b = blk_for_tok.unsqueeze(1).expand(num_seqs, hk, seq_len).long()
    dst_base = blk_b * bytes_per_block + base_data
    rng_k = torch.arange(KEY_DATA_BYTES, device=device)
    rng_v = torch.arange(KEY_DATA_BYTES, device=device) + KEY_DATA_BYTES
    k_dst = (dst_base.unsqueeze(-1) + rng_k.view(1, 1, 1, -1)).reshape(-1)
    v_dst = (dst_base.unsqueeze(-1) + rng_v.view(1, 1, 1, -1)).reshape(-1)
    flat = kv_cache.view(-1)
    flat[k_dst] = k_packed.reshape(-1)
    flat[v_dst] = v_packed.reshape(-1)

    meta_off = kv_block_size * hk * DATA_BYTES_PER_SLOT
    meta_view = kv_cache.view(torch.float16).view(total_blocks, -1)
    meta_off_hw = meta_off // 2
    common = h_ax.view(
        1, hk, 1
    ) * NUM_SOA_FIELDS * kv_block_size + slot_for_tok.unsqueeze(1)
    blk_flat = blk_for_tok.unsqueeze(1).expand(num_seqs, hk, seq_len)
    knorm_off = meta_off_hw + common + SOA_K_NORM * kv_block_size
    vscale_off = meta_off_hw + common + SOA_V_SCALE * kv_block_size
    vzero_off = meta_off_hw + common + SOA_V_ZERO * kv_block_size
    meta_view[blk_flat.reshape(-1).long(), knorm_off.reshape(-1).long()] = (
        knorm.reshape(-1)
    )
    meta_view[blk_flat.reshape(-1).long(), vscale_off.reshape(-1).long()] = (
        vscale.reshape(-1)
    )
    meta_view[blk_flat.reshape(-1).long(), vzero_off.reshape(-1).long()] = (
        vzero.reshape(-1)
    )

    seq_lens = torch.full((num_seqs,), seq_len, dtype=torch.int32, device=device)
    kv_cache_4d = kv_cache.view(total_blocks, kv_block_size, hk, slot_size_aligned)
    return (
        centroids,
        q_bf16,
        kv_cache_4d,
        block_table,
        seq_lens,
        k_ref.cpu(),
        v_ref.cpu(),
    )


def _reference_attention(q_bf16, k_ref, v_ref, seq_lens, scale):
    """fp32 softmax attention on dequantized keys/values (ground truth)."""
    num_seqs, hq, d = q_bf16.shape
    hk = k_ref.shape[1]
    qg = hq // hk
    q = q_bf16.float().reshape(num_seqs, hk, qg, d)
    out = torch.zeros(num_seqs, hq, d, dtype=torch.float32)
    for s in range(num_seqs):
        for h in range(hk):
            ql = int(seq_lens[s].item())
            k = k_ref[s, h, :ql]
            v = v_ref[s, h, :ql]
            scores = (q[s, h] @ k.T) * scale
            m = scores.max(dim=-1, keepdim=True).values
            e = torch.exp(scores - m)
            p = e / e.sum(dim=-1, keepdim=True)
            out[s, h * qg : (h + 1) * qg] = p @ v
    return out


def _run_flydsl(query, centroids, kv_cache, block_table, seq_lens, seq_len, scale):
    identity = torch.eye(HEAD_SIZE, dtype=torch.float32, device="cuda")
    return flydsl_turboquant_decode_attention(
        query=query,
        kv_cache=kv_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        Pi=identity,
        centroids=centroids,
        scale=scale,
        mse_bits=4,
        key_packed_size=KEY_DATA_BYTES + 2,
        value_quant_bits=4,
        value_packed_size=KEY_DATA_BYTES + 4,
        key_fp8=False,
        norm_correction=False,
        PiT=identity.T.contiguous(),
        max_seq_len=seq_len,
        max_num_kv_splits=32,
        sinks=None,
    )


@pytest.mark.parametrize("num_seqs", [1, 4, 16])
@pytest.mark.parametrize("seq_len", [256, 1024, 4096])
@pytest.mark.parametrize(
    "num_kv_heads,qg",
    [
        (8, 8),  # Qwen2.5-72B class
        (8, 16),  # Qwen3-32B class
        (4, 6),  # MiniMax-M2.5 class
    ],
)
@pytest.mark.parametrize("kv_block_size", [16, 32])
def test_flydsl_matches_reference(num_seqs, seq_len, num_kv_heads, qg, kv_block_size):
    """FlyDSL decode must match fp32 attention on dequantized KV."""
    centroids, q_bf16, kv_cache, block_table, seq_lens, k_ref, v_ref = _build_cache(
        num_seqs, num_kv_heads, seq_len, qg, kv_block_size
    )
    scale = 1.0 / (HEAD_SIZE**0.5)

    out = _run_flydsl(
        q_bf16, centroids, kv_cache, block_table, seq_lens, seq_len, scale
    )

    ref = _reference_attention(q_bf16.cpu(), k_ref, v_ref, seq_lens.cpu(), scale)

    torch.testing.assert_close(out.cpu().float(), ref, atol=ATOL, rtol=0.0)


@pytest.mark.parametrize("num_kv_heads,qg", [(8, 8), (8, 16), (4, 6)])
def test_flydsl_fp16_query(num_kv_heads, qg):
    """fp16 models: output keeps the query dtype and matches the reference."""
    seq_len = 1024
    centroids, q_bf16, kv_cache, block_table, seq_lens, k_ref, v_ref = _build_cache(
        4, num_kv_heads, seq_len, qg, 16
    )
    # x16 peaks the softmax: at the default 0.1 scale attention is near-uniform,
    # so a misread Q still lands within ATOL. x16 is exact in fp16 and bf16.
    q_fp16 = (q_bf16 * 16).to(torch.float16)
    scale = 1.0 / (HEAD_SIZE**0.5)

    out = _run_flydsl(
        q_fp16, centroids, kv_cache, block_table, seq_lens, seq_len, scale
    )

    assert out.dtype == torch.float16
    ref = _reference_attention(q_fp16.cpu(), k_ref, v_ref, seq_lens.cpu(), scale)
    torch.testing.assert_close(out.cpu().float(), ref, atol=ATOL, rtol=0.0)


def test_flydsl_kv_cache_over_4gib():
    """Blocks past 4 GiB: the block base must not wrap the 32-bit voffset."""
    num_seqs, num_kv_heads, qg, seq_len, kv_block_size = 1, 8, 8, 256, 16
    slot_size_aligned = (KEY_DATA_BYTES + 2) + (KEY_DATA_BYTES + 4)
    bytes_per_block = kv_block_size * num_kv_heads * slot_size_aligned
    # Enough leading blocks that every referenced block starts above 4 GiB.
    lead_blocks = (1 << 32) // bytes_per_block + 1
    need = (lead_blocks + num_seqs * (seq_len // kv_block_size) + 8) * bytes_per_block
    if current_platform.mem_get_info()[0] < need + (1 << 30):
        pytest.skip("needs >5 GiB free device memory")

    centroids, q_bf16, kv_cache, block_table, seq_lens, k_ref, v_ref = _build_cache(
        num_seqs,
        num_kv_heads,
        seq_len,
        qg,
        kv_block_size,
        lead_blocks=lead_blocks,
    )
    assert kv_cache.numel() > (1 << 32)
    scale = 1.0 / (HEAD_SIZE**0.5)

    out = _run_flydsl(
        q_bf16, centroids, kv_cache, block_table, seq_lens, seq_len, scale
    )

    ref = _reference_attention(q_bf16.cpu(), k_ref, v_ref, seq_lens.cpu(), scale)
    torch.testing.assert_close(out.cpu().float(), ref, atol=ATOL, rtol=0.0)
