# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 paged KV formats read by FlashMLA (PR #221).

A page stores ``block_size`` data rows followed by ``block_size`` scale rows.

* V4.1 fp8, 528 B/token: 512 e4m3 values (RoPE dims quantized too) in 16 tiles
  of 32, one ue8m0 scale per tile: ``2**ceil(log2(clamp_min(amax / 448,
  1e-4)))``. Used for the sliding-window cache.
* V4.1 fp4, 288 B/token: 512 e2m1 values packed two per byte (even element in
  the low nibble) in 32 tiles of 16, one e4m3 scale ``clamp(amax / 6, 2^-9,
  448)`` per tile. Used for the compressed cache next to a V4.1 fp8 SWA cache.

Insert kernels take the compressor's bf16 latent (or the SWA KV row), apply
GPT-J RoPE at the group position and quantize; gather kernels dequantize
selected rows to bf16 for prefill.
"""

import torch

from vllm.models.deepseek_v4.common.ops.fused_indexer_q import _fp32x2_to_fp4x2
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

V41_FP8_BYTES = 528
V41_FP4_BYTES = 288
_GATHER_WORKERS = 128


@triton.jit
def _roped_row(
    latent,
    t,
    position,
    cos_sin,
    COS_STRIDE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
):
    """GPT-J RoPE of ``latent[t]`` at the group position; bf16-rounded fp32."""
    d = tl.arange(0, 512)
    normed = tl.load(latent + t.to(tl.int64) * 512 + d).to(tl.float32)
    even, odd = tl.split(tl.reshape(normed, (256, 2)))
    pair = tl.arange(0, 256) - 224
    cs = cos_sin + (position // COMPRESS_RATIO * COMPRESS_RATIO) * COS_STRIDE
    c = tl.load(cs + tl.maximum(pair, 0), pair >= 0, other=1.0).to(tl.float32)
    s = tl.load(cs + 32 + tl.maximum(pair, 0), pair >= 0, other=0.0).to(tl.float32)
    row = tl.interleave(even * c - odd * s, odd * c + even * s)
    return row.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _v41_fp8_insert_kernel(
    latent,
    positions,
    cos_sin,
    cache,
    cache_slots,
    COS_STRIDE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
):
    t = tl.program_id(0)
    slot = tl.load(cache_slots + t)
    if slot < 0:
        return
    position = tl.load(positions + t)
    if (position + 1) % COMPRESS_RATIO != 0:
        return
    row = _roped_row(latent, t, position, cos_sin, COS_STRIDE, COMPRESS_RATIO)
    tiles = tl.reshape(row, (16, 32))
    scale_inv = tl.maximum(tl.max(tl.abs(tiles), 1) * (1.0 / 448.0), 1e-4)
    # ceil(log2(scale_inv)) from the fp32 bit pattern (exact, unlike lg2.approx).
    bits = scale_inv.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    inv_scale = ((127 - exponent) << 23).to(tl.float32, bitcast=True)
    fp8 = tl.clamp(tiles * tl.reshape(inv_scale, (16, 1)), -448.0, 448.0)
    fp8 = fp8.to(tl.float8e4nv)
    page = cache + (slot // CACHE_BLOCK).to(tl.int64) * CACHE_STRIDE
    d = tl.arange(0, 512)
    tl.store(
        page + (slot % CACHE_BLOCK) * 512 + d,
        tl.reshape(fp8.to(tl.uint8, bitcast=True), (512,)),
    )
    scale_row = page + CACHE_BLOCK * 512 + (slot % CACHE_BLOCK) * 16
    tl.store(scale_row + tl.arange(0, 16), (exponent + 127).to(tl.uint8))


@triton.jit
def _v41_fp4_insert_kernel(
    latent,
    positions,
    cos_sin,
    cache,
    cache_slots,
    COS_STRIDE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
):
    t = tl.program_id(0)
    slot = tl.load(cache_slots + t)
    if slot < 0:
        return
    position = tl.load(positions + t)
    if (position + 1) % COMPRESS_RATIO != 0:
        return
    row = _roped_row(latent, t, position, cos_sin, COS_STRIDE, COMPRESS_RATIO)
    tiles = tl.reshape(row, (32, 16))
    amax = tl.max(tl.abs(tiles), 1)
    scale = tl.clamp(amax / 6.0, 0.001953125, 448.0).to(tl.float8e4nv)
    # IEEE division: Triton's default div.full misplaces exact e2m1 ties.
    scaled = tl.math.div_rn(tiles, tl.reshape(scale.to(tl.float32), (32, 1)))
    scaled = tl.reshape(scaled, (512,))
    lo, hi = tl.split(tl.reshape(scaled, (256, 2)))
    packed = _fp32x2_to_fp4x2(lo, hi)
    page = cache + (slot // CACHE_BLOCK).to(tl.int64) * CACHE_STRIDE
    tl.store(page + (slot % CACHE_BLOCK) * 256 + tl.arange(0, 256), packed)
    scale_row = page + CACHE_BLOCK * 256 + (slot % CACHE_BLOCK) * 32
    tl.store(scale_row + tl.arange(0, 32), scale.to(tl.uint8, bitcast=True))


def _check_insert_args(
    latent: torch.Tensor,
    positions: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    compress_ratio: int,
    bytes_per_token: int,
) -> int:
    assert compress_ratio in (1, 2)
    assert latent.shape[1] == 512 and latent.dtype == torch.bfloat16
    assert latent.is_contiguous()
    assert kv_cache.dtype == torch.uint8 and kv_cache.shape[-1] == bytes_per_token
    assert kv_cache.stride(-1) == 1
    num_tokens = slot_mapping.numel()
    assert num_tokens <= min(latent.shape[0], positions.numel())
    return num_tokens


def _insert(
    kernel,
    bytes_per_token: int,
    latent: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    compress_ratio: int,
) -> None:
    num_tokens = _check_insert_args(
        latent, positions, kv_cache, slot_mapping, compress_ratio, bytes_per_token
    )
    if num_tokens == 0:
        return
    launch_kwargs = {"launch_pdl": False} if current_platform.is_cuda() else {}
    kernel[(num_tokens,)](
        latent,
        positions,
        cos_sin_cache,
        kv_cache,
        slot_mapping,
        COS_STRIDE=cos_sin_cache.stride(0),
        CACHE_STRIDE=kv_cache.stride(0),
        CACHE_BLOCK=kv_cache.shape[1],
        COMPRESS_RATIO=compress_ratio,
        num_warps=4,
        **launch_kwargs,
    )


def rope_v41_fp8_insert(
    latent: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    compress_ratio: int,
) -> None:
    """RoPE ``latent`` rows and store them as V4.1 fp8 (528 B) paged rows."""
    _insert(
        _v41_fp8_insert_kernel,
        V41_FP8_BYTES,
        latent,
        positions,
        cos_sin_cache,
        kv_cache,
        slot_mapping,
        compress_ratio,
    )


def rope_v41_fp4_insert(
    latent: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    compress_ratio: int,
) -> None:
    """RoPE ``latent`` rows and store them as V4.1 fp4 (288 B) paged rows."""
    _insert(
        _v41_fp4_insert_kernel,
        V41_FP4_BYTES,
        latent,
        positions,
        cos_sin_cache,
        kv_cache,
        slot_mapping,
        compress_ratio,
    )


@triton.jit
def _v41_gather_kernel(
    out_ptr,
    out_stride0,
    out_stride1,
    k_cache_ptr,
    seq_lens_ptr,
    block_table_ptr,
    offset,
    gather_lens_ptr,
    max_blocks_per_seq: tl.constexpr,
    cache_block_size: tl.constexpr,
    block_stride: tl.constexpr,
    FP4: tl.constexpr,
):
    """Gather the last ``gather_len`` rows of each request into bf16 ``out``."""
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    if gather_lens_ptr is not None:  # noqa: SIM108
        gather_len = tl.load(gather_lens_ptr + batch_idx)
    else:
        gather_len = seq_len
    start_pos = seq_len - gather_len
    d = tl.arange(0, 512)

    for i in range(worker_id, gather_len, num_workers):
        pos = start_pos + i
        block_in_seq = pos // cache_block_size
        pos_in_block = pos % cache_block_size
        physical_block_idx = tl.load(
            block_table_ptr + batch_idx * max_blocks_per_seq + block_in_seq
        )
        page = k_cache_ptr + physical_block_idx.to(tl.int64) * block_stride
        if FP4:
            packed = tl.load(page + pos_in_block * 256 + tl.arange(0, 256))
            codes = tl.interleave(
                (packed & 0xF).to(tl.int32), (packed >> 4).to(tl.int32)
            )
            mag_code = codes & 7
            e = (mag_code >> 1).to(tl.float32)
            m = (mag_code & 1).to(tl.float32)
            mag = tl.where(mag_code < 2, m * 0.5, (1.0 + m * 0.5) * tl.exp2(e - 1.0))
            vals = tl.where(codes >= 8, -mag, mag)
            sf = tl.load(
                page + cache_block_size * 256 + pos_in_block * 32 + tl.arange(0, 32)
            )
            scale = sf.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            row = tl.reshape(
                tl.reshape(vals, (32, 16)) * tl.reshape(scale, (32, 1)), (512,)
            )
        else:
            raw = tl.load(page + pos_in_block * 512 + d)
            vals = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            e8m0 = tl.load(
                page + cache_block_size * 512 + pos_in_block * 16 + tl.arange(0, 16)
            )
            scale = tl.exp2(e8m0.to(tl.float32) - 127.0)
            row = tl.reshape(
                tl.reshape(vals, (16, 32)) * tl.reshape(scale, (16, 1)), (512,)
            )
        out_row = (
            out_ptr
            + batch_idx.to(tl.int64) * out_stride0
            + (offset + i).to(tl.int64) * out_stride1
        )
        tl.store(out_row + d, row.to(tl.bfloat16))


def gather_dequant_v41(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    """Dequantize V4.1 fp8 or fp4 pages (by ``k_cache.shape[-1]``) into ``out``.

    Same contract as ``dequantize_and_gather_k_cache``: ``out`` is
    ``[num_reqs, max_tokens, 512]`` bf16 and request ``r`` receives its last
    ``gather_lens[r]`` (or all ``seq_lens[r]``) rows starting at column
    ``offset``.
    """
    bytes_per_token = k_cache.shape[-1]
    assert bytes_per_token in (V41_FP8_BYTES, V41_FP4_BYTES), bytes_per_token
    assert k_cache.dtype == torch.uint8 and out.dtype == torch.bfloat16
    num_reqs = seq_lens.shape[0]
    if num_reqs == 0:
        return
    _v41_gather_kernel[(num_reqs, _GATHER_WORKERS)](
        out,
        out.stride(0),
        out.stride(1),
        k_cache,
        seq_lens,
        block_table,
        offset,
        gather_lens,
        max_blocks_per_seq=block_table.shape[-1],
        cache_block_size=block_size,
        block_stride=k_cache.stride(0),
        FP4=bytes_per_token == V41_FP4_BYTES,
    )
