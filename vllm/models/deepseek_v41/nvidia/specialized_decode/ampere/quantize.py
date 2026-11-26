# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The FP4 quantizers of DeepSeek-V4.1 on SM80, in CuTe DSL.

Three per-token kernels write the FP4 data that the stock SM100 kernels
produce with conversion instructions SM80 lacks (codecs.py encodes with
comparisons and integer operations instead). Each follows its
`reference` function, which states the rounding:

* `IndexerQuery` -- the indexer queries: RoPE on dims 64-127, then MXFP4
  (one UE8M0 scale per 32 dims), and the head weights times the softmax and
  head scales (`reference.indexer_q_rope_quant`). A warp per (token, head),
  four dims per lane.
* `IndexKey` -- the index keys of group-boundary latents: RMSNorm, RoPE at
  the group's first position, MXFP4, stored into the paged key cache
  (`reference.indexer_k_norm_rope` and `store_index_keys`). A warp per
  token, four dims per lane.
* `CompressedRecord` -- the 288-byte NVFP4 compressed KV record of a
  group-boundary latent (`reference.nvfp4_kv_record`), stored into the
  compressed cache. A warp per token, one 16-dim scale tile per lane.

The two storing kernels skip rows whose slot is negative or whose position
does not close a group. Byte outputs are written by address.
"""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_stream, make_ptr

from vllm.models.common.ops.moe_tail_fusion.primitives import current_cu_stream

from .codecs import (
    ceil_log2,
    e4m3,
    store_u8,
    store_u16,
    store_u32,
    to_e2m1,
    to_e4m3,
    ue8m0,
)

THREADS = 256
WARPS = THREADS // 32
INDEX_DIM = 128
INDEX_ROPE_FIRST = 64
KV_DIM = 512
KV_ROPE_FIRST = 448
NVFP4_RECORD = 288
# float32(1 / 6), the smallest MXFP4 block amax (6 * 2^-126) and the NVFP4
# scale range (the smallest normal E4M3 2^-9 and the largest, 448).
ONE_SIXTH = 0.1666666716337204
MIN_AMAX = 6.0 * 2.0**-126
MIN_NVFP4_SCALE = 2.0**-9
MAX_NVFP4_SCALE = 448.0


@cute.jit
def _rope(x: cute.Tensor, i: int, c: Float32, s: Float32):
    """GPT-J RoPE of the pair `x[2i], x[2i + 1]` in place."""
    even = x[2 * i]
    odd = x[2 * i + 1]
    x[2 * i] = even * c - odd * s
    x[2 * i + 1] = odd * c + even * s


@cute.jit
def _mxfp4_lane(x: cute.Tensor) -> tuple[Uint32, Uint32]:
    """MXFP4 of this lane's four values of a 32-value block spread over eight
    adjacent lanes: the two packed bytes as one 16-bit word (even element in
    the low nibble) and the block's UE8M0 scale byte."""
    amax = cute.arch.fmax(
        cute.arch.fmax(cute.math.absf(x[0]), cute.math.absf(x[1])),
        cute.arch.fmax(cute.math.absf(x[2]), cute.math.absf(x[3])),
    )
    for offset in cutlass.range_constexpr(3):
        amax = cute.arch.fmax(
            amax, cute.arch.shuffle_sync_bfly(amax, offset=1 << offset)
        )
    amax = cute.arch.fmax(amax, Float32(MIN_AMAX))
    exponent = ceil_log2(amax * Float32(ONE_SIXTH))
    exponent = cutlass.max(cutlass.min(exponent, Int32(127)), Int32(-127))
    inverse = ue8m0(Uint32(127 - exponent))
    word = Uint32(0)
    for i in cutlass.range_constexpr(4):
        word = word | (to_e2m1(x[i] * inverse) << (4 * i))
    return word, Uint32(exponent + 127)


class IndexerQuery:
    """MXFP4 indexer queries and scaled head weights; see the module
    docstring."""

    @cute.jit
    def __call__(
        self,
        q: cute.Pointer,
        positions: cute.Pointer,
        cos_sin: cute.Pointer,
        weights: cute.Pointer,
        weights_out: cute.Pointer,
        values: Int64,
        scales: Int64,
        tokens: Int32,
        heads: Int32,
        q_row_stride: Int32,
        cos_sin_stride: Int32,
        weights_scale: Float32,
        stream: cuda.CUstream,
    ):
        items = tokens * heads
        self.kernel(
            q,
            positions,
            cos_sin,
            weights,
            weights_out,
            values,
            scales,
            tokens,
            heads,
            q_row_stride,
            cos_sin_stride,
            weights_scale,
        ).launch(
            grid=((items + WARPS - 1) // WARPS, 1, 1),
            block=[THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q: cute.Pointer,
        positions: cute.Pointer,
        cos_sin: cute.Pointer,
        weights: cute.Pointer,
        weights_out: cute.Pointer,
        values: Int64,
        scales: Int64,
        tokens: Int32,
        heads: Int32,
        q_row_stride: Int32,
        cos_sin_stride: Int32,
        weights_scale: Float32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        lane = tid % 32
        item = block * WARPS + tid // 32
        if item < tokens * heads:
            token = item // heads
            dim = lane * 4
            gQ = cute.make_tensor(
                q + Int64(token) * Int64(q_row_stride) + (item % heads) * INDEX_DIM,
                cute.make_layout(INDEX_DIM),
            )
            x = cute.make_rmem_tensor(cute.make_layout(4), Float32)
            for i in cutlass.range_constexpr(4):
                x[i] = Float32(gQ[dim + i])
            if dim >= INDEX_ROPE_FIRST:
                gPositions = cute.make_tensor(positions, cute.make_layout(tokens))
                gCosSin = cute.make_tensor(
                    cos_sin + gPositions[token] * Int64(cos_sin_stride),
                    cute.make_layout(64),
                )
                for i in cutlass.range_constexpr(2):
                    pair = (dim - INDEX_ROPE_FIRST) // 2 + i
                    _rope(x, i, gCosSin[pair], gCosSin[32 + pair])
                for i in cutlass.range_constexpr(4):
                    x[i] = Float32(BFloat16(x[i]))
            word, scale = _mxfp4_lane(x)
            store_u16(values + Int64(item) * 64 + lane * 2, word)
            if lane % 8 == 0:
                store_u8(scales + Int64(item) * 4 + lane // 8, scale)
            if lane == 0:
                gWeights = cute.make_tensor(weights, cute.make_layout(tokens * heads))
                gOut = cute.make_tensor(weights_out, cute.make_layout(tokens * heads))
                gOut[item] = Float32(gWeights[item]) * weights_scale


class IndexKey:
    """MXFP4 index keys into the paged key cache; see the module docstring."""

    @cute.jit
    def __call__(
        self,
        k: cute.Pointer,
        positions: cute.Pointer,
        slots: cute.Pointer,
        norm_weight: cute.Pointer,
        cos_sin: cute.Pointer,
        cache: Int64,
        tokens: Int32,
        k_row_stride: Int32,
        cos_sin_stride: Int32,
        block_size: Int32,
        block_bytes: Int64,
        compress_ratio: Int32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            k,
            positions,
            slots,
            norm_weight,
            cos_sin,
            cache,
            tokens,
            k_row_stride,
            cos_sin_stride,
            block_size,
            block_bytes,
            compress_ratio,
            eps,
        ).launch(
            grid=((tokens + WARPS - 1) // WARPS, 1, 1),
            block=[THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        k: cute.Pointer,
        positions: cute.Pointer,
        slots: cute.Pointer,
        norm_weight: cute.Pointer,
        cos_sin: cute.Pointer,
        cache: Int64,
        tokens: Int32,
        k_row_stride: Int32,
        cos_sin_stride: Int32,
        block_size: Int32,
        block_bytes: Int64,
        compress_ratio: Int32,
        eps: Float32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        lane = tid % 32
        token = block * WARPS + tid // 32
        slot = Int64(-1)
        position = Int64(0)
        if token < tokens:
            slot = Int64(cute.make_tensor(slots, cute.make_layout(tokens))[token])
            position = Int64(
                cute.make_tensor(positions, cute.make_layout(tokens))[token]
            )
        if slot >= 0 and (position + 1) % compress_ratio == 0:
            dim = lane * 4
            gK = cute.make_tensor(
                k + Int64(token) * Int64(k_row_stride), cute.make_layout(INDEX_DIM)
            )
            gWeight = cute.make_tensor(norm_weight, cute.make_layout(INDEX_DIM))
            x = cute.make_rmem_tensor(cute.make_layout(4), Float32)
            squares = Float32(0.0)
            for i in cutlass.range_constexpr(4):
                x[i] = Float32(gK[dim + i])
                squares = squares + x[i] * x[i]
            for offset in cutlass.range_constexpr(5):
                squares = squares + cute.arch.shuffle_sync_bfly(
                    squares, offset=1 << offset
                )
            inverse_rms = cute.math.rsqrt(squares / INDEX_DIM + eps)
            for i in cutlass.range_constexpr(4):
                x[i] = Float32(BFloat16(x[i] * inverse_rms * Float32(gWeight[dim + i])))
            if dim >= INDEX_ROPE_FIRST:
                first = position // compress_ratio * compress_ratio
                gCosSin = cute.make_tensor(
                    cos_sin + first * Int64(cos_sin_stride), cute.make_layout(64)
                )
                for i in cutlass.range_constexpr(2):
                    pair = (dim - INDEX_ROPE_FIRST) // 2 + i
                    _rope(x, i, gCosSin[pair], gCosSin[32 + pair])
                for i in cutlass.range_constexpr(4):
                    x[i] = Float32(BFloat16(x[i]))
            word, scale = _mxfp4_lane(x)
            page = cache + (slot // block_size) * block_bytes
            row = slot % block_size
            store_u16(page + row * 64 + lane * 2, word)
            if lane % 8 == 0:
                store_u8(page + Int64(block_size) * 64 + row * 4 + lane // 8, scale)


class CompressedRecord:
    """NVFP4 compressed KV records into the paged compressed cache; see the
    module docstring."""

    @cute.jit
    def __call__(
        self,
        latent: cute.Pointer,
        positions: cute.Pointer,
        slots: cute.Pointer,
        cos_sin: cute.Pointer,
        cache: Int64,
        tokens: Int32,
        cos_sin_stride: Int32,
        block_size: Int32,
        block_bytes: Int64,
        compress_ratio: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            latent,
            positions,
            slots,
            cos_sin,
            cache,
            tokens,
            cos_sin_stride,
            block_size,
            block_bytes,
            compress_ratio,
        ).launch(
            grid=((tokens + WARPS - 1) // WARPS, 1, 1),
            block=[THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        latent: cute.Pointer,
        positions: cute.Pointer,
        slots: cute.Pointer,
        cos_sin: cute.Pointer,
        cache: Int64,
        tokens: Int32,
        cos_sin_stride: Int32,
        block_size: Int32,
        block_bytes: Int64,
        compress_ratio: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        lane = tid % 32
        token = block * WARPS + tid // 32
        slot = Int64(-1)
        position = Int64(0)
        if token < tokens:
            slot = Int64(cute.make_tensor(slots, cute.make_layout(tokens))[token])
            position = Int64(
                cute.make_tensor(positions, cute.make_layout(tokens))[token]
            )
        if slot >= 0 and (position + 1) % compress_ratio == 0:
            dim = lane * 16
            gLatent = cute.make_tensor(
                latent + Int64(token) * KV_DIM, cute.make_layout(KV_DIM)
            )
            x = cute.make_rmem_tensor(cute.make_layout(16), Float32)
            for i in cutlass.range_constexpr(16):
                x[i] = Float32(gLatent[dim + i])
            if dim >= KV_ROPE_FIRST:
                first = position // compress_ratio * compress_ratio
                gCosSin = cute.make_tensor(
                    cos_sin + first * Int64(cos_sin_stride), cute.make_layout(64)
                )
                for i in cutlass.range_constexpr(8):
                    pair = (dim - KV_ROPE_FIRST) // 2 + i
                    _rope(x, i, gCosSin[pair], gCosSin[32 + pair])
            amax = Float32(0.0)
            for i in cutlass.range_constexpr(16):
                amax = cute.arch.fmax(amax, cute.math.absf(x[i]))
            scaled = amax * Float32(ONE_SIXTH)
            scaled = cute.arch.fmax(scaled, Float32(MIN_NVFP4_SCALE))
            scaled = cute.arch.fmin(scaled, Float32(MAX_NVFP4_SCALE))
            scale_code = to_e4m3(scaled)
            scale = e4m3(scale_code)
            record = (
                cache
                + (slot // block_size) * block_bytes
                + (slot % block_size) * NVFP4_RECORD
            )
            for w in cutlass.range_constexpr(2):
                word = Uint32(0)
                for i in cutlass.range_constexpr(8):
                    word = word | (to_e2m1(x[8 * w + i] / scale) << (4 * i))
                store_u32(record + dim // 2 + 4 * w, word)
            store_u8(record + KV_DIM // 2 + lane, scale_code)


def _fake(dtype: type[cutlass.Numeric], align: int):
    """A placeholder pointer for compilation."""
    return make_ptr(dtype, align, cute.AddressSpace.gmem, assumed_align=align)


def _pointer(dtype: type[cutlass.Numeric], tensor: torch.Tensor, align: int):
    """A CuTe DSL global-memory pointer to `tensor`'s data."""
    return make_ptr(
        dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=align
    )


_DTYPES: dict[torch.dtype, type[cutlass.Numeric]] = {
    torch.bfloat16: BFloat16,
    torch.float32: Float32,
    torch.int32: Int32,
    torch.int64: cutlass.Int64,
}


@cache
def _query_kernel(device_index: int, weights_dtype: torch.dtype):
    """`IndexerQuery` compiled for a device and head-weight dtype."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            IndexerQuery(),
            _fake(BFloat16, 2),
            _fake(cutlass.Int64, 8),
            _fake(Float32, 4),
            _fake(_DTYPES[weights_dtype], 2),
            _fake(Float32, 4),
            Int64(0),
            Int64(0),
            *(Int32(1),) * 4,
            Float32(1.0),
            make_fake_stream(),
        )


@cache
def _key_kernel(device_index: int, slot_dtype: torch.dtype):
    """`IndexKey` compiled for a device and slot-mapping dtype."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            IndexKey(),
            _fake(BFloat16, 2),
            _fake(cutlass.Int64, 8),
            _fake(_DTYPES[slot_dtype], 4),
            _fake(BFloat16, 2),
            _fake(Float32, 4),
            Int64(0),
            *(Int32(1),) * 4,
            Int64(1),
            Int32(1),
            Float32(1e-6),
            make_fake_stream(),
        )


@cache
def _record_kernel(device_index: int, slot_dtype: torch.dtype):
    """`CompressedRecord` compiled for a device and slot-mapping dtype."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            CompressedRecord(),
            _fake(BFloat16, 2),
            _fake(cutlass.Int64, 8),
            _fake(_DTYPES[slot_dtype], 4),
            _fake(Float32, 4),
            Int64(0),
            *(Int32(1),) * 3,
            Int64(1),
            Int32(1),
            make_fake_stream(),
        )


def _cache_geometry(cache: torch.Tensor, record: int) -> tuple[int, int]:
    """(block size, block stride in bytes) of a uint8 paged cache whose
    pages hold `block size` rows of `record` bytes."""
    assert cache.dtype == torch.uint8 and cache.stride(-1) == 1
    assert cache.shape[1] * record <= cache.stride(0)
    return cache.shape[1], cache.stride(0)


def indexer_q_rope_quant(
    positions: torch.Tensor,
    q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    weights_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`reference.indexer_q_rope_quant` on the GPU."""
    tokens, heads, dim = q.shape
    assert dim == INDEX_DIM and q.dtype == torch.bfloat16
    assert q.stride(2) == 1 and q.stride(1) == INDEX_DIM
    assert positions.dtype == torch.int64 and positions.is_contiguous()
    assert cos_sin_cache.dtype == torch.float32 and cos_sin_cache.stride(1) == 1
    assert weights.is_contiguous() and weights.shape == (tokens, heads)
    values = torch.empty((tokens, heads, 64), dtype=torch.uint8, device=q.device)
    scales = torch.empty((tokens, heads, 4), dtype=torch.uint8, device=q.device)
    weights_out = torch.empty((tokens, heads), dtype=torch.float32, device=q.device)
    if tokens > 0:
        _query_kernel(q.device.index, weights.dtype)(
            _pointer(BFloat16, q, 2),
            _pointer(cutlass.Int64, positions, 8),
            _pointer(Float32, cos_sin_cache, 4),
            _pointer(_DTYPES[weights.dtype], weights, 2),
            _pointer(Float32, weights_out, 4),
            Int64(values.data_ptr()),
            Int64(scales.data_ptr()),
            Int32(tokens),
            Int32(heads),
            Int32(q.stride(0)),
            Int32(cos_sin_cache.stride(0)),
            Float32(weights_scale),
            current_cu_stream(),
        )
    return values, scales, weights_out


def store_index_keys(
    k_pre: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    cos_sin_cache: torch.Tensor,
    compress_ratio: int,
    k_cache: torch.Tensor,
) -> None:
    """`reference.indexer_k_norm_rope` then `store_index_keys`, fused."""
    tokens = slot_mapping.numel()
    if tokens == 0:
        return
    assert k_pre.dtype == torch.bfloat16 and k_pre.shape[1] == INDEX_DIM
    assert k_pre.stride(1) == 1 and k_pre.shape[0] >= tokens
    assert positions.dtype == torch.int64 and positions.is_contiguous()
    assert slot_mapping.is_contiguous() and slot_mapping.dtype in _DTYPES
    assert norm_weight.dtype == torch.bfloat16 and norm_weight.is_contiguous()
    assert cos_sin_cache.dtype == torch.float32 and cos_sin_cache.stride(1) == 1
    block_size, block_bytes = _cache_geometry(k_cache, 68)
    _key_kernel(k_pre.device.index, slot_mapping.dtype)(
        _pointer(BFloat16, k_pre, 2),
        _pointer(cutlass.Int64, positions, 8),
        _pointer(_DTYPES[slot_mapping.dtype], slot_mapping, 4),
        _pointer(BFloat16, norm_weight, 2),
        _pointer(Float32, cos_sin_cache, 4),
        Int64(k_cache.data_ptr()),
        Int32(tokens),
        Int32(k_pre.stride(0)),
        Int32(cos_sin_cache.stride(0)),
        Int32(block_size),
        Int64(block_bytes),
        Int32(compress_ratio),
        Float32(eps),
        current_cu_stream(),
    )


def store_compressed_records(
    latent: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    compress_ratio: int,
    kv_cache: torch.Tensor,
) -> None:
    """`reference.nvfp4_kv_record` of the group-boundary latents, stored at
    their slots of the compressed cache `[blocks, block_size, 288]`."""
    tokens = slot_mapping.numel()
    if tokens == 0:
        return
    assert latent.dtype == torch.bfloat16 and latent.is_contiguous()
    assert latent.shape[1] == KV_DIM and latent.shape[0] >= tokens
    assert positions.dtype == torch.int64 and positions.is_contiguous()
    assert slot_mapping.is_contiguous() and slot_mapping.dtype in _DTYPES
    assert cos_sin_cache.dtype == torch.float32 and cos_sin_cache.stride(1) == 1
    assert kv_cache.shape[-1] == NVFP4_RECORD
    block_size, block_bytes = _cache_geometry(kv_cache, NVFP4_RECORD)
    _record_kernel(latent.device.index, slot_mapping.dtype)(
        _pointer(BFloat16, latent, 2),
        _pointer(cutlass.Int64, positions, 8),
        _pointer(_DTYPES[slot_mapping.dtype], slot_mapping, 4),
        _pointer(Float32, cos_sin_cache, 4),
        Int64(kv_cache.data_ptr()),
        Int32(tokens),
        Int32(cos_sin_cache.stride(0)),
        Int32(block_size),
        Int64(block_bytes),
        Int32(compress_ratio),
        current_cu_stream(),
    )
