# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native sparse MLA for Flash DP attention on gfx950.

Cache rows contain 448 FP8 NoPE bytes and fourteen duplicated UE8M0 scales
in a 512-byte record. A BF16 RoPE side pool uses the same physical row IDs.
The planner reserves one fifth of the KV budget for this side pool.
"""

from dataclasses import dataclass

import torch

import vllm.envs as envs
from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import (
    _rope_quant_insert_kernel,
)
from vllm.triton_utils import tl, triton


def enabled() -> bool:
    return envs.VLLM_ROCM_USE_AITER_ASM_SPARSE_MLA


def _view_key(cache: torch.Tensor) -> tuple[int, int]:
    # KV groups alias one allocation with different rows per block (e.g. a
    # 64-row compressed view and a 32-row SWA view at the same address), so
    # the address alone does not identify a view's geometry.
    return cache.data_ptr(), cache.shape[1]


@dataclass(frozen=True)
class Geometry:
    offset: int
    stride: int
    block: int
    count: int


class NativeCachePool:
    def __init__(self, caches: list[torch.Tensor]):
        storage = caches[0].untyped_storage()
        rows = storage.nbytes() // 512
        if rows >= 2**31:
            raise ValueError("ASM KV cache exceeds the int32 row index range")
        self.kv = caches[0].as_strided((rows, 512), (512, 1), 0)
        self.rope = torch.zeros((rows, 64), dtype=torch.bfloat16, device=self.kv.device)
        self.entries: dict[tuple[int, int], Geometry] = {}
        self.tables: dict = {}
        for cache in caches:
            assert cache.untyped_storage().data_ptr() == storage.data_ptr(), (
                "ASM sparse MLA requires a shared block-outermost KV allocation"
            )
            assert cache.dtype == torch.uint8 and cache.shape[-1] == 512
            offset = cache.data_ptr() - storage.data_ptr()
            assert offset % 512 == 0 and cache.stride(0) % 512 == 0
            assert cache.stride(1) == 512 and cache.stride(2) == 1
            self.entries[_view_key(cache)] = Geometry(
                offset // 512, cache.stride(0) // 512, cache.shape[1], cache.shape[0]
            )

    def rope_view(self, cache: torch.Tensor) -> torch.Tensor:
        g = self.entries[_view_key(cache)]
        return self.rope.as_strided(
            (g.count, g.block, 64), (g.stride * 64, 64, 1), g.offset * 64
        )


_pool: NativeCachePool | None = None
_contexts: list[dict] = []


def reset_pool() -> None:
    global _pool
    _pool = None


def configure(vllm_config) -> None:
    from vllm.platforms.rocm import on_gfx950

    if (
        not on_gfx950()
        or vllm_config.parallel_config.tensor_parallel_size != 1
        or vllm_config.parallel_config.data_parallel_size < 2
        or vllm_config.kernel_config.monokernel
        or not vllm_config.use_v2_model_runner
        or vllm_config.kv_transfer_config is not None
        or vllm_config.cache_config.kv_offloading_size is not None
    ):
        raise ValueError(
            "Native ASM MLA requires gfx950 DP attention, TP1, MRV2, "
            "without monokernel or KV transfer/offloading"
        )
    context = vllm_config.compilation_config.static_forward_context
    if not any(context is c for c in _contexts):
        _contexts.append(context)


def pool() -> NativeCachePool:
    global _pool
    if _pool is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Bind the native ASM cache before graph capture")
        caches = {}
        for layer in (layer for ctx in _contexts for layer in ctx.values()):
            cache = getattr(layer, "kv_cache", None)
            if (
                isinstance(cache, torch.Tensor)
                and cache.ndim == 3
                and cache.shape[-1] == 512
                and cache.dtype == torch.uint8
            ):
                caches[_view_key(cache)] = cache
        if not caches:
            raise RuntimeError("ASM MLA requires bound native KV caches")
        _pool = NativeCachePool(list(caches.values()))
    return _pool


def rope_view(cache: torch.Tensor) -> torch.Tensor:
    return pool().rope_view(cache)


@triton.jit
def _prepare_qk(
    Q,
    K,
    Positions,
    Slots,
    CS,
    Cache,
    CacheRope,
    Out,
    Packed,
    QR,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    CS0: tl.constexpr,
    C0: tl.constexpr,
    R0: tl.constexpr,
    BLOCK: tl.constexpr,
    HEADS: tl.constexpr,
    PACK_ROWS: tl.constexpr,
    INSERT_ROWS: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    if tile == HEADS // 4:
        if row < INSERT_ROWS:
            _rope_quant_insert_kernel(
                K, Positions, CS, Cache, Slots, CS0, C0, BLOCK, 1, True, CacheRope, R0
            )
    else:
        h = tile * 4 + tl.arange(0, 4)
        d = tl.arange(0, 512)
        x = tl.load(Q + row * QS0 + h[:, None] * QS1 + d[None, :]).to(tl.float32)
        pos = tl.load(Positions + row)
        even, odd = tl.split(tl.reshape(x, (4, 256, 2)))
        pair = tl.arange(0, 256) - 224
        c = tl.load(CS + pos * CS0 + tl.maximum(pair, 0), pair >= 0, 1.0)
        s = tl.load(CS + pos * CS0 + 32 + tl.maximum(pair, 0), pair >= 0, 0.0)
        rotated = tl.reshape(tl.join(even * c - odd * s, odd * c + even * s), (4, 512))
        value = rotated.to(tl.bfloat16)
        base = (row * HEADS + h[:, None]) * 512
        if row < PACK_ROWS:
            # Flash's query is already normalized before wq_b; no head RMSNorm.
            blocks = tl.reshape(value.to(tl.float32), (4, 8, 64))
            amax = tl.maximum(tl.max(tl.abs(blocks), 2), 1e-4)
            exp = tl.ceil(tl.log2(amax / 448.0))
            quant = tl.clamp(blocks * tl.exp2(-exp[:, :, None]), -448.0, 448.0)
            quant = tl.reshape(quant, (4, 512)).to(tl.float8e4nv)
            tl.store(
                Packed + base + d[None, :],
                quant.to(tl.uint8, bitcast=True),
                d[None, :] < 448,
            )
            g = tl.arange(0, 8)
            code = (exp + 127).to(tl.uint8)
            tl.store(Packed + base + 448 + g[None, :] * 2, code, g[None, :] < 7)
            tl.store(Packed + base + 449 + g[None, :] * 2, code, g[None, :] < 7)
            tl.store(
                QR + (row * HEADS + h[:, None]) * 64 + d[None, :] - 448,
                value,
                d[None, :] >= 448,
            )
        else:
            tl.store(Out + base + d[None, :], value)


def prepare_qk(attn, q, kv, positions, metadata):
    cache = attn.swa_cache_layer.kv_cache
    rope = rope_view(cache)
    rows, heads, _ = q.shape
    td = metadata.num_decode_tokens
    packed_rows = td if td >= 16 and attn.compress_ratio > 0 else 0
    out = torch.empty_like(q)
    qp = torch.empty((packed_rows, heads, 512), dtype=torch.uint8, device=q.device)
    qr = torch.empty((packed_rows, heads, 64), dtype=q.dtype, device=q.device)
    _prepare_qk[(rows, heads // 4 + 1)](
        q,
        kv,
        positions,
        metadata.slot_mapping,
        attn.rotary_emb.cos_sin_cache,
        cache,
        rope,
        out,
        qp,
        qr,
        q.stride(0),
        q.stride(1),
        attn.rotary_emb.cos_sin_cache.stride(0),
        cache.stride(0),
        rope.stride(0),
        cache.shape[1],
        heads,
        packed_rows,
        metadata.slot_mapping.numel(),
        num_warps=4,
        enable_fp_fusion=False,
    )
    attn._asm_q = (qp, qr) if packed_rows else None
    return out


def insert_kv(attn, kv, positions, slots):
    cache = attn.swa_cache_layer.kv_cache
    rope = rope_view(cache)
    _rope_quant_insert_kernel[(slots.numel(),)](
        kv,
        positions,
        attn.rotary_emb.cos_sin_cache,
        cache,
        slots,
        attn.rotary_emb.cos_sin_cache.stride(0),
        cache.stride(0),
        cache.shape[1],
        1,
        True,
        rope,
        rope.stride(0),
        num_warps=4,
        enable_fp_fusion=False,
    )


@triton.jit
def _merge_indices(
    Mi,
    Mp,
    Ei,
    Ep,
    Indices,
    Indptr,
    MO,
    MS,
    MB: tl.constexpr,
    EO,
    ES,
    EB: tl.constexpr,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    EXTRA: tl.constexpr,
):
    row = tl.program_id(0)
    j = tl.arange(0, WIDTH)
    ms, me = tl.load(Mp + row), tl.load(Mp + row + 1)
    es, ee = 0, 0
    if EXTRA:
        es, ee = tl.load(Ep + row), tl.load(Ep + row + 1)
    nm, ne = me - ms, ee - es
    m = tl.load(Mi + ms + j, j < nm, 0)
    m = MO + m // MB * MS + m % MB
    e = tl.full((WIDTH,), 0, tl.int32)
    if EXTRA:
        e = tl.load(Ei + es + j - nm, (j >= nm) & (j < nm + ne), 0)
        e = EO + e // EB * ES + e % EB
    tl.store(Indices + ms + es + j, tl.where(j < nm, m, e), j < nm + ne)
    tl.store(Indptr + row, ms + es)
    if row == ROWS - 1:
        tl.store(Indptr + ROWS, me + ee)


@triton.jit
def _zero_empty_rows(
    Out, Indptr, S0: tl.constexpr, S1: tl.constexpr, HEADS: tl.constexpr
):
    row = tl.program_id(0)
    if tl.load(Indptr + row) == tl.load(Indptr + row + 1):
        d = tl.arange(0, HEADS * 512)
        tl.store(Out + row * S0 + d // 512 * S1 + d % 512, 0)


def decode(
    q, main, mi, mp, extra, ei, ep, sink, out, max_keys, packed=None, cache_pool=None
):
    from aiter.mla import get_mla_v4_nm_num_kv_splits, mla_decode_fwd_v4_nm

    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _rocm_sparse_attn_decode_ragged_bf16_triton,
    )

    p = cache_pool if cache_pool is not None else pool()
    rows, heads, _ = q.shape
    indices = torch.empty(rows * max_keys, dtype=torch.int32, device=q.device)
    indptr = torch.empty(rows + 1, dtype=torch.int32, device=q.device)
    gm = p.entries[_view_key(main)]
    ge = p.entries[_view_key(extra)] if extra is not None else gm
    _merge_indices[(rows,)](
        mi,
        mp,
        ei,
        ep,
        indices,
        indptr,
        gm.offset,
        gm.stride,
        gm.block,
        ge.offset,
        ge.stride,
        ge.block,
        rows,
        triton.next_power_of_2(max_keys),
        extra is not None,
    )
    if packed is None:
        _rocm_sparse_attn_decode_ragged_bf16_triton(
            q,
            p.kv,
            indices,
            indptr,
            512**-0.5,
            sink,
            448,
            64,
            8 if rows < 16 else 4,
            out=out,
            native_rope=p.rope,
        )
        return
    qp, qr = packed
    key = (rows, heads, max_keys)
    if key not in p.tables:
        qptr = torch.arange(rows + 1, dtype=torch.int32, device=q.device)
        splits = get_mla_v4_nm_num_kv_splits(rows, heads, max_keys)
        p.tables[key] = (qptr, splits, qptr * splits)
    qptr, splits, splitptr = p.tables[key]
    mla_decode_fwd_v4_nm(
        qp.view(torch.float8_e4m3fn),
        qr,
        p.kv.view(torch.float8_e4m3fn).view(-1, 1, 1, 512),
        p.rope.view(-1, 1, 1, 64),
        out,
        qptr,
        indptr,
        indices,
        1,
        sink=sink,
        num_kv_splits=splits,
        split_indptr=splitptr,
    )
    _zero_empty_rows[(rows,)](
        out, indptr, out.stride(0), out.stride(1), heads, num_warps=8
    )
