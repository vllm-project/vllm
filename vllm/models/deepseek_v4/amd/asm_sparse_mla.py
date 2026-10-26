# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 sparse MLA decode on aiter's v4 nm ASM kernels with FP8 Q
(VLLM_ROCM_USE_AITER_ASM_SPARSE_MLA).

Layout. Every DeepSeek-V4 attention cache page (SWA and compressed, target and
DSpark draft) holds 512-byte rows, the packed row of aiter's v4 nm kernels:
448 fp8 NoPE | 14 UE8M0 bytes (the 7 per-64 scales, each twice) | 50 pad.
The bf16 RoPE (64 values) of a row lives in a side pool outside vLLM's KV
allocation. Rows are named by r = (byte offset in the KV backing buffer) / 512,
the same number in both pools: the kernels read row r at kv_base + 512 r and its
RoPE at rope_base + 128 r. The backing buffer is block-outermost, so a layer's
rows are not contiguous; r stays linear because every page offset and the block
stride are multiples of 512 (asserted in register()). The side pool spans the
whole buffer, 1/4 of its bytes, so the KV planner gives the cache 4/5 of the
memory.

A layer's cache view [NB, R, 512] (R rows a page) is Geom(c, k, R): the page of
block b starts at row c + b k. Index lists are kept relative to a per-call base
row, so the layer-independent part f(slot) = (slot // R) k + slot % R is computed
once a step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

import vllm.envs as envs
from vllm.utils.math_utils import cdiv, round_up

ENABLED = envs.VLLM_ROCM_USE_AITER_ASM_SPARSE_MLA

ROW_BYTES = 512
NOPE_DIM = 448
ROPE_DIM = 64
HEADS = 128  # the v4 nm entry point used here (gqa 128, one query row a sequence)
MAX_SPLITS = 16  # aiter never picks more for v4 nm


@dataclass(frozen=True)
class Geom:
    c: int  # row of block 0's page (512-byte rows from the buffer start)
    k: int  # rows between consecutive blocks
    R: int  # rows (states) a page
    nb: int  # blocks


class _State:
    table_size: int = 0
    storage_ptr: int = -1
    raw: torch.Tensor | None = None  # uint8 [rows, 512], the whole KV buffer
    raw_fp8: torch.Tensor | None = None  # same bytes as fp8
    rope: torch.Tensor | None = None  # bf16 [rows, 64]
    arange: torch.Tensor | None = None  # int32 [table_size + 1]
    zeros: torch.Tensor | None = None  # int32 [table_size]
    split_table: torch.Tensor | None = None  # int32 [MAX_SPLITS, table_size + 1]
    rope_tables: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    weights: dict[int, torch.Tensor] = {}
    ps_workspace: Any = None


_S = _State()


def configure(vllm_config) -> None:
    """Size the index tables: tokens a step, or the C128A top-k row width."""
    from vllm.models.deepseek_v4.sparse_mla import _C128A_TOPK_ALIGNMENT
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        raise NotImplementedError("ASM sparse MLA requires gfx950.")
    from vllm.platforms.rocm import on_gfx950

    config = vllm_config.model_config.hf_config
    if not on_gfx950() or config.model_type != "deepseek_v4":
        raise NotImplementedError("ASM sparse MLA requires DeepSeek V4 on gfx950.")
    if (
        vllm_config.parallel_config.tensor_parallel_size != 1
        or config.num_attention_heads != HEADS
        or config.head_dim != 512
        or config.qk_rope_head_dim != ROPE_DIM
    ):
        raise NotImplementedError(
            "ASM sparse MLA requires 128 local heads of size 512."
        )
    if vllm_config.cache_config.cache_dtype != "fp8_ds_mla":
        raise ValueError("ASM sparse MLA requires fp8_ds_mla KV caches.")
    if (
        vllm_config.kv_transfer_config is not None
        or vllm_config.cache_config.kv_offloading_size is not None
    ):
        raise NotImplementedError("ASM sparse MLA does not yet transfer the RoPE pool.")

    c128a_width = round_up(
        cdiv(vllm_config.model_config.max_model_len, 128), _C128A_TOPK_ALIGNMENT
    )
    _S.table_size = max(
        _S.table_size,
        vllm_config.scheduler_config.max_num_batched_tokens,
        c128a_width,
    )
    _persistent_workspace(torch.device("cuda"))


def _persistent_workspace(device):
    if _S.ps_workspace is None:
        assert not torch.cuda.is_current_stream_capturing(), (
            "Prepare persistent attention workspace before graph capture"
        )
        import aiter.mla as am

        _S.ps_workspace = am.get_mla_v4_nm_ps_workspace(device, num_partitions=128)
    return _S.ps_workspace


def register(cache: torch.Tensor) -> Geom:
    """Check a bound cache view [NB, R, 512] and return its geometry; the first
    call for a backing buffer allocates the side pool and the decode tables
    (KV binding runs before CUDA-graph capture)."""
    assert ENABLED and _S.table_size > 0
    assert cache.dtype == torch.uint8 and cache.dim() == 3, (cache.dtype, cache.shape)
    nb, R, row = cache.shape
    assert row == ROW_BYTES and cache.stride(2) == 1 and cache.stride(1) == ROW_BYTES, (
        f"cache view must be [NB, R, 512] with 512-byte rows, got "
        f"{tuple(cache.shape)} / {cache.stride()}"
    )
    storage = cache.untyped_storage()
    base = storage.data_ptr()
    off = cache.data_ptr() - base
    assert off % ROW_BYTES == 0 and cache.stride(0) % ROW_BYTES == 0, (
        f"page offset {off} / block stride {cache.stride(0)} not 512-aligned"
    )
    if _S.storage_ptr != base:
        rows = storage.nbytes() // ROW_BYTES
        dev = cache.device
        _S.raw = cache.as_strided((rows, ROW_BYTES), (ROW_BYTES, 1), 0)
        _S.raw_fp8 = _S.raw.view(torch.float8_e4m3fn)
        _S.rope = None  # free a previous (profiling) pool first
        _S.rope = torch.zeros((rows, ROPE_DIM), dtype=torch.bfloat16, device=dev)
        _S.storage_ptr = base
        if _S.arange is None:
            n = _S.table_size + 1
            _S.arange = torch.arange(n, dtype=torch.int32, device=dev)
            _S.zeros = torch.zeros(_S.table_size, dtype=torch.int32, device=dev)
            splits = torch.arange(1, MAX_SPLITS + 1, dtype=torch.int32, device=dev)
            _S.split_table = splits[:, None] * _S.arange[None, :]
    k = cache.stride(0) // ROW_BYTES
    c = off // ROW_BYTES
    assert c < k, (c, k)
    return Geom(c=c, k=k, R=R, nb=nb)


def rope_view(g: Geom) -> torch.Tensor:
    """The layer's RoPE rows as [NB, R, 64] bf16 (block stride k * 64)."""
    assert _S.rope is not None
    return _S.rope.as_strided(
        (g.nb, g.R, ROPE_DIM), (g.k * ROPE_DIM, ROPE_DIM, 1), g.c * ROPE_DIM
    )


def _rows(slots: torch.Tensor, g: Geom) -> torch.Tensor:
    """f(slot) = (slot // R) k + slot % R, -1 where slot < 0 (int32)."""
    s = slots.to(torch.int32)
    r = torch.where(
        s >= 0,
        torch.div(s, g.R, rounding_mode="floor") * g.k + torch.remainder(s, g.R),
        -1,
    )
    return r.to(torch.int32)


def _step() -> dict:
    """Index lists shared by the layers of one step, kept on the forward
    context so a new step never sees an old entry."""
    from vllm.forward_context import get_forward_context

    fc = get_forward_context()
    st = getattr(fc, "_asm_sparse_mla_step", None)
    if st is None:
        st = {}
        fc._asm_sparse_mla_step = st  # type: ignore[attr-defined]
    return st


def _prepare_tables(attn) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """bf16 cos / sin [max_pos, 32] and the bf16 kv_norm weight of a layer."""
    cs = attn.rotary_emb.cos_sin_cache
    t = _S.rope_tables.get(cs.data_ptr())
    w = _S.weights.get(id(attn))
    if t is None or w is None:
        assert not torch.cuda.is_current_stream_capturing(), (
            "tables must be prepared before CUDA-graph capture"
        )
        if t is None:
            half = cs.shape[-1] // 2
            t = (
                cs[:, :half].to(torch.bfloat16).contiguous(),
                cs[:, half:].to(torch.bfloat16).contiguous(),
            )
            _S.rope_tables[cs.data_ptr()] = t
        if w is None:
            w = attn.kv_norm.weight.data.to(torch.bfloat16).contiguous()
            _S.weights[id(attn)] = w
    return t[0], t[1], w


def qk_insert(
    attn,
    q: torch.Tensor,
    kv_raw: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_decode_tokens: int,
) -> torch.Tensor:
    """Q RMSNorm + RoPE (+ FP8 group quant for decode rows) and K RMSNorm + RoPE
    + UE8M0 quant + SWA row write, on aiter's fused_qk_norm_rope_group_quant.

    ``kv_raw`` is the pre-norm kv (the kernel applies kv_norm). Decode rows'
    packed Q goes to ``attn._v4_q`` for the ASM decode; the returned bf16 q holds
    the prefill rows (decode rows are left unset)."""
    from aiter import dtypes
    from aiter.ops.fused_qk_norm_rope_cache_quant import (
        fused_qk_norm_rope_group_quant,
    )

    assert _S.raw_fp8 is not None and _S.rope is not None and _S.zeros is not None
    g: Geom = attn.swa_cache_layer._v4_geom
    cos, sin, kw = _prepare_tables(attn)
    T, H, D = q.shape
    st = _step()
    key = ("swa_rows", slot_mapping.data_ptr(), g.R, g.k, T)
    dest = st.get(key)
    if dest is None:
        dest = _rows(slot_mapping[:T], g)
        st[key] = dest
    nope_rows = _S.raw_fp8[g.c :]
    rope_rows = _S.rope[g.c :]
    bid = _S.zeros[:T]
    q_out = torch.empty((T, H, D), dtype=q.dtype, device=q.device)
    td = min(num_decode_tokens, T)
    common: dict[str, Any] = dict(
        k_weight=kw,
        cos_cache=cos,
        sin_cache=sin,
        eps=attn.eps,
        is_neox=False,
        swa_nope_scale_buff=nope_rows,
        swa_rope_buff=rope_rows,
    )
    if td > 0:
        qp = torch.empty((td, H, D), dtype=dtypes.fp8, device=q.device)
        qr = torch.empty((td, H, ROPE_DIM), dtype=q.dtype, device=q.device)
        fused_qk_norm_rope_group_quant(
            q[:td],
            kv_raw[:td],
            positions=positions[:td],
            q_nope_scale_buff=qp,
            q_rope_buff=qr,
            swa_dest_row=dest[:td],
            batch_id_per_token=bid[:td],
            **common,
        )
        attn._v4_q = (qp, qr)
    if td < T:
        fused_qk_norm_rope_group_quant(
            q[td:],
            kv_raw[td:],
            positions=positions[td:],
            q_nope_scale_buff=q_out[td:],
            swa_dest_row=dest[td:],
            batch_id_per_token=bid[: T - td],
            **common,
        )
    return q_out


def _swa_dense(swa_md, gs: Geom, T: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Step-shared dense SWA rows f_s [T, W] (-1 beyond each row's length) and
    its 0/1 validity."""
    assert _S.arange is not None
    st = _step()
    key = ("swa_dense", id(swa_md))
    hit = st.get(key)
    if hit is None:
        w = swa_md.decode_swa_width
        dense = swa_md.decode_swa_indices.reshape(-1, w)[:T]
        lens = swa_md.decode_swa_lens[:T]
        col = _S.arange[:w]
        valid = (col[None, :] < lens[:, None]) & (dense >= 0)
        f = torch.where(valid, _rows(dense, gs), -1).to(torch.int32)
        hit = (f, valid.to(torch.int32))
        st[key] = hit
    return hit


def _row_lens(swa_md, T: int, width: int) -> torch.Tensor:
    st = _step()
    key = ("row_lens", id(swa_md), T, width)
    hit = st.get(key)
    if hit is None:
        valid = swa_md.is_valid_token[:T]
        hit = torch.where(valid, width, 0).to(torch.int32)
        st[key] = hit
    return hit


def decode(
    attn,
    output: torch.Tensor,
    swa_md,
    attn_md,
    swa_only: bool,
) -> None:
    """Decode attention with one merged list of SWA and compressed KV rows."""
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        build_ragged_indices_from_dense,
    )

    assert _S.raw_fp8 is not None and _S.rope is not None
    assert _S.arange is not None and _S.split_table is not None
    qp, qr = attn._v4_q
    # Drop the layer's reference: held across layers it would keep every
    # layer's packed Q alive.
    attn._v4_q = None
    T = qp.shape[0]
    assert qp.shape[1] == HEADS and output.shape[0] == T
    gs: Geom = attn.swa_cache_layer._v4_geom
    st = _step()
    sink = attn.attn_sink
    if swa_only:
        key: tuple = ("swa_ragged", id(swa_md))
        hit = st.get(key)
        if hit is None:
            idx = _rows(swa_md.decode_swa_ragged_indices, gs)
            hit = (idx, swa_md.decode_swa_ragged_indptr[: T + 1])
            st[key] = hit
        idx, indptr = hit
        base = gs.c
        kv_len = max(attn.window_size, swa_md.decode_swa_width)
    else:
        gc: Geom = attn._v4_geom
        alpha = gs.c - gc.c
        swa_f, swa_mask = _swa_dense(swa_md, gs, T)
        w = swa_f.shape[1]
        if attn.compress_ratio == 4:
            from vllm.models.deepseek_v4.common.ops.cache_utils import (
                ComputeGlobalTopkIndicesAndLensKernel,
            )

            # Base row = this layer's SWA page: the SWA columns are shared by
            # the CSA layers of the step, the top-k columns get the layer's
            # compressed-page offset in the kernel.
            topk = attn.topk_indices_buffer[:T]
            K = topk.shape[-1]
            key = ("csa_dense", id(swa_md), T, K)
            hit = st.get(key)
            if hit is None:
                dense = torch.empty((T, w + K), dtype=torch.int32, device=qp.device)
                dense[:, :w].copy_(swa_f)
                hit = (dense, torch.empty(T, dtype=torch.int32, device=qp.device))
                st[key] = hit
            dense, lens = hit
            num_decodes = swa_md.num_decodes
            block_table = attn_md.block_table[:num_decodes]
            ComputeGlobalTopkIndicesAndLensKernel.kernel[(T,)](
                dense[:, w:],
                w + K,
                lens,
                topk,
                topk.stride(0),
                K,
                swa_md.token_to_req_indices[:T],
                block_table,
                block_table.stride(0),
                gc.R,
                swa_md.is_valid_token[:T],
                TRITON_BLOCK_SIZE=1024,
                ROW_BLOCK_STRIDE=gc.k,
                ROW_OFFSET=gc.c - gs.c,
            )
            idx, indptr = build_ragged_indices_from_dense(
                dense, _row_lens(swa_md, T, w + K)
            )
            kv_len = w + K
            base = gs.c
        else:
            key = ("hca", id(swa_md), id(attn_md))
            hit = st.get(key)
            if hit is None:
                # The builder's view is buffer[:n, :active_width] (the active
                # width varies per step); read the persistent buffer at its full
                # row width so a captured graph stays valid, masked by the lens.
                c128 = attn_md.c128a_global_decode_topk_indices
                wc = c128.stride(0)
                c128 = c128.as_strided((T, wc), (wc, 1), c128.storage_offset())
                lens = attn_md.c128a_decode_topk_lens[:T]
                col = _S.arange[:wc]
                valid = (col[None, :] < lens[:, None]) & (c128 >= 0)
                hf = torch.where(valid, _rows(c128, gc), -1).to(torch.int32)
                dense = torch.cat([swa_f, hf], dim=1)
                mask = torch.cat([swa_mask, torch.zeros_like(hf)], dim=1)
                row_lens = _row_lens(swa_md, T, w + wc)
                merged, indptr = build_ragged_indices_from_dense(dense, row_lens)
                mask_r, _ = build_ragged_indices_from_dense(
                    torch.where(dense >= 0, mask, -1).to(torch.int32), row_lens
                )
                hit = (merged, mask_r, indptr, w + wc)
                st[key] = hit
            merged, mask_r, indptr, kv_len = hit
            idx = torch.add(merged, mask_r, alpha=alpha)
            base = gc.c
    kv2d = _S.raw_fp8[base:]
    rope2d = _S.rope[base:]
    decode_packed(
        qp,
        qr,
        kv2d,
        rope2d,
        indptr,
        idx,
        sink,
        output,
        max_kv_len=kv_len,
        query_indptr=_S.arange,
        split_table=_S.split_table,
        compress_ratio=0 if swa_only else attn.compress_ratio,
    )


def decode_packed(
    q: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    kv_rope: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sink: torch.Tensor,
    output: torch.Tensor,
    *,
    max_kv_len: int,
    query_indptr: torch.Tensor,
    split_table: torch.Tensor,
    compress_ratio: int = 0,
) -> None:
    """Run packed decode with FP32 short-row correction on the larger batches.

    Bounded short rows execute the corrected kernel directly. Large compressed
    batches use single-pass CSA or persistent HCA; smaller long batches retain
    the conservative split-KV path. Tables remain graph-stable.
    """
    import aiter.mla as am

    rows = q.shape[0]
    from vllm.models.deepseek_v4.amd.asm_sparse_mla_short import (
        correct_short_packed_rows,
    )

    if max_kv_len <= 512:
        correct_short_packed_rows(
            q,
            q_rope,
            kv,
            kv_rope,
            kv_indptr,
            kv_indices,
            sink,
            output,
            max_short_kv=max_kv_len,
        )
        return
    fast = rows >= 84 and compress_ratio in (4, 128)
    if fast and compress_ratio == 128:
        am.mla_decode_fwd_v4_nm_ps(
            q,
            q_rope,
            kv,
            kv_rope,
            kv_indptr,
            kv_indices,
            sink,
            _persistent_workspace(q.device),
            out=output,
        )
        correct_short_packed_rows(
            q, q_rope, kv, kv_rope, kv_indptr, kv_indices, sink, output
        )
        return
    s = am.get_mla_v4_nm_num_kv_splits(rows, HEADS, max_kv_len)
    s = 1 if fast else max(2, min(s, MAX_SPLITS))
    am.mla_decode_fwd_v4_nm(
        q,
        q_rope,
        kv.view(-1, 1, 1, ROW_BYTES),
        kv_rope.view(-1, 1, 1, ROPE_DIM),
        output,
        query_indptr[: rows + 1],
        kv_indptr,
        kv_indices,
        1,
        sink=sink,
        num_kv_splits=s,
        split_indptr=split_table[s - 1, : rows + 1],
    )
    correct_short_packed_rows(
        q, q_rope, kv, kv_rope, kv_indptr, kv_indices, sink, output
    )


def context_kv_insert(
    attn, kv_raw: torch.Tensor, positions: torch.Tensor, slot_mapping: torch.Tensor
) -> None:
    """DSpark context KV: kv_norm + RoPE + UE8M0 quant + SWA row write of the
    pre-norm ``kv_raw`` (with a one-head dummy query that is discarded)."""
    from aiter.ops.fused_qk_norm_rope_cache_quant import (
        fused_qk_norm_rope_group_quant,
    )

    assert _S.raw_fp8 is not None and _S.rope is not None and _S.zeros is not None
    g: Geom = attn.swa_cache_layer._v4_geom
    cos, sin, kw = _prepare_tables(attn)
    n = kv_raw.shape[0]
    dest = _rows(slot_mapping[:n], g)
    dummy_q = torch.zeros(
        (n, 1, kv_raw.shape[-1]), dtype=kv_raw.dtype, device=kv_raw.device
    )
    fused_qk_norm_rope_group_quant(
        dummy_q,
        kv_raw,
        k_weight=kw,
        positions=positions[:n],
        cos_cache=cos,
        sin_cache=sin,
        eps=attn.eps,
        is_neox=False,
        q_nope_scale_buff=torch.empty_like(dummy_q),
        swa_nope_scale_buff=_S.raw_fp8[g.c :],
        swa_rope_buff=_S.rope[g.c :],
        swa_dest_row=dest,
        batch_id_per_token=_S.zeros[:n],
    )
