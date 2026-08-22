# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""rel_mha extend Gluon kernel for AMD GFX950.

This handles ragged, multi-token queries against a paged KV cache. The query
axis is tiled into the MFMA ``M`` dimension (prefill-style): ``BLOCK_M`` query
rows of a single q-head share each paged KV tile, so every KV tile is loaded
once and reused across all rows in the tile. The grid is
``(blocks_per_req, batch, n_heads)`` -- all host-known sizes -- and each program
self-locates its request from ``cu_seqlens_q`` / ``cache_seqlens`` in-kernel, so
the launch stays CUDA-graph static with no device->host sync.

Visibility per query row depends on ``is_causal`` and the optional sliding
window:

* ``is_causal=False``: every query token attends the full visible cache, i.e.
  ``visible_kv = cache_seqlens[batch]``.
* ``is_causal=True``: query tokens are a causal suffix, so the ``i``-th query
  token of a request (0-indexed) attends ``prefix + i + 1`` tokens, where
  ``prefix = cache_seqlens[batch] - query_len[batch]``.

Causal masking only touches the few KV tiles that reach the diagonal; the long
prefix is a mask-free fast path. Sinks and sliding windows (causal or not) are
applied per tile. This is the sole Gluon extend implementation.
"""

import math

import torch

from vllm.models.inkling.amd.ops.gluon.utils import (
    _INV_LN2_VALUE,
    _LN2,
    InputStrides,
    PagedKVStrides,
    max,
    maximum,
)
from vllm.triton_utils import aggregate, gl, gluon

cdna4 = gl.amd.cdna4
async_copy = cdna4.async_copy


# ===-----------------------------------------------------------------------===#
# Query-batched path
#
# This path tiles BLOCK_M query rows (of a single q-head) into the MFMA M
# dimension -- like the prefill kernel -- so each paged KV tile is loaded once
# and reused across all BLOCK_M rows. KV still comes from the paged cache
# (decode-style async page loads). Causal masking only touches the few KV tiles
# that reach the diagonal; the long prefix is a mask-free fast path.
# ===-----------------------------------------------------------------------===#

_EXTEND_SHORT_Q_BLOCK_M = 64
_EXTEND_SHORT_Q_NUM_WARPS = 2
_EXTEND_LONG_Q_BLOCK_M = 128
_EXTEND_LONG_Q_NUM_WARPS = 4


def _select_extend_tile(max_seqlen_q: int, page_size: int) -> tuple[int, int, int]:
    """Return (BLOCK_M, BLOCK_N, NUM_WARPS) for the given max query length.

    Queries that fit in a single short tile use it (least padding, most
    occupancy); longer ones use the tall tile that covers more rows per
    shared-KV pass.
    """
    if max_seqlen_q <= _EXTEND_SHORT_Q_BLOCK_M:
        return _EXTEND_SHORT_Q_BLOCK_M, page_size, _EXTEND_SHORT_Q_NUM_WARPS
    return _EXTEND_LONG_Q_BLOCK_M, page_size, _EXTEND_LONG_Q_NUM_WARPS


@aggregate
class ExtendConfig:
    N_HEADS: gl.constexpr
    N_KV_HEADS: gl.constexpr
    GROUP_SIZE: gl.constexpr
    HEAD_DIM: gl.constexpr
    SM_SCALE: gl.constexpr
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    NUM_WARPS: gl.constexpr
    PAGE_SIZE: gl.constexpr
    PAGE_TABLE_STRIDE: gl.constexpr
    IS_CAUSAL: gl.constexpr
    HAS_LSE: gl.constexpr
    WINDOW_LEFT: gl.constexpr
    REL_EXTENT: gl.constexpr
    REL_BIAS_QK_SCALE: gl.constexpr
    IS_FP8: gl.constexpr
    q_strides: InputStrides
    rel_strides: InputStrides
    k_strides: PagedKVStrides
    v_strides: PagedKVStrides
    qk_layout: gl.constexpr
    pv_layout: gl.constexpr
    q_layout: gl.constexpr
    k_layout: gl.constexpr
    p_layout: gl.constexpr
    v_layout: gl.constexpr
    load_layout: gl.constexpr
    store_layout: gl.constexpr
    k_smem_layout: gl.constexpr
    v_smem_layout: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        N_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
        SM_SCALE,
        BLOCK_M,
        BLOCK_N,
        NUM_WARPS,
        PAGE_SIZE,
        PAGE_TABLE_STRIDE,
        IS_CAUSAL,
        HAS_LSE,
        WINDOW_LEFT,
        REL_EXTENT,
        REL_BIAS_QK_SCALE,
        IS_FP8,
        q_strides,
        rel_strides,
        k_strides,
        v_strides,
    ):
        assert HEAD_DIM in (64, 128)
        assert BLOCK_N == PAGE_SIZE

        instr_shape = [32, 32, 16]
        mfma_layout = gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=instr_shape,
            transposed=True,
            warps_per_cta=[NUM_WARPS, 1],
        )
        qk_layout = mfma_layout
        pv_layout = mfma_layout
        # Elements loaded per lane depend on the input dtype, matching qk_kw.
        # load_threads is how many lanes span HEAD_DIM.
        load_vec = 16 if IS_FP8 else 8
        load_threads = HEAD_DIM // load_vec
        load_layout = gl.BlockedLayout(
            [1, load_vec], [64 // load_threads, load_threads], [NUM_WARPS, 1], [1, 0]
        )
        # Output is always 16-bit, so a 128-bit store has 8 elements.
        # store_threads is how many lanes span HEAD_DIM.
        store_vec = 8
        store_threads = HEAD_DIM // store_vec
        store_layout = gl.BlockedLayout(
            [1, store_vec], [64 // store_threads, store_threads], [NUM_WARPS, 1], [1, 0]
        )
        # Padding interval is 64 lanes * load_vec elems.
        pad_interval = 64 * load_vec
        # Empirically tuned.
        pad_k = 16 if IS_FP8 else 8
        pad_v = 16 if IS_FP8 else 32
        k_smem_layout = gl.PaddedSharedLayout.with_identity_for(
            [[pad_interval, pad_k]], [BLOCK_N, HEAD_DIM], [1, 0]
        )
        v_smem_layout = gl.PaddedSharedLayout.with_identity_for(
            [[pad_interval, pad_v]], [BLOCK_N, HEAD_DIM], [1, 0]
        )

        self.N_HEADS = gl.constexpr(N_HEADS)
        self.N_KV_HEADS = gl.constexpr(N_KV_HEADS)
        self.GROUP_SIZE = gl.constexpr(N_HEADS // N_KV_HEADS)
        self.HEAD_DIM = gl.constexpr(HEAD_DIM)
        self.SM_SCALE = gl.constexpr(SM_SCALE)
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.NUM_WARPS = gl.constexpr(NUM_WARPS)
        self.PAGE_SIZE = gl.constexpr(PAGE_SIZE)
        self.PAGE_TABLE_STRIDE = gl.constexpr(PAGE_TABLE_STRIDE)
        self.IS_CAUSAL = gl.constexpr(IS_CAUSAL)
        self.HAS_LSE = gl.constexpr(HAS_LSE)
        self.WINDOW_LEFT = gl.constexpr(WINDOW_LEFT)
        self.REL_EXTENT = gl.constexpr(REL_EXTENT)
        self.REL_BIAS_QK_SCALE = gl.constexpr(REL_BIAS_QK_SCALE)
        self.IS_FP8 = gl.constexpr(IS_FP8)
        self.q_strides = q_strides
        self.rel_strides = rel_strides
        self.k_strides = k_strides
        self.v_strides = v_strides
        self.qk_layout = gl.constexpr(qk_layout)
        self.pv_layout = gl.constexpr(pv_layout)
        # qk_kw is derived from a 128-bit load / dtype bitwidth.
        # pv_kw is empirically tuned.
        qk_kw = 16 if IS_FP8 else 8
        pv_kw = 16 if IS_FP8 else 4
        self.q_layout = gl.constexpr(gl.DotOperandLayout(0, qk_layout, k_width=qk_kw))
        self.k_layout = gl.constexpr(gl.DotOperandLayout(1, qk_layout, k_width=qk_kw))
        self.p_layout = gl.constexpr(gl.DotOperandLayout(0, pv_layout, k_width=pv_kw))
        self.v_layout = gl.constexpr(gl.DotOperandLayout(1, pv_layout, k_width=pv_kw))
        self.load_layout = gl.constexpr(load_layout)
        self.store_layout = gl.constexpr(store_layout)
        self.k_smem_layout = gl.constexpr(k_smem_layout)
        self.v_smem_layout = gl.constexpr(v_smem_layout)


@aggregate
class ExtendProgram:
    cfg: gl.constexpr
    q_ptr: gl.tensor
    rel_logits_ptr: gl.tensor
    k_cache_ptr: gl.tensor
    v_cache_ptr: gl.tensor
    page_table_ptr: gl.tensor
    output_ptr: gl.tensor
    lse_ptr: gl.tensor
    batch: gl.tensor
    q_head: gl.tensor
    kv_head: gl.tensor
    q_start: gl.tensor
    seq_base: gl.tensor
    seq_len: gl.tensor
    prefix: gl.tensor
    cache_len: gl.tensor

    @gluon.constexpr_function
    def __init__(
        self,
        cfg,
        q_ptr,
        rel_logits_ptr,
        k_cache_ptr,
        v_cache_ptr,
        page_table_ptr,
        output_ptr,
        lse_ptr,
        batch,
        q_head,
        kv_head,
        q_start,
        seq_base,
        seq_len,
        prefix,
        cache_len,
    ):
        self.cfg = gl.constexpr(cfg)
        self.q_ptr = q_ptr
        self.rel_logits_ptr = rel_logits_ptr
        self.k_cache_ptr = k_cache_ptr
        self.v_cache_ptr = v_cache_ptr
        self.page_table_ptr = page_table_ptr
        self.output_ptr = output_ptr
        self.lse_ptr = lse_ptr
        self.batch = batch
        self.q_head = q_head
        self.kv_head = kv_head
        self.q_start = q_start
        self.seq_base = seq_base
        self.seq_len = seq_len
        self.prefix = prefix
        self.cache_len = cache_len

    @gluon.jit
    def create(
        cfg,
        q_ptr,
        rel_logits_ptr,
        k_cache_ptr,
        v_cache_ptr,
        page_table_ptr,
        output_ptr,
        lse_ptr,
        cu_seqlens_q_ptr,
        cache_seqlens_ptr,
    ):
        # Gluon treats ``cfg`` as the constexpr factory argument, while mypy
        # models the first parameter of this aggregate method as ``self``.
        block_in_req = gl.program_id(0)
        batch = gl.program_id(1)
        q_head = gl.program_id(2)
        q_start = block_in_req * cfg.BLOCK_M  # type: ignore[attr-defined]
        seq_base = gl.load(cu_seqlens_q_ptr + batch)
        seq_end = gl.load(cu_seqlens_q_ptr + batch + 1)
        seq_len = seq_end - seq_base
        cache_len = gl.load(cache_seqlens_ptr + batch)
        prefix = cache_len - seq_len
        kv_head = q_head // cfg.GROUP_SIZE  # type: ignore[attr-defined]
        return ExtendProgram(
            gl.constexpr(cfg),
            q_ptr,
            rel_logits_ptr,
            k_cache_ptr,
            v_cache_ptr,
            page_table_ptr,
            output_ptr,
            lse_ptr,
            batch,
            q_head,
            kv_head,
            q_start,
            seq_base,
            seq_len,
            prefix,
            cache_len,
        )

    @gluon.jit
    def load_q(self):
        cfg = self.cfg
        offs_m = self.q_start + gl.arange(
            0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.q_layout)
        )
        offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.q_layout))
        row = self.seq_base + offs_m
        offsets = cfg.q_strides.offsets(row[:, None], self.q_head, offs_d[None, :])
        mask = offs_m[:, None] < self.seq_len
        return cdna4.buffer_load(self.q_ptr, offsets, mask=mask, other=0.0)

    @gluon.jit
    def load_page(self, start_n):
        cfg = self.cfg
        page_index = start_n // cfg.PAGE_SIZE
        valid = start_n < self.cache_len
        return gl.load(
            self.page_table_ptr + self.batch * cfg.PAGE_TABLE_STRIDE + page_index,
            mask=valid,
            other=0,
        )

    @gluon.jit
    def issue_load_k(self, physical_page, k_smem):
        cfg = self.cfg
        offs_n = gl.arange(0, cfg.BLOCK_N, layout=gl.SliceLayout(1, cfg.load_layout))
        offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.load_layout))
        offsets = cfg.k_strides.offsets(
            physical_page,
            offs_n[:, None],
            self.kv_head,
            offs_d[None, :],
        )
        async_copy.global_load_to_shared(k_smem, self.k_cache_ptr + offsets)
        async_copy.commit_group()

    @gluon.jit
    def issue_load_v(self, physical_page, v_smem):
        cfg = self.cfg
        offs_n = gl.arange(0, cfg.BLOCK_N, layout=gl.SliceLayout(1, cfg.load_layout))
        offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.load_layout))
        offsets = cfg.v_strides.offsets(
            physical_page,
            offs_n[:, None],
            self.kv_head,
            offs_d[None, :],
        )
        async_copy.global_load_to_shared(v_smem, self.v_cache_ptr + offsets)
        async_copy.commit_group()

    @gluon.jit
    def shared_load_k(self, k_smem):
        cfg = self.cfg
        k_buffer = k_smem.permute([1, 0])
        return k_buffer.load(cfg.k_layout)

    @gluon.jit
    def shared_load_v(self, v_smem):
        cfg = self.cfg
        return v_smem.load(cfg.v_layout)

    @gluon.jit
    def compute_qk(self, q, k):
        cfg = self.cfg
        qk = gl.zeros(
            [cfg.BLOCK_M, cfg.BLOCK_N], dtype=gl.float32, layout=cfg.qk_layout
        )
        return cdna4.mfma(q, k, qk)

    @gluon.jit
    def apply_rel_bias(self, qk, start_n):
        cfg = self.cfg
        offs_m = self.q_start + gl.arange(
            0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.qk_layout)
        )
        offs_n_abs = start_n + gl.arange(
            0, cfg.BLOCK_N, layout=gl.SliceLayout(0, cfg.qk_layout)
        )
        q_pos = self.prefix + offs_m
        rel_dist = q_pos[:, None] - offs_n_abs[None, :]
        rel_valid = (rel_dist >= 0) & (rel_dist < cfg.REL_EXTENT)
        rel_idx = gl.where(rel_dist >= 0, rel_dist, 0)
        rel_idx = gl.where(rel_idx < cfg.REL_EXTENT, rel_idx, cfg.REL_EXTENT - 1)
        offsets = cfg.rel_strides.offsets(
            self.seq_base + offs_m[:, None], self.q_head, rel_idx
        )
        mask = (offs_m[:, None] < self.seq_len) & rel_valid
        rel_bias = cdna4.buffer_load(
            self.rel_logits_ptr, offsets, mask=mask, other=0.0
        ).to(gl.float32)
        return qk + rel_bias * cfg.REL_BIAS_QK_SCALE

    @gluon.jit
    def compute_pv(self, p, v, acc):
        return cdna4.mfma(p, v, acc)

    @gluon.jit
    def init_attention_state(self):
        cfg = self.cfg
        m_i = gl.full(
            [cfg.BLOCK_M],
            value=-float("inf"),
            dtype=gl.float32,
            layout=gl.SliceLayout(1, cfg.pv_layout),
        )
        l_i = gl.full(
            [cfg.BLOCK_M],
            value=0,
            dtype=gl.float32,
            layout=gl.SliceLayout(1, cfg.pv_layout),
        )
        acc = gl.zeros(
            [cfg.BLOCK_M, cfg.HEAD_DIM], dtype=gl.float32, layout=cfg.pv_layout
        )
        return m_i, l_i, acc

    @gluon.jit
    def softmax(self, qk, m_i, l_i, acc):
        cfg = self.cfg
        # In sliding window case, some rows can see fully masked tiles before
        # any valid KV. Guard the online softmax state so `-inf - -inf` does not
        # produce NaNs.
        HAS_INVALID: gl.constexpr = cfg.WINDOW_LEFT >= 0

        row_max = max(qk, 1)
        m_new = maximum(m_i, row_max)
        m_new_scaled = m_new * cfg.SM_SCALE
        if HAS_INVALID:
            invalid = m_new == -float("inf")
            m_new_scaled = gl.where(invalid, 0.0, m_new_scaled)

        qk_shifted = qk * cfg.SM_SCALE - m_new_scaled[:, None]
        p = gl.exp2(qk_shifted)
        m_diff = m_i * cfg.SM_SCALE - m_new_scaled
        if HAS_INVALID:
            m_diff = gl.where(invalid, 0.0, m_diff)

        alpha = gl.exp2(m_diff)
        l_ij = gl.sum(p, axis=1)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        p = p.to(self.q_ptr.dtype.element_ty)
        p = gl.convert_layout(p, cfg.p_layout)
        return p, m_new, l_i, acc

    @gluon.jit
    def store_output(self, output):
        cfg = self.cfg
        offs_m = self.q_start + gl.arange(
            0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.store_layout)
        )
        offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.store_layout))
        offsets = (
            ((self.seq_base + offs_m[:, None]) * cfg.N_HEADS + self.q_head)
            * cfg.HEAD_DIM
            + offs_d[None, :]
        ).to(gl.int32)
        mask = offs_m[:, None] < self.seq_len
        output = output.to(self.output_ptr.dtype.element_ty)
        cdna4.buffer_store(output, self.output_ptr, offsets, mask=mask)

    @gluon.jit
    def store_lse(self, l_i, m_i):
        cfg = self.cfg
        if cfg.HAS_LSE:
            offs_m = self.q_start + gl.arange(
                0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.pv_layout)
            )
            offsets = ((self.seq_base + offs_m) * cfg.N_HEADS + self.q_head).to(
                gl.int32
            )
            mask = offs_m < self.seq_len
            lse_l_i = gl.where(l_i > 0.0, l_i, 1.0)
            # Softmax runs in base-2 (exp2 hardware fast path), so m_i*SM_SCALE +
            # log2(l_i) is the LSE in base-2 units. Convert to natural log (the
            # public op contract / torch.logsumexp convention) by scaling by ln2.
            lse = (m_i * cfg.SM_SCALE + gl.log2(lse_l_i)) * _LN2
            cdna4.buffer_store(lse, self.lse_ptr, offsets, mask=mask)


@gluon.jit
def _rel_mha_extend_fp16(
    q_ptr,
    rel_logits_ptr,
    k_cache_ptr,
    v_cache_ptr,
    page_table_ptr,
    output_ptr,
    lse_ptr,
    cu_seqlens_q_ptr,
    cache_seqlens_ptr,
    Q_STRIDE_T: gl.constexpr,
    Q_STRIDE_H: gl.constexpr,
    Q_STRIDE_D: gl.constexpr,
    K_STRIDE_B: gl.constexpr,
    K_STRIDE_P: gl.constexpr,
    K_STRIDE_H: gl.constexpr,
    K_STRIDE_D: gl.constexpr,
    V_STRIDE_B: gl.constexpr,
    V_STRIDE_P: gl.constexpr,
    V_STRIDE_H: gl.constexpr,
    V_STRIDE_D: gl.constexpr,
    SM_SCALE: gl.constexpr,
    PAGE_TABLE_STRIDE: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    N_HEADS: gl.constexpr,
    N_KV_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    HAS_LSE: gl.constexpr,
    WINDOW_LEFT: gl.constexpr,
    REL_STRIDE_T: gl.constexpr,
    REL_STRIDE_H: gl.constexpr,
    REL_STRIDE_E: gl.constexpr,
    REL_EXTENT: gl.constexpr,
    REL_BIAS_QK_SCALE: gl.constexpr,
    IS_FP8: gl.constexpr,
):
    cfg = ExtendConfig(
        N_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
        SM_SCALE,
        BLOCK_M,
        BLOCK_N,
        NUM_WARPS,
        PAGE_SIZE,
        PAGE_TABLE_STRIDE,
        IS_CAUSAL,
        HAS_LSE,
        WINDOW_LEFT,
        REL_EXTENT,
        REL_BIAS_QK_SCALE,
        IS_FP8,
        InputStrides(Q_STRIDE_T, Q_STRIDE_H, Q_STRIDE_D),
        InputStrides(REL_STRIDE_T, REL_STRIDE_H, REL_STRIDE_E),
        PagedKVStrides(K_STRIDE_B, K_STRIDE_P, K_STRIDE_H, K_STRIDE_D),
        PagedKVStrides(V_STRIDE_B, V_STRIDE_P, V_STRIDE_H, V_STRIDE_D),
    )
    program = ExtendProgram.create(
        cfg,
        q_ptr,
        rel_logits_ptr,
        k_cache_ptr,
        v_cache_ptr,
        page_table_ptr,
        output_ptr,
        lse_ptr,
        cu_seqlens_q_ptr,
        cache_seqlens_ptr,
    )
    # Over-provisioned tile past this request's real query rows: nothing to do.
    if program.q_start >= program.seq_len:
        return
    k_smem = gl.allocate_shared_memory(
        k_cache_ptr.dtype.element_ty, [cfg.BLOCK_N, cfg.HEAD_DIM], cfg.k_smem_layout
    )
    v_smem = gl.allocate_shared_memory(
        v_cache_ptr.dtype.element_ty, [cfg.BLOCK_N, cfg.HEAD_DIM], cfg.v_smem_layout
    )

    q = program.load_q()
    m_i, l_i, acc = program.init_attention_state()

    offs_m_q = program.q_start + gl.arange(
        0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.qk_layout)
    )
    offs_n_q = gl.arange(0, cfg.BLOCK_N, layout=gl.SliceLayout(0, cfg.qk_layout))
    diag_row = program.prefix + offs_m_q

    if IS_CAUSAL:
        kv_end = min(program.cache_len, program.prefix + program.q_start + cfg.BLOCK_M)
    else:
        kv_end = program.cache_len

    # Sliding window (inclusive-left, matches flash-attn window_size=(W, 0)):
    # skip KV tiles entirely below the window's lower edge. The tile's top query
    # row sits at absolute position pos_top; its window opens at
    # pos_top - WINDOW_LEFT (that key is visible -> W + 1 keys total). The min()
    # form clamps to 0 without the shadowed builtin max().
    if cfg.WINDOW_LEFT >= 0:
        pos_top = program.prefix + program.q_start
        kv_start = pos_top - min(pos_top, cfg.WINDOW_LEFT)
        kv_start = (kv_start // cfg.BLOCK_N) * cfg.BLOCK_N
    else:
        kv_start = 0

    for start_n in range(kv_start, kv_end, cfg.BLOCK_N):
        physical_page = program.load_page(start_n)
        program.issue_load_k(physical_page, k_smem)
        program.issue_load_v(physical_page, v_smem)

        async_copy.wait_group(1)
        k = program.shared_load_k(k_smem)
        qk = program.compute_qk(q, k)
        qk = program.apply_rel_bias(qk, start_n)

        if cfg.WINDOW_LEFT >= 0:
            # Window lower edge + cache bound always apply; causal upper edge only
            # when IS_CAUSAL (independent layering keeps non-causal + window correct).
            offs_n_abs = start_n + offs_n_q
            mask = (offs_n_abs[None, :] >= diag_row[:, None] - cfg.WINDOW_LEFT) & (
                offs_n_abs[None, :] < program.cache_len
            )
            if IS_CAUSAL:
                mask &= offs_n_abs[None, :] <= diag_row[:, None]
            qk = gl.where(mask, qk, -float("inf"))
        elif IS_CAUSAL:
            if start_n + cfg.BLOCK_N > program.prefix + program.q_start:
                offs_n_abs = start_n + offs_n_q
                mask = (offs_n_abs[None, :] <= diag_row[:, None]) & (
                    offs_n_abs[None, :] < program.cache_len
                )
                qk = gl.where(mask, qk, -float("inf"))
        else:
            if start_n + cfg.BLOCK_N > program.cache_len:
                offs_n_abs = start_n + offs_n_q
                qk = gl.where(
                    offs_n_abs[None, :] < program.cache_len, qk, -float("inf")
                )

        p, m_i, l_i, acc = program.softmax(qk, m_i, l_i, acc)

        async_copy.wait_group(0)
        v = program.shared_load_v(v_smem)
        acc = program.compute_pv(p, v, acc)

    denom = gl.where(l_i > 0.0, l_i, 1.0)
    output = acc * (1.0 / denom)[:, None]
    output = gl.convert_layout(output, cfg.store_layout)
    program.store_output(output)
    program.store_lse(l_i, m_i)


def gluon_rel_mha_extend_gfx950(
    q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    window_left: int = -1,
    return_lse: bool = False,
    max_seqlen_q: int = 1,
    max_seqlen_k: int = 1,
    rel_logits: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    assert rel_logits is not None
    head_dim = q.shape[2]
    n_heads = q.shape[1]
    n_kv_heads = k_cache.shape[2]
    page_size = k_cache.shape[1]
    block_m, block_n, num_warps = _select_extend_tile(max_seqlen_q, page_size)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    sm_scale = softmax_scale * _INV_LN2_VALUE
    rel_bias_qk_scale = 1.0 / softmax_scale

    # max_seqlen_q must be >= the true max query length; extra tiles early-exit.
    batch = cu_seqlens_q.shape[0] - 1
    safe_max_q = max_seqlen_q if max_seqlen_q > 0 else 1
    blocks_per_req = (safe_max_q + block_m - 1) // block_m
    cu_q_i32 = cu_seqlens_q.to(torch.int32).contiguous()
    cache_i32 = cache_seqlens.to(torch.int32).contiguous()

    is_fp8 = q.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    out_dtype = torch.bfloat16 if is_fp8 else q.dtype
    if page_size not in (64, 128, 256):
        raise ValueError(
            "gfx950 Gluon relative-attention extend requires page size "
            f"64, 128, or 256; got {page_size}"
        )
    output = (
        torch.empty(q.shape, device=q.device, dtype=out_dtype) if out is None else out
    )
    if return_lse:
        lse = torch.empty((q.shape[0], n_heads), device=q.device, dtype=torch.float32)
        lse_arg = lse
    else:
        lse = None
        lse_arg = q
    grid = (blocks_per_req, batch, n_heads)
    _rel_mha_extend_fp16[grid](
        q,
        rel_logits,
        k_cache,
        v_cache,
        page_table,
        output,
        lse_arg,
        cu_q_i32,
        cache_i32,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        sm_scale,
        page_table.stride(0),
        page_size,
        n_heads,
        n_kv_heads,
        head_dim,
        block_m,
        block_n,
        num_warps,
        True,
        return_lse,
        window_left,
        rel_logits.stride(0),
        rel_logits.stride(1),
        rel_logits.stride(2),
        rel_logits.shape[2],
        rel_bias_qk_scale,
        is_fp8,
        num_warps=num_warps,
    )
    if return_lse:
        assert lse is not None
        return output, lse
    return output
