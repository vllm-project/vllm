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

"""rel_mha decode Gluon kernel for AMD GFX950."""

import math
from typing import NamedTuple

import torch

from vllm.models.inkling.amd.ops.gluon.utils import (
    _INV_LN2_VALUE,
    InputStrides,
    PagedKVStrides,
    max,
    maximum,
)
from vllm.triton_utils import aggregate, gl, gluon

cdna4 = gl.amd.cdna4
async_copy = cdna4.async_copy
cdiv = gl.cdiv
_GFX950_SM_COUNT = 256


# ===-----------------------------------------------------------------------===#
# Kernel Config
# ===-----------------------------------------------------------------------===#


@aggregate
class AttentionConfig:
    SM_SCALE: gl.constexpr
    PAGE_TABLE_STRIDE: gl.constexpr
    PAGE_SIZE: gl.constexpr
    NUM_KV_SPLITS: gl.constexpr
    MAX_SEQLEN_Q: gl.constexpr
    NUM_Q_HEADS: gl.constexpr
    NUM_KV_HEADS: gl.constexpr
    HEAD_DIM: gl.constexpr
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    IS_SLIDING: gl.constexpr
    WINDOW_LEFT: gl.constexpr
    REL_EXTENT: gl.constexpr
    REL_BIAS_QK_SCALE: gl.constexpr
    IS_FP8: gl.constexpr
    GROUP_SIZE: gl.constexpr
    NUM_GROUPS: gl.constexpr
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
    reduce_layout: gl.constexpr
    k_smem_layout: gl.constexpr
    v_smem_layout: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        SM_SCALE,
        PAGE_TABLE_STRIDE,
        PAGE_SIZE,
        NUM_KV_SPLITS,
        MAX_SEQLEN_Q,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        BLOCK_M,
        BLOCK_N,
        IS_SLIDING,
        WINDOW_LEFT,
        REL_EXTENT,
        REL_BIAS_QK_SCALE,
        IS_FP8,
        q_strides,
        rel_strides,
        k_strides,
        v_strides,
    ):
        assert NUM_Q_HEADS % NUM_KV_HEADS == 0
        assert HEAD_DIM in (64, 128)
        assert BLOCK_N == PAGE_SIZE
        if IS_SLIDING:
            assert WINDOW_LEFT >= 0
        else:
            assert WINDOW_LEFT == -1

        mfma_layout = gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[16, 16, 32],
            transposed=True,
            warps_per_cta=[1, 1],
        )
        qk_layout = mfma_layout
        pv_layout = mfma_layout
        # qk_kw is derived from a 128-bit load / dtype bitwidth.
        # pv_kw is empirically tuned.
        qk_kw = 16 if IS_FP8 else 8
        pv_kw = 8 if IS_FP8 else 4
        q_layout = gl.DotOperandLayout(0, qk_layout, k_width=qk_kw)
        k_layout = gl.DotOperandLayout(1, qk_layout, k_width=qk_kw)
        p_layout = gl.DotOperandLayout(0, pv_layout, k_width=pv_kw)
        v_layout = gl.DotOperandLayout(1, pv_layout, k_width=pv_kw)
        # Elements loaded per lane depend on the input dtype, matching qk_kw.
        # load_threads is how many lanes span HEAD_DIM.
        load_vec = 16 if IS_FP8 else 8
        load_threads = HEAD_DIM // load_vec
        load_layout = gl.BlockedLayout(
            [1, load_vec], [64 // load_threads, load_threads], [1, 1], [1, 0]
        )
        # Output is always 16-bit, so a 128-bit store has 8 elements.
        # store_threads is how many lanes span HEAD_DIM.
        store_vec = 8
        store_threads = HEAD_DIM // store_vec
        store_layout = gl.BlockedLayout(
            [1, store_vec], [64 // store_threads, store_threads], [1, 1], [1, 0]
        )
        reduce_layout = gl.BlockedLayout([1, HEAD_DIM // 64], [1, 64], [1, 1], [1, 0])
        # Padding interval is 64 lanes * load_vec elems.
        pad_interval = 64 * load_vec
        # Empirically tuned.
        pad_k = 16 if IS_FP8 else 8
        pad_v = 32
        k_smem_layout = gl.PaddedSharedLayout.with_identity_for(
            [[pad_interval, pad_k]], [BLOCK_N, HEAD_DIM], [1, 0]
        )
        v_smem_layout = gl.PaddedSharedLayout.with_identity_for(
            [[pad_interval, pad_v]], [BLOCK_N, HEAD_DIM], [1, 0]
        )

        self.SM_SCALE = gl.constexpr(SM_SCALE)
        self.PAGE_TABLE_STRIDE = gl.constexpr(PAGE_TABLE_STRIDE)
        self.PAGE_SIZE = gl.constexpr(PAGE_SIZE)
        self.NUM_KV_SPLITS = gl.constexpr(NUM_KV_SPLITS)
        self.MAX_SEQLEN_Q = gl.constexpr(MAX_SEQLEN_Q)
        self.NUM_Q_HEADS = gl.constexpr(NUM_Q_HEADS)
        self.NUM_KV_HEADS = gl.constexpr(NUM_KV_HEADS)
        self.HEAD_DIM = gl.constexpr(HEAD_DIM)
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.IS_SLIDING = gl.constexpr(IS_SLIDING)
        self.WINDOW_LEFT = gl.constexpr(WINDOW_LEFT)
        self.REL_EXTENT = gl.constexpr(REL_EXTENT)
        self.REL_BIAS_QK_SCALE = gl.constexpr(REL_BIAS_QK_SCALE)
        self.IS_FP8 = gl.constexpr(IS_FP8)
        self.GROUP_SIZE = gl.constexpr(NUM_Q_HEADS // NUM_KV_HEADS)
        self.NUM_GROUPS = gl.constexpr((self.GROUP_SIZE + BLOCK_M - 1) // BLOCK_M)
        self.q_strides = q_strides
        self.rel_strides = rel_strides
        self.k_strides = k_strides
        self.v_strides = v_strides
        self.qk_layout = gl.constexpr(qk_layout)
        self.pv_layout = gl.constexpr(pv_layout)
        self.q_layout = gl.constexpr(q_layout)
        self.k_layout = gl.constexpr(k_layout)
        self.p_layout = gl.constexpr(p_layout)
        self.v_layout = gl.constexpr(v_layout)
        self.load_layout = gl.constexpr(load_layout)
        self.store_layout = gl.constexpr(store_layout)
        self.reduce_layout = gl.constexpr(reduce_layout)
        self.k_smem_layout = gl.constexpr(k_smem_layout)
        self.v_smem_layout = gl.constexpr(v_smem_layout)


# ===-----------------------------------------------------------------------===#
# Kernel Program
# ===-----------------------------------------------------------------------===#


@aggregate
class AttentionProgram:
    cfg: gl.constexpr
    q_ptr: gl.tensor
    rel_logits_ptr: gl.tensor
    k_cache_ptr: gl.tensor
    v_cache_ptr: gl.tensor
    page_table_ptr: gl.tensor
    cache_seqlens_ptr: gl.tensor
    mid_o_ptr: gl.tensor
    mid_lse_ptr: gl.tensor
    q_index: gl.tensor
    batch: gl.tensor
    kv_head: gl.tensor
    group_start: gl.tensor
    split_id: gl.tensor
    cache_len: gl.tensor
    kv_start: gl.tensor
    split_start: gl.tensor
    split_end: gl.tensor

    @gluon.constexpr_function
    def __init__(
        self,
        cfg,
        q_ptr,
        rel_logits_ptr,
        k_cache_ptr,
        v_cache_ptr,
        page_table_ptr,
        cache_seqlens_ptr,
        mid_o_ptr,
        mid_lse_ptr,
        q_index,
        batch,
        kv_head,
        group_start,
        split_id,
        cache_len,
        kv_start,
        split_start,
        split_end,
    ):
        self.cfg = gl.constexpr(cfg)
        self.q_ptr = q_ptr
        self.rel_logits_ptr = rel_logits_ptr
        self.k_cache_ptr = k_cache_ptr
        self.v_cache_ptr = v_cache_ptr
        self.page_table_ptr = page_table_ptr
        self.cache_seqlens_ptr = cache_seqlens_ptr
        self.mid_o_ptr = mid_o_ptr
        self.mid_lse_ptr = mid_lse_ptr
        self.q_index = q_index
        self.batch = batch
        self.kv_head = kv_head
        self.group_start = group_start
        self.split_id = split_id
        self.cache_len = cache_len
        self.kv_start = kv_start
        self.split_start = split_start
        self.split_end = split_end

    @gluon.jit
    def create(
        cfg,
        q_ptr,
        rel_logits_ptr,
        k_cache_ptr,
        v_cache_ptr,
        page_table_ptr,
        cache_seqlens_ptr,
        mid_o_ptr,
        mid_lse_ptr,
    ):
        # Gluon treats ``cfg`` as the constexpr factory argument, while mypy
        # models the first parameter of this aggregate method as ``self``.
        q_index = gl.program_id(0)
        batch = q_index // cfg.MAX_SEQLEN_Q  # type: ignore[attr-defined]
        q_pos = q_index - batch * cfg.MAX_SEQLEN_Q  # type: ignore[attr-defined]
        head_block = gl.program_id(1)
        kv_head = head_block // cfg.NUM_GROUPS  # type: ignore[attr-defined]
        group_block = head_block - kv_head * cfg.NUM_GROUPS  # type: ignore[attr-defined]
        group_start = group_block * cfg.BLOCK_M  # type: ignore[attr-defined]
        split_id = gl.program_id(2)
        cache_len = gl.load(cache_seqlens_ptr + batch)
        cache_len = cache_len - (
            cfg.MAX_SEQLEN_Q - 1 - q_pos  # type: ignore[attr-defined]
        )
        cache_len = maximum(cache_len, 0)
        if cfg.IS_SLIDING:  # type: ignore[attr-defined]
            # WINDOW_LEFT is defined as exclusive (keys strictly to the left, not
            # counting the current token), so the window is WINDOW_LEFT + 1 keys
            # once the current token is included. E.g. with
            # WINDOW_LEFT = 127 and cache_len = 500 (current token at index 499),
            # kv_start = 500 - (127 + 1) = 372 keeps keys [372, 499] = 128 keys.
            # Without the + 1, kv_start = 373 would drop the leftmost key.
            kv_start = cache_len - min(
                cache_len,
                cfg.WINDOW_LEFT + 1,  # type: ignore[attr-defined]
            )
        else:
            kv_start = cache_len - cache_len
        first_page = kv_start // cfg.PAGE_SIZE  # type: ignore[attr-defined]
        end_page = cdiv(cache_len, cfg.PAGE_SIZE)  # type: ignore[attr-defined]
        num_pages = end_page - first_page
        pages_per_split = cdiv(
            num_pages,
            cfg.NUM_KV_SPLITS,  # type: ignore[attr-defined]
        )
        split_start_page = first_page + split_id * pages_per_split
        split_end_page = min(split_start_page + pages_per_split, end_page)
        split_start = (
            split_start_page * cfg.PAGE_SIZE  # type: ignore[attr-defined]
        )
        split_end = min(
            split_end_page * cfg.PAGE_SIZE,  # type: ignore[attr-defined]
            cache_len,
        )
        return AttentionProgram(
            gl.constexpr(cfg),
            q_ptr,
            rel_logits_ptr,
            k_cache_ptr,
            v_cache_ptr,
            page_table_ptr,
            cache_seqlens_ptr,
            mid_o_ptr,
            mid_lse_ptr,
            q_index,
            batch,
            kv_head,
            group_start,
            split_id,
            cache_len,
            kv_start,
            split_start,
            split_end,
        )

    @gluon.jit
    def load_q(self):
        cfg = self.cfg
        offs_m = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.q_layout))
        offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.q_layout))
        q_heads = self.kv_head * cfg.GROUP_SIZE + self.group_start + offs_m
        valid = (self.group_start + offs_m) < cfg.GROUP_SIZE
        offsets = cfg.q_strides.offsets(self.q_index, q_heads[:, None], offs_d[None, :])
        return cdna4.buffer_load(self.q_ptr, offsets, mask=valid[:, None], other=0.0)

    @gluon.jit
    def init_state(self):
        cfg = self.cfg
        m_i = gl.full(
            [cfg.BLOCK_M],
            value=-float("inf"),
            dtype=gl.float32,
            layout=gl.SliceLayout(1, cfg.pv_layout),
        )
        l_i = gl.full(
            [cfg.BLOCK_M],
            value=0.0,
            dtype=gl.float32,
            layout=gl.SliceLayout(1, cfg.pv_layout),
        )
        acc = gl.zeros(
            [cfg.BLOCK_M, cfg.HEAD_DIM], dtype=gl.float32, layout=cfg.pv_layout
        )
        return m_i, l_i, acc

    @gluon.jit
    def load_page(self, start_n):
        cfg = self.cfg
        page_index = start_n // cfg.PAGE_SIZE
        valid = start_n < self.split_end
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
        # can't use buffer_load: paged KV offsets may exceed its 32-bit range.
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
        # can't use buffer_load: paged KV offsets may exceed its 32-bit range.
        async_copy.global_load_to_shared(v_smem, self.v_cache_ptr + offsets)
        async_copy.commit_group()

    @gluon.jit
    def shared_load_k(self, k_smem):
        return k_smem.permute([1, 0]).load(self.cfg.k_layout)

    @gluon.jit
    def shared_load_v(self, v_smem):
        return v_smem.load(self.cfg.v_layout)

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
        offs_m = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.qk_layout))
        offs_n = start_n + gl.arange(
            0, cfg.BLOCK_N, layout=gl.SliceLayout(0, cfg.qk_layout)
        )
        q_heads = self.kv_head * cfg.GROUP_SIZE + self.group_start + offs_m
        rel_dist = (self.cache_len - 1) - offs_n
        rel_valid = (rel_dist >= 0) & (rel_dist < cfg.REL_EXTENT)
        rel_idx = gl.where(rel_dist >= 0, rel_dist, 0)
        rel_idx = gl.where(rel_idx < cfg.REL_EXTENT, rel_idx, cfg.REL_EXTENT - 1)
        offsets = cfg.rel_strides.offsets(
            self.q_index, q_heads[:, None], rel_idx[None, :]
        )
        head_valid = (self.group_start + offs_m) < cfg.GROUP_SIZE
        rel_bias = cdna4.buffer_load(
            self.rel_logits_ptr,
            offsets,
            mask=head_valid[:, None] & rel_valid[None, :],
            other=0.0,
        ).to(gl.float32)
        return qk + rel_bias * cfg.REL_BIAS_QK_SCALE

    @gluon.jit
    def apply_kv_mask(self, qk, start_n):
        cfg = self.cfg
        offs_n = gl.arange(0, cfg.BLOCK_N, layout=gl.SliceLayout(0, cfg.qk_layout))
        tokens = start_n + offs_n[None, :]
        mask = (tokens >= self.kv_start) & (tokens < self.split_end)
        return gl.where(mask, qk, -float("inf"))

    @gluon.jit
    def softmax(self, qk, m_i, l_i, acc):
        cfg = self.cfg
        row_max = max(qk, axis=1)
        row_max = gl.convert_layout(row_max, gl.SliceLayout(1, cfg.pv_layout))
        m_new = maximum(m_i, row_max)
        m_new_scaled = m_new * cfg.SM_SCALE
        qk_shifted = qk * cfg.SM_SCALE - m_new_scaled[:, None]
        p = gl.exp2(qk_shifted)
        m_diff = m_i * cfg.SM_SCALE - m_new_scaled
        alpha = gl.exp2(m_diff)
        l_ij = gl.sum(p, axis=1)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        p = p.to(self.q_ptr.dtype.element_ty)
        p = gl.convert_layout(p, cfg.p_layout)
        return p, m_new, l_i, acc

    @gluon.jit
    def compute_pv(self, p, v, acc):
        return cdna4.mfma(p, v, acc)

    @gluon.jit
    def store_split(self, acc, l_i, m_i):
        cfg = self.cfg
        offs_m = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.store_layout))
        offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.store_layout))
        q_heads = self.kv_head * cfg.GROUP_SIZE + self.group_start + offs_m
        valid = ((self.group_start + offs_m) < cfg.GROUP_SIZE) & (
            self.split_start < self.split_end
        )
        acc = gl.convert_layout(acc, cfg.store_layout)
        l_i = gl.convert_layout(l_i, gl.SliceLayout(1, cfg.store_layout))
        m_i = gl.convert_layout(m_i, gl.SliceLayout(1, cfg.store_layout))
        recip_l_i = 1.0 / l_i
        part_o = acc * recip_l_i[:, None]
        part_lse = m_i * cfg.SM_SCALE + gl.log2(l_i)
        mid_o_offsets = (
            (self.q_index * cfg.NUM_Q_HEADS + q_heads[:, None]) * cfg.NUM_KV_SPLITS
            + self.split_id
        ) * cfg.HEAD_DIM + offs_d[None, :]
        mid_lse_offsets = (
            self.q_index * cfg.NUM_Q_HEADS + q_heads
        ) * cfg.NUM_KV_SPLITS + self.split_id
        cdna4.buffer_store(part_o, self.mid_o_ptr, mid_o_offsets, mask=valid[:, None])
        cdna4.buffer_store(part_lse, self.mid_lse_ptr, mid_lse_offsets, mask=valid)

    @gluon.jit
    def store_output(self, acc, l_i):
        cfg = self.cfg
        offs_m = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.store_layout))
        offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.store_layout))
        q_heads = self.kv_head * cfg.GROUP_SIZE + self.group_start + offs_m
        valid = (self.group_start + offs_m) < cfg.GROUP_SIZE
        acc = gl.convert_layout(acc, cfg.store_layout)
        l_i = gl.convert_layout(l_i, gl.SliceLayout(1, cfg.store_layout))
        output = acc * (1.0 / l_i)[:, None]
        output = output.to(self.mid_o_ptr.dtype.element_ty)
        offsets = (self.q_index * cfg.NUM_Q_HEADS + q_heads[:, None]) * cfg.HEAD_DIM
        offsets += offs_d[None, :]
        cdna4.buffer_store(output, self.mid_o_ptr, offsets, mask=valid[:, None])


# ===-----------------------------------------------------------------------===#
# Entry Point
# ===-----------------------------------------------------------------------===#


@gluon.jit
def _rel_mha_decode_fp16(
    q_ptr,
    rel_logits_ptr,
    k_cache_ptr,
    v_cache_ptr,
    page_table_ptr,
    cache_seqlens_ptr,
    mid_o_ptr,
    mid_lse_ptr,
    Q_STRIDE_B: gl.constexpr,
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
    NUM_KV_SPLITS: gl.constexpr,
    MAX_SEQLEN_Q: gl.constexpr,
    NUM_Q_HEADS: gl.constexpr,
    NUM_KV_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    IS_SLIDING: gl.constexpr,
    WINDOW_LEFT: gl.constexpr,
    REL_STRIDE_T: gl.constexpr,
    REL_STRIDE_H: gl.constexpr,
    REL_STRIDE_E: gl.constexpr,
    REL_EXTENT: gl.constexpr,
    REL_BIAS_QK_SCALE: gl.constexpr,
    IS_FP8: gl.constexpr,
):
    cfg = AttentionConfig(
        SM_SCALE,
        PAGE_TABLE_STRIDE,
        PAGE_SIZE,
        NUM_KV_SPLITS,
        MAX_SEQLEN_Q,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        BLOCK_M,
        BLOCK_N,
        IS_SLIDING,
        WINDOW_LEFT,
        REL_EXTENT,
        REL_BIAS_QK_SCALE,
        IS_FP8,
        InputStrides(Q_STRIDE_B, Q_STRIDE_H, Q_STRIDE_D),
        InputStrides(REL_STRIDE_T, REL_STRIDE_H, REL_STRIDE_E),
        PagedKVStrides(K_STRIDE_B, K_STRIDE_P, K_STRIDE_H, K_STRIDE_D),
        PagedKVStrides(V_STRIDE_B, V_STRIDE_P, V_STRIDE_H, V_STRIDE_D),
    )
    program = AttentionProgram.create(
        cfg,
        q_ptr,
        rel_logits_ptr,
        k_cache_ptr,
        v_cache_ptr,
        page_table_ptr,
        cache_seqlens_ptr,
        mid_o_ptr,
        mid_lse_ptr,
    )
    k_smem = gl.allocate_shared_memory(
        k_cache_ptr.dtype.element_ty, [cfg.BLOCK_N, cfg.HEAD_DIM], cfg.k_smem_layout
    )
    v_smem = gl.allocate_shared_memory(
        v_cache_ptr.dtype.element_ty, [cfg.BLOCK_N, cfg.HEAD_DIM], cfg.v_smem_layout
    )

    q = program.load_q()
    m_i, l_i, acc = program.init_state()

    physical_page = program.load_page(program.split_start)

    for start_n in range(program.split_start, program.split_end, cfg.BLOCK_N):
        program.issue_load_k(physical_page, k_smem)
        program.issue_load_v(physical_page, v_smem)
        physical_page = program.load_page(start_n + cfg.BLOCK_N)

        async_copy.wait_group(1)
        k = program.shared_load_k(k_smem)
        qk = program.compute_qk(q, k)
        qk = program.apply_rel_bias(qk, start_n)
        qk = program.apply_kv_mask(qk, start_n)
        p, m_i, l_i, acc = program.softmax(qk, m_i, l_i, acc)

        async_copy.wait_group(0)
        v = program.shared_load_v(v_smem)
        acc = program.compute_pv(p, v, acc)

    program.store_split(acc, l_i, m_i)


@gluon.jit
def _rel_mha_decode_sliding_fp16(
    q_ptr,
    rel_logits_ptr,
    k_cache_ptr,
    v_cache_ptr,
    page_table_ptr,
    cache_seqlens_ptr,
    out_ptr,
    Q_STRIDE_B: gl.constexpr,
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
    MAX_SEQLEN_Q: gl.constexpr,
    NUM_Q_HEADS: gl.constexpr,
    NUM_KV_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    IS_SLIDING: gl.constexpr,
    WINDOW_LEFT: gl.constexpr,
    REL_STRIDE_T: gl.constexpr,
    REL_STRIDE_H: gl.constexpr,
    REL_STRIDE_E: gl.constexpr,
    REL_EXTENT: gl.constexpr,
    REL_BIAS_QK_SCALE: gl.constexpr,
    IS_FP8: gl.constexpr,
):
    cfg = AttentionConfig(
        SM_SCALE,
        PAGE_TABLE_STRIDE,
        PAGE_SIZE,
        1,
        MAX_SEQLEN_Q,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        BLOCK_M,
        BLOCK_N,
        IS_SLIDING,
        WINDOW_LEFT,
        REL_EXTENT,
        REL_BIAS_QK_SCALE,
        IS_FP8,
        InputStrides(Q_STRIDE_B, Q_STRIDE_H, Q_STRIDE_D),
        InputStrides(REL_STRIDE_T, REL_STRIDE_H, REL_STRIDE_E),
        PagedKVStrides(K_STRIDE_B, K_STRIDE_P, K_STRIDE_H, K_STRIDE_D),
        PagedKVStrides(V_STRIDE_B, V_STRIDE_P, V_STRIDE_H, V_STRIDE_D),
    )
    program = AttentionProgram.create(
        cfg,
        q_ptr,
        rel_logits_ptr,
        k_cache_ptr,
        v_cache_ptr,
        page_table_ptr,
        cache_seqlens_ptr,
        out_ptr,
        out_ptr,
    )
    k_smem = gl.allocate_shared_memory(
        k_cache_ptr.dtype.element_ty, [cfg.BLOCK_N, cfg.HEAD_DIM], cfg.k_smem_layout
    )
    v_smem = gl.allocate_shared_memory(
        v_cache_ptr.dtype.element_ty, [cfg.BLOCK_N, cfg.HEAD_DIM], cfg.v_smem_layout
    )

    q = program.load_q()
    m_i, l_i, acc = program.init_state()

    for start_n in range(program.split_start, program.split_end, cfg.BLOCK_N):
        physical_page = program.load_page(start_n)
        program.issue_load_k(physical_page, k_smem)
        program.issue_load_v(physical_page, v_smem)
        async_copy.wait_group(1)
        k = program.shared_load_k(k_smem)
        qk = program.compute_qk(q, k)
        qk = program.apply_rel_bias(qk, start_n)
        qk = program.apply_kv_mask(qk, start_n)
        p, m_i, l_i, acc = program.softmax(qk, m_i, l_i, acc)

        async_copy.wait_group(0)
        v = program.shared_load_v(v_smem)
        acc = program.compute_pv(p, v, acc)

    program.store_output(acc, l_i)


@gluon.jit
def _rel_mha_decode_reduce_fp16(
    mid_o_ptr,
    mid_lse_ptr,
    out_ptr,
    cache_seqlens_ptr,
    SM_SCALE: gl.constexpr,
    PAGE_TABLE_STRIDE: gl.constexpr,
    NUM_KV_SPLITS: gl.constexpr,
    MAX_SEQLEN_Q: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    NUM_Q_HEADS: gl.constexpr,
    NUM_KV_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    IS_SLIDING: gl.constexpr,
    WINDOW_LEFT: gl.constexpr,
    IS_FP8: gl.constexpr,
):
    cfg = AttentionConfig(
        SM_SCALE,
        PAGE_TABLE_STRIDE,
        PAGE_SIZE,
        NUM_KV_SPLITS,
        MAX_SEQLEN_Q,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        BLOCK_M,
        BLOCK_N,
        IS_SLIDING,
        WINDOW_LEFT,
        1,  # REL_EXTENT
        1.0,  # REL_BIAS_QK_SCALE
        IS_FP8,
        InputStrides(1, 1, 1),
        InputStrides(1, 1, 1),
        PagedKVStrides(1, 1, 1, 1),
        PagedKVStrides(1, 1, 1, 1),
    )
    q_index = gl.program_id(0)
    batch = q_index // MAX_SEQLEN_Q
    q_pos = q_index - batch * MAX_SEQLEN_Q
    q_head = gl.program_id(1)
    cache_len = gl.load(cache_seqlens_ptr + batch)
    cache_len = cache_len - (MAX_SEQLEN_Q - 1 - q_pos)
    cache_len = maximum(cache_len, 0)
    if cfg.IS_SLIDING:
        kv_start = cache_len - min(cache_len, cfg.WINDOW_LEFT + 1)
    else:
        kv_start = cache_len - cache_len
    first_page = kv_start // cfg.PAGE_SIZE
    end_page = cdiv(cache_len, cfg.PAGE_SIZE)
    num_pages = end_page - first_page
    pages_per_split = cdiv(num_pages, cfg.NUM_KV_SPLITS)

    # SPLIT_TILE pads NUM_KV_SPLITS up to a power of 2.
    SPLIT_TILE: gl.constexpr = 1 << (NUM_KV_SPLITS - 1).bit_length()
    offs_s = gl.arange(0, SPLIT_TILE, layout=gl.SliceLayout(1, cfg.reduce_layout))
    offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.reduce_layout))
    # split_valid masks out empty splits and the power-of-2 padding tail.
    split_start_page = first_page + offs_s * pages_per_split
    split_end_page_raw = split_start_page + pages_per_split
    split_end_page = gl.where(
        split_end_page_raw < end_page, split_end_page_raw, end_page
    )
    split_start_tok = split_start_page * cfg.PAGE_SIZE
    split_end_raw = split_end_page * cfg.PAGE_SIZE
    split_end_tok = gl.where(split_end_raw < cache_len, split_end_raw, cache_len)
    split_valid = (split_start_tok < split_end_tok) & (offs_s < cfg.NUM_KV_SPLITS)
    # Load every split's partial output and lse.
    base = (q_index * cfg.NUM_Q_HEADS + q_head) * cfg.NUM_KV_SPLITS + offs_s
    part_lse = gl.load(mid_lse_ptr + base, mask=split_valid, other=-float("inf"))
    o_off = base[:, None] * cfg.HEAD_DIM + offs_d[None, :]
    part_o = cdna4.buffer_load(mid_o_ptr, o_off, mask=split_valid[:, None], other=0.0)

    # Global softmax max over all splits.
    m_i = max(part_lse, axis=0)
    # Weighted sum of the split partials, normalized by the total softmax mass.
    beta = gl.exp2(part_lse - m_i)
    l_i = gl.sum(beta, axis=0)
    acc = gl.sum(part_o * beta[:, None], axis=0)

    out_base = (q_index * cfg.NUM_Q_HEADS + q_head) * cfg.HEAD_DIM
    output = acc * (1.0 / l_i)
    output = output.to(out_ptr.dtype.element_ty)
    cdna4.buffer_store(output, out_ptr, out_base + offs_d)


def _select_num_kv_splits(
    *,
    batch: int,
    num_kv_heads: int,
    num_groups: int,
    num_pages: int,
    sm_count: int,
) -> int:
    """Pick num_kv_splits to balance occupancy against reduce overhead.

    The launch grid is (batch * num_kv_heads * num_groups) * num_kv_splits
    work-groups. Too few splits under-fill the machine at low batch; too many
    leave each split with a handful of pages, so the reduce kernel dominates.

    Return the smaller of two candidate counts: splits_for_occupancy (enough to
    fill ~wave_target waves of CUs) and splits_for_pages (~min_pages_per_split
    pages per split), with the pages candidate clamped to [min_page_splits,
    max_page_splits] so a short context still splits without launching empty work
    and a long one does not over-split where reduce cost outgrows the decode win.
    """
    wave_target = 2
    min_pages_per_split = 2
    min_page_splits = 8
    max_page_splits = 32

    base_ctas = batch * num_kv_heads * num_groups
    target_ctas = sm_count * wave_target
    splits_for_occupancy = (target_ctas + base_ctas - 1) // base_ctas

    splits_for_pages = num_pages // min_pages_per_split
    min_page_splits = min(min_page_splits, num_pages)
    if splits_for_pages < min_page_splits:
        splits_for_pages = min_page_splits
    if splits_for_pages > max_page_splits:
        splits_for_pages = max_page_splits
    return min(splits_for_occupancy, splits_for_pages)


class LaunchConfig(NamedTuple):
    num_q_heads: int
    num_kv_heads: int
    num_groups: int
    head_dim: int
    page_size: int
    num_kv_splits: int
    block_m: int
    block_n: int
    sm_scale: float
    rel_bias_qk_scale: float
    is_sliding: bool
    window_left: int


def get_config(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    max_seqlen_k: int,
    window_left: int,
    softmax_scale: float | None,
) -> LaunchConfig:
    head_dim = q.shape[2]
    page_size = k_cache.shape[1]
    block_m = 16
    block_n = page_size
    group_size = q.shape[1] // k_cache.shape[2]
    num_groups = math.ceil(group_size / block_m)
    is_sliding = window_left >= 0
    window_left = window_left if is_sliding else -1
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    sm_scale = softmax_scale
    effective_seqlen_k = (
        min(max_seqlen_k, window_left + 1) if is_sliding else max_seqlen_k
    )
    num_pages = (effective_seqlen_k + page_size - 1) // page_size
    if is_sliding:
        num_kv_splits = 4
    else:
        num_kv_splits = _select_num_kv_splits(
            batch=q.shape[0],
            num_kv_heads=k_cache.shape[2],
            num_groups=num_groups,
            num_pages=num_pages,
            sm_count=_GFX950_SM_COUNT,
        )
    return LaunchConfig(
        num_q_heads=q.shape[1],
        num_kv_heads=k_cache.shape[2],
        num_groups=num_groups,
        head_dim=head_dim,
        page_size=page_size,
        num_kv_splits=num_kv_splits,
        block_m=block_m,
        block_n=block_n,
        sm_scale=sm_scale * _INV_LN2_VALUE,
        rel_bias_qk_scale=1.0 / softmax_scale,
        is_sliding=is_sliding,
        window_left=window_left,
    )


def gluon_rel_mha_decode_gfx950(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    rel_logits: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int = 1,
    window_left: int = -1,
    softmax_scale: float | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    total_q = q.shape[0]

    config = get_config(
        q=q,
        k_cache=k_cache,
        max_seqlen_k=max_seqlen_k,
        window_left=window_left,
        softmax_scale=softmax_scale,
    )

    is_fp8 = q.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    out_dtype = torch.bfloat16 if is_fp8 else q.dtype
    if k_cache.shape[1] not in (64, 128, 256):
        raise ValueError(
            "gfx950 Gluon relative-attention decode requires page size "
            f"64, 128, or 256; got {k_cache.shape[1]}"
        )
    output = (
        torch.empty(q.shape, device=q.device, dtype=out_dtype) if out is None else out
    )

    # Always use split-k for full attention and for the small-batch sliding
    # decode path. Sliding uses a fixed 8 splits, one page per split for the
    # TP-4 local-attention shape with 512-token window and 64-token pages.
    mid_o = torch.empty(
        (total_q, config.num_q_heads, config.num_kv_splits, config.head_dim),
        device=q.device,
        dtype=torch.float32,
    )
    mid_lse = torch.empty(
        (total_q, config.num_q_heads, config.num_kv_splits),
        device=q.device,
        dtype=torch.float32,
    )

    grid = (
        total_q,
        config.num_kv_heads * config.num_groups,
        config.num_kv_splits,
    )
    _rel_mha_decode_fp16[grid](
        q,
        rel_logits,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        mid_o,
        mid_lse,
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
        config.sm_scale,
        page_table.stride(0),
        config.page_size,
        config.num_kv_splits,
        max_seqlen_q,
        config.num_q_heads,
        config.num_kv_heads,
        config.head_dim,
        config.block_m,
        config.block_n,
        config.is_sliding,
        config.window_left,
        rel_logits.stride(0),
        rel_logits.stride(1),
        rel_logits.stride(2),
        rel_logits.shape[2],
        config.rel_bias_qk_scale,
        is_fp8,
        num_warps=1,
    )

    reduce_grid = (total_q, config.num_q_heads)
    _rel_mha_decode_reduce_fp16[reduce_grid](
        mid_o,
        mid_lse,
        output,
        cache_seqlens,
        config.sm_scale,
        page_table.stride(0),
        config.num_kv_splits,
        max_seqlen_q,
        config.page_size,
        config.num_q_heads,
        config.num_kv_heads,
        config.head_dim,
        config.block_m,
        config.block_n,
        config.is_sliding,
        config.window_left,
        is_fp8,
        num_warps=1,
    )
    return output
