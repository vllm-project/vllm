# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/sgl-project/sglang/blob/97cb762bb65ebf05025eb342de03c184660427a3/python/sglang/srt/layers/attention/triton_ops/prefill_attention.py
# Changes:
# - Add support for sliding window attention

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Memory-efficient attention for prefill.
It supports page size = 1.
"""

# Adapted from
# https://github.com/ModelTC/lightllm/blob/f2a54f0912293f683bf1d1695fd12c4098a5bf82/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py#L1
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import RCP_LN2


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    B_Start_Loc,
    B_Seqlen,
    Out,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    kv_group_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SLIDING_WINDOW_Q: tl.constexpr,
    SLIDING_WINDOW_K: tl.constexpr,
    Lk: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    off_k = offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None]
    off_v = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :]

    mask_d = offs_d < Lk

    q = tl.load(
        Q + off_q,
        mask=(offs_m[:, None] < cur_batch_seq_len) & (mask_d[None, :]),
        other=0.0,
    )

    k_ptrs = K + off_k
    v_ptrs = V + off_v

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)

    # Calculate the end position for attention computation
    end_n = cur_batch_seq_len

    # Apply causal attention pruning and sliding window attention pruning
    end_n = tl.minimum(end_n, (start_m + 1) * BLOCK_M) if IS_CAUSAL else end_n

    # Calculate the start position for backward sliding window
    start_n_limit = 0
    end_n_limit = block_mask * end_n

    for start_n in range(start_n_limit, end_n_limit, BLOCK_N):
        # -- prepare attention mask ----
        # Position indices in the sequence
        pos_q = offs_m[:, None]  # Query positions [BLOCK_M, 1]
        pos_k = start_n + offs_n[None, :]  # Key positions [1, BLOCK_N]

        # Valid sequence mask
        mask = pos_k < cur_batch_seq_len
        # Causal mask
        if IS_CAUSAL:
            mask &= pos_q >= pos_k

        # Bidirectional sliding window masks
        sliding_mask_q = (
            pos_q - pos_k <= SLIDING_WINDOW_Q if SLIDING_WINDOW_Q > 0 else None
        )
        sliding_mask_k = (
            pos_k - pos_q <= SLIDING_WINDOW_K if SLIDING_WINDOW_K > 0 else None
        )
        if sliding_mask_q is not None:
            mask &= sliding_mask_q
        if sliding_mask_k is not None:
            mask &= sliding_mask_k

        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(
            k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
            mask=(pos_k < cur_batch_seq_len) & (mask_d[:, None]),
            other=0.0,
        )

        qk = tl.dot(q, k)
        qk = tl.where(mask, qk * sm_scale, -1.0e8)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(
            v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
            mask=((start_n + offs_n[:, None]) < cur_batch_seq_len) & (mask_d[None, :]),
            other=0.0,
        )
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)
        # update m_i
        m_i = m_ij

    acc = acc / l_i[:, None]
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :]
    )
    out_ptrs = Out + off_o
    tl.store(
        out_ptrs, acc, mask=(offs_m[:, None] < cur_batch_seq_len) & (mask_d[None, :])
    )


def get_block_size(dtype: torch.dtype) -> int:
    if dtype == torch.float32:
        return 32
    elif current_platform.is_cuda_alike() and current_platform.has_device_capability(
        80
    ):
        return 128
    else:
        return 64


def context_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    max_input_len: int,
    is_causal: bool = True,
    softmax_scale: float | None = None,
    sliding_window_q: int | None = None,
    sliding_window_k: int | None = None,
):
    """
    q, k, v: [b * s, head, head_dim]
    b_start_loc: [b]
    b_seq_len: [b]
    out: [b * s, head, head_dim]
    """
    BLOCK = get_block_size(q.dtype)

    Lq, Lk, _ = q.shape[-1], k.shape[-1], v.shape[-1]

    sm_scale = 1.0 / (Lq**0.5) if softmax_scale is None else softmax_scale
    # rescale with 1/ln(2) for triton exp2
    sm_scale *= RCP_LN2

    batch, head = b_seq_len.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))
    num_warps = 4 if Lk <= 64 else 8

    sliding_window_q = sliding_window_q if sliding_window_q is not None else 0
    sliding_window_k = sliding_window_k if sliding_window_k is not None else 0

    _fwd_kernel[grid](
        q,
        k,
        v,
        sm_scale,
        b_start_loc,
        b_seq_len,
        o,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        o.stride(0),
        o.stride(1),
        kv_group_num=kv_group_num,
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=triton.next_power_of_2(Lk),
        BLOCK_N=BLOCK,
        IS_CAUSAL=is_causal,
        SLIDING_WINDOW_Q=sliding_window_q,
        SLIDING_WINDOW_K=sliding_window_k,
        num_warps=num_warps,
        num_stages=1,
        Lk=Lk,
    )
