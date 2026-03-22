# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""前缀预填充注意力操作模块。

本模块实现了上下文注意力（Context Attention）的前向传播操作，负责：
- 实现 Triton 前向传播 kernel（支持非标准块大小如 544）
- 支持 FP8 KV 缓存反量化
- 支持滑动窗口注意力
- 支持 Sink token
- 支持 ALIBI 斜率
- 支持 FP8 输出量化
- 支持分页 KV 缓存

主要函数：
- _fwd_kernel: 主前向传播 Triton kernel
- _fwd_kernel_alibi: 支持 ALIBI 的 Triton kernel
- context_attention_fwd: 上下文注意力前向传播封装函数
"""

# The kernels in this file are adapted from LightLLM's context_attention_fwd:
# https://github.com/ModelTC/lightllm/blob/main/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

# 静态 kernel 参数
BASE_BLOCK = 128 if current_platform.has_device_capability(80) else 64
NUM_WARPS = 4 if current_platform.is_rocm() else 8

# 兼容性检查
IS_TURING = current_platform.get_device_capability() == (7, 5)
float8_info = torch.finfo(current_platform.fp8_dtype())


# 以下是 autotuning 配置示例。该配置确实能提供性能提升，但会显著增加
# triton 3.2 中首次调用的延迟。由于这种权衡，目前注释掉。
# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, \
#                         "num_unroll_cache": 4, \
#                         "num_unroll_request": 1 } | \
#                         ({"kpack": 2, "waves_per_eu": 2} \
#                             if current_platform.is_rocm() else {}), \
#                         num_warps=4, \
#                         num_stages=1)
#     ],
#     key=["BLOCK_SIZE", "MAX_Q_LEN", "MAX_CTX_LEN"]
# )
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    K_cache,
    V_cache,
    sink_ptr,
    B_Loc,
    sm_scale,
    k_scale,
    v_scale,
    out_scale_inv,
    B_Start_Loc,
    B_Seqlen,
    x: tl.constexpr,
    Out,
    stride_b_loc_b,
    stride_b_loc_s,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_obs,
    stride_oh,
    stride_od,
    stride_k_cache_bs,
    stride_k_cache_h,
    stride_k_cache_d,
    stride_k_cache_bl: tl.constexpr,
    stride_k_cache_x,
    stride_v_cache_bs,
    stride_v_cache_h,
    stride_v_cache_d,
    stride_v_cache_bl,
    num_queries_per_kv: tl.constexpr,
    IN_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DMODEL_PADDED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PHYSICAL_BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    num_unroll_cache: tl.constexpr,
    num_unroll_request: tl.constexpr,
    SKIP_DECODE: tl.constexpr,
    USE_SINKS: tl.constexpr,
    USE_FP8: tl.constexpr,
    MAX_Q_LEN: tl.constexpr = 0,
    MAX_CTX_LEN: tl.constexpr = 0,
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    """前向传播 Triton kernel。

    实现上下文注意力的前向传播计算，支持：
    - 非标准块大小（如 544）
    - FP8 KV 缓存反量化
    - 滑动窗口注意力
    - Sink token
    - FP8 输出量化
    - 分页 KV 缓存

    Args:
        Q: Query 张量指针
        K: Key 张量指针
        V: Value 张量指针
        K_cache: Key 缓存指针
        V_cache: Value 缓存指针
        sink_ptr: Sink 指针
        B_Loc: 块表指针
        sm_scale: Softmax 缩放因子
        k_scale: K 缩放因子
        v_scale: V 缩放因子
        out_scale_inv: 输出缩放因子的倒数
        B_Start_Loc: 批次起始位置指针
        B_Seqlen: 序列长度指针
        x: K 缓存分块参数
        Out: 输出张量指针
        stride_b_loc_b: 块表第 B 维步幅
        stride_b_loc_s: 块表第 S 维步幅
        stride_qbs: Q 第 0 维步幅
        stride_qh: Q 第 H 维步幅
        stride_qd: Q 第 D 维步幅
        stride_kbs: K 第 0 维步幅
        stride_kh: K 第 H 维步幅
        stride_kd: K 第 D 维步幅
        stride_vbs: V 第 0 维步幅
        stride_vh: V 第 H 维步幅
        stride_vd: V 第 D 维步幅
        stride_obs: 输出第 0 维步幅
        stride_oh: 输出第 H 维步幅
        stride_od: 输出第 D 维步幅
        stride_k_cache_bs: K 缓存第 0 维步幅
        stride_k_cache_h: K 缓存第 H 维步幅
        stride_k_cache_d: K 缓存第 D 维步幅
        stride_k_cache_bl: K 缓存块大小维步幅
        stride_k_cache_x: K 缓存 x 维步幅
        stride_v_cache_bs: V 缓存第 0 维步幅
        stride_v_cache_h: V 缓存第 H 维步幅
        stride_v_cache_d: V 缓存第 D 维步幅
        stride_v_cache_bl: V 缓存块大小维步幅
        num_queries_per_kv: 每个 KV 头对应的 Query 头数（GQA 比例）
        IN_PRECISION: 输入精度
        BLOCK_M: M 维度块大小
        BLOCK_DMODEL: 头维度
        BLOCK_DMODEL_PADDED: 填充后的头维度（2 的幂）
        BLOCK_SIZE: 块大小
        PHYSICAL_BLOCK_SIZE: 物理块大小
        BLOCK_N: N 维度块大小
        SLIDING_WINDOW: 滑动窗口大小
        num_unroll_cache: 缓存循环展开数
        num_unroll_request: 请求循环展开数
        SKIP_DECODE: 是否跳图解码
        USE_SINKS: 是否使用 sink
        USE_FP8: 是否使用 FP8
        MAX_Q_LEN: 最大 Query 长度
        MAX_CTX_LEN: 最大上下文长度
        FP8_MIN: FP8 最小值
        FP8_MAX: FP8 最大值
    """
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // num_queries_per_kv

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_in_all_stop_index = tl.load(B_Start_Loc + cur_batch + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
    cur_batch_ctx_len = cur_batch_seq_len - cur_batch_query_len

    if SKIP_DECODE and cur_batch_query_len == 1:
        return

    # start position inside of the query
    # generally, N goes over kv, while M goes over query_len
    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    # [BLOCK_SIZE]; starts at 0
    offs_bs_n = tl.arange(0, BLOCK_SIZE)
    # [N]; starts at 0
    offs_n = tl.arange(0, BLOCK_N)
    # [D]; starts at 0
    offs_d = tl.arange(0, BLOCK_DMODEL_PADDED)
    # [M]; starts at current position in query
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # [M,D]
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :] * stride_qd
    )

    dim_mask = tl.where(tl.arange(0, BLOCK_DMODEL_PADDED) < BLOCK_DMODEL, 1, 0).to(
        tl.int1
    )  # [D]

    q = tl.load(
        Q + off_q,
        mask=dim_mask[None, :] & (offs_m[:, None] < cur_batch_query_len),
        other=0.0,
    )  # [M,D]

    # initialize pointer to m and l
    if not USE_SINKS:
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    else:
        m_i = tl.load(
            sink_ptr + tl.full([BLOCK_M], cur_head, dtype=tl.int64),
            mask=(offs_m < cur_batch_query_len),
            other=float("-inf"),
        ).to(dtype=tl.float32)
        l_i = tl.where(m_i > float("-inf"), 1.0, 0.0)

    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_PADDED], dtype=tl.float32)  # [M,D]

    # compute query against context (no causal mask here)
    for start_n in tl.range(
        0, cur_batch_ctx_len, BLOCK_SIZE, loop_unroll_factor=num_unroll_cache
    ):
        # Under a block size of 544 (Qwen/Qwen3-Next-80B-A3B-Thinking),
        # replace one physical block every 17 32-Tile blocks
        # Calculate the logical block index of each of the 32 tokens
        # in the current Tile (handling cross-block cases).
        token_indices = start_n + offs_bs_n
        bn_logical_indices = token_indices // PHYSICAL_BLOCK_SIZE

        # 2. Vectorized loading of physical block IDs from B_Loc
        bn = tl.load(
            B_Loc + cur_batch * stride_b_loc_b + bn_logical_indices * stride_b_loc_s
        ).to(tl.int64)

        # 3. Calculate the exact offset of
        # each token within its physical block.
        internal_offsets = token_indices % PHYSICAL_BLOCK_SIZE

        # Addressing of K (5D)
        off_k = (
            bn[None, :] * stride_k_cache_bs
            + cur_kv_head * stride_k_cache_h
            + (offs_d[:, None] // x) * stride_k_cache_d
            + internal_offsets[None, :] * stride_k_cache_bl
            + (offs_d[:, None] % x) * stride_k_cache_x
        )

        # Addressing of V (4D)
        off_v = (
            bn[:, None] * stride_v_cache_bs
            + cur_kv_head * stride_v_cache_h
            + offs_d[None, :] * stride_v_cache_d
            + internal_offsets[:, None] * stride_v_cache_bl
        )

        if (
            start_n + BLOCK_SIZE > cur_batch_ctx_len
            or BLOCK_DMODEL != BLOCK_DMODEL_PADDED
        ):
            k_load = tl.load(
                K_cache + off_k,
                mask=dim_mask[:, None]
                & ((start_n + offs_bs_n[None, :]) < cur_batch_ctx_len),
                other=0.0,
            )  # [D,N]
        else:
            k_load = tl.load(K_cache + off_k)

        if k_load.dtype.is_fp8():
            k = (k_load.to(tl.float32) * tl.load(k_scale)).to(q.dtype)
        else:
            k = k_load

        # qk = tl.zeros([BLOCK_M, BLOCK_SIZE], dtype=tl.float32)  # [M,N]
        qk = sm_scale * tl.dot(q, k, input_precision=IN_PRECISION)
        qk = tl.where(
            (start_n + offs_bs_n[None, :]) < cur_batch_ctx_len, qk, float("-inf")
        )
        # qk *= sm_scale
        if SLIDING_WINDOW > 0:
            # (cur_batch_ctx_len + offs_m[:, None]) are the positions of
            # Q entries in sequence
            # (start_n + offs_bs_n[None, :]) are the positions of
            # KV entries in sequence
            # So the condition makes sure each entry in Q only attends
            # to KV entries not more than SLIDING_WINDOW away.
            #
            # We can't use -inf here, because the
            # sliding window may lead to the entire row being masked.
            # This then makes m_ij contain -inf, which causes NaNs in
            # exp().
            qk = tl.where(
                (cur_batch_ctx_len + offs_m[:, None]) - (start_n + offs_bs_n[None, :])
                < SLIDING_WINDOW,
                qk,
                float("-inf"),
            )

        # compute running maximum
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        p = tl.where(m_ij[:, None] == float("-inf"), 0.0, p)
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        alpha = tl.where(m_i == float("-inf"), 0.0, alpha)
        acc = acc * alpha[:, None]

        # update acc
        if (
            start_n + BLOCK_SIZE > cur_batch_ctx_len
            or BLOCK_DMODEL != BLOCK_DMODEL_PADDED
        ):
            v_load = tl.load(
                V_cache + off_v,
                mask=dim_mask[None, :]
                & ((start_n + offs_bs_n[:, None]) < cur_batch_ctx_len),
                other=0.0,
            )  # [N,D]
        else:
            v_load = tl.load(V_cache + off_v)

        if v_load.dtype.is_fp8():
            v = (v_load.to(tl.float32) * tl.load(v_scale)).to(q.dtype)
        else:
            v = v_load
        p = p.to(v.dtype)

        acc = tl.dot(p, v, acc=acc, input_precision=IN_PRECISION)
        # # update m_i and l_i
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    off_k = (
        offs_n[None, :] * stride_kbs
        + cur_kv_head * stride_kh
        + offs_d[:, None] * stride_kd
    )
    off_v = (
        offs_n[:, None] * stride_vbs
        + cur_kv_head * stride_vh
        + offs_d[None, :] * stride_vd
    )
    k_ptrs = K + off_k
    v_ptrs = V + off_v

    # block_mask is 0 when we're already past the current query length
    block_mask = tl.where(block_start_loc < cur_batch_query_len, 1, 0)

    # compute query against itself (with causal mask)
    for start_n in tl.range(
        0,
        block_mask * (start_m + 1) * BLOCK_M,
        BLOCK_N,
        loop_unroll_factor=num_unroll_request,
    ):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(
            k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
            mask=dim_mask[:, None]
            & ((start_n + offs_n[None, :]) < cur_batch_query_len),
            other=0.0,
        )

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, acc=qk, input_precision=IN_PRECISION)
        qk *= sm_scale
        # apply causal mask
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        if SLIDING_WINDOW > 0:
            qk = tl.where(
                offs_m[:, None] - (start_n + offs_n[None, :]) < SLIDING_WINDOW,
                qk,
                float("-inf"),
            )

        # compute running maximum
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        p = tl.where(m_ij[:, None] == float("-inf"), 0.0, p)
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        # To prevent NaN from appearing in the first round
        alpha = tl.where(m_i == float("-inf"), 0.0, alpha)
        acc = acc * alpha[:, None]

        # update acc
        v = tl.load(
            v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
            mask=dim_mask[None, :]
            & ((start_n + offs_n[:, None]) < cur_batch_query_len),
            other=0.0,
        )
        p = p.to(v.dtype)

        acc = tl.dot(p, v, acc=acc, input_precision=IN_PRECISION)
        # update m_i and l_i
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    acc = acc / (l_i[:, None] + 1e-10)

    # initialize pointers to output
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :] * stride_od
    )
    out_ptrs = Out + off_o
    if USE_FP8:
        acc = acc * tl.load(out_scale_inv)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)
    tl.store(
        out_ptrs, acc, mask=dim_mask[None, :] & (offs_m[:, None] < cur_batch_query_len)
    )
    return


@triton.jit
def _fwd_kernel_alibi(
    Q,
    K,
    V,
    K_cache,
    V_cache,
    B_Loc,
    sm_scale,
    k_scale,
    v_scale,
    B_Start_Loc,
    B_Seqlen,
    Alibi_slopes,
    block_size,
    x,
    Out,
    stride_b_loc_b,
    stride_b_loc_s,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_obs,
    stride_oh,
    stride_od,
    stride_k_cache_bs,
    stride_k_cache_h,
    stride_k_cache_d,
    stride_k_cache_bl,
    stride_k_cache_x,
    stride_v_cache_bs,
    stride_v_cache_h,
    stride_v_cache_d,
    stride_v_cache_bl,
    num_queries_per_kv: int,
    IN_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,  # head size
    BLOCK_DMODEL_PADDED: tl.constexpr,  # head size padded to a power of 2
    BLOCK_N: tl.constexpr,
    SKIP_DECODE: tl.constexpr,
):
    """支持 ALIBI 的前向传播 Triton kernel。

    实现带有 ALIBI（Attention with Linear Biases）的上下文注意力计算。
    ALIBI 通过添加与距离成比例的偏置来扩展模型的上下文长度。

    Args:
        Q: Query 张量指针
        K: Key 张量指针
        V: Value 张量指针
        K_cache: Key 缓存指针
        V_cache: Value 缓存指针
        B_Loc: 块表指针
        sm_scale: Softmax 缩放因子
        k_scale: K 缩放因子
        v_scale: V 缩放因子
        B_Start_Loc: 批次起始位置指针
        B_Seqlen: 序列长度指针
        Alibi_slopes: ALIBI 斜率指针
        block_size: 块大小
        x: K 缓存分块参数
        Out: 输出张量指针
        stride_b_loc_b: 块表第 B 维步幅
        stride_b_loc_s: 块表第 S 维步幅
        stride_qbs: Q 第 0 维步幅
        stride_qh: Q 第 H 维步幅
        stride_qd: Q 第 D 维步幅
        stride_kbs: K 第 0 维步幅
        stride_kh: K 第 H 维步幅
        stride_kd: K 第 D 维步幅
        stride_vbs: V 第 0 维步幅
        stride_vh: V 第 H 维步幅
        stride_vd: V 第 D 维步幅
        stride_obs: 输出第 0 维步幅
        stride_oh: 输出第 H 维步幅
        stride_od: 输出第 D 维步幅
        stride_k_cache_bs: K 缓存第 0 维步幅
        stride_k_cache_h: K 缓存第 H 维步幅
        stride_k_cache_d: K 缓存第 D 维步幅
        stride_k_cache_bl: K 缓存块大小维步幅
        stride_k_cache_x: K 缓存 x 维步幅
        stride_v_cache_bs: V 缓存第 0 维步幅
        stride_v_cache_h: V 缓存第 H 维步幅
        stride_v_cache_d: V 缓存第 D 维步幅
        stride_v_cache_bl: V 缓存块大小维步幅
        num_queries_per_kv: 每个 KV 头对应的 Query 头数
        IN_PRECISION: 输入精度
        BLOCK_M: M 维度块大小
        BLOCK_DMODEL: 头维度
        BLOCK_DMODEL_PADDED: 填充后的头维度（2 的幂）
        BLOCK_N: N 维度块大小
        SKIP_DECODE: 是否跳图解码
    """
    # attn_bias[]
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // num_queries_per_kv

    # cur_batch_seq_len: the length of prompts
    # cur_batch_ctx_len: the length of prefix
    # cur_batch_in_all_start_index: the start id of the dim=0
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_in_all_stop_index = tl.load(B_Start_Loc + cur_batch + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
    cur_batch_ctx_len = cur_batch_seq_len - cur_batch_query_len

    if SKIP_DECODE and cur_batch_query_len == 1:
        return

    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL_PADDED)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :] * stride_qd
    )

    dim_mask = tl.where(tl.arange(0, BLOCK_DMODEL_PADDED) < BLOCK_DMODEL, 1, 0).to(
        tl.int1
    )

    q = tl.load(
        Q + off_q,
        mask=dim_mask[None, :]
        & (offs_m[:, None] < cur_batch_seq_len - cur_batch_ctx_len),
        other=0.0,
    )

    # # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_PADDED], dtype=tl.float32)

    alibi_slope = tl.load(Alibi_slopes + cur_head)
    alibi_start_q = tl.arange(0, BLOCK_M) + block_start_loc + cur_batch_ctx_len
    alibi_start_k = 0
    for start_n in range(0, cur_batch_ctx_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        bn = tl.load(
            B_Loc
            + cur_batch * stride_b_loc_b
            + ((start_n + offs_n) // block_size) * stride_b_loc_s,
            mask=(start_n + offs_n) < cur_batch_ctx_len,
            other=0,
        ).to(tl.int64)
        off_k = (
            bn[None, :] * stride_k_cache_bs
            + cur_kv_head * stride_k_cache_h
            + (offs_d[:, None] // x) * stride_k_cache_d
            + ((start_n + offs_n[None, :]) % block_size) * stride_k_cache_bl
            + (offs_d[:, None] % x) * stride_k_cache_x
        )
        off_v = (
            bn[:, None] * stride_v_cache_bs
            + cur_kv_head * stride_v_cache_h
            + offs_d[None, :] * stride_v_cache_d
            + (start_n + offs_n[:, None]) % block_size * stride_v_cache_bl
        )
        k_load = tl.load(
            K_cache + off_k,
            mask=dim_mask[:, None] & ((start_n + offs_n[None, :]) < cur_batch_ctx_len),
            other=0.0,
        )  # [D,N]

        if k_load.dtype.is_fp8():
            k = (k_load.to(tl.float32) * tl.load(k_scale)).to(q.dtype)
        else:
            k = k_load

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, acc=qk, input_precision=IN_PRECISION)
        qk = tl.where(
            (start_n + offs_n[None, :]) < cur_batch_ctx_len, qk, float("-inf")
        )
        qk *= sm_scale

        # load alibi
        alibi = (
            tl.arange(0, BLOCK_N)[None, :] + alibi_start_k - alibi_start_q[:, None]
        ) * alibi_slope
        alibi = tl.where(
            (alibi <= 0) & (alibi_start_q[:, None] < cur_batch_seq_len),
            alibi,
            float("-inf"),
        )
        qk += alibi
        alibi_start_k += BLOCK_N

        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        p = tl.math.exp(qk - m_i_new[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i

        alpha = tl.math.exp(m_i - m_i_new)
        l_i_new = alpha * l_i + l_ij
        # -- update output accumulator --
        # scale p
        # scale acc
        acc_scale = alpha
        # acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        # update acc
        v_load = tl.load(
            V_cache + off_v,
            mask=dim_mask[None, :] & ((start_n + offs_n[:, None]) < cur_batch_ctx_len),
            other=0.0,
        )
        if v_load.dtype.is_fp8():
            v = (v_load.to(tl.float32) * tl.load(v_scale)).to(q.dtype)
        else:
            v = v_load
        p = p.to(v.dtype)

        acc = tl.dot(p, v, acc=acc, input_precision="ieee")
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    off_k = (
        offs_n[None, :] * stride_kbs
        + cur_kv_head * stride_kh
        + offs_d[:, None] * stride_kd
    )
    off_v = (
        offs_n[:, None] * stride_vbs
        + cur_kv_head * stride_vh
        + offs_d[None, :] * stride_vd
    )
    k_ptrs = K + off_k
    v_ptrs = V + off_v

    block_mask = tl.where(block_start_loc < cur_batch_seq_len - cur_batch_ctx_len, 1, 0)

    # init alibi
    alibi_slope = tl.load(Alibi_slopes + cur_head)
    alibi_start_q = tl.arange(0, BLOCK_M) + block_start_loc + cur_batch_ctx_len
    alibi_start_k = cur_batch_ctx_len
    # # init debugger
    # offset_db_q = tl.arange(0, BLOCK_M) + block_start_loc
    # offset_db_k = tl.arange(0, BLOCK_N)
    # calc q[BLOCK_M, BLOCK_MODEL] mul k[prefix_len: , BLOCK_DMODEL]
    for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(
            k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
            mask=dim_mask[:, None]
            & ((start_n + offs_n[None, :]) < cur_batch_seq_len - cur_batch_ctx_len),
            other=0.0,
        )

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, acc=qk, input_precision="ieee")
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

        # load alibi
        alibi = (
            tl.arange(0, BLOCK_N)[None, :] + alibi_start_k - alibi_start_q[:, None]
        ) * alibi_slope
        alibi = tl.where(
            (alibi <= 0) & (alibi_start_q[:, None] < cur_batch_seq_len),
            alibi,
            float("-inf"),
        )
        qk += alibi
        alibi_start_k += BLOCK_N

        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        p = tl.math.exp(qk - m_i_new[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i

        alpha = tl.math.exp(m_i - m_i_new)
        l_i_new = alpha * l_i + l_ij
        # -- update output accumulator --
        # scale p
        # scale acc
        acc_scale = alpha
        # acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        # update acc
        v = tl.load(
            v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
            mask=dim_mask[None, :]
            & ((start_n + offs_n[:, None]) < cur_batch_seq_len - cur_batch_ctx_len),
            other=0.0,
        )
        p = p.to(v.dtype)

        acc = tl.dot(p, v, acc=acc, input_precision="ieee")
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    acc = acc / l_i[:, None]

    # initialize pointers to output
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :] * stride_od
    )
    out_ptrs = Out + off_o
    tl.store(
        out_ptrs,
        acc,
        mask=dim_mask[None, :]
        & (offs_m[:, None] < cur_batch_seq_len - cur_batch_ctx_len),
    )
    return


@torch.inference_mode()
def context_attention_fwd(
    q,
    k,
    v,
    o,
    kv_cache_dtype: str,
    k_cache,
    v_cache,
    b_loc,
    b_start_loc,
    b_seq_len,
    max_seq_len,
    max_input_len,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    alibi_slopes=None,
    sliding_window=None,
    sm_scale=None,
    skip_decode=False,
    fp8_out_scale=None,
    sinks=None,
    is_block_table_ptr: bool = False,
):
    """上下文注意力前向传播封装函数。

    该函数处理上下文注意力（Context Attention）的前向传播计算，支持：
    - FP8 KV 缓存反量化
    - 滑动窗口注意力
    - ALIBI 斜率
    - Sink token
    - FP8 输出量化
    - 非标准块大小（如 544）

    Args:
        q: Query 张量
        k: Key 张量
        v: Value 张量
        o: 输出张量
        kv_cache_dtype: KV 缓存数据类型
        k_cache: Key 缓存
        v_cache: Value 缓存
        b_loc: 块表
        b_start_loc: 批次起始位置
        b_seq_len: 序列长度
        max_seq_len: 最大序列长度
        max_input_len: 最大输入长度
        k_scale: K 缩放因子
        v_scale: V 缩放因子
        alibi_slopes: ALIBI 斜率（可选）
        sliding_window: 滑动窗口大小（可选）
        sm_scale: Softmax 缩放因子（可选）
        skip_decode: 是否跳图解码
        fp8_out_scale: FP8 输出缩放因子（可选）
        sinks: Sink token 张量（可选）
        is_block_table_ptr: b_loc 是否为指针
    """
    q_dtype_is_f32 = q.dtype is torch.float32

    # Turing does have tensor core for float32 multiplication
    # use ieee as fallback for triton kernels work. There is also
    # warning on vllm/config.py to inform users this fallback
    # implementation
    IN_PRECISION = "ieee" if IS_TURING and q_dtype_is_f32 else None

    # Conversion of FP8 Tensor from uint8 storage to
    # appropriate torch.dtype for interpretation by Triton
    if "fp8" in kv_cache_dtype:
        assert k_cache.dtype in [torch.uint8, current_platform.fp8_dtype()]
        assert v_cache.dtype in [torch.uint8, current_platform.fp8_dtype()]

        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            target_dtype = current_platform.fp8_dtype()
        elif kv_cache_dtype == "fp8_e5m2":
            target_dtype = torch.float8_e5m2
        else:
            raise ValueError("Unsupported FP8 dtype:", kv_cache_dtype)

        k_cache = k_cache.view(target_dtype)
        v_cache = v_cache.view(target_dtype)

    if (
        k_cache.dtype == torch.uint8
        or v_cache.dtype == torch.uint8
        and kv_cache_dtype == "auto"
    ):
        raise ValueError(
            "kv_cache_dtype='auto' unsupported for\
            FP8 KV Cache prefill kernel"
        )

    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    # round up Lk to a power of 2 - this is required for Triton block size
    Lk_padded = triton.next_power_of_2(Lk)

    if sm_scale is None:
        sm_scale = 1.0 / (Lq**0.5)
    batch, head = b_seq_len.shape[0], q.shape[1]
    num_queries_per_kv = q.shape[1] // k.shape[1]

    assert batch + 1 == len(b_start_loc)

    # 0 means "disable"
    if sliding_window is None or sliding_window <= 0:
        sliding_window = 0

    if is_block_table_ptr:
        kv_element_size = k_cache.element_size()
        block_byte_stride = k_cache.stride(0) * kv_element_size
        # The physical starting point of the obtained KV Cache Pool
        base_addr = k_cache.data_ptr()

        mask = b_loc > 0
        processed_b_loc = torch.where(
            mask, (b_loc - base_addr) // block_byte_stride, b_loc
        ).to(torch.int32)
    else:
        processed_b_loc = b_loc.to(torch.int32)

    if alibi_slopes is not None:
        assert sinks is None, "Sinks arg is not supported with alibi"
        assert fp8_out_scale is None, "FP8 output not supported with alibi"
        # need to reduce num. blocks when using fp32
        # due to increased use of GPU shared memory
        # if q.dtype is torch.float32:
        BLOCK = BASE_BLOCK // 2 if q_dtype_is_f32 else BASE_BLOCK
        # batch, head,
        grid = (batch, head, triton.cdiv(max_input_len, BLOCK))
        _fwd_kernel_alibi[grid](
            q,
            k,
            v,
            k_cache,
            v_cache,
            b_loc,
            sm_scale,
            k_scale,
            v_scale,
            b_start_loc,
            b_seq_len,
            alibi_slopes,
            v_cache.shape[3],
            k_cache.shape[4],
            o,
            b_loc.stride(0),
            b_loc.stride(1),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
            k_cache.stride(4),  # [num_blocks, num_kv_heads, head_size/x, block_size, x]
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            v_cache.stride(3),  # [num_blocks, num_kv_heads, head_size, block_size]
            num_queries_per_kv=num_queries_per_kv,
            IN_PRECISION=IN_PRECISION,
            BLOCK_M=BLOCK,
            BLOCK_DMODEL=Lk,
            BLOCK_DMODEL_PADDED=Lk_padded,
            BLOCK_N=BLOCK,
            SKIP_DECODE=skip_decode,
            num_warps=NUM_WARPS,
            num_stages=1,
        )
        return

    max_seq_len = 0 if max_seq_len is None else max_seq_len
    extra_kargs = {}
    if current_platform.is_rocm():
        extra_kargs = {}

    real_block_size = v_cache.shape[3]
    is_pow2 = real_block_size > 0 and (real_block_size & (real_block_size - 1) == 0)
    # For standard models involving powers of 2,
    # follow the original logic (Llama 128/64)
    # For non-standard models (Qwen3-next block_size 544), set to 32.
    if is_pow2:
        BLOCK_M = 128
        BLOCK_N = 64
    else:
        BLOCK_M = 32
        BLOCK_N = 32

    # TRITON_BLOCK_SIZE is kept at 32 to ensure
    # correct alignment logic when the kernel handles
    # non-standard sizes (such as 544).
    TRITON_BLOCK_SIZE = 32

    grid_fn = lambda META: (batch, head, triton.cdiv(max_input_len, META["BLOCK_M"]))
    _fwd_kernel[grid_fn](
        q,
        k,
        v,
        k_cache,
        v_cache,
        sinks,
        processed_b_loc,
        sm_scale,
        k_scale,
        v_scale,
        1.0 / fp8_out_scale if fp8_out_scale is not None else 1.0,
        b_start_loc,
        b_seq_len,
        k_cache.shape[4],
        o,
        processed_b_loc.stride(0),
        processed_b_loc.stride(1),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        stride_k_cache_bs=k_cache.stride(0),
        stride_k_cache_h=k_cache.stride(1),
        stride_k_cache_d=k_cache.stride(2),
        stride_k_cache_bl=k_cache.stride(3),
        stride_k_cache_x=k_cache.stride(4),
        stride_v_cache_bs=v_cache.stride(0),
        stride_v_cache_h=v_cache.stride(1),
        stride_v_cache_d=v_cache.stride(2),
        stride_v_cache_bl=v_cache.stride(3),
        BLOCK_SIZE=TRITON_BLOCK_SIZE,
        PHYSICAL_BLOCK_SIZE=real_block_size,
        num_queries_per_kv=num_queries_per_kv,
        IN_PRECISION=IN_PRECISION,
        BLOCK_DMODEL=Lk,
        BLOCK_DMODEL_PADDED=Lk_padded,
        SLIDING_WINDOW=sliding_window,
        SKIP_DECODE=skip_decode,
        USE_FP8=fp8_out_scale is not None,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_unroll_cache=4,
        num_unroll_request=1,
        num_warps=4,
        num_stages=1,
        USE_SINKS=sinks is not None,
        **extra_kargs,
    )
    return
