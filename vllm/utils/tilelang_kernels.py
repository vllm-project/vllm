# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import tilelang
import tilelang.language as T
import torch

tilelang.set_log_level("WARNING")

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
}

FP8 = "float8_e4m3"
BF16 = "bfloat16"
FP32 = "float32"


def fast_log2_ceil(x):
    bits_x = T.reinterpret("uint32", x)
    exp_x = (bits_x >> 23) & 0xFF
    man_bits = bits_x & ((1 << 23) - 1)
    return T.Cast("int32", exp_x - 127 + T.if_then_else(man_bits != 0, 1, 0))


def fast_pow2(x):
    bits_x = (x + 127) << 23
    return T.reinterpret("float32", bits_x)


def fast_round_scale(amax, fp8_max_inv):
    return fast_pow2(fast_log2_ceil(amax * fp8_max_inv))


@tilelang.jit(pass_configs=pass_configs)
def act_quant_kernel(N,
                     in_dtype=BF16,
                     out_dtype=FP8,
                     scale_dtype=FP32,
                     round_scale=False):
    M = T.symbolic("M")
    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1 / fp8_max
    num_stages = 0 if round_scale else 2
    blk_m = 32
    group_size = 128

    @T.prim_func
    def act_quant_kernel_(
        X: T.Tensor[(M, N), in_dtype],
        Y: T.Tensor[(M, N), out_dtype],
        S: T.Tensor[(M, T.ceildiv(N, group_size)), scale_dtype],
    ):
        with T.Kernel(T.ceildiv(M, blk_m),
                      T.ceildiv(N, group_size),
                      threads=128) as (
                          pid_m,
                          pid_n,
                      ):
            x_shared = T.alloc_shared((blk_m, group_size), in_dtype)
            x_local = T.alloc_fragment((blk_m, group_size), in_dtype)
            amax_local = T.alloc_fragment((blk_m, ), scale_dtype)
            s_local = T.alloc_fragment((blk_m, ), scale_dtype)
            y_local = T.alloc_fragment((blk_m, group_size), out_dtype)
            y_shared = T.alloc_shared((blk_m, group_size), out_dtype)

            for _ in T.Pipelined(1, num_stages=num_stages):
                T.copy(X[pid_m * blk_m, pid_n * group_size], x_shared)
                T.copy(x_shared, x_local)
                T.reduce_absmax(x_local, amax_local, dim=1)
                for i in T.Parallel(blk_m):
                    amax_local[i] = T.max(amax_local[i], 1e-4)
                    if round_scale:
                        s_local[i] = fast_round_scale(amax_local[i],
                                                      fp8_max_inv)
                    else:
                        s_local[i] = amax_local[i] * fp8_max_inv
                for i, j in T.Parallel(blk_m, group_size):
                    y_local[i, j] = T.clamp(x_local[i, j] / s_local[i],
                                            fp8_min, fp8_max)
                for i in T.Parallel(blk_m):
                    S[pid_m * blk_m + i, pid_n] = s_local[i]
                T.copy(y_local, y_shared)
                T.copy(y_shared, Y[pid_m * blk_m, pid_n * group_size])

    return act_quant_kernel_


def act_quant(
        x: torch.Tensor,
        block_size: int = 128,
        scale_fmt: Optional[str] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.

    Args:
    - x (torch.Tensor): 
        The input tensor to be quantized. 
        Must be contiguous and its last dimension size 
        must be divisible by `block_size`.
    - block_size (int, optional): 
        The size of the blocks to be used for quantization. Default is 128.
    - scale_fmt (Optional[str], optional): 
        The format of the scale. Default is None.
    Returns:
    - tuple[torch.Tensor, torch.Tensor]: A tuple containing:
        - The quantized tensor with dtype `torch.float8_e4m3fn`.
        - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.size(-1) % block_size == 0, (
        f"Last dimension must be divisible by block_size={block_size}")
    N = x.size(-1)
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], N // block_size, dtype=torch.float32)
    kernel = act_quant_kernel(N, round_scale=scale_fmt is not None)
    kernel(x.view(-1, N), y.view(-1, N), s.view(-1, N // block_size))
    return y, s


@tilelang.jit(pass_configs=pass_configs)
def fp8_gemm_kernel(N, K, out_dtype=BF16, accum_dtype="float32"):
    assert out_dtype in [BF16, "float32"]

    M = T.symbolic("M")
    group_size = 128
    block_M = 32
    block_N = 128
    block_K = 128

    @T.prim_func
    def fp8_gemm_kernel_(
        A: T.Tensor[(M, K), FP8],
        B: T.Tensor[(N, K), FP8],
        C: T.Tensor[(M, N), out_dtype],
        scales_a: T.Tensor[(M, T.ceildiv(K, group_size)), FP32],
        scales_b: T.Tensor[(T.ceildiv(N, group_size),
                            T.ceildiv(K, group_size)), FP32],
    ):
        with T.Kernel(T.ceildiv(N, block_N),
                      T.ceildiv(M, block_M),
                      threads=128) as (
                          bx,
                          by,
                      ):
            A_shared = T.alloc_shared((block_M, block_K), FP8)
            B_shared = T.alloc_shared((block_N, block_K), FP8)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)
            Scale_C_shared = T.alloc_shared((block_M), FP32)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_local_accum = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Improve L2 Cache
            T.use_swizzle(panel_size=10)

            T.clear(C_local)
            T.clear(C_local_accum)
            K_iters = T.ceildiv(K, block_K)
            for k in T.Pipelined(K_iters, num_stages=4):
                # Load A into shared memory
                T.copy(A[by * block_M, k * block_K], A_shared)
                # Load B into shared memory
                T.copy(B[bx * block_N, k * block_K], B_shared)
                # Load scale into shared memory
                Scale_B = scales_b[bx * block_N // group_size, k]
                for i in T.Parallel(block_M):
                    Scale_C_shared[i] = scales_a[by * block_M + i, k] * Scale_B

                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                # Promote to enable 2xAcc
                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[i, j] += C_local[i, j] * Scale_C_shared[i]
                T.clear(C_local)
            # TMA store
            T.copy(C_local_accum, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return fp8_gemm_kernel_


def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor,
             b_s: torch.Tensor) -> torch.Tensor:
    """
    Perform a matrix multiplication using FP8 precision.

    Args:
    - a (torch.Tensor): 
        The first input matrix, must be contiguous.
    - a_s (torch.Tensor): 
        The scaling factor for the first input matrix, must be contiguous.
    - b (torch.Tensor): 
        The second input matrix, must be contiguous.
    - b_s (torch.Tensor):
        The scaling factor for the second input matrix, must be contiguous.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """
    assert a.is_contiguous() and b.is_contiguous(
    ), "Input tensors must be contiguous"
    assert a_s.is_contiguous() and b_s.is_contiguous(), (
        "Scaling factor tensors must be contiguous")
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    kernel = fp8_gemm_kernel(N, K)
    kernel(a.view(M, K), b, c.view(M, N), a_s.view(M, -1), b_s)
    return c


@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def fp8_index_kernel(h: int, d: int):
    b = T.symbolic("b")
    m = T.symbolic("m")
    n = T.symbolic("n")

    blk_n1 = 512
    blk_n2 = 128

    @T.prim_func
    def fp8_index_kernel_(
        q: T.Tensor[(b, m, h, d), FP8],
        q_s: T.Tensor[(b, m, h), FP32],
        k: T.Tensor[(b, n, d), FP8],
        k_s: T.Tensor[(b, n), FP32],
        o: T.Tensor[(b, m, n), FP32],
    ) -> None:
        with T.Kernel(b, m, T.ceildiv(n, blk_n1)) as (i_b, i_m, i1_n):
            q_smem = T.alloc_shared((h, d), FP8)
            T.copy(q[i_b, i_m, 0, 0], q_smem)

            q_s_frag = T.alloc_fragment(h, FP32)
            T.copy(q_s[i_b, i_m, 0], q_s_frag)

            for i2_n in T.Pipelined(blk_n1 // blk_n2, num_stages=2):
                k_smem = T.alloc_shared((blk_n2, d), FP8)
                T.copy(k[i_b, i1_n * blk_n1 + i2_n * blk_n2, 0], k_smem)

                k_s_frag = T.alloc_fragment(blk_n2, FP32)
                T.copy(k_s[i_b, i1_n * blk_n1 + i2_n * blk_n2], k_s_frag)

                logits = T.alloc_fragment((blk_n2, h), FP32)
                T.gemm(
                    k_smem,
                    q_smem,
                    logits,
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=True,
                )

                for i_h, i3_n in T.Parallel(h, blk_n2):
                    logits[i3_n,
                           i_h] = T.max(logits[i3_n, i_h], 0) * q_s_frag[i_h]

                logits_sum = T.alloc_fragment(blk_n2, FP32)
                T.reduce_sum(logits, logits_sum, dim=1)

                for i3_n in T.Parallel(blk_n2):
                    logits_sum[i3_n] *= k_s_frag[i3_n]

                T.copy(logits_sum, o[i_b, i_m, i1_n * blk_n1 + i2_n * blk_n2])

    return fp8_index_kernel_


def fp8_index(
    q: torch.Tensor,
    q_s: torch.Tensor,
    k: torch.Tensor,
    k_s: torch.Tensor,
) -> torch.Tensor:
    """
    Perform index score using FP8 precision.

    Args:
    - q (torch.Tensor): 
        The Q tensor, must be contiguous.
    - q_s (torch.Tensor): 
        The scaling factor for Q (float), must be contiguous.
    - k (torch.Tensor): 
        The K tensor, must be contiguous.
    - k_s (torch.Tensor): 
        The scaling factor for K (e8m0 here), must be contiguous.

        fp8 q @ fp8 k -> fp32 logits
        relu(fp32 logits) * q_s (weights) -> fp32 logits
        fp32 logits -> fp32 logits_sum
        fp32 logits_sum * k_s (e8m0) -> fp32 index_score
    """
    return fp8_index_kernel(q.shape[2], q.shape[3])(q, q_s, k, k_s)


def convert_to_uint16(x):
    hval = T.Cast("float16", x)
    bits_uint = T.reinterpret("uint16", hval)
    bits_uint = T.if_then_else(x < 0, ~bits_uint & (0xFFFF),
                               bits_uint | (0x8000))
    return bits_uint >> 8


def convert_to_uint32(x):
    bits_uint = T.reinterpret("uint32", x)
    bits_uint = T.if_then_else(
        x < 0,
        ~bits_uint & T.Cast("uint32", (0xFFFFFFFF)),
        bits_uint | T.Cast("uint32", (0x80000000)),
    )
    return bits_uint


@tilelang.jit(pass_configs=pass_configs)
def tl_topk_impl(topk, in_dtype="float32", out_dtype="int32"):
    batch = T.symbolic("batch")
    seq_len = T.symbolic("seq_len")
    RADIX = 1 << 8
    BLOCK_SIZE = 1024
    # assume the threshold bucket size after first pass is less than 4K
    SMEM_INPUT_SIZE = 4096

    @T.prim_func
    def tl_topk_kernel(
            input: T.Tensor[(batch, seq_len), in_dtype],
            index: T.Tensor[(batch, topk), out_dtype],
            starts: T.Tensor[(batch), "int32"],  # noqa: F821
            ends: T.Tensor[(batch), "int32"],  # noqa: F821
    ):
        with T.Kernel(batch, threads=BLOCK_SIZE) as (bx):
            tx = T.get_thread_binding()

            s_threshold_bin_id = T.alloc_shared([1], "int32")
            s_histogram = T.alloc_shared([RADIX + 1], "int32")
            s_num_input = T.alloc_shared([2], "int32")
            s_input_idx = T.alloc_shared([2, SMEM_INPUT_SIZE], "int32")

            l_threshold_bin_id = T.alloc_var("int32")
            l_new_topk = T.alloc_var("int32")
            l_num_input = T.alloc_var("int32")
            l_bin_id32 = T.alloc_var("int32")
            l_val = T.alloc_var("int32")
            l_start_pos = T.alloc_var("int32")
            l_start_idx = T.alloc_var("int32")
            l_end_idx = T.alloc_var("int32")
            l_out_pos = T.alloc_var("int32")

            l_new_topk = topk
            l_start_idx = starts[bx]
            l_end_idx = ends[bx]

            # stage 1: use 8bit to do quick topk
            T.fill(s_histogram, 0)
            T.fill(s_num_input[0], 0)

            T.sync_threads()
            for s in T.serial(T.ceildiv(seq_len, BLOCK_SIZE)):
                input_idx = s * BLOCK_SIZE + tx
                if (input_idx < l_end_idx and input_idx >= l_start_idx
                        and input_idx < seq_len):
                    inval_int16 = convert_to_uint16(input[bx, input_idx])
                    T.atomic_add(s_histogram[inval_int16], 1)
            T.sync_threads()

            # cumsum
            if tx < RADIX:
                for i in T.serial(8):
                    offset = 1 << i
                    T.sync_threads(3, RADIX)
                    if tx < RADIX - offset:
                        l_val = s_histogram[tx] + s_histogram[tx + offset]
                    T.sync_threads(3, RADIX)
                    if tx < RADIX - offset:
                        s_histogram[tx] = l_val

                # find threshold bin id
                T.sync_threads(3, RADIX)
                if s_histogram[tx] > l_new_topk and s_histogram[
                        tx + 1] <= l_new_topk:
                    s_threshold_bin_id[0] = tx
            T.sync_threads()
            l_threshold_bin_id = s_threshold_bin_id[0]
            l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
            T.sync_threads()

            # collect all elements with exponent ≥ threshold
            for s in T.serial(T.ceildiv(seq_len, BLOCK_SIZE)):
                T.sync_threads()
                input_idx = s * BLOCK_SIZE + tx
                if (input_idx < l_end_idx and input_idx >= l_start_idx
                        and input_idx < seq_len):
                    bin_id = convert_to_uint16(input[bx, input_idx])
                    l_bin_id32 = T.Cast("int32", bin_id)
                    if l_bin_id32 > l_threshold_bin_id:
                        # need a pos = T.atomic_add(s_histogram[bin_id32+1], 1)
                        pos = T.atomic_add(s_histogram[l_bin_id32 + 1],
                                           1,
                                           return_prev=True)
                        index[bx, pos] = input_idx

                    elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                        # pos = s_num_input[0]
                        pos = T.atomic_add(s_num_input[0], 1, return_prev=True)
                        s_input_idx[0, pos] = input_idx

            # stage 2: tail pass
            for round in T.serial(4):
                if l_new_topk <= 0:
                    T.loop_break()

                r_idx = round % 2
                l_start_pos = topk - l_new_topk

                T.sync_threads()
                T.fill(s_histogram, 0)
                if tx == 0:
                    s_num_input[r_idx ^ 1] = 0
                T.sync_threads()

                l_num_input = s_num_input[r_idx]
                for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                    if s * BLOCK_SIZE + tx < l_num_input:
                        l_bin_id32 = T.Cast(
                            "int32", ((convert_to_uint32(input[bx, s_input_idx[
                                r_idx, s * BLOCK_SIZE + tx]]) >>
                                       (24 - round * 8)) & 0xFF))
                        T.atomic_add(s_histogram[l_bin_id32], 1)
                T.sync_threads()
                # cumsum
                if tx < RADIX:
                    for i in T.serial(8):
                        offset = 1 << i
                        T.sync_threads(3, RADIX)
                        if tx < RADIX - offset:
                            l_val = s_histogram[tx] + s_histogram[tx + offset]
                        T.sync_threads(3, RADIX)
                        if tx < RADIX - offset:
                            s_histogram[tx] = l_val

                    # find threshold bin id
                    T.sync_threads(3, RADIX)
                    if s_histogram[tx] > l_new_topk and s_histogram[
                            tx + 1] <= l_new_topk:
                        s_threshold_bin_id[0] = tx
                T.sync_threads()

                l_threshold_bin_id = s_threshold_bin_id[0]
                l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
                T.sync_threads()

                for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                    T.sync_threads()
                    if s * BLOCK_SIZE + tx < l_num_input:
                        l_bin_id32 = T.Cast(
                            "int32", ((convert_to_uint32(input[bx, s_input_idx[
                                r_idx, s * BLOCK_SIZE + tx]]) >>
                                       (24 - round * 8)) & 0xFF))
                        if l_bin_id32 > l_threshold_bin_id:
                            pos = T.atomic_add(s_histogram[l_bin_id32 + 1],
                                               1,
                                               return_prev=True) + l_start_pos
                            index[bx, pos] = s_input_idx[r_idx,
                                                         s * BLOCK_SIZE + tx]
                        elif (l_bin_id32 == l_threshold_bin_id
                              and l_new_topk > 0):
                            if round == 3:
                                l_out_pos = T.atomic_add(
                                    s_histogram[l_bin_id32 + 1],
                                    1,
                                    return_prev=True) + l_start_pos
                                if l_out_pos < topk:
                                    index[bx,
                                          l_out_pos] = s_input_idx[r_idx, s *
                                                                   BLOCK_SIZE +
                                                                   tx]
                            else:
                                pos = T.atomic_add(s_num_input[r_idx ^ 1],
                                                   1,
                                                   return_prev=True)
                                s_input_idx[r_idx ^ 1,
                                            pos] = s_input_idx[r_idx,
                                                               s * BLOCK_SIZE +
                                                               tx]

    return tl_topk_kernel


def tl_topk(input, starts, ends, topk):
    batch, seq_len = input.shape
    indexes = torch.zeros(batch, topk, dtype=torch.int32, device=input.device)
    kernel = tl_topk_impl(topk)
    kernel(input, indexes, starts, ends)
    return indexes
