# SPDX-License-Identifier: Apache-2.0
"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm
See https://tridao.me/publications/flash2/flash2.pdf

Credits:
AMD Triton kernels team
OpenAI kernel team

Currently only the forward kernel is supported, and contains these features:

1) Fwd with causal masking
2) Arbitrary Q and KV sequence lengths
3) Arbitrary head sizes
4) Multi and grouped query attention
5) Variable sequence lengths
6) ALiBi and matrix bias
7) FP8 support

"""

from typing import Optional

import torch
import triton
import triton.language as tl

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

SUPPORTED_LAYOUTS = ['thd', 'bhsd', 'bshd']

default_eight_bit_dtype_triton = tl.float8e4b8
default_eight_bit_dtype_torch = current_platform.fp8_dtype()
default_float8_info = torch.finfo(default_eight_bit_dtype_torch)

FP8_MIN = triton.language.constexpr(default_float8_info.min)

# According to https://github.com/vllm-project/vllm/blob/main
#              /csrc/quantization/utils.cuh#L31,
# need to make the max for the uz datatype be 224.0 for accuracy reasons.
FP8_MAX = triton.language.constexpr(
    default_float8_info.max if default_eight_bit_dtype_torch !=
    torch.float8_e4m3fnuz else 224.0)


class MetaData:
    cu_seqlens_q = None
    cu_seqlens_k = None
    max_seqlens_q = 0
    max_seqlens_k = 0
    bias = None
    alibi_slopes = None
    causal = False
    num_contexts = 0
    varlen = False
    eight_bit = False
    layout = None
    return_encoded_softmax = False
    eight_bit_dtype_triton = default_eight_bit_dtype_triton
    eight_bit_dtype_torch = default_eight_bit_dtype_torch
    output_dtype = None

    # Note about layouts:
    #
    # thd - [num_tokens, num_heads, head_size]
    # bshd - [batch_size, seq_len, num_heads, head_size]
    # bhsd - [batch_size, num_heads, seq_len, head_size]
    #
    # This is for each tensor, all tensors must have same layout.
    # Q can have num_heads and seq_len differ from  from K and V,
    # however K and V must agree on this.
    #
    # Notes about varlen and bias:
    # Only one or the other is implemented, meaning can't combine
    # both varlen and bias right now.
    #
    # Note about quantization:
    # Only 8-bit quantization supported (for now) and specifically fp8.
    # Scales must be tensors.
    # o_scale: This is 'output scaling', but comes from parameter called
    # 'input_scale', this is applied to the output from the kernel.
    # o_scale should be None if none of the other quantization parameters
    # are used.
    #
    # NOTE: Object is in a tentatively good state after initialized, however,
    # to verify, call check_args(q,k,v,o) where o is the output tensor.
    def __init__(
        self,
        sm_scale=1.0,
        layout=None,  # layout can be 'bshd', 'bhsd', or 'thd'
        output_dtype=None,
        max_seqlens_q=0,
        max_seqlens_k=0,
        # varlen params
        cu_seqlens_q=None,  # only 'thd' layout supported for varlen 
        cu_seqlens_k=None,
        # quant params
        q_descale=None,
        k_descale=None,
        v_descale=None,
        p_scale=None,
        o_scale=None,
        # bias params
        bias=None,  # varlen not implemented for bias
        seqlen_q=None,
        seqlen_k=None,
        # alibi params
        alibi_slopes=None,
        alibi_batch=None,
        alibi_nheads=None,
        # causal
        causal=None,
    ):
        self.sm_scale = sm_scale
        self.output_dtype = output_dtype
        self.max_seqlens_q = max_seqlens_q
        self.max_seqlens_k = max_seqlens_k
        self.layout = layout
        if cu_seqlens_q is not None or cu_seqlens_k is not None:
            assert cu_seqlens_q is not None and cu_seqlens_k is not None
            assert layout is None or layout not in [
                'bshd', 'bhsd'
            ], "Varlen only implemented for thd layout"
            self.set_varlen_params(cu_seqlens_q, cu_seqlens_k)
        quant_params = [q_descale, k_descale, v_descale, p_scale, o_scale]
        if any(x is not None for x in quant_params):
            p_descale = 1.0 / p_scale if p_scale is not None else None
            self.set_eight_bit_params(q_descale, k_descale, v_descale, p_scale,
                                      p_descale, o_scale)
        if bias is not None:
            self.need_bias(bias, seqlen_q, seqlen_k)
        if alibi_slopes is not None:
            self.need_alibi(alibi_slopes, alibi_batch, alibi_nheads)
        if causal is not None and causal:
            self.need_causal()

    def set_varlen_params(self, cu_seqlens_q, cu_seqlens_k):
        self.varlen = True
        self.layout = 'thd'
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k
        # Without "varlen", there should still be one sequence.
        assert len(cu_seqlens_q) >= 2
        assert len(cu_seqlens_q) == len(cu_seqlens_k)
        self.num_contexts = len(cu_seqlens_q) - 1
        for i in range(0, self.num_contexts):
            self.max_seqlens_q = max(
                cu_seqlens_q[i + 1].item() - cu_seqlens_q[i].item(),
                self.max_seqlens_q)
            self.max_seqlens_k = max(
                cu_seqlens_k[i + 1].item() - cu_seqlens_k[i].item(),
                self.max_seqlens_k)

    def set_eight_bit_params(self, q_descale, k_descale, v_descale, p_scale,
                             p_descale, o_scale):
        self.eight_bit = True
        self.q_descale = q_descale
        self.k_descale = k_descale
        self.v_descale = v_descale
        self.p_scale = p_scale
        self.p_descale = p_descale
        self.o_scale = o_scale
        self.use_p_scale = (p_scale is not None) and (
            p_descale is not None) and (v_descale is not None)
        self.eight_bit_kv = ((q_descale is None) and (k_descale is not None)
                             and (v_descale is not None))
        self.eight_bit_dtype_torch = default_eight_bit_dtype_torch

    def need_bias(self, bias, seqlen_q, seqlen_k):
        assert bias is not None
        assert bias.is_cuda
        assert bias.dim() == 4
        assert bias.shape[0] == 1
        assert bias.shape[2:] == (seqlen_q, seqlen_k)
        self.bias = bias

    def need_alibi(self, alibi_slopes, batch, nheads):
        assert alibi_slopes.is_cuda
        assert alibi_slopes.dim() == 2
        assert alibi_slopes.shape[0] == batch
        assert alibi_slopes.shape[1] == nheads
        self.alibi_slopes = alibi_slopes

    def need_causal(self):
        self.causal = True

    def check_args(self, q, k, v, o):
        assert q.dim() == k.dim() and q.dim() == v.dim()

        batch, nheads_q, nheads_k, head_size = get_shape_from_layout(
            q, k, self)
        if self.varlen:
            assert q.dim() == 3
            assert self.cu_seqlens_q is not None
            assert self.cu_seqlens_k is not None
            assert len(self.cu_seqlens_q) == len(self.cu_seqlens_k)
            # TODO: Remove once bias is supported with varlen
            assert self.bias is None
            assert not self.return_encoded_softmax
        else:
            assert q.dim() == 4
            assert self.max_seqlens_q > 0 and self.max_seqlens_k > 0
            assert self.cu_seqlens_q is None and self.cu_seqlens_k is None
        assert k.shape == v.shape
        assert q.shape[-1] == k.shape[-1] and q.shape[-1] == v.shape[-1]
        # TODO: Change assert if we support qkl f8 and v f16
        if self.eight_bit:
            if self.eight_bit_kv:
                assert (v.dtype == k.dtype
                        and k.dtype == self.eight_bit_dtype_torch)
                assert q.dtype != k.dtype
                assert (self.v_descale is not None) and (self.k_descale
                                                         is not None)
            else:
                assert (q.dtype == k.dtype and q.dtype == v.dtype
                        and q.dtype == self.eight_bit_dtype_torch)
                assert (self.q_descale
                        is not None) and (self.k_descale
                                          is not None) and (self.v_descale
                                                            is not None)
                if self.use_p_scale:
                    assert (self.p_scale is not None) and (self.p_descale
                                                           is not None)
        else:
            assert (q.dtype == k.dtype) and (q.dtype == v.dtype)
        assert head_size <= 256
        assert o.shape == q.shape
        assert (nheads_q % nheads_k) == 0
        assert self.layout is not None
        assert self.layout == 'thd' or not self.varlen


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def max_fn(x, y):
    return tl.math.max(x, y)


# Convenience function to load with optional boundary checks.
# "First" is the major dim, "second" is the minor dim.
@triton.jit
def masked_load(ptrs, offset_first, offset_second, boundary_first,
                boundary_second):
    if offset_first is not None and offset_second is not None:
        mask = (offset_first[:, None] < boundary_first) & \
               (offset_second[None, :] < boundary_second)
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_first is not None:
        mask = offset_first[:, None] < boundary_first
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_second is not None:
        mask = offset_second[None, :] < boundary_second
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    else:
        tensor = tl.load(ptrs)
    return tensor


@triton.jit
def compute_alibi_block(alibi_slope,
                        seqlen_q,
                        seqlen_k,
                        offs_m,
                        offs_n,
                        transpose=False):
    # when seqlen_k and seqlen_q are different we want the diagonal to stick to
    # the bottom right of the attention matrix
    # for casual mask we want something like this where (1 is kept and 0 is
    # masked)
    # seqlen_q = 2 and seqlen_k = 5
    #   1 1 1 1 0
    #   1 1 1 1 1
    # seqlen_q = 5 and seqlen_k = 2
    #        0 0
    #        0 0
    #        0 0
    #        1 0
    #        1 1
    # for alibi the diagonal is 0 indicating no penalty for attending to that
    # spot and increasing penalty for attending further from the diagonal
    # e.g. alibi_slope = 1, seqlen_q = 2, seqlen_k = 5,
    # offs_m = [0, 1, 2, 3], offs_n = [0, 1, 2, 3, 4], transpose = False
    # 1. offs_m[:,None] = [[0],
    #                       [1],
    # 2. offs_m[:,None] + seqlen_k = [[5],
    #                                  [6],
    # 3. offs_m[:,None] + seqlen_k - seqlen_q = [[3],
    #                                             [4],
    # 4. offs_m[:,None] + seqlen_k - seqlen_q - offs_n[None,:] =
    # [[3], - [[0, 1, 2, 3, 4]] =  [[ 3, 2, 1, 0,-1], [4], [ 4, 3, 2, 1, 0]]
    # 5. -1 * alibi_slope * tl.abs(relative_pos_block) = [[ -3, -2, -1, 0,-1],
    #                                                    [ -4, -3, -2, -1, 0]],
    relative_pos_block = (offs_m[:, None] + seqlen_k - seqlen_q -
                          offs_n[None, :])
    alibi_block = -1 * alibi_slope * tl.abs(relative_pos_block)
    if transpose:
        return alibi_block.T
    else:
        return alibi_block


def compute_alibi_tensor(alibi_slopes, seqlen_q, seqlen_k):
    q_idx = torch.arange(seqlen_q, dtype=torch.int32,
                         device="cuda").unsqueeze(-1)  # (N_CTX_Q, 1)
    k_idx = torch.arange(seqlen_k, dtype=torch.int32,
                         device="cuda").unsqueeze(0)  # (1, N_CTX_K)
    relative_pos = torch.abs(q_idx + seqlen_k - seqlen_q -
                             k_idx)  # (N_CTX_Q, N_CTX_K)
    return -1 * alibi_slopes.unsqueeze(-1).unsqueeze(
        -1) * relative_pos  # (Z, H, N_CTX_Q, N_CTX_K)


@triton.jit
def quant_fp8(x, scale):
    x *= scale
    x = tl.clamp(x, FP8_MIN, FP8_MAX)
    return x


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    k_ptrs,
    v_ptrs,
    bias_ptrs,
    stride_kn,
    stride_vk,
    stride_bn,
    start_m,
    actual_seqlen_k,
    actual_seqlen_q,
    philox_seed,
    batch_philox_offset,
    encoded_sm_ptrs,
    block_min,
    block_max,
    offs_n_causal,
    masked_blocks,
    n_extra_tokens,
    alibi_slope,
    q_descale,
    k_descale,
    v_descale,
    p_scale,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    OFFS_M: tl.constexpr,
    OFFS_N: tl.constexpr,
    SHOULD_PRE_LOAD_V: tl.constexpr,
    SHOULD_MASK_STEPS: tl.constexpr,
    SHOULD_RETURN_ENCODED_SOFTMAX: tl.constexpr,
    USE_PADDED_HEAD: tl.constexpr,
    IS_ACTUAL_BLOCK_DMODEL: tl.constexpr,
    QK_SCALE: tl.constexpr,
    IS_EIGHT_BIT_GEMM: tl.constexpr,
    USE_P_SCALE: tl.constexpr,
    IS_EIGHT_BIT_KV: tl.constexpr,
    QUANT_DTYPE: tl.constexpr = default_eight_bit_dtype_triton,
):

    # loop over k, v, and update accumulator
    for start_n in range(block_min, block_max, BLOCK_N):
        # For padded blocks, we will overrun the tensor size if
        # we load all BLOCK_N. For others, the blocks are all within range.
        k_offs_n = start_n + tl.arange(0,
                                       BLOCK_N) if SHOULD_MASK_STEPS else None
        k_offs_k = None if not USE_PADDED_HEAD else tl.arange(0, BLOCK_DMODEL)
        k = masked_load(k_ptrs, k_offs_k, k_offs_n, IS_ACTUAL_BLOCK_DMODEL,
                        actual_seqlen_k)
        if SHOULD_PRE_LOAD_V:
            # We can use the same offsets as k, just with dims transposed.
            v = masked_load(v_ptrs, k_offs_n, k_offs_k, actual_seqlen_k,
                            IS_ACTUAL_BLOCK_DMODEL)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # We start from end of seqlen_k so only the first iteration would need
        # to be checked for padding if it is not a multiple of block_n
        # TODO: This can be optimized to only be true for the padded block.
        if SHOULD_MASK_STEPS:  # noqa: SIM102
            # If this is the last block / iteration, we want to
            # mask if the sequence length is not a multiple of block size
            # a solution is to always do BLOCK_M // BLOCK_N + 1 steps if not
            # is_modulo_mn. last step might get wasted but that is okay.
            # check if this masking works for that case.
            if (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0):
                boundary_m = tl.full([BLOCK_M],
                                     actual_seqlen_k,
                                     dtype=tl.int32)
                size_n = start_n + OFFS_N[None, :]
                mask = size_n < boundary_m[:, None]
                qk = tl.where(mask, qk, float("-inf"))
        if IS_CAUSAL:
            causal_boundary = start_n + offs_n_causal
            causal_mask = OFFS_M[:, None] >= causal_boundary[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))

        # -- compute qk ----
        if IS_EIGHT_BIT_GEMM:
            qk += ((((tl.dot(q, k).to(tl.float32) * q_descale)) * k_descale) *
                   QK_SCALE)
        else:
            if IS_EIGHT_BIT_KV:
                k = (k * k_descale).to(q.type.element_ty)
            qk += (tl.dot(q, k) * QK_SCALE)

        if bias_ptrs is not None:
            bias_offs_n = start_n + tl.arange(
                0, BLOCK_N) if SHOULD_MASK_STEPS else None
            bias = masked_load(bias_ptrs, OFFS_M, bias_offs_n, actual_seqlen_q,
                               actual_seqlen_k)
            # While bias is added after multiplying qk with sm_scale,
            # our optimization to use 2^x instead of e^x results in an
            # additional scale factor of log2(e) which we must also multiply
            # the bias with.
            qk += (bias * 1.44269504089)

        if alibi_slope is not None:
            # Compute the global position of each token within the sequence
            global_m_positions = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            global_n_positions = start_n + tl.arange(0, BLOCK_N)
            alibi_block = compute_alibi_block(alibi_slope, actual_seqlen_q,
                                              actual_seqlen_k,
                                              global_m_positions,
                                              global_n_positions)
            qk += (alibi_block * 1.44269504089)  # scale factor of log2(e)

        # softmax
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)

        # CAVEAT: Must update l_ij before applying dropout
        l_ij = tl.sum(p, 1)
        if SHOULD_RETURN_ENCODED_SOFTMAX:
            tl.store(encoded_sm_ptrs, p.to(encoded_sm_ptrs.type.element_ty))
        # -- update output accumulator --
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        if not SHOULD_PRE_LOAD_V:
            v = masked_load(v_ptrs, k_offs_n, k_offs_k, actual_seqlen_k,
                            IS_ACTUAL_BLOCK_DMODEL)
        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij

        if IS_EIGHT_BIT_GEMM:
            if USE_P_SCALE:
                p = quant_fp8(p, p_scale).to(QUANT_DTYPE)
                acc += tl.dot(p, v)
            else:
                # v is in eight_bit but p is not, we want the gemm in p's type
                acc += tl.dot(p, v.to(p.type.element_ty))
        else:
            if IS_EIGHT_BIT_KV:
                v = (v * v_descale).to(p.type.element_ty)
            acc += tl.dot(p.to(v.type.element_ty), v)

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
        if bias_ptrs is not None:
            bias_ptrs += BLOCK_N * stride_bn
        if SHOULD_RETURN_ENCODED_SOFTMAX:
            encoded_sm_ptrs += BLOCK_N
    return acc, l_i, m_i


def get_cdna_autotune_configs():
    return [
        triton.Config(
            {
                'BLOCK_M': 128,
                'BLOCK_N': 128,
                'waves_per_eu': 2,
                'SHOULD_PRE_LOAD_V': False,
                'GRID_CU_MULTIP': 2
            },
            num_stages=1,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 128,
                'BLOCK_N': 64,
                'waves_per_eu': 2,
                'SHOULD_PRE_LOAD_V': False,
                'GRID_CU_MULTIP': 2
            },
            num_stages=1,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 128,
                'BLOCK_N': 64,
                'waves_per_eu': 3,
                'SHOULD_PRE_LOAD_V': False,
                'GRID_CU_MULTIP': 2
            },
            num_stages=1,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 128,
                'BLOCK_N': 64,
                'waves_per_eu': 1,
                'SHOULD_PRE_LOAD_V': False,
                'GRID_CU_MULTIP': 2
            },
            num_stages=1,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 128,
                'BLOCK_N': 32,
                'waves_per_eu': 2,
                'SHOULD_PRE_LOAD_V': False,
                'GRID_CU_MULTIP': 2
            },
            num_stages=1,
            num_warps=4),
    ], [
        'IS_CAUSAL', 'MAX_SEQLENS_Q', 'MAX_SEQLENS_K',
        'IS_ACTUAL_BLOCK_DMODEL', 'VARLEN', 'HQ', 'HK'
    ]


def get_rdna_autotune_configs():
    return [
        triton.Config(
            {
                'BLOCK_M': 32,
                'BLOCK_N': 32,
                'waves_per_eu': 4,
                'SHOULD_PRE_LOAD_V': False,
                'GRID_CU_MULTIP': 2
            },
            num_stages=1,
            num_warps=2),
        triton.Config(
            {
                'BLOCK_M': 32,
                'BLOCK_N': 32,
                'waves_per_eu': 2,
                'SHOULD_PRE_LOAD_V': False,
                'GRID_CU_MULTIP': 2
            },
            num_stages=1,
            num_warps=2),
        triton.Config(
            {
                'BLOCK_M': 32,
                'BLOCK_N': 16,
                'waves_per_eu': 4,
                'SHOULD_PRE_LOAD_V': False,
                'GRID_CU_MULTIP': 2
            },
            num_stages=1,
            num_warps=2),
        triton.Config(
            {
                'BLOCK_M': 32,
                'BLOCK_N': 16,
                'waves_per_eu': 2,
                'SHOULD_PRE_LOAD_V': False,
                'GRID_CU_MULTIP': 2
            },
            num_stages=1,
            num_warps=2),
        triton.Config(
            {
                'BLOCK_M': 16,
                'BLOCK_N': 16,
                'waves_per_eu': 4,
                'SHOULD_PRE_LOAD_V': False,
                'GRID_CU_MULTIP': 2
            },
            num_stages=1,
            num_warps=2),
        triton.Config(
            {
                'BLOCK_M': 16,
                'BLOCK_N': 16,
                'waves_per_eu': 2,
                'SHOULD_PRE_LOAD_V': False,
                'GRID_CU_MULTIP': 2
            },
            num_stages=1,
            num_warps=2),
        # Fall-back config.
        triton.Config(
            {
                'BLOCK_M': 16,
                'BLOCK_N': 16,
                'waves_per_eu': 1,
                'SHOULD_PRE_LOAD_V': False,
                'GRID_CU_MULTIP': 2
            },
            num_stages=1,
            num_warps=2),
    ], [
        'IS_CAUSAL', 'MAX_SEQLENS_Q', 'MAX_SEQLENS_K',
        'IS_ACTUAL_BLOCK_DMODEL', 'VARLEN', 'HQ', 'HK'
    ]


def get_general_autotune_configs():
    return [
        triton.Config(
            {
                'BLOCK_M': 128,
                'BLOCK_N': 128,
                'SHOULD_PRE_LOAD_V': False,
                'GRID_CU_MULTIP': 2
            },
            num_stages=1,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 128,
                'BLOCK_N': 64,
                'SHOULD_PRE_LOAD_V': False,
                'GRID_CU_MULTIP': 2
            },
            num_stages=1,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 128,
                'BLOCK_N': 32,
                'SHOULD_PRE_LOAD_V': False,
                'GRID_CU_MULTIP': 2
            },
            num_stages=1,
            num_warps=4),
    ], [
        'IS_CAUSAL', 'MAX_SEQLENS_Q', 'MAX_SEQLENS_K',
        'IS_ACTUAL_BLOCK_DMODEL', 'VARLEN', 'HQ', 'HK'
    ]


def has_cdna_target():
    ROCM_CDNA_TARGETS = ["gfx940", "gfx941", "gfx942", "gfx90a", "gfx908"]
    return triton.runtime.driver.active.get_current_target(
    ).arch in ROCM_CDNA_TARGETS


def is_rocm_cdna():
    return current_platform.is_rocm() and has_cdna_target()


def get_autotune_configs():
    if is_rocm_cdna():
        return get_cdna_autotune_configs()
    elif current_platform.is_rocm():
        return get_rdna_autotune_configs()
    else:
        return get_general_autotune_configs()


autotune_configs, autotune_keys = get_autotune_configs()


@triton.autotune(
    configs=autotune_configs,
    key=autotune_keys,
    use_cuda_graph=True,
)
@triton.jit
def attn_fwd(
    Q,
    K,
    V,
    bias,
    SM_SCALE: tl.constexpr,
    L,
    Out,
    stride_qz: tl.int64,
    stride_qh: tl.int64,
    stride_qm: tl.int64,
    stride_qk: tl.int64,
    stride_kz: tl.int64,
    stride_kh: tl.int64,
    stride_kn: tl.int64,
    stride_kk: tl.int64,
    stride_vz: tl.int64,
    stride_vh: tl.int64,
    stride_vk: tl.int64,
    stride_vn: tl.int64,
    stride_oz: tl.int64,
    stride_oh: tl.int64,
    stride_om: tl.int64,
    stride_on: tl.int64,
    stride_bz: tl.int64,
    stride_bh: tl.int64,
    stride_bm: tl.int64,
    stride_bn: tl.int64,
    stride_az: tl.int64,
    stride_ah: tl.int64,
    q_descale_ptr,
    k_descale_ptr,
    p_scale_ptr,
    p_descale_ptr,
    o_descale_ptr,
    v_descale_ptr,
    q_descale_has_singleton: tl.constexpr,
    k_descale_has_singleton: tl.constexpr,
    p_descale_has_singleton: tl.constexpr,
    v_descale_has_singleton: tl.constexpr,
    cu_seqlens_q,
    cu_seqlens_k,
    philox_seed,
    NUM_CU: tl.constexpr,
    GRID_CU_MULTIP: tl.constexpr,
    B: tl.constexpr,
    philox_offset_base,
    encoded_softmax,
    alibi_slopes,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    IS_ACTUAL_BLOCK_DMODEL: tl.constexpr,
    MAX_SEQLENS_Q: tl.constexpr,
    MAX_SEQLENS_K: tl.constexpr,
    VARLEN: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SHOULD_PRE_LOAD_V: tl.constexpr,
    USE_BIAS: tl.constexpr,
    SHOULD_RETURN_ENCODED_SOFTMAX: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    IS_EIGHT_BIT: tl.constexpr,
    USE_P_SCALE: tl.constexpr,
    IS_EIGHT_BIT_KV: tl.constexpr,
    QUANT_DTYPE: tl.constexpr = default_eight_bit_dtype_triton,
):

    if o_descale_ptr is not None:
        o_descale = tl.load(o_descale_ptr)

    start_m: tl.int64 = tl.program_id(0)
    off_h_q: tl.int64 = tl.program_id(1)
    off_z: tl.int64 = tl.program_id(2)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_n = tl.arange(0, BLOCK_N).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_DMODEL).to(tl.int64)

    # as we can't have return statements inside while loop in Triton
    continue_condition = True

    if VARLEN:
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        # We have a one-size-fits-all grid in id(0). Some seqlens might be
        # too small for all start_m so for those we return early.
        if start_m * BLOCK_M > seqlen_q:
            continue_condition = False
            # return
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
    else:
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = MAX_SEQLENS_Q
        seqlen_k = MAX_SEQLENS_K

    if continue_condition:
        # Now we compute whether we need to exit early due to causal
        # masking. This is because for seqlen_q > seqlen_k, M rows of the
        # attn scores are completely masked, resulting in 0s written to the
        # output, and inf written to LSE. We don't need to do any GEMMs in
        # this case. This block of code determines what N is, and if this
        # WG is operating on those M rows.
        n_blocks = cdiv_fn(seqlen_k, BLOCK_N)
        if (IS_CAUSAL):
            # If seqlen_q == seqlen_k, the attn scores are a square matrix.
            # If seqlen_q != seqlen_k, attn scores are rectangular which
            # means the causal mask boundary is bottom right aligned, and
            # ends at either the top edge (seqlen_q < seqlen_k) or left
            # edge. This captures the decrease in n_blocks if we have a
            # rectangular attn matrix
            n_blocks_seqlen = cdiv_fn(
                (start_m + 1) * BLOCK_M + seqlen_k - seqlen_q, BLOCK_N)
            # This is what adjusts the block_max for the current WG, only
            # if IS_CAUSAL. Otherwise we want to always iterate through all
            # n_blocks
            n_blocks = min(n_blocks, n_blocks_seqlen)
            # If we have no blocks after adjusting for seqlen deltas, this
            # WG is part of the blocks that are all 0. We exit early.
            if n_blocks <= 0:
                o_offset = (Out + off_z * stride_oz + off_h_q * stride_oh +
                            cu_seqlens_q_start * stride_om)
                o_ptrs = (o_offset + offs_m[:, None] * stride_om +
                          offs_d[None, :] * stride_on)
                acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
                o_ptrs_mask = (offs_m[:, None] < seqlen_q).broadcast_to(
                    [BLOCK_M, BLOCK_DMODEL])
                # We still need to write 0s to the result
                tl.store(o_ptrs, acc, mask=o_ptrs_mask)
                # The tensor allocated for L is based on MAX_SEQLENS_Q as
                # that is statically known.
                l_ptrs = (L + off_z * HQ * MAX_SEQLENS_Q +
                          off_h_q * MAX_SEQLENS_Q + offs_m)
                # We store inf to LSE, not -inf because in the bwd pass,
                # we subtract this from qk which makes it -inf, such that
                # exp(qk - inf) = 0 for these masked blocks.
                l_value = tl.full([BLOCK_M],
                                  value=float("inf"),
                                  dtype=tl.float32)
                l_ptrs_mask = offs_m < MAX_SEQLENS_Q
                tl.store(l_ptrs, l_value, mask=l_ptrs_mask)
                # TODO: Should dropout and return encoded softmax be
                # handled here too?
                continue_condition = False
                # return

        if continue_condition:
            # If MQA / GQA, set the K and V head offsets appropriately.
            GROUP_SIZE: tl.constexpr = HQ // HK
            off_h_k = off_h_q // GROUP_SIZE if GROUP_SIZE != 1 else off_h_q
            n_extra_tokens = 0
            if seqlen_k < BLOCK_N:
                n_extra_tokens = BLOCK_N - seqlen_k
            elif seqlen_k % BLOCK_N:
                n_extra_tokens = seqlen_k % BLOCK_N
            USE_PADDED_HEAD: tl.constexpr = (IS_ACTUAL_BLOCK_DMODEL
                                             != BLOCK_DMODEL)

            # Compute pointers for all the tensors used in this kernel.
            q_offset = (Q + off_z * stride_qz + off_h_q * stride_qh +
                        cu_seqlens_q_start * stride_qm)
            q_ptrs = (q_offset + offs_m[:, None] * stride_qm +
                      offs_d[None, :] * stride_qk)
            k_offset = (K + off_z * stride_kz + off_h_k * stride_kh +
                        cu_seqlens_k_start * stride_kn)
            k_ptrs = (k_offset + offs_d[:, None] * stride_kk +
                      offs_n[None, :] * stride_kn)
            v_offset = (V + off_z * stride_vz + off_h_k * stride_vh +
                        cu_seqlens_k_start * stride_vk)
            v_ptrs = (v_offset + offs_n[:, None] * stride_vk +
                      offs_d[None, :] * stride_vn)
            # Compute pointers for all scale tensors used in this kernel.

            IS_EIGHT_BIT_GEMM: tl.constexpr = IS_EIGHT_BIT & (
                not IS_EIGHT_BIT_KV)
            if IS_EIGHT_BIT:
                if k_descale_has_singleton:
                    k_descale_ptrs = k_descale_ptr
                else:
                    k_descale_ptrs = k_descale_ptr + off_h_k

                if v_descale_has_singleton:
                    v_descale_ptrs = v_descale_ptr
                else:
                    v_descale_ptrs = v_descale_ptr + off_h_k

                if not IS_EIGHT_BIT_KV:
                    if q_descale_has_singleton:
                        q_descale_ptrs = q_descale_ptr
                    else:
                        q_descale_ptrs = q_descale_ptr + off_h_q
                if USE_P_SCALE:
                    if p_descale_has_singleton:
                        p_scale_ptrs = p_scale_ptr
                        p_descale_ptrs = p_descale_ptr
                    else:
                        p_scale_ptrs = p_scale_ptr + off_h_q
                        p_descale_ptrs = p_descale_ptr + off_h_q

            if USE_BIAS:
                bias_offset = off_h_q * stride_bh
                bias_ptrs = (bias + bias_offset + offs_m[:, None] * stride_bm +
                             offs_n[None, :] * stride_bn)
            else:
                bias_ptrs = None

            if USE_ALIBI:
                a_offset = off_z * stride_az + off_h_q * stride_ah
                alibi_slope = tl.load(alibi_slopes + a_offset)
            else:
                alibi_slope = None

            batch_philox_offset = 0
            # We can ask to return the dropout mask without doing any
            # dropout. In this case, we return an invalid pointer so
            # indicate the mask is not valid.
            if SHOULD_RETURN_ENCODED_SOFTMAX:
                encoded_sm_base = (encoded_softmax +
                                   off_h_q * seqlen_q * seqlen_k)
                encoded_sm_ptrs = (encoded_sm_base +
                                   offs_m[:, None] * seqlen_k +
                                   offs_n[None, :])
            else:
                encoded_sm_ptrs = None
            # initialize pointer to m and l
            m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
            l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
            acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
            # scale sm_scale by log_2(e) and use 2^x in the loop as we do
            # not have native e^x support in HW.
            QK_SCALE: tl.constexpr = SM_SCALE * 1.44269504089
            # Q is loaded once at the beginning and shared by all N blocks.
            q_ptrs_mask = offs_m[:, None] < seqlen_q
            if USE_PADDED_HEAD:
                q_ptrs_mask = q_ptrs_mask & (offs_d[None, :]
                                             < IS_ACTUAL_BLOCK_DMODEL)
            q = tl.load(q_ptrs, mask=q_ptrs_mask, other=0.0)

            if IS_EIGHT_BIT:
                k_descale = tl.load(k_descale_ptrs)
                v_descale = tl.load(v_descale_ptrs)
                q_descale = None if IS_EIGHT_BIT_KV else tl.load(
                    q_descale_ptrs)
                if USE_P_SCALE:
                    p_scale = tl.load(p_scale_ptrs)
                    p_descale = tl.load(p_descale_ptrs)
                else:
                    p_scale = None
                    p_descale = None
            else:
                q_descale = None
                k_descale = None
                v_descale = None
                p_scale = None
                p_descale = None
            # Here we compute how many full and masked blocks we have.
            padded_block_k = n_extra_tokens != 0
            is_modulo_mn = not padded_block_k and (seqlen_q % BLOCK_M == 0)
            if IS_CAUSAL:
                # There are always at least BLOCK_M // BLOCK_N masked
                # blocks.  Additionally there might be one more due to
                # dissimilar seqlens.
                masked_blocks = BLOCK_M // BLOCK_N + (not is_modulo_mn)
            else:
                # Padding on Q does not need to be masked in the FA loop.
                masked_blocks = padded_block_k
            # if IS_CAUSAL, not is_modulo_mn does not always result in an
            # additional block. In this case we might exceed n_blocks so
            # pick the min.
            masked_blocks = min(masked_blocks, n_blocks)
            n_full_blocks = n_blocks - masked_blocks
            block_min = 0
            block_max = n_blocks * BLOCK_N
            # Compute for full blocks. Here we set causal to false
            # regardless of its actual value because there is no masking.
            # Similarly we do not need padding.
            if n_full_blocks > 0:
                block_max = (n_blocks - masked_blocks) * BLOCK_N
                acc, l_i, m_i = _attn_fwd_inner(
                    acc,
                    l_i,
                    m_i,
                    q,
                    k_ptrs,
                    v_ptrs,
                    bias_ptrs,
                    stride_kn,
                    stride_vk,
                    stride_bn,
                    start_m,
                    seqlen_k,
                    seqlen_q,
                    philox_seed,
                    batch_philox_offset,
                    encoded_sm_ptrs,
                    # _, _, offs_n_causal, masked_blocks, n_extra_tokens, _
                    block_min,
                    block_max,
                    0,
                    0,
                    0,
                    alibi_slope,
                    q_descale,
                    k_descale,
                    v_descale,
                    p_scale,
                    # IS_CAUSAL, ....
                    False,
                    BLOCK_M,
                    BLOCK_DMODEL,
                    BLOCK_N,
                    offs_m,
                    offs_n,
                    # _, SHOULD_MASK_STEPS, ...
                    SHOULD_PRE_LOAD_V,
                    False,
                    SHOULD_RETURN_ENCODED_SOFTMAX,
                    USE_PADDED_HEAD,
                    IS_ACTUAL_BLOCK_DMODEL,
                    QK_SCALE,
                    IS_EIGHT_BIT_GEMM,
                    USE_P_SCALE,
                    IS_EIGHT_BIT_KV,
                    QUANT_DTYPE)
                block_min = block_max
                block_max = n_blocks * BLOCK_N

            tl.debug_barrier()
            # Remaining blocks, if any, are full / not masked.
            if (masked_blocks > 0):
                if IS_CAUSAL:
                    offs_n_causal = offs_n + (seqlen_q - seqlen_k)
                else:
                    offs_n_causal = 0
                k_ptrs += n_full_blocks * BLOCK_N * stride_kn
                v_ptrs += n_full_blocks * BLOCK_N * stride_vk
                if USE_BIAS:
                    bias_ptrs += n_full_blocks * BLOCK_N * stride_bn
                if SHOULD_RETURN_ENCODED_SOFTMAX:
                    encoded_sm_ptrs += n_full_blocks * BLOCK_N
                acc, l_i, m_i = _attn_fwd_inner(
                    acc,
                    l_i,
                    m_i,
                    q,
                    k_ptrs,
                    v_ptrs,
                    bias_ptrs,
                    stride_kn,
                    stride_vk,
                    stride_bn,
                    start_m,
                    seqlen_k,
                    seqlen_q,
                    philox_seed,
                    batch_philox_offset,
                    encoded_sm_ptrs,
                    block_min,
                    block_max,
                    offs_n_causal,
                    masked_blocks,
                    n_extra_tokens,
                    alibi_slope,
                    q_descale,
                    k_descale,
                    v_descale,
                    p_scale,
                    IS_CAUSAL,
                    BLOCK_M,
                    BLOCK_DMODEL,
                    BLOCK_N,
                    offs_m,
                    offs_n,
                    # _, SHOULD_MASK_STEPS, ...
                    SHOULD_PRE_LOAD_V,
                    True,
                    SHOULD_RETURN_ENCODED_SOFTMAX,
                    USE_PADDED_HEAD,
                    IS_ACTUAL_BLOCK_DMODEL,
                    QK_SCALE,
                    IS_EIGHT_BIT_GEMM,
                    USE_P_SCALE,
                    IS_EIGHT_BIT_KV,
                    QUANT_DTYPE)

            if IS_EIGHT_BIT and not IS_EIGHT_BIT_KV:
                if USE_P_SCALE:
                    acc *= p_descale
                acc *= v_descale

            # epilogue
            # This helps the compiler do Newton Raphson on l_i vs on acc
            # which is much larger.
            l_recip = 1 / l_i[:, None]
            acc = acc * l_recip

            # If seqlen_q > seqlen_k but the delta is not a multiple of
            # BLOCK_M, then we have one block with a row of all NaNs which
            # come from computing softmax over a row of all
            # -infs (-inf - inf = NaN). We check for that here and store 0s
            # where there are NaNs as these rows should've been zeroed out.
            end_m_idx = (start_m + 1) * BLOCK_M
            start_m_idx = start_m * BLOCK_M
            causal_start_idx = seqlen_q - seqlen_k
            if IS_EIGHT_BIT and not IS_EIGHT_BIT_KV:  # noqa: SIM102
                if o_descale_ptr is not None:
                    acc = quant_fp8(acc, o_descale)

            acc = acc.to(Out.type.element_ty)
            if IS_CAUSAL:  # noqa: SIM102
                if (causal_start_idx > start_m_idx
                        and causal_start_idx < end_m_idx):
                    out_mask_boundary = tl.full((BLOCK_DMODEL, ),
                                                causal_start_idx,
                                                dtype=tl.int32)
                    mask_m_offsets = start_m_idx + tl.arange(0, BLOCK_M)
                    out_ptrs_mask = (mask_m_offsets[:, None]
                                     >= out_mask_boundary[None, :])
                    z = tl.zeros((1, ), tl.float32)
                    acc = tl.where(out_ptrs_mask, acc,
                                   z.to(acc.type.element_ty))
            # write back LSE
            l_ptrs = (L + off_z * HQ * MAX_SEQLENS_Q +
                      off_h_q * MAX_SEQLENS_Q + offs_m)
            # If seqlen_q not multiple of BLOCK_M, we need to mask out the
            # last few rows. This is only true for the last M block.
            # For others, overflow_size will be -ve
            overflow_size = end_m_idx - seqlen_q
            if overflow_size > 0:
                boundary = tl.full((BLOCK_M, ),
                                   BLOCK_M - overflow_size,
                                   dtype=tl.int32)
                l_ptrs_mask = tl.arange(0, BLOCK_M) < boundary
                tl.store(l_ptrs, m_i + tl.math.log2(l_i), mask=l_ptrs_mask)
            else:
                tl.store(l_ptrs, m_i + tl.math.log2(l_i))

            # write back O
            o_offset = (Out + off_z * stride_oz + off_h_q * stride_oh +
                        cu_seqlens_q_start * stride_om)
            o_ptrs = (o_offset + offs_m[:, None] * stride_om +
                      offs_d[None, :] * stride_on)
            o_ptrs_mask = tl.full([BLOCK_M, BLOCK_DMODEL], 1, dtype=tl.int1)
            if overflow_size > 0:
                o_ptrs_mask = o_ptrs_mask & (offs_m[:, None] < seqlen_q)
            if USE_PADDED_HEAD:
                o_ptrs_mask = o_ptrs_mask & (offs_d[None, :]
                                             < IS_ACTUAL_BLOCK_DMODEL)
            tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=o_ptrs_mask)


def get_shape_from_layout(q, k, metadata):
    assert metadata.layout in SUPPORTED_LAYOUTS, "Got unsupported layout."

    if metadata.layout == 'thd':
        nheads_q, nheads_k = q.shape[1], k.shape[1]
        head_size = q.shape[-1]
        batch = metadata.num_contexts
    elif metadata.layout == 'bhsd':
        batch, nheads_q, _, head_size = q.shape
        nheads_k = k.shape[1]
    elif metadata.layout == 'bshd':
        batch, _, nheads_q, head_size = q.shape
        nheads_k = k.shape[2]
    return batch, nheads_q, nheads_k, head_size


def get_strides_from_layout(q, k, v, o, metadata):
    assert metadata.layout in SUPPORTED_LAYOUTS, "Got unsupported layout."

    STRIDE_PERMUTATIONS = {
        'thd': (None, 1, 0, 2),
        'bhsd': (0, 1, 2, 3),
        'bshd': (0, 2, 1, 3),
    }

    perm = STRIDE_PERMUTATIONS[metadata.layout]
    stride = lambda x, p: (0 if p is None else x.stride(p))
    strides = lambda x: (stride(x, p) for p in perm)

    return tuple(strides(x) for x in [q, k, v, o])


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, o, metadata: MetaData):
        # NOTE: a large bias tensor leads to overflow during pointer arithmetic
        if (metadata.bias is not None):
            assert (metadata.bias.numel() < 2**31)

        if o is None:
            if metadata.eight_bit:
                o = torch.empty_like(
                    q,
                    dtype=metadata.output_dtype if metadata.output_dtype
                    is not None else metadata.eight_bit_dtype_torch)
            else:
                o = torch.empty_like(q, dtype=q.dtype)

        metadata.check_args(q, k, v, o)

        batch, nheads_q, nheads_k, head_size = get_shape_from_layout(
            q, k, metadata)
        q_strides, k_strides, v_strides, o_strides = get_strides_from_layout(
            q, k, v, o, metadata)

        # Get closest power of 2 over or equal to 32.
        padded_d_model = 1 << (head_size - 1).bit_length()
        # Smallest head_dim supported is 16. If smaller, the tile in the
        # kernel is padded - there is no padding in memory for any dims.
        padded_d_model = max(padded_d_model, 16)

        # encoded_softmax is used to validate dropout behavior vs the
        # PyTorch SDPA math backend reference.  We zero this out to give a
        # consistent starting point and then populate it with the output of
        # softmax with the sign bit set according to the dropout mask.
        # The resulting return allows this mask to be fed into the reference
        # implementation for testing only.  This return holds no useful output
        # aside from debugging.
        if metadata.return_encoded_softmax:
            encoded_softmax = torch.zeros(
                (q.shape[0], q.shape[1], q.shape[2], k.shape[2]),
                device=q.device,
                dtype=torch.float32)
        else:
            encoded_softmax = None

        M = torch.empty((batch, nheads_q, metadata.max_seqlens_q),
                        device=q.device,
                        dtype=torch.float32)

        # Seed the RNG so we get reproducible results for testing.
        philox_seed = 0x1BF52
        philox_offset = 0x1D4B42

        if metadata.bias is not None:
            bias_strides = (metadata.bias.stride(0), metadata.bias.stride(1),
                            metadata.bias.stride(2), metadata.bias.stride(3))
        else:
            bias_strides = (0, 0, 0, 0)

        if metadata.alibi_slopes is not None:
            alibi_strides = (metadata.alibi_slopes.stride(0),
                             metadata.alibi_slopes.stride(1))
        else:
            alibi_strides = (0, 0)

        if metadata.eight_bit:
            q_descale, k_descale, p_scale, p_descale, v_descale, o_scale = (
                metadata.q_descale, metadata.k_descale, metadata.p_scale,
                metadata.p_descale, metadata.v_descale, metadata.o_scale)
            o_descale = 1.0 / o_scale if o_scale is not None else None
        else:
            q_descale = k_descale = p_scale = None
            p_descale = v_descale = o_descale = None

        # number of compute units available
        NUM_CU = torch.cuda.get_device_properties("cuda").multi_processor_count

        grid = lambda META: (triton.cdiv(metadata.max_seqlens_q, META[
            'BLOCK_M']), nheads_q, batch)

        attn_fwd[grid](
            q,
            k,
            v,
            metadata.bias,
            metadata.sm_scale,
            M,
            o,
            *q_strides,
            *k_strides,
            *v_strides,
            *o_strides,
            *bias_strides,
            *alibi_strides,
            q_descale,
            k_descale,
            p_scale,
            p_descale,
            o_descale,
            v_descale,
            q_descale.numel() == 1 if q_descale is not None else False,
            k_descale.numel() == 1 if k_descale is not None else False,
            p_descale.numel() == 1 if p_descale is not None else False,
            v_descale.numel() == 1 if v_descale is not None else False,
            metadata.cu_seqlens_q,
            metadata.cu_seqlens_k,
            philox_seed=philox_seed,
            philox_offset_base=philox_offset,
            encoded_softmax=encoded_softmax,
            alibi_slopes=metadata.alibi_slopes,
            HQ=nheads_q,
            HK=nheads_k,
            IS_ACTUAL_BLOCK_DMODEL=head_size,
            MAX_SEQLENS_Q=metadata.max_seqlens_q,
            MAX_SEQLENS_K=metadata.max_seqlens_k,
            IS_CAUSAL=metadata.causal,
            VARLEN=metadata.varlen,
            BLOCK_DMODEL=padded_d_model,
            USE_BIAS=metadata.bias is not None,
            USE_ALIBI=metadata.alibi_slopes is not None,
            SHOULD_RETURN_ENCODED_SOFTMAX=metadata.return_encoded_softmax,
            IS_EIGHT_BIT=metadata.eight_bit,
            USE_P_SCALE=metadata.eight_bit and metadata.use_p_scale,
            IS_EIGHT_BIT_KV=metadata.eight_bit and metadata.eight_bit_kv,
            NUM_CU=NUM_CU,
            B=batch,
            QUANT_DTYPE=metadata.eight_bit_dtype_triton)

        ctx.grid = grid
        ctx.sm_scale = metadata.sm_scale
        ctx.BLOCK_DMODEL = head_size
        ctx.causal = metadata.causal
        ctx.alibi_slopes = metadata.alibi_slopes
        ctx.philox_seed = philox_seed
        ctx.philox_offset = philox_offset
        ctx.encoded_softmax = encoded_softmax
        ctx.return_encoded_softmax = metadata.return_encoded_softmax
        return o, encoded_softmax


triton_attention_rocm = _attention.apply


def scale_fp8(t, scale=None):
    t_scaled, scale_out = ops.scaled_fp8_quant(t.reshape(-1, t.shape[-1]),
                                               scale)
    return t_scaled.reshape(t.shape), scale_out


def maybe_quantize_fp8(t, scale):
    eight_bit_dtype = current_platform.fp8_dtype()
    if t.dtype != eight_bit_dtype:
        t, _ = scale_fp8(t, scale)
    return t


def check_and_maybe_quantize_qkv(q, k, v, fp8_scales):
    (q_scale, k_scale, v_scale, p_scale) = fp8_scales

    q = maybe_quantize_fp8(q, q_scale)
    k = maybe_quantize_fp8(k, k_scale)
    v = maybe_quantize_fp8(v, v_scale)

    return q, k, v


# query   - [num_tokens, num_heads, head_size]
# key     - [num_tokens, num_kv_heads, head_size]
# value   - [num_tokens, num_kv_heads, head_size
# output  - [num_tokens, num_heads, head_size]
def triton_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlens_q: int,
    max_seqlens_k: int,
    causal: bool = False,
    sm_scale: float = 1.0,
    bias: Optional[torch.Tensor] = None,
    fp8_scales: Optional[tuple[float, ...]] = None,
    input_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if fp8_scales is not None:
        q_descale, k_descale, v_descale, p_scale = fp8_scales
    else:
        q_descale = k_descale = v_descale = p_scale = None

    attn_metadata = MetaData(sm_scale=sm_scale,
                             max_seqlens_q=max_seqlens_q,
                             max_seqlens_k=max_seqlens_k,
                             causal=causal,
                             bias=bias,
                             output_dtype=q.dtype,
                             cu_seqlens_q=cu_seqlens_q,
                             cu_seqlens_k=cu_seqlens_k,
                             q_descale=q_descale,
                             k_descale=k_descale,
                             v_descale=v_descale,
                             p_scale=p_scale,
                             o_scale=input_scale)

    if fp8_scales is not None:
        q, k, v = check_and_maybe_quantize_qkv(q, k, v, fp8_scales)

    return triton_attention_rocm(q, k, v, o, attn_metadata)
