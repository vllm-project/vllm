# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from math import log2
from typing import Optional

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    deep_gemm_block_shape)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import (fp8_m_grouped_gemm_nt_masked,
                                  is_deep_gemm_e8m0_used)

logger = init_logger(__name__)


@triton.jit
def _silu_mul_fp8_quant_deep_gemm(
    # Pointers ------------------------------------------------------------
    input_ptr,  # 16-bit activations (E, T, 2*H)
    y_q_ptr,  # fp8 quantized activations (E, T, H)
    y_s_ptr,  # 16-bit scales (E, T, G)
    counts_ptr,  # int32 num tokens per expert (E)
    # Sizes ---------------------------------------------------------------
    H: tl.constexpr,  # hidden dimension (per output)
    GROUP_SIZE: tl.constexpr,  # elements per group (usually 128)
    # Strides for input (elements) ---------------------------------------
    stride_i_e,
    stride_i_t,
    stride_i_h,
    # Strides for y_q (elements) -----------------------------------------
    stride_yq_e,
    stride_yq_t,
    stride_yq_h,
    # Strides for y_s (elements) -----------------------------------------
    stride_ys_e,
    stride_ys_t,
    stride_ys_g,
    # Stride for counts (elements)
    stride_counts_e,
    # Numeric params ------------------------------------------------------
    eps: tl.constexpr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    use_ue8m0: tl.constexpr,
    # Meta ---------------------------------------------------------------
    BLOCK: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    G = H // GROUP_SIZE

    # map program id -> (e, g)
    pid = tl.program_id(0)
    e = pid // G
    g = pid % G

    e = e.to(tl.int64)
    g = g.to(tl.int64)

    # number of valid tokens for this expert
    n_tokens = tl.load(counts_ptr + e * stride_counts_e).to(tl.int64)

    cols = tl.arange(0, BLOCK).to(tl.int64)
    mask = cols < BLOCK

    base_input_offset = e * stride_i_e + g * GROUP_SIZE * stride_i_h
    base_gate_offset = base_input_offset + cols * stride_i_h
    base_up_offset = base_input_offset + H * stride_i_h + cols * stride_i_h
    base_yq_offset = (e * stride_yq_e + g * GROUP_SIZE * stride_yq_h +
                      cols * stride_yq_h)
    base_ys_offset = e * stride_ys_e + g * stride_ys_g

    for t in tl.range(0, n_tokens, num_stages=NUM_STAGES):
        gate = tl.load(input_ptr + base_gate_offset + t * stride_i_t,
                       mask=mask,
                       other=0.0).to(tl.float32)
        up = tl.load(input_ptr + base_up_offset + t * stride_i_t,
                     mask=mask,
                     other=0.0)

        gate = gate * (1.0 / (1.0 + tl.exp(-gate)))
        y = gate * up

        y_s = tl.maximum(tl.max(tl.abs(y)), eps) / fp8_max
        if use_ue8m0:
            y_s = tl.exp2(tl.ceil(tl.log2(y_s)))

        y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

        tl.store(y_q_ptr + base_yq_offset + t * stride_yq_t, y_q, mask=mask)
        tl.store(y_s_ptr + base_ys_offset + t * stride_ys_t, y_s)


def silu_mul_fp8_quant_deep_gemm_cuda(
    y: torch.Tensor,  # (E, T, 2*H)
    tokens_per_expert: torch.Tensor,  # (E,) number of valid tokens per expert
    num_parallel_tokens=16,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize silu(y[..., :H]) * y[..., H:] to FP8 with group per-token scales
    y has shape (E, T, 2*H). The first half of the last dimension is
    silu-activated, multiplied by the second half, then quantized into FP8.
    Returns `(y_q, y_s)` where
    * `y_q`: FP8 tensor, shape (E, T, H), same layout as y[..., :H]
    * `y_s`: FP32 tensor, shape (E, T, H // group_size), strides (T*G, 1, T)
    """
    assert y.ndim == 3, "y must be (E, T, 2*H)"
    E, T, H2 = y.shape
    assert H2 % 2 == 0, "last dim of y must be even (2*H)"
    H = H2 // 2
    G = (H + group_size - 1) // group_size
    assert H % 8 == 0, "H must be divisible by 8"
    assert group_size == 128, "H must be divisible by 8"
    assert tokens_per_expert.ndim == 1 and tokens_per_expert.shape[0] == E

    tokens_per_expert = tokens_per_expert.to(device=y.device,
                                             dtype=torch.int32)

    fp8_dtype = torch.float8_e4m3fn
    y_q = torch.empty((E, T, H), dtype=fp8_dtype, device=y.device)

    stride_ys_e = T * G
    stride_ys_t = 1
    stride_ys_g = T
    y_s = torch.empty_strided((E, T, G),
                              (stride_ys_e, stride_ys_t, stride_ys_g),
                              dtype=torch.float32,
                              device=y.device)

    use_ue8m0 = is_deep_gemm_e8m0_used()

    if E <= 16:
        max_empirical_parallelism = 64
    elif E <= 32:
        max_empirical_parallelism = 16
    else:
        max_empirical_parallelism = 4

    # We never want to launch more than Tx number of threads
    # This computes the clip.
    num_parallel_tokens = max(
        1,
        min(max_empirical_parallelism, 2**int(log2(min(num_parallel_tokens,
                                                       T)))))
    cuda_arch = current_platform.get_device_capability(
        device_id=y.device.index).to_int()

    if cuda_arch >= 80:
        torch.ops._C.silu_mul_fp8_quant_deep_gemm_cuda(y, tokens_per_expert,
                                                       y_q, y_s, group_size,
                                                       use_ue8m0,
                                                       num_parallel_tokens)
    else:
        # Default to triton if not on cuda or if arch is too old
        y_q = torch.empty((E, T, H), dtype=fp8_dtype, device=y.device)

        stride_cnt_e = tokens_per_expert.stride()[0]

        # Static grid over experts and H-groups.
        # A loop inside the kernel handles the token dim
        grid = (E * G, )
        # strides (elements)
        stride_i_e, stride_i_t, stride_i_h = y.stride()
        stride_yq_e, stride_yq_t, stride_yq_h = y_q.stride()

        # desired scale strides (elements): (T*G, 1, T)
        stride_ys_e = T * G
        stride_ys_t = 1
        stride_ys_g = T
        y_s = torch.empty_strided(
            (E, T, G),
            (stride_ys_e, stride_ys_t, stride_ys_g),
            dtype=torch.float32,
            device=y.device,
        )
        f_info = torch.finfo(fp8_dtype)
        fp8_max = f_info.max
        fp8_min = f_info.min
        eps: float = 1e-10
        _silu_mul_fp8_quant_deep_gemm[grid](
            y,
            y_q,
            y_s,
            tokens_per_expert,
            H,
            group_size,
            stride_i_e,
            stride_i_t,
            stride_i_h,
            stride_yq_e,
            stride_yq_t,
            stride_yq_h,
            stride_ys_e,
            stride_ys_t,
            stride_ys_g,
            stride_cnt_e,
            eps,
            fp8_min,
            fp8_max,
            is_deep_gemm_e8m0_used(),
            BLOCK=group_size,
            NUM_STAGES=4,
            num_warps=1,
        )

    return y_q, y_s


class BatchedDeepGemmExperts(mk.FusedMoEPermuteExpertsUnpermute):

    def __init__(
        self,
        max_num_tokens: int,
        num_dispatchers: int,
        quant_config: FusedMoEQuantConfig,
    ):
        """
        max_num_tokens: Maximum number of tokens from a DP Rank
        num_dispatchers: The number of DP dispatchers.
        quant_config: Quantization configuration
        """
        super().__init__(quant_config)
        assert self.block_shape == deep_gemm_block_shape()
        self.max_num_tokens = max_num_tokens
        self.num_dispatchers = num_dispatchers

    @property
    def activation_formats(
        self
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        return (mk.FusedMoEActivationFormat.BatchedExperts,
                mk.FusedMoEActivationFormat.BatchedExperts)

    def supports_chunking(self) -> bool:
        return False

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # Let PrepareAndFinalize::finalize() decide the impl.
        return TopKWeightAndReduceDelegate()

    def workspace_shapes(
        self,
        a: torch.Tensor,
        aq: torch.Tensor,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_metadata: Optional[mk.ExpertTokensMetadata],
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], torch.dtype]:
        assert a.dim() == 2
        # FIXME (varun): We should be able to dispatch only from the leader
        # DP ranks in the case of TP > 1. At the moment, all the Ranks
        # end up sending their tokens. This needs to be fixed.
        num_dispatchers = self.num_dispatchers
        num_experts = local_num_experts
        max_num_tokens = a.size(
            0) if self.max_num_tokens is None else self.max_num_tokens
        workspace13 = (num_experts, max_num_tokens * num_dispatchers,
                       max(K, N))
        workspace2 = (num_experts, max_num_tokens * num_dispatchers, (N // 2))
        output = (num_experts, max_num_tokens * num_dispatchers, K)
        return (workspace13, workspace2, output, a.dtype)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: Optional[torch.Tensor],
        a1q_scale: Optional[torch.Tensor],
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: Optional[mk.ExpertTokensMetadata],
        apply_router_weight_on_input: bool,
    ):
        assert expert_tokens_meta is not None
        expert_num_tokens = expert_tokens_meta.expert_num_tokens

        assert hidden_states.ndim == 3
        assert self.block_shape is not None

        a1q = hidden_states
        _, N, K = w1.size()

        assert w2.size(1) == K

        E, max_num_tokens, N, K, top_k_num = mk._moe_problem_size(
            hidden_states, w1, w2, topk_ids)

        workspace1 = _resize_cache(workspace13, (E, max_num_tokens, N))

        # (from deepgemm docs) : A value hint (which is a value on CPU)
        # for the M expectation of each batch, correctly setting this value
        # may lead to better performance.
        expected_m = max_num_tokens
        fp8_m_grouped_gemm_nt_masked((a1q, a1q_scale), (w1, self.w1_scale),
                                     workspace1, expert_num_tokens, expected_m)

        a2q, a2q_scale = silu_mul_fp8_quant_deep_gemm_cuda(
            workspace1, expert_num_tokens)

        fp8_m_grouped_gemm_nt_masked((a2q, a2q_scale), (w2, self.w2_scale),
                                     output, expert_num_tokens, expected_m)
