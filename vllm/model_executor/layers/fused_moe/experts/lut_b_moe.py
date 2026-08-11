# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Triton fused MoE expert GEMMs for logical LUT-B weights."""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.lut_b_utils import (
    LUT_B_BLOCK_K,
    LUT_B_BLOCK_N,
    LUT_B_CODEBOOK_SIZE,
    LUT_B_PACKED_TILE_BYTES,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kLutBStatic,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

LUT_B_BLOCK_M = 16
LUT_B_GEMM_BLOCK_N = 64


@triton.jit
def _lut_b_grouped_gemm_kernel(
    a_ptr,
    b_ptr,
    codebook_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K: tl.constexpr,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bnt,
    stride_bkt,
    stride_bb,
    stride_ce,
    stride_cnt,
    stride_ckt,
    stride_ci,
    stride_cm,
    stride_cn,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    LUT_BLOCK_N: tl.constexpr,
    LUT_BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_M >= num_tokens_post_padded:
        return

    offs_m = tl.arange(0, BLOCK_M).to(tl.int64)
    offs_token_id = pid_m * BLOCK_M + offs_m
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    n_mask = offs_n < N
    c_ptrs = c_ptr + offs_token[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = token_mask[:, None] & n_mask[None, :]
    if expert == -1:
        tl.store(c_ptrs, 0.0, mask=c_mask)
        return

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    n_tile = offs_n // LUT_BLOCK_N
    n_in_tile = offs_n % LUT_BLOCK_N

    for k_start in range(0, K, LUT_BLOCK_K):
        offs_k = k_start + tl.arange(0, LUT_BLOCK_K).to(tl.int64)
        a_ptrs = (
            a_ptr
            + (offs_token[:, None] // top_k) * stride_am
            + offs_k[None, :] * stride_ak
        )
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K),
            other=0.0,
        ).to(compute_type)

        k_tile = offs_k // LUT_BLOCK_K
        k_in_tile = offs_k % LUT_BLOCK_K
        flat_index = k_in_tile[:, None] + n_in_tile[None, :] * LUT_BLOCK_K
        index_group = flat_index // 8
        bit_index = (flat_index % 8) * 3
        byte_index = index_group * 3 + bit_index // 8
        bit_shift = bit_index % 8
        weight_mask = (offs_k[:, None] < K) & n_mask[None, :]
        packed_ptrs = (
            b_ptr
            + expert * stride_be
            + n_tile[None, :] * stride_bnt
            + k_tile[:, None] * stride_bkt
            + byte_index * stride_bb
        )
        low = tl.load(packed_ptrs, mask=weight_mask, other=0).to(tl.int32)
        high = tl.load(
            packed_ptrs + stride_bb,
            mask=weight_mask & (bit_shift > 5),
            other=0,
        ).to(tl.int32)
        lut_index = ((low >> bit_shift) | (high << (8 - bit_shift))) & 0x7
        codebook_ptrs = (
            codebook_ptr
            + expert * stride_ce
            + n_tile[None, :] * stride_cnt
            + k_tile[:, None] * stride_ckt
            + lut_index * stride_ci
        )
        b = tl.load(codebook_ptrs, mask=weight_mask, other=0.0).to(compute_type)
        accumulator += tl.dot(a, b)

    if MUL_ROUTED_WEIGHT:
        routed_weight = tl.load(
            topk_weights_ptr + offs_token,
            mask=token_mask,
            other=0.0,
        )
        accumulator *= routed_weight[:, None]
    tl.store(c_ptrs, accumulator, mask=c_mask)


def invoke_lut_b_grouped_gemm(
    a: torch.Tensor,
    packed_weight: torch.Tensor,
    codebooks: torch.Tensor,
    c: torch.Tensor,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    mul_routed_weight: bool,
    top_k: int,
    logical_n: int,
) -> None:
    """Launch a grouped GEMM that decodes LUT-B values in the K loop."""
    if mul_routed_weight:
        assert topk_weights is not None
    assert packed_weight.ndim == 4
    assert codebooks.shape == (*packed_weight.shape[:3], LUT_B_CODEBOOK_SIZE)
    assert packed_weight.shape[-1] == LUT_B_PACKED_TILE_BYTES
    assert a.shape[-1] % LUT_B_BLOCK_K == 0

    if a.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif a.dtype == torch.float16:
        compute_type = tl.float16
    else:
        raise ValueError(f"LUT-B fused MoE requires BF16 or FP16, got {a.dtype}")

    grid = (
        triton.cdiv(sorted_token_ids.shape[0], LUT_B_BLOCK_M),
        triton.cdiv(logical_n, LUT_B_GEMM_BLOCK_N),
    )
    _lut_b_grouped_gemm_kernel[grid](
        a,
        packed_weight,
        codebooks,
        c,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        logical_n,
        a.shape[-1],
        a.shape[0] * top_k,
        a.stride(0),
        a.stride(1),
        packed_weight.stride(0),
        packed_weight.stride(1),
        packed_weight.stride(2),
        packed_weight.stride(3),
        codebooks.stride(0),
        codebooks.stride(1),
        codebooks.stride(2),
        codebooks.stride(3),
        c.stride(1),
        c.stride(2),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        LUT_BLOCK_N=LUT_B_BLOCK_N,
        LUT_BLOCK_K=LUT_B_BLOCK_K,
        BLOCK_M=LUT_B_BLOCK_M,
        BLOCK_N=LUT_B_GEMM_BLOCK_N,
        num_warps=4,
        num_stages=2,
    )


class LutBTritonExperts(TritonExperts):
    """Routed experts backed by two fused LUT-B decode-and-GEMM launches."""

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_cuda_alike()

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return weight_key == kLutBStatic and activation_key is None

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    @staticmethod
    def supports_lora() -> bool:
        return False

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        assert w1.ndim == 4 and w2.ndim == 4
        num_experts = w1.shape[0]
        n = w1.shape[1] * LUT_B_BLOCK_N
        return num_experts, a1.shape[0], n, a1.shape[-1], topk_ids.shape[1]

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        del a1q_scale, a2_scale, expert_tokens_meta
        assert hidden_states.is_contiguous()
        assert w1.is_contiguous() and w2.is_contiguous()
        assert self.w1_scale is not None and self.w2_scale is not None

        num_experts, num_tokens, n, k, top_k = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )
        if global_num_experts == -1:
            global_num_experts = num_experts

        intermediate_cache1 = _resize_cache(
            workspace2,
            (num_tokens, top_k, n),
        )
        activation_n = self.adjust_N_for_activation(n, activation)
        intermediate_cache2 = _resize_cache(
            workspace13,
            (num_tokens * top_k, activation_n),
        )
        intermediate_cache3 = _resize_cache(
            workspace2,
            (num_tokens, top_k, k),
        )
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids,
            LUT_B_BLOCK_M,
            global_num_experts,
            expert_map,
        )

        invoke_lut_b_grouped_gemm(
            hidden_states,
            w1,
            self.w1_scale,
            intermediate_cache1,
            None,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight=False,
            top_k=top_k,
            logical_n=n,
        )
        self.activation(
            activation,
            intermediate_cache2,
            intermediate_cache1.view(-1, n),
        )
        invoke_lut_b_grouped_gemm(
            intermediate_cache2,
            w2,
            self.w2_scale,
            intermediate_cache3,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight=not apply_router_weight_on_input,
            top_k=1,
            logical_n=k,
        )
        self.moe_sum(intermediate_cache3, output)


def make_lut_b_moe_kernel(
    moe_config,
    quant_config: FusedMoEQuantConfig,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
) -> mk.FusedMoEKernel:
    """Build the modular prepare/experts/finalize LUT-B kernel."""
    from vllm.model_executor.layers.fused_moe.all2all_utils import (
        maybe_make_prepare_finalize,
    )

    prepare_finalize = maybe_make_prepare_finalize(
        moe=moe_config,
        quant_config=quant_config,
        routing_tables=routing_tables,
        allow_new_interface=True,
        use_monolithic=False,
    )
    assert prepare_finalize is not None
    experts = LutBTritonExperts(
        moe_config=moe_config,
        quant_config=quant_config,
    )
    return mk.FusedMoEKernel(prepare_finalize, experts)
