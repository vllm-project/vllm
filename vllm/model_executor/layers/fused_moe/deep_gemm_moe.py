# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from tqdm import tqdm

import vllm.envs as env
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    fp8_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    compute_aligned_M,
    deepgemm_moe_permute,
    deepgemm_unpermute_and_reduce,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.utils.deep_gemm import (
    get_mk_alignment_for_contiguous_layout,
    m_grouped_fp8_gemm_nt_contiguous,
)
from vllm.utils.func_utils import run_once
from vllm.utils.import_utils import has_deep_gemm

logger = init_logger(__name__)


def _valid_deep_gemm_shape(M: int, N: int, K: int) -> bool:
    align = get_mk_alignment_for_contiguous_layout()[0]
    return align <= M and N % align == 0 and K % align == 0


def _valid_deep_gemm(
    hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> bool:
    """
    Check if the given problem size is supported by the DeepGemm grouped
    gemm kernel.  All of M, N, K and the quantization block_shape must be
    aligned by `dg.get_m_alignment_for_contiguous_layout()`.
    """
    if not has_deep_gemm():
        logger.debug_once("DeepGemm disabled: deep_gemm not available.")
        return False

    M = hidden_states.size(0)
    _, K, N = w2.size()

    align = get_mk_alignment_for_contiguous_layout()[0]

    if not _valid_deep_gemm_shape(M, N, K):
        logger.debug_once(
            "DeepGemm disabled due to unaligned problem size. "
            "M: %s, N: %s, K: %s. M should >= %s "
            "and N and K must be multiples of %s. "
            "This is not an error and we will fall back to triton.",
            M,
            N,
            K,
            align,
            align,
        )
        return False
    elif N <= 512:
        logger.debug_once(
            "DeepGemm disabled for N <= 512. M: %s, N: %s, K: %s. "
            "This means we will fallback to triton "
            "for this specific shape for further speed up.",
            M,
            N,
            K,
        )
        return False

    if w1.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        logger.debug_once(
            "DeepGemm disabled: invalid weight dtype(s). w1.dtype: %s, w2.dtype: %s",
            w1.dtype,
            w2.dtype,
        )
        return False

    if (
        not hidden_states.is_contiguous()
        or not w1.is_contiguous()
        or not w2.is_contiguous()
    ):
        logger.debug_once(
            "DeepGemm disabled: weights or activations not contiguous. "
            "hidden_states.is_contiguous(): %s, w1.is_contiguous(): %s, "
            "w2.is_contiguous(): %s",
            hidden_states.is_contiguous(),
            w1.is_contiguous(),
            w2.is_contiguous(),
        )
        return False

    return True


@run_once
def warmup_deepgemm_gg_contiguous_kernels(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    num_topk: int,
):
    """
    DeepGemm JITs the grouped-gemm kernels. The JIT'ing happens based on the
    input tensor shapes. In this function, we construct all possible input
    tensor shapes so all the kernels are JIT'ed and cached.
    Note that this warmup is expected to happen during the model profile
    call and not during actual model inference.
    """

    assert w1.size(0) == w2.size(0), "w1 and w2 must have the same number of experts"

    block_m = get_mk_alignment_for_contiguous_layout()[0]
    num_experts = w1.size(0)
    device = w1.device

    # This is the maximum GroupedGemm M size that we expect to run
    # the grouped_gemm with.
    MAX_M = compute_aligned_M(
        env.VLLM_FUSED_MOE_CHUNK_SIZE,
        num_topk,
        num_experts,
        block_m,
        expert_tokens_meta=None,
    )
    # Distribute expert-ids evenly.
    MAX_BLOCKS = MAX_M // block_m
    expert_ids_block = torch.randint(
        low=0, high=num_experts, size=(MAX_BLOCKS,), device=device, dtype=torch.int32
    )
    expert_ids = torch.repeat_interleave(expert_ids_block, block_m, dim=0)

    def _warmup(w: torch.Tensor, w_scale: torch.Tensor):
        _, n, k = w.size()
        a1q = torch.empty((MAX_M, k), device=device).to(torch.float8_e4m3fn)
        a1q_scales = torch.empty(
            (MAX_M, k // block_m), device=device, dtype=torch.float32
        )
        out = torch.empty((MAX_M, n), device=device, dtype=torch.bfloat16)

        pbar = tqdm(
            total=MAX_BLOCKS, desc=f"DeepGemmExperts GEMM warmup (MAX_M={MAX_M})"
        )
        num_tokens = MAX_M
        while num_tokens > 0:
            m_grouped_fp8_gemm_nt_contiguous(
                (a1q[:num_tokens], a1q_scales[:num_tokens]),
                (w, w_scale),
                out[:num_tokens],
                expert_ids[:num_tokens],
            )
            pbar.update(1)
            num_tokens = num_tokens - block_m

    _warmup(w1, w1_scale)
    _warmup(w2, w2_scale)


class DeepGemmExperts(mk.FusedMoEPermuteExpertsUnpermute):
    def __init__(self, quant_config: FusedMoEQuantConfig):
        super().__init__(quant_config)
        assert quant_config.block_shape == get_mk_alignment_for_contiguous_layout()
        assert quant_config.quant_dtype == torch.float8_e4m3fn
        assert not quant_config.per_act_token_quant
        assert not quant_config.per_out_ch_quant

    @property
    def activation_formats(
        self,
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        return (
            mk.FusedMoEActivationFormat.Standard,
            mk.FusedMoEActivationFormat.Standard,
        )

    def supports_chunking(self) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        assert self.block_shape is not None
        block_m = self.block_shape[0]
        M_sum = compute_aligned_M(
            M, topk, local_num_experts, block_m, expert_tokens_meta
        )
        assert M_sum % block_m == 0

        workspace1 = (M_sum, N)
        workspace2 = (M_sum, max(N // 2, K))
        output = (M, K)
        return (workspace1, workspace2, output)

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
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        assert a1q_scale is not None
        assert a2_scale is None
        assert self.block_shape is not None
        assert self.w1_scale is not None
        assert self.w2_scale is not None

        a1q = hidden_states
        _, N, K = w1.size()

        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        assert w2.size(1) == K

        M_sum = compute_aligned_M(
            M=topk_ids.size(0),
            num_topk=topk_ids.size(1),
            local_num_experts=local_num_experts,
            alignment=get_mk_alignment_for_contiguous_layout()[0],
            expert_tokens_meta=expert_tokens_meta,
        )

        a1q_perm = _resize_cache(workspace2.view(dtype=torch.float8_e4m3fn), (M_sum, K))
        mm1_out = _resize_cache(workspace13, (M_sum, N))
        act_out = _resize_cache(workspace2, (M_sum, N // 2))
        quant_out = _resize_cache(
            workspace13.view(dtype=torch.float8_e4m3fn), (M_sum, N // 2)
        )
        mm2_out = _resize_cache(workspace2, (M_sum, K))

        a1q, a1q_scale, expert_ids, inv_perm = deepgemm_moe_permute(
            aq=a1q,
            aq_scale=a1q_scale,
            topk_ids=topk_ids,
            local_num_experts=local_num_experts,
            expert_map=expert_map,
            expert_tokens_meta=expert_tokens_meta,
            aq_out=a1q_perm,
        )
        assert a1q.size(0) == M_sum

        m_grouped_fp8_gemm_nt_contiguous(
            (a1q, a1q_scale), (w1, self.w1_scale), mm1_out, expert_ids
        )

        self.activation(activation, act_out, mm1_out.view(-1, N))

        a2q_scale: torch.Tensor | None = None
        a2q, a2q_scale = per_token_group_quant_fp8(
            act_out, self.block_shape[1], column_major_scales=True, out_q=quant_out
        )

        m_grouped_fp8_gemm_nt_contiguous(
            (a2q, a2q_scale), (w2, self.w2_scale), mm2_out, expert_ids
        )

        if apply_router_weight_on_input:
            topk_weights = torch.ones_like(topk_weights)

        deepgemm_unpermute_and_reduce(
            a=mm2_out,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            inv_perm=inv_perm,
            expert_map=expert_map,
            output=output,
        )


def deep_gemm_moe_fp8(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    apply_router_weight_on_input=False,
) -> torch.Tensor:
    """
    This function computes a a8w8-quantized Mixture of Experts (MoE) layer
    using two sets of quantized weights, w1_q and w2_q, and top-k gating
    mechanism. The matrix multiplications are implemented with DeepGemm
    grouped gemm.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
        Shape: [M, K]
    - w1 (torch.Tensor): The first set of fp8 quantized expert weights.
        Shape: [num_experts, K, 2N] (the weights are passed transposed)
    - w2 (torch.Tensor): The second set of fp8 quantized expert weights.
        Shape: [num_experts, N, K] (the weights are passed transposed)
    - w1_scale (torch.Tensor): The fp32 scale to dequantize w1_q.
        Shape: [num_experts] or [num_experts, 2N]
    - w2_scale (torch.Tensor): The fp32 scale to dequantize w2_q.
        Shape: [num_experts] or [num_experts, K]
    - topk_weights (torch.Tensor): The weights of each token->expert mapping.
    - topk_ids (torch.Tensor): The token->expert mapping for topk_weights.
    - inplace (bool): If True, perform the operation in-place.
        Defaults to False.
    - activation (str): The activation function to apply after the first
        MoE layer.
    - global_num_experts (int): The total number of experts in the global
        expert space.
    - expert_map (Optional[torch.Tensor]):  A tensor mapping expert indices
        from the global expert space to the local expert space of the expert
        parallel shard.
    - a1_scale (Optional[torch.Tensor]): The optional fp32 scale to quantize a.
        Shape: scalar or [M]
    - a2_scale (Optional[torch.Tensor]): The optional fp32 scale to
        quantize the intermediate result between the gemms.
        Shape: scalar or [M]

    Returns:
    - torch.Tensor: The bfloat16 output tensor after applying the MoE layer.
    """
    quant_config = fp8_w8a8_moe_quant_config(
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=get_mk_alignment_for_contiguous_layout(),
    )

    fn = mk.FusedMoEModularKernel(
        MoEPrepareAndFinalizeNoEP(),
        DeepGemmExperts(quant_config),
    )
    return fn(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=inplace,
        activation=activation,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
