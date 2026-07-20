# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
    TopKWeightAndReduceNaiveBatched,
)
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input,
    normalize_scales_shape,
)


class BatchedPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """
    A reference prepare/finalize class that reorganizes the tokens into
    expert batched format, i.e. E x max_num_tokens x K.  This is the format
    that the batched dispatch/combine kernels use.
    """

    def __init__(
        self,
        max_num_tokens: int,
        num_local_experts: int,
        num_dispatchers: int,
        rank: int,
    ):
        super().__init__()
        self.max_num_tokens = max_num_tokens
        self.num_local_experts = num_local_experts
        self.rank = rank
        self.num_dispatchers_ = num_dispatchers

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_num_tokens

    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        return False

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        if defer_input_quant:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support defer_input_quant=True. "
                "Please select an MoE kernel that accepts quantized inputs."
            )
        assert a1.dim() == 2
        assert topk_ids.dim() == 2
        assert topk_ids.size(0) == a1.size(0)

        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            # TODO: this only works for topK=1, will need to update for topK>1
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1.mul_(topk_weights.to(a1.dtype))

        num_tokens, hidden_dim = a1.size()
        topk = topk_ids.size(1)

        tokens_per_expert = torch.zeros(num_experts, dtype=torch.int, device=a1.device)

        num_local_experts = self.num_local_experts

        if quant_config.quant_dtype is None:
            b_type = a1.dtype
        else:
            b_type = quant_config.quant_dtype

        b_a1 = torch.zeros(
            (num_local_experts, self.max_num_tokens, hidden_dim),
            dtype=b_type,
            device=a1.device,
        )

        if quant_config.is_quantized:
            scale_shape = quant_config.batched_scale_shape(
                num_local_experts, self.max_num_tokens, hidden_dim
            )

            b_a1_scale = torch.empty(scale_shape, dtype=torch.float32, device=a1.device)
        else:
            assert quant_config.a1_scale is None
            b_a1_scale = None

        first_expert = num_local_experts * self.rank
        last_expert = first_expert + num_local_experts

        a1_scale = normalize_scales_shape(quant_config.a1_scale)

        for expert_id in range(first_expert, last_expert):
            topks = torch.any(topk_ids == expert_id, dim=1).flatten()
            rows = torch.count_nonzero(topks.flatten())
            if rows == 0:
                continue
            idx = expert_id - first_expert
            tokens_per_expert[idx] = rows
            rhs = a1[: topks.numel()][topks]
            if quant_config.quant_dtype is not None:
                if a1_scale is not None:
                    if quant_config.is_per_act_token:
                        rhs_a1_scale = a1_scale[: topks.numel()][topks]
                    else:
                        rhs_a1_scale = a1_scale
                else:
                    rhs_a1_scale = None
                b_a1[idx, :rows, :], b_s = moe_kernel_quantize_input(
                    rhs,
                    rhs_a1_scale,
                    quant_config.quant_dtype,
                    quant_config.per_act_token_quant,
                    quant_config.block_shape,
                )
                assert b_s is not None
                if quant_config.is_per_act_token:
                    b_a1_scale[idx, :rows] = b_s[:rows]
                else:
                    b_a1_scale[idx, : b_s.shape[0]] = b_s
            else:
                b_a1[idx, :rows, :] = rhs

        assert b_a1_scale is None or b_a1_scale.ndim == 3

        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=tokens_per_expert, expert_num_tokens_cpu=None
        )

        return b_a1, b_a1_scale, expert_tokens_meta, None, None

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
            weight_and_reduce_impl = TopKWeightAndReduceNaiveBatched(self.rank)
        weight_and_reduce_impl.apply(
            output=output,
            fused_expert_output=fused_expert_output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
