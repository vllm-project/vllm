# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, Optional

import deep_ep
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous, TopKWeightAndReduceDelegate)
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input)


class DeepEPHTPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """
    Prepare/Finalize using DeepEP High-Throughput kernels.
    """

    def __init__(self, buffer: deep_ep.Buffer, num_dispatchers: int,
                 dp_size: int, rank_expert_offset: int):
        super().__init__()
        self.buffer = buffer
        self.num_dispatchers_ = num_dispatchers
        self.dp_size = dp_size
        self.rank_expert_offset = rank_expert_offset
        # The dispatch function returns a handle that the combine function
        # requires. We store the handle here so it is available to the
        # combine function.
        self.handle = None

        # From https://github.com/deepseek-ai/DeepEP/blob/9fe9021f29c9083cd1808ab36b740208524d9f63/deep_ep/buffer.py#L164
        self.available_rank_configs = [2, 4, 8, 16, 24, 32, 64, 128, 144, 160]

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> Optional[int]:
        return None

    def topk_indices_dtype(self) -> Optional[torch.dtype]:
        return torch.int64

    def _get_dispatch_config(self) -> Optional[deep_ep.Config]:
        if self.dp_size not in self.available_rank_configs:
            return None
        return deep_ep.Buffer.get_dispatch_config(self.dp_size)

    def _get_combine_config(self) -> Optional[deep_ep.Config]:
        if self.dp_size not in self.available_rank_configs:
            return None
        return deep_ep.Buffer.get_combine_config(self.dp_size)

    def _do_dispatch(self, tokens: torch.Tensor,
                     token_scales: Optional[torch.Tensor],
                     rank_topk_ids: torch.Tensor,
                     rank_topk_weights: torch.Tensor, num_experts: int):

        has_scales = token_scales is not None

        (num_tokens_per_rank, num_tokens_per_rdma_rank,
         dispatch_expert_num_tokens, is_token_in_rank,
         event) = self.buffer.get_dispatch_layout(
             topk_idx=rank_topk_ids,
             num_experts=num_experts,
             previous_event=None,
             async_finish=False,
             allocate_on_comm_stream=False)

        token_data = tokens
        if has_scales:
            token_data = (tokens, token_scales)

        (
            token_data, expert_topk_ids, expert_topk_weights,
            expert_num_tokens_per_expert_list, self.handle, event
        ) = self.buffer.dispatch(
            x=token_data,
            handle=None,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=dispatch_expert_num_tokens,
            topk_idx=rank_topk_ids,
            topk_weights=rank_topk_weights,
            # expert_alignment rounds the number of tokens per expert
            # to this value.
            expert_alignment=1,
            config=self._get_dispatch_config(),
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=False)

        if has_scales:
            expert_x, expert_x_scale = token_data
        else:
            expert_x, expert_x_scale = token_data, None

        # The existing MOE kernels assume that all entries of topk_ids are
        # valid. To that effect, set the -1s in expert_topk_ids to some expert
        # outside this rank so the expert_map can remap it to -1 when safe.
        # With Expert Parallel, the experts are divided amongst the rank
        # sequentially. For rank 0, set it to num_experts - 1 and for all other
        # ranks set it to 0 as we know that expert_map will have a -1 in those
        # regions for those ranks.
        #
        # DeepEP's topk_ids output refers to the local experts directly. Offset
        # the topk_ids to move it back to the global experts space so it aligns
        # with existing vLLM interfaces.
        expert_topk_ids = torch.where(
            expert_topk_ids == -1,
            num_experts - 1 if self.rank_expert_offset == 0 else 0,
            expert_topk_ids + self.rank_expert_offset)

        # Makes a GPU-CPU copy.
        # TODO (varun): Maybe it is better to re-compute the expert_num_tokens
        # on GPU.
        expert_tokens_meta = mk.ExpertTokensMetadata.make_from_list(
            expert_num_tokens_per_expert_list, device=expert_x.device)

        return (expert_x, expert_x_scale, expert_tokens_meta, expert_topk_ids,
                expert_topk_weights)

    def prepare(
        self, a1: torch.Tensor, a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor], topk_weights: torch.Tensor,
        topk_ids: torch.Tensor, num_experts: int,
        expert_map: Optional[torch.Tensor], apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        extra_prepare_args: Optional[dict[str, Any]]
    ) -> tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[mk.ExpertTokensMetadata], Optional[torch.Tensor],
               Optional[torch.Tensor]]:

        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            # TODO: this only works for topK=1, will need to update for topK>1
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1")
            a1 = a1 * topk_weights.to(a1.dtype)

        if quant_config.is_block_quantized:
            # Quant and Dispatch
            a1q, a1q_scale = moe_kernel_quantize_input(
                a1,
                a1_scale,
                quant_dtype=quant_config.quant_dtype,
                per_act_token_quant=quant_config.per_act_token_quant,
                block_shape=quant_config.block_shape,
            )
            if a1q_scale is not None and a1q_scale.numel() == 1:
                a1q_scale = a1q_scale.view(1, 1)
            (expert_x, expert_x_scale, expert_tokens_meta, expert_topk_ids,
             expert_topk_weights) = self._do_dispatch(
                 tokens=a1q,
                 token_scales=a1q_scale,
                 rank_topk_ids=topk_ids,
                 rank_topk_weights=topk_weights,
                 num_experts=num_experts)
        else:
            # Dispatch and Quant
            # DeepEP kernels only support dispatching block-quantized
            # activation scales.
            # Dispatch in bfloat16
            (expert_x, _, expert_tokens_meta, expert_topk_ids,
             expert_topk_weights) = self._do_dispatch(
                 tokens=a1,
                 token_scales=None,
                 rank_topk_ids=topk_ids,
                 rank_topk_weights=topk_weights,
                 num_experts=num_experts)
            # Quantize after dispatch.
            expert_x_scale = None
            if expert_x.numel() != 0:
                expert_x, expert_x_scale = moe_kernel_quantize_input(
                    expert_x,
                    a1_scale,
                    quant_dtype=quant_config.quant_dtype,
                    per_act_token_quant=False,
                    block_shape=quant_config.block_shape)

        return (expert_x, expert_x_scale, expert_tokens_meta, expert_topk_ids,
                expert_topk_weights)

    def finalize(self, output: torch.Tensor, fused_expert_output: torch.Tensor,
                 topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                 apply_router_weight_on_input: bool,
                 weight_and_reduce_impl: mk.TopKWeightAndReduce,
                 extra_finalize_args: Optional[dict[str, Any]]) -> None:

        assert self.handle is not None

        # fused_expert_output can have 0 tokens - This happens when none of the
        # tokens from the all2all reach this EP rank.
        if fused_expert_output.numel() != 0:
            if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
                weight_and_reduce_impl = TopKWeightAndReduceContiguous()
            fused_expert_output = weight_and_reduce_impl.apply(
                output=None,
                fused_expert_output=fused_expert_output,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )

        combined_x, _, event = self.buffer.combine(
            x=fused_expert_output,
            handle=self.handle,
            topk_weights=None,
            config=self._get_combine_config(),
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=False)
        # Respect inplace outputs.
        output.copy_(combined_x, non_blocking=True)
