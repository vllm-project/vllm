# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import deep_ep
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input)


class DeepEPHTPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """
    Prepare/Finalize using DeepEP High-Throughput kernels.
    """

    def __init__(self,
                 buffer: deep_ep.Buffer,
                 world_size: int,
                 rank: int,
                 dp_size: int,
                 rank_expert_offset: int,
                 quant_dtype: Optional[torch.dtype] = None,
                 block_shape: Optional[list[int]] = None):
        super().__init__()
        self.buffer = buffer
        self.world_size = world_size
        self.rank = rank
        self.dp_size = dp_size
        self.rank_expert_offset = rank_expert_offset
        self.quant_dtype = quant_dtype
        self.block_shape = block_shape
        # The dispatch function returns a handle that the combine function
        # requires. We store the handle here so it is available to the
        # combine function.
        self.handle = None

        # From https://github.com/deepseek-ai/DeepEP/blob/9fe9021f29c9083cd1808ab36b740208524d9f63/deep_ep/buffer.py#L164
        self.available_rank_configs = [2, 4, 8, 16, 24, 32, 64, 128, 144, 160]

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

    def _do_quant(self, tokens: torch.Tensor,
                  token_scales: Optional[torch.Tensor], per_act_token: bool):
        tokens, token_scales = moe_kernel_quantize_input(
            tokens, token_scales, self.quant_dtype, per_act_token,
            self.block_shape)
        return tokens, token_scales

    def _do_dispatch(self, tokens: torch.Tensor,
                     token_scales: Optional[torch.Tensor],
                     rank_topk_ids: torch.Tensor,
                     rank_topk_weights: torch.Tensor, num_experts: int):

        has_scales = token_scales is not None

        (num_tokens_per_rank, num_tokens_per_rdma_rank, expert_num_tokens,
         is_token_in_rank, event) = self.buffer.get_dispatch_layout(
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
            num_tokens_per_expert=expert_num_tokens,
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

        return (expert_x, expert_x_scale, expert_num_tokens, expert_topk_ids,
                expert_topk_weights)

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        rank_topk_weights: torch.Tensor,
        rank_topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor], Optional[torch.Tensor]]:

        if apply_router_weight_on_input:
            topk = rank_topk_ids.size(1)
            # TODO: this only works for topK=1, will need to update for topK>1
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1")
            a1 = a1 * rank_topk_weights.to(a1.dtype)

        # Check if there is a block_shape / or if we can infer the quantization
        # schemes from the scales.
        per_token_quant = None
        if all([x is None for x in [self.block_shape, a1_scale, a2_scale]
                ]) and self.quant_dtype is not None:
            # Quantization required despite none of the inputs suggesting
            # quantization. Fallback to per_token_dynamic quant.
            per_token_quant = True
        else:
            per_token_quant = ((self.block_shape is not None) or
                               (a1_scale is not None and a1_scale.numel() != 1)
                               or (a2_scale is not None
                                   and a2_scale.numel() != 1))

        if per_token_quant:
            a1q, a1q_scale = self._do_quant(a1, a1_scale, per_act_token=True)
            (expert_x, expert_x_scale, expert_num_tokens, expert_topk_ids,
             expert_topk_weights) = self._do_dispatch(
                 tokens=a1q,
                 token_scales=a1q_scale,
                 rank_topk_ids=rank_topk_ids,
                 rank_topk_weights=rank_topk_weights,
                 num_experts=num_experts)
        else:
            # DeepEP kernels only support dispatching per-token-quant
            # quantization. dispatch in bfloat16.
            (expert_x, _, expert_num_tokens, expert_topk_ids,
             expert_topk_weights) = self._do_dispatch(
                 tokens=a1,
                 token_scales=None,
                 rank_topk_ids=rank_topk_ids,
                 rank_topk_weights=rank_topk_weights,
                 num_experts=num_experts)
            # quantize now
            expert_x_scale = None
            if expert_x.numel() != 0:
                expert_x, expert_x_scale = self._do_quant(expert_x,
                                                          a1_scale,
                                                          per_act_token=False)

        return (expert_x, expert_x_scale, expert_num_tokens, expert_topk_ids,
                expert_topk_weights)

    def _apply_weights_and_reduce(self, num_tokens: int,
                                  fused_expert_output: torch.Tensor,
                                  topk_weights: torch.Tensor,
                                  apply_router_weight_on_input: bool,
                                  output_dtype: torch.dtype):

        hidden_dim = fused_expert_output.size(-1)
        if fused_expert_output.ndim == 2:
            fused_expert_output = fused_expert_output.view(
                num_tokens, -1, hidden_dim)

        if not apply_router_weight_on_input:
            # The DeepEP combine kernels don't do the topk weight
            # multiplication. We multiply the weights locally.
            m_x_topk = fused_expert_output.size(0)
            fused_expert_output.mul_(topk_weights.view(m_x_topk, -1, 1))

        out = torch.empty((num_tokens, hidden_dim),
                          device=fused_expert_output.device,
                          dtype=output_dtype)
        ops.moe_sum(fused_expert_output, out)

        return out

    def finalize(self, output: torch.Tensor, fused_expert_output: torch.Tensor,
                 topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                 apply_router_weight_on_input: bool) -> None:

        assert self.handle is not None

        # fused_expert_output can have 0 tokens - This happens when none of the
        # tokens from the all2all reach this EP rank.
        if fused_expert_output.numel() != 0:
            fused_expert_output = self._apply_weights_and_reduce(
                num_tokens=topk_ids.size(0),
                fused_expert_output=fused_expert_output,
                topk_weights=topk_weights,
                apply_router_weight_on_input=apply_router_weight_on_input,
                output_dtype=output.dtype)

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
