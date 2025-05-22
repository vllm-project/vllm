# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import deep_ep
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input)


class DeepEPLLPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):

    # DeepEP low-latency kernels are compiled only for certain
    # specific hidden sizes.
    SUPPORTED_HIDDEN_SIZES = [2560, 4096, 5120, 7168]

    # TODO (varun) : Expose internode / low-latency mode kernels
    def __init__(
            self,
            buffer: deep_ep.Buffer,
            # maybe just pass in ep rank ??
            world_size: int,
            rank: int,
            dp_size: int,
            rank_expert_offset: int,
            max_tokens_per_rank: int,
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
        self.max_tokens_per_rank = max_tokens_per_rank
        # The dispatch function returns a handle that the combine function
        # requires. We store the handle here so it is available to the
        # combine function.
        self.handle = None

    def max_num_tokens_per_dp_rank(self) -> Optional[int]:
        return self.max_tokens_per_rank

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        rank_topk_weights: torch.Tensor,
        rank_topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],  # TODO (varun) : Unused - remove
        apply_router_weight_on_input: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:

        hidden_size = a1.size(1)
        assert hidden_size in self.SUPPORTED_HIDDEN_SIZES, \
            (f"Hidden Size {hidden_size} not in supported list of hidden sizes"
            "{self.SUPPORTED_HIDDEN_SIZES}")

        # Quantize
        per_act_token = a1_scale.numel() != 1 if a1_scale is not None else (
            a2_scale.numel() != 1 if a2_scale is not None else False)
        assert not per_act_token, (
            "low_latency kernels don't support per-act-token quant")

        if apply_router_weight_on_input:
            topk = rank_topk_ids.size(1)
            # TODO: this only works for topK=1, will need to update for topK>1
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1")
            a1 = a1 * rank_topk_weights.to(a1.dtype)

        # Dispatch
        expert_x, expert_num_tokens, self.handle, event, hook = \
                self.buffer.low_latency_dispatch(a1,
                                                rank_topk_ids,
                                                self.max_tokens_per_rank,
                                                num_experts,
                                                use_fp8=False,
                                                async_finish=False,
                                                return_recv_hook=False)

        num_experts = expert_x.size(0)
        hidden_dim = expert_x.size(-1)

        expert_x = expert_x.view((-1, expert_x.size(-1)))
        expert_x, expert_x_scale = moe_kernel_quantize_input(
            expert_x, a1_scale, self.quant_dtype, per_act_token,
            self.block_shape)
        expert_x = expert_x.view((num_experts, -1, hidden_dim))

        return (expert_x, expert_x_scale, expert_num_tokens, None, None)

    def finalize(self, output: torch.Tensor, fused_expert_output: torch.Tensor,
                 topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                 apply_router_weight_on_input: bool) -> torch.Tensor:

        assert self.handle is not None

        _, event, hook = self.buffer.low_latency_combine(
            fused_expert_output,
            topk_ids,
            topk_weights,
            self.handle,
            async_finish=False,
            zero_copy=False,
            return_recv_hook=False,
            out=output)
        return output
