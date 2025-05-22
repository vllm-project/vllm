# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import deep_ep
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops


class DeepEPLLPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):

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

        if apply_router_weight_on_input:
            topk = rank_topk_ids.size(1)
            # TODO: this only works for topK=1, will need to update for topK>1
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1")
            a1 = a1 * rank_topk_weights.to(a1.dtype)

        #per_act_token = a1_scale.numel() != 1 if a1_scale is not None else (
        #    a2_scale.numel() != 1 if a2_scale is not None else False)

        expert_x, expert_num_tokens, self.handle, event, hook = \
                self.buffer.low_latency_dispatch(a1,
                                                rank_topk_ids,
                                                self.max_tokens_per_rank,
                                                num_experts,
                                                use_fp8=False,
                                                async_finish=False,
                                                return_recv_hook=False)

        expert_x_scale = None
        return (expert_x, expert_x_scale, expert_num_tokens, None, None)

    def finalize(self, output: torch.Tensor, fused_expert_output: torch.Tensor,
                 topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                 apply_router_weight_on_input: bool) -> torch.Tensor:

        assert self.handle is not None

        # The DeepEP kernels don't seem to do the topk weight multiplication.
        # We multiply the weights locally.
        if not apply_router_weight_on_input:
            # TODO (varun) : fix inefficiencies
            fused_expert_output.mul_(
                topk_weights.view(fused_expert_output.size(0), -1, 1))
            num_tokens = topk_ids.size(0)
            hidden_dim = fused_expert_output.size(-1)
            local_out_shape = (num_tokens, hidden_dim)
            local_out = torch.zeros(local_out_shape,
                                    device="cuda",
                                    dtype=torch.float32)
            ops.moe_sum(fused_expert_output, local_out)
            fused_expert_output = local_out

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
