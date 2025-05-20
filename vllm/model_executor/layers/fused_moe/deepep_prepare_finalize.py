# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import deep_ep
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input)


class DeepEPPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):

    # TODO (varun) : Expose internode / low-latency mode kernels
    def __init__(
            self,
            buffer: deep_ep.Buffer,
            # maybe just pass in ep rank ??
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
        # TODO (varun) : Exercise x2_scale in tests
        do_debug = False

        if do_debug:
            torch.cuda.synchronize()
            s = f"topk_ids {rank_topk_ids} \n"
            s += f"num_experts {num_experts} \n"
            print(s, flush=True)

        # TODO (varun) : application of router weights and quantization is
        # common to all implementations - factor out
        if apply_router_weight_on_input:
            topk = rank_topk_ids.size(1)
            # TODO: this only works for topK=1, will need to update for topK>1
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1")
            a1 = a1 * rank_topk_weights.to(a1.dtype)

        per_act_token = a1_scale.numel() != 1 if a1_scale is not None else (
            a2_scale.numel() != 1 if a2_scale is not None else False)

        a1q, a1q_scale = moe_kernel_quantize_input(a1, a1_scale,
                                                   self.quant_dtype,
                                                   per_act_token,
                                                   self.block_shape)

        (num_tokens_per_rank, num_tokens_per_rdma_rank, expert_num_tokens,
         is_token_in_rank, event) = self.buffer.get_dispatch_layout(
             topk_idx=rank_topk_ids,
             num_experts=num_experts,
             previous_event=None,
             async_finish=False,
             allocate_on_comm_stream=False)

        if do_debug:
            torch.cuda.synchronize()
            s = f"num_tokens_per_rank {num_tokens_per_rank} \n"
            s += f"num_tokens_per_rdma_rank {num_tokens_per_rdma_rank} \n"
            s += f"num_tokens_per_expert {expert_num_tokens} \n"
            s += f"is_token_in_rank {is_token_in_rank} \n"
            print(s, flush=True)

        token_data = a1q
        if a1q_scale is not None:
            token_data = (a1q, a1q_scale)

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
            expert_alignment=
            1,  # TODO (varun) : set this properly and avoid moe_alighn kernel ?
            config=None,
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=False)

        if do_debug:
            torch.cuda.synchronize()
            s = ""
            if isinstance(token_data, tuple):
                s += f"expert_x : {token_data[0].shape} {token_data[0]}"
                s += f"expert_x_scale : {token_data[1].shape} {token_data[1]}"
            else:
                s += f"expert_x : {token_data.shape} {token_data} \n"
            s += f"expert_topk_ids : {expert_topk_ids} \n"
            s += f"expert_topk_weights : {expert_topk_weights} \n"
            s += ("expert_num_tokens_per_expert_list : "
                  "{expert_num_tokens_per_expert_list} \n")
            print(s, flush=True)

        if self.quant_dtype is not None:
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

        combined_x, combined_topk_weights, event = self.buffer.combine(
            x=fused_expert_output,
            handle=self.handle,
            topk_weights=topk_weights,
            config=None,
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=False)
        # TODO (varun) : update interface to avoid this explicit copy
        output.copy_(combined_x)
