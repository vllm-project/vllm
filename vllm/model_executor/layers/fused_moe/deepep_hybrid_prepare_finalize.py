# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Callable, Optional, Union

import deep_ep
import torch
import torch.nn.functional as F

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous, TopKWeightAndReduceDelegate)
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input)
from vllm.utils import round_up
from vllm.v1.worker.ubatching import (
    dbo_current_ubatch_id, dbo_enabled, dbo_switch_to_comm,
    dbo_switch_to_compute, dbo_switch_to_compute_sync,
    dbo_yield_and_switch_from_comm_to_compute,
    dbo_yield_and_switch_from_compute_to_comm)


class DeepEPHybridPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """
    Prepare/Finalize using DeepEP High-Throughput kernels.
    """

    @staticmethod
    def maybe_roundup_layer_hidden_size(hidden_size: int,
                                        dtype: torch.dtype) -> int:
        # Round up hidden size so it is compatible with DeepEP High Throughput
        # kernels.
        # DeepEP intranode kernels make copies in units of,
        # 32(warp-size) int4 elements. Round up hidden size to respect this.
        # For example, an input hidden size of 2880 with dtype torch.bfloat16
        # will be rounded up to 3072.
        hidden_size_bytes = hidden_size * dtype.itemsize
        xfer_atom_size = 512  # 32 * 16 (size(int4))
        if hidden_size_bytes % xfer_atom_size == 0:
            return hidden_size

        hidden_size_bytes = round_up(hidden_size_bytes, xfer_atom_size)
        return hidden_size_bytes // dtype.itemsize

    def __init__(self, buffer: deep_ep.HybridEpBuffer, num_dispatchers: int,
                 dp_size: int, rank_expert_offset: int):
        super().__init__()
        self.buffer = buffer
        self.num_dispatchers_ = num_dispatchers
        self.dp_size = dp_size
        self.rank_expert_offset = rank_expert_offset #?
        self.handle = None
        self.expert_probs = None

        # From https://github.com/deepseek-ai/DeepEP/blob/9fe9021f29c9083cd1808ab36b740208524d9f63/deep_ep/buffer.py#L164
        self.available_rank_configs = [2, 4, 8, 16, 24, 32, 64, 128, 144, 160]

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        return True

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> Optional[int]:
        return None

    def topk_indices_dtype(self) -> Optional[torch.dtype]:
        return torch.int64

    def _get_dispatch_config(self) -> Optional[deep_ep.Config]:
        if self.num_dispatchers_ not in self.available_rank_configs:
            return None
        return deep_ep.Buffer.get_dispatch_config(self.num_dispatchers_)

    def _get_combine_config(self) -> Optional[deep_ep.Config]:
        if self.num_dispatchers_ not in self.available_rank_configs:
            return None
        return deep_ep.Buffer.get_combine_config(self.num_dispatchers_)

    def supports_async(self) -> bool:
        return False # combine async not supported

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
    ) -> mk.PrepareResultType:

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
                quant_config.a1_scale,
                quant_dtype=quant_config.quant_dtype,
                per_act_token_quant=quant_config.per_act_token_quant,
                block_shape=quant_config.block_shape,
            )
            if a1q_scale is not None and a1q_scale.numel() == 1:
                a1q_scale = a1q_scale.view(1, 1)
            a1_post_scale = None
        else:
            a1q = a1
            a1q_scale = torch.ones(1, device=a1.device, dtype=torch.float32) # hack
            a1_post_scale = quant_config.a1_scale

        # use dispatch_with_permute/combine_with_unpermute?
        if True:
            (
                expert_x, expert_probs, expert_x_scale, handle
            ) = self.buffer.dispatch(
                hidden=a1q,
                scaling_factor=a1q_scale,
                topk_idx=topk_ids,
                topk_weights=topk_weights,
                routing_map=None, # None = generated dynamically
                handle=None,
                num_dispatched_tokens=-1, #??
            )

            self.handle = handle
            self.expert_probs = expert_probs

            (sparse_to_dense_map,
             rdma_to_attn_map,
             attn_to_rdma_map,
             num_of_tokens_for_experts,
             local_expert_routing_map, #
             num_tokens) = self.handle

        else:
            (
                expert_x, expert_probs, expert_x_scale, tokens_per_expert, handle
            ) = self.buffer.dispatch_with_permute(
                hidden=a1q,
                scaling_factor=a1q_scale,
                topk_idx=topk_ids,
                topk_weights=topk_weights,
                routing_map=None, # None = generated dynamically
                handle=None,
                num_dispatched_tokens=-1, #??
            )

            self.handle = handle
            self.expert_probs = expert_probs

            (sparse_to_dense_map,
             rdma_to_attn_map,
             attn_to_rdma_map,
             num_dispatched_tokens_tensor,
             local_expert_routing_map,
             row_id_map,
             num_tokens) = self.handle

        topk = topk_ids.size(1)
        if topk == 1 and expert_probs.dim() == 1:
            expert_probs = expert_probs.view(expert_probs.shape(0), topk)


        #num_of_tokens_for_experts = num_of_tokens_for_experts.cpu()

        print(f"POST DISPATCH {self.rank_expert_offset} {topk} {expert_x.shape} {expert_probs.shape}")

        if False:
            print(f"STUFF\n"
                  f"rank_exp_offset = {self.rank_expert_offset}\n"
                  f"a={a1q.shape}/{a1q.dtype} -> {expert_x.shape}/{expert_x.dtype}\n"
                  f"topk_ids={topk_ids.shape}\n"
                  f"tok_for_exp={num_of_tokens_for_experts}\n"
                  f"probs={expert_probs.shape}\n"
                  f"lem shape={local_expert_routing_map.shape}, {local_expert_routing_map[:num_of_tokens_for_experts].shape}\n"
                  f"lem numel={local_expert_routing_map.nonzero().numel()}\n"
                  #f"lem={local_expert_routing_map}\n"
                  f"lem sum={local_expert_routing_map.sum(dim=1).shape}\n"
                  f"sparse_to_dense_map={sparse_to_dense_map.shape} {sparse_to_dense_map.dtype} {sparse_to_dense_map}\n"
                  f"rdma_to_attn_map={rdma_to_attn_map.shape} {rdma_to_attn_map.dtype} {rdma_to_attn_map}\n"
                  f"attn_to_rdma_map={attn_to_rdma_map.shape} {attn_to_rdma_map.dtype}\n"
                  f"num_tokens={num_tokens}\n"
                  )

        #local_expert_routing_map = local_expert_routing_map[:num_of_tokens_for_experts.item()]

        # TBD
        new_topk_ids = None

        # N/A
        expert_tokens_meta = None

        # Dispatch and Quant
        # DeepEP kernels only support dispatching block-quantized
        # activation scales.
        # Dispatch in bfloat16 and quantize afterwards
        if not quant_config.is_block_quantized:
            # Quantize after dispatch.
            expert_x_scale = None
            if expert_x.numel() != 0:
                expert_x, expert_x_scale = moe_kernel_quantize_input(
                    expert_x,
                    a1_post_scale,
                    quant_dtype=quant_config.quant_dtype,
                    per_act_token_quant=False,
                    block_shape=quant_config.block_shape)

        return (expert_x, expert_x_scale, expert_tokens_meta, new_topk_ids, expert_probs)

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        # assert isinstance(
        #     weight_and_reduce_impl, TopKWeightAndReduceDelegate
        # ), f"Weight application and reduction happens in the combine kernel. {weight_and_reduce_impl}"

        # fused_expert_output can have 0 tokens - This happens when none of the
        # tokens from the all2all reach this EP rank.
        if False and fused_expert_output.numel() != 0:
            if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
                weight_and_reduce_impl = TopKWeightAndReduceContiguous()
            fused_expert_output = weight_and_reduce_impl.apply(
                output=None,
                fused_expert_output=fused_expert_output,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )

        if False:
            print(f"\nCOMBINE START({self.rank_expert_offset})\n")

        if True:
            combined_x, combined_probs = self.buffer.combine(
                hidden=fused_expert_output,
                probs=self.expert_probs,  # None?
                handle=self.handle,
            )
        else:
            combined_x, combined_probs = self.buffer.combine_with_unpermute(
                hidden=fused_expert_output,
                probs=self.expert_probs,  # None?
                handle=self.handle,
            )

        top_k = topk_ids.shape[1]

        combined_x = combined_x / top_k

        # multiply by probs?
        if False:
            print(f"\nCOMBINE END({self.rank_expert_offset}) {combined_x.shape}/{combined_x.dtype}\n")

        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
            weight_and_reduce_impl = TopKWeightAndReduceContiguous()

        weight_and_reduce_impl.apply(
            output=combined_x,
            fused_expert_output=output,
            topk_weights=combined_probs,
            topk_ids=topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input)

        # TODO(lucas): support this case with the refactored modular kernel
        # Respect inplace outputs.
        # apply weights???
        #output.copy_(combined_x, non_blocking=True)
