# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import deep_ep
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk

from vllm.distributed.communication_op import tensor_model_parallel_all_gather
from vllm.distributed.parallel_state import get_dp_group
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.utils import cdiv, round_up


def indices_to_map(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_of_tokens: int,
    num_of_experts: int,
):
    """
    Map the map to the indices.
    """
    # Generate the routing map and the probs according to the topk_ids and topk_weights.
    assert topk_ids is not None
    routing_map = torch.zeros(
        num_of_tokens, num_of_experts, device="cuda", dtype=torch.bool
    )
    routing_map = routing_map.scatter(1, topk_ids.to(torch.int64), 1).bool()
    if topk_weights is not None:
        probs = torch.zeros(
            num_of_tokens, num_of_experts, device="cuda", dtype=torch.float32
        )
        probs = probs.scatter(1, topk_ids.to(torch.int64), topk_weights)
    else:
        probs = None

    topk = topk_ids.shape[1]
    print(
        f"ROUTING_MAP = {routing_map.shape} {routing_map.nonzero().flatten().sum().item()}"
    )
    print(f"topk = {topk}, num_experts = {num_of_experts}")
    # print(f"TOPK_WEIGHTS={topk_weights}")

    # unscat = torch.gather(probs, 1, topk_ids)
    # print(f"UNSCAT {unscat}")

    return routing_map, probs, topk_ids


def balanced_indices_to_map(
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_of_tokens: int,
    num_of_experts: int,
):
    seq_len, topk = topk_idx.size()
    routing_map = torch.zeros(
        num_of_tokens, num_of_experts, device="cuda", dtype=torch.bool
    )
    probs = torch.zeros(
        num_of_tokens, num_of_experts, device="cuda", dtype=torch.float32
    )
    for i in range(seq_len):
        # topk_weights[i, :] = torch.rand(topk, device="cuda", dtype=torch.float32)
        # Force balanced routing for testing
        if True:
            selected_experts = torch.tensor(
                [
                    ((i * topk) % num_of_experts + val) % num_of_experts
                    for val in range(topk)
                ],
                dtype=torch.int64,
                device="cuda",
            )
        else:
            selected_experts = (
                topk_idx  # torch.randperm(num_of_experts, device="cuda")[:topk]
            )
        topk_idx[i, :] = selected_experts.to(torch.int64)
        routing_map[i, selected_experts] = True
        probs[i, selected_experts] = topk_weights[i, :]

    print(
        f"ROUTING_MAP = {routing_map.shape} {routing_map.nonzero().flatten().sum().item()}, {probs.shape}"
    )

    return routing_map, probs, topk_idx


class DeepEPHybridPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """
    Prepare/Finalize using DeepEP Hybrid kernels.
    """

    @staticmethod
    def maybe_roundup_layer_hidden_size(hidden_size: int, dtype: torch.dtype) -> int:
        # Round up hidden size so it is compatible with DeepEP Hybrid
        # kernels.
        alignment = 128
        return round_up(hidden_size, alignment)

    def __init__(
        self,
        buffer: deep_ep.HybridEPBuffer,
        num_dispatchers: int,
        dp_size: int,  # Needed?
        rank_expert_offset: int,
        num_local_experts: int,
    ):
        super().__init__()
        self.buffer = buffer
        self.num_dispatchers_ = num_dispatchers
        self.dp_size = dp_size
        self.rank_expert_offset = rank_expert_offset  # TODO: not needed
        self.handle = None
        self.expert_probs = None
        self.do_permute = False
        self.num_local_experts = num_local_experts

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        return True

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int64

    def supports_async(self) -> bool:
        return False

    def setup_handle(self, hidden_dim, num_of_tokens, num_of_experts, routing_map):
        config = self.buffer.update_template_config(
            hidden_dim=hidden_dim,
            max_num_of_tokens_per_rank=num_of_tokens,
        )

        if True:
            # print(f"ALL GATHER BEGIN {routing_map.shape}")
            global_routing_map = get_dp_group().all_gather(routing_map, dim=0)
            # print(f"ALL GATHER END {global_routing_map.shape, (num_of_tokens * self.buffer.group_size, num_of_experts)}")
        else:
            # The hybrid-ep kernel requires the routing info from all ranks.
            global_routing_map = torch.empty(
                num_of_tokens * self.buffer.group_size,
                num_of_experts,
                device="cuda",
                dtype=torch.bool,
            )
            torch.distributed.all_gather_into_tensor(
                global_routing_map, routing_map, self.buffer.group
            )

        # Run the metadata preprocessing kernel.
        (
            sparse_to_dense_map,
            rdma_to_attn_map,
            attn_to_rdma_map,
            num_dispatched_tokens_tensor,
            local_expert_routing_map,
        ) = self.buffer.runtime.metadata_preprocessing(
            config=config,
            routing_map=global_routing_map,
            num_of_tokens_per_rank=num_of_tokens,
        )
        # Create the handle using the data generated by the preprocessing kernel.
        return (
            sparse_to_dense_map,
            rdma_to_attn_map,
            attn_to_rdma_map,
            num_dispatched_tokens_tensor,
            local_expert_routing_map,
            num_of_tokens,
            config,
        )

    def p(self, msg, force=False):
        if force:
            print(msg)

    def pp(self, msg, t, force=False):
        if force:
            print(
                f"{msg}[{self.rank_expert_offset}] = {t.shape if t is not None else None}"
            )
        elif False:
            print(
                f"{msg}[{self.rank_expert_offset}] = {t.shape if t is not None else None}\n{t}"
            )

    def create_new_topk_data(
        self,
        expert_x,
        local_expert_routing_map,
        sparse_to_dense_map,
        expert_probs,
        topk_ids,
        topk_weights,
    ):  # -> tuple[torch.Tensor,torch.Tensor]:
        if self.do_permute:
            M_sum, K = expert_x.shape
            # Are these interleaved?
            return topk_ids.view(-1, 1)[:M_sum], topk_weights.view(-1, 1)[:M_sum]

        # TODO: use all_gatherv
        all_topk_ids = get_dp_group().all_gather(topk_ids, dim=0)
        all_topk_weights = get_dp_group().all_gather(topk_weights, dim=0)

        #self.pp("ALL_TOPK_IDS", all_topk_ids)

        start = self.rank_expert_offset
        end = self.rank_expert_offset + self.num_local_experts

        # subtract? use oob expert idx?
        oob_idx = self.num_local_experts if self.rank_expert_offset == 0 else 0
        assert (all_topk_ids == oob_idx).all() == False
        new_topk_ids = torch.where((all_topk_ids >= start) & (all_topk_ids < end), all_topk_ids, oob_idx)
        new_topk_weights = torch.where(all_topk_ids != oob_idx, all_topk_weights, 0.0)

        mask = ~torch.all(new_topk_ids == oob_idx, dim=1)
        #self.pp("MASK", mask)
        new_topk_ids = new_topk_ids[mask]
        new_topk_weights = new_topk_weights[mask]

        #self.pp("NEW_TOPK_IDS_PRE", new_topk_ids)

        assert new_topk_ids.shape[0] == expert_x.shape[0], f"{new_topk_ids.shape} == {expert_x.shape}"

        return new_topk_ids, new_topk_weights

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
    ) -> mk.PrepareResultType:
        torch.set_printoptions(profile="full")

        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            # TODO: this only works for topK=1, will need to update for topK>1
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        M, K = a1.shape

        # topk_ids = torch.arange(topk_ids.size(1), device=topk_ids.device).expand(topk_ids.size(0), -1)

        if quant_config.is_block_quantized:
            # Quant and Dispatch
            assert quant_config.block_shape == [128, 128]  # TODO: use constant
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
            assert a1q_scale.shape == quant_config.scale_shape(M, K)
        else:
            a1q = a1
            a1_post_scale = quant_config.a1_scale
            if quant_config.quant_dtype is not None:
                a1q_scale = torch.ones(
                    (M, cdiv(K, 128)), device=a1.device, dtype=torch.float32
                )  # hack
            else:
                a1q_scale = None

        self.p(
            f"DISPATCH BEGIN[{self.rank_expert_offset}], a1q={a1q.shape} a1q_s={a1q_scale.shape if a1q_scale is not None else None}"
        )
        self.pp("TOPK_IDS", topk_ids)
        self.p(f"M, K [{self.rank_expert_offset}] = {M, K}")

        # self.p(f"TOPK_WEIGHTS[{self.rank_expert_offset}] = {topk_weights.shape}\n{topk_weights}")

        assert num_experts > 0

        if False:
            routing_map, probs, _ = indices_to_map(
                topk_ids, topk_weights, M, num_experts
            )
        elif False:
            routing_map, probs, topk_ids = balanced_indices_to_map(
                topk_ids, topk_weights, M, num_experts
            )
        else:
            routing_map = None
            probs = None

        if False and routing_map is not None:
            handle = self.setup_handle(K, M, num_experts, routing_map)
        else:
            handle = None

        self.pp("A1Q", a1q)

        if not self.do_permute:
            (expert_x, expert_probs, expert_x_scale, handle) = self.buffer.dispatch(
                hidden=a1q,
                scaling_factor=a1q_scale,
                topk_idx=topk_ids,
                topk_weights=topk_weights,
                routing_map=routing_map,
                probs=probs,
                handle=handle,
                num_dispatched_tokens=None,
                num_of_experts=num_experts,
            )

            self.pp("EXPERT_X", expert_x)

            (
                sparse_to_dense_map,
                rdma_to_attn_map,
                attn_to_rdma_map,
                num_of_tokens_for_experts,
                local_expert_routing_map,  #
                num_tokens,
                config,
            ) = handle

            self.p(
                f"NUM_TOK_PER_EXPERT[{self.rank_expert_offset}]={num_of_tokens_for_experts.item()}"
            )
            num_dispatched = num_of_tokens_for_experts.item()
        else:
            topk = topk_ids.shape[1]
            (expert_x, expert_probs, expert_x_scale, tokens_per_expert, handle) = (
                self.buffer.dispatch_with_permute(
                    hidden=a1q,
                    scaling_factor=a1q_scale,
                    topk_idx=topk_ids,
                    topk_weights=topk_weights,
                    routing_map=routing_map,
                    probs=probs,
                    handle=handle,
                    num_dispatched_tokens=None,
                    num_of_experts=num_experts,
                    # pad_multiple=topk,
                )
            )

            self.pp("TOKENS_PER_EXPERT", tokens_per_expert)

            (
                sparse_to_dense_map,
                rdma_to_attn_map,
                attn_to_rdma_map,
                num_dispatched_tokens_tensor,
                local_expert_routing_map,
                row_id_map,
                num_tokens,
                config,
            ) = handle

            num_dispatched = num_dispatched_tokens_tensor.item()

            self.p(f"NUM_TOKENS = {num_tokens}")
            self.pp("NUM_DISPATCHED_TOKENS_TENSOR", num_dispatched_tokens_tensor)
            # self.pp("ROW_ID_MAP", row_id_map)

        self.handle = handle
        self.expert_probs = expert_probs
        assert self.handle is not None

        self.pp("PROBS", self.expert_probs)
        self.pp("S2D", sparse_to_dense_map)

        self.p(
            f"DISPATCH END[{self.rank_expert_offset}], x={expert_x.shape} x_s={expert_x_scale.shape if expert_x_scale is not None else None}"
        )

        #assert num_dispatched == local_expert_routing_map.shape[0]

        self.pp("LERM", local_expert_routing_map)

        new_topk_ids, new_topk_weights = self.create_new_topk_data(
            expert_x,
            local_expert_routing_map,
            sparse_to_dense_map,
            expert_probs,
            topk_ids,
            topk_weights,
        )

        self.pp("NEW_TOPK_IDS", new_topk_ids)
        self.pp("NEW_TOPK_WEIGHTS", new_topk_weights)

        # N/A
        expert_tokens_meta = None

        # Dispatch and Quant
        # DeepEP kernels only support dispatching block-quantized
        # activation scales.
        # Dispatch in bfloat16 and quantize afterwards
        if not quant_config.is_block_quantized and quant_config.quant_dtype is not None:
            # Quantize after dispatch.
            expert_x_scale = None
            if expert_x.numel() != 0:
                expert_x, expert_x_scale = moe_kernel_quantize_input(
                    expert_x,
                    a1_post_scale,
                    quant_dtype=quant_config.quant_dtype,
                    per_act_token_quant=quant_config.per_act_token_quant,
                    block_shape=quant_config.block_shape,
                )

        # TODO
        assert new_topk_ids.shape[0] == expert_x.shape[0], f"{topk_ids.shape[0]} == {expert_x.shape[0]}"
        assert new_topk_weights.shape[0] == expert_x.shape[0], f"{topk_weights.shape[0]} == {expert_x.shape[0]}"

        return (
            expert_x,
            expert_x_scale,
            expert_tokens_meta,
            new_topk_ids,
            new_topk_weights,
        )

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        # fused_expert_output can have 0 tokens - This happens when none of the
        # tokens from the all2all reach this EP rank.
        # TODO(bnell): check if this is still needed
        self.p(f"M, K [{self.rank_expert_offset}] = {output.shape}")

        if False and fused_expert_output.numel() == 0:
            fused_expert_output = torch.empty((1, fused_expert_output.shape[1]), device=fused_expert_output.device, dtype=fused_expert_output.dtype)
        self.p(
            f"COMBINE BEGIN[{self.rank_expert_offset}] {fused_expert_output.dtype} out={output.shape} fe_out={fused_expert_output.shape}"
        )

        #assert self.expert_probs.numel() > 0
        if self.expert_probs.numel() == 0:
            print(f"PROBS NUMEL == 0 {self.expert_probs.shape}")


        if not self.do_permute:
            combined_x, combined_probs = self.buffer.combine(
                hidden=fused_expert_output,
                probs=None, #self.expert_probs,  # None?
                handle=self.handle,
            )
        else:
            topk = topk_ids.shape[1]
            combined_x, combined_probs = self.buffer.combine_with_unpermute(
                hidden=fused_expert_output,
                probs=self.expert_probs,  # None?
                handle=self.handle,
                # pad_multiple=topk,
            )
        self.p(f"COMBINE END[{self.rank_expert_offset}] {combined_x.shape} {combined_probs.shape if combined_probs is not None else None}")

        # TODO(bnell): Double check this
        # top_k = topk_ids.shape[1]
        # combined_x = combined_x / top_k

        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
            weight_and_reduce_impl = TopKWeightAndReduceContiguous()

        self.p(f"REDUCDER = {weight_and_reduce_impl}")

        weight_and_reduce_impl.apply(
            output=combined_x,
            fused_expert_output=output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
