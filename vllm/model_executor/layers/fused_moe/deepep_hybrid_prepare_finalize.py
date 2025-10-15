# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import deep_ep
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk

# from vllm.distributed.communication_op import tensor_model_parallel_all_gather
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


def do_some_shit(routing_map, probs):
    num_tokens, num_experts = routing_map.shape
    routing_map = routing_map.bool().T.contiguous()
    token_indices = (
        torch.arange(num_tokens, device=routing_map.device)
        .unsqueeze(0)
        .expand(num_experts, -1)
    )
    sorted_indices = token_indices.masked_select(routing_map)
    print(f"SORTED_INDICES = {sorted_indices.shape, sorted_indices}")


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

    def pp(self, msg, t):
        if False:
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
        num_dispatched,
    ):  # -> tuple[torch.Tensor,torch.Tensor]:
        if self.do_permute:
            M_sum, K = expert_x.shape
            # Are these interleaved?
            return topk_ids.view(-1, 1)[:M_sum], topk_weights.view(-1, 1)[:M_sum]

        start = self.rank_expert_offset
        end = self.rank_expert_offset + self.num_local_experts

        # subtract? use oob expert idx?
        oob_idx = self.num_local_experts if self.rank_expert_offset == 0 else 0
        assert (topk_ids == oob_idx).all() == False
        new_topk_ids = torch.where((topk_ids >= start) & (topk_ids < end), topk_ids, oob_idx)
        new_topk_weights = torch.where(topk_ids != oob_idx, topk_weights, 0.0)

        return new_topk_ids[:num_dispatched], new_topk_weights[:num_dispatched]

        num_dispatched = local_expert_routing_map.shape[0]
        topk = topk_ids.shape[1]

        # Initialize output tensors with zeros
        new_topk_ids = torch.zeros(
            (num_dispatched, topk),
            dtype=torch.int64,
            device=local_expert_routing_map.device,
        )
        new_topk_weights = torch.zeros(
            (num_dispatched, topk), dtype=topk_weights.dtype, device=topk_weights.device
        )

        # Extract expert assignments from local_expert_routing_map
        # For each dispatched token, find which local experts it's assigned to
        token_indices, local_expert_indices = local_expert_routing_map.nonzero(
            as_tuple=True
        )

        # Convert local expert indices to global expert IDs by adding rank offset
        global_expert_ids = local_expert_indices + self.rank_expert_offset

        # Count how many experts each token is assigned to
        experts_per_token = local_expert_routing_map.sum(dim=1)

        # Create position indices for each expert assignment within a token
        # This helps us place the expert IDs in the correct position within the topk dimension
        cumsum = torch.cat(
            [
                torch.tensor([0], device=local_expert_routing_map.device),
                experts_per_token.cumsum(0)[:-1],
            ]
        )
        positions = (
            torch.arange(len(token_indices), device=local_expert_routing_map.device)
            - cumsum[token_indices]
        )

        # Only keep assignments that fit within topk
        valid_mask = positions < topk
        valid_token_indices = token_indices[valid_mask]
        valid_positions = positions[valid_mask]
        valid_global_expert_ids = global_expert_ids[valid_mask]
        valid_local_expert_indices = local_expert_indices[valid_mask]

        # Fill in new_topk_ids with global expert IDs
        new_topk_ids[valid_token_indices, valid_positions] = valid_global_expert_ids

        # For weights, we need to look up the original weights from the source tokens
        # sparse_to_dense_map[i] gives the original token index for dispatched token i
        if sparse_to_dense_map is not None and len(valid_token_indices) > 0:
            # Get original token indices for the valid dispatched tokens
            original_token_indices = sparse_to_dense_map[valid_token_indices]

            # For each valid assignment, find the weight from the original topk_weights
            # We need to find which position in the original topk_ids matches our expert
            for i in range(len(valid_token_indices)):
                dispatched_tok = valid_token_indices[i]
                orig_tok = original_token_indices[i]
                pos = valid_positions[i]
                expert_id = valid_global_expert_ids[i]

                # Find this expert in the original topk_ids for this token
                orig_topk_mask = topk_ids[orig_tok] == expert_id
                if orig_topk_mask.any():
                    # Get the weight from the original topk_weights
                    weight_idx = orig_topk_mask.nonzero(as_tuple=True)[0][0]
                    new_topk_weights[dispatched_tok, pos] = topk_weights[
                        orig_tok, weight_idx
                    ]

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

        print(
            f"DISPATCH BEGIN[{self.rank_expert_offset}], a1q={a1q.shape} a1q_s={a1q_scale.shape if a1q_scale is not None else None}"
        )
        self.pp("TOPK_IDS", topk_ids)
        print(f"M, K [{self.rank_expert_offset}] = {M, K}")

        # print(f"TOPK_WEIGHTS[{self.rank_expert_offset}] = {topk_weights.shape}\n{topk_weights}")

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

            self.handle = handle
            self.expert_probs = expert_probs
            assert self.handle is not None

            (
                sparse_to_dense_map,
                rdma_to_attn_map,
                attn_to_rdma_map,
                num_of_tokens_for_experts,
                local_expert_routing_map,  #
                num_tokens,
                config,
            ) = self.handle

            print(
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

            self.handle = handle
            self.expert_probs = expert_probs
            assert self.handle is not None

            (
                sparse_to_dense_map,
                rdma_to_attn_map,
                attn_to_rdma_map,
                num_dispatched_tokens_tensor,
                local_expert_routing_map,
                row_id_map,
                num_tokens,
                config,
            ) = self.handle

            num_dispatched = num_dispatched_tokens_tensor.item()

            print(f"NUM_TOKENS = {num_tokens}")
            self.pp("NUM_DISPATCHED_TOKENS_TENSOR", num_dispatched_tokens_tensor)
            # self.pp("ROW_ID_MAP", row_id_map)

        self.pp("PROBS", expert_probs)
        self.pp("S2D", sparse_to_dense_map)

        print(
            f"DISPATCH END[{self.rank_expert_offset}], x={expert_x.shape} x_s={expert_x_scale.shape if expert_x_scale is not None else None}"
        )

        #assert num_dispatched == local_expert_routing_map.shape[0]

        self.pp("LERM", local_expert_routing_map)

        # Trim local_expert_routing_map to actual number of dispatched tokens
        local_expert_routing_map = local_expert_routing_map[:num_dispatched]

        new_topk_ids, new_topk_weights = self.create_new_topk_data(
            expert_x,
            local_expert_routing_map,
            sparse_to_dense_map,
            expert_probs,
            topk_ids,
            topk_weights,
            num_dispatched,
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
        # assert new_topk_ids.shape[0] == expert_x.shape[0], f"{topk_ids.shape[0]} == {expert_x.shape[0]}"

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
        print(f"M, K [{self.rank_expert_offset}] = {output.shape}")

        if True or fused_expert_output.numel() != 0:
            print(
                f"COMBINE BEGIN[{self.rank_expert_offset}] {fused_expert_output.dtype} out={output.shape} fe_out={fused_expert_output.shape}"
            )
            if not self.do_permute:
                combined_x, combined_probs = self.buffer.combine(
                    hidden=fused_expert_output,
                    probs=self.expert_probs,  # None?
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
            print(f"COMBINE END[{self.rank_expert_offset}] {combined_x.shape}")
        else:
            combined_x = None
            combined_probs = topk_weights
            output = fused_expert_output
            print(f"COMBINE EMPTY END [{self.rank_expert_offset}]")

        # TODO(bnell): Double check this
        # top_k = topk_ids.shape[1]
        # combined_x = combined_x / top_k

        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
            weight_and_reduce_impl = TopKWeightAndReduceContiguous()

        weight_and_reduce_impl.apply(
            output=combined_x,
            fused_expert_output=output,
            topk_weights=combined_probs,
            topk_ids=topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
