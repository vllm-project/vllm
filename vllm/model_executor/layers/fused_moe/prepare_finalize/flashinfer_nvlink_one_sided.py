# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_ep_group
from vllm.distributed.device_communicators.base_device_communicator import (
    All2AllManagerBase,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.utils.flashinfer import nvfp4_block_scale_interleave


def get_local_sizes():
    dp_metadata = get_forward_context().dp_metadata
    assert dp_metadata is not None
    return dp_metadata.get_chunk_sizes_across_dp_rank()


class FlashInferNVLinkOneSidedPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """FlashInfer implementation using the Moe AlltoAll kernel."""

    all2all_manager: All2AllManagerBase

    def __init__(
        self,
        max_num_tokens: int,
        top_k: int,
        num_experts: int,
        hidden_size: int,
        num_dispatchers: int = 1,
        dispatch_dtype_bytes_per_elem: int = 0,
        dispatch_scale_bytes_per_token: int = 0,
    ):
        super().__init__()
        self.max_num_tokens = max_num_tokens
        self.top_k = top_k
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.num_dispatchers_ = num_dispatchers
        self.scale_elems_per_token = dispatch_scale_bytes_per_token

        device_communicator = get_ep_group().device_communicator
        assert device_communicator is not None
        all2all_manager = device_communicator.all2all_manager
        assert all2all_manager is not None
        self.all2all_manager = all2all_manager
        self.all2all_manager.initialize(  # type: ignore[attr-defined]
            max_num_tokens=self.max_num_tokens,
            top_k=self.top_k,
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            dispatch_dtype_bytes_per_elem=dispatch_dtype_bytes_per_elem,
            dispatch_scale_bytes_per_token=dispatch_scale_bytes_per_token,
        )

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        return False

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int32

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
        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1.mul_(topk_weights.to(a1.dtype))

        global_num_tokens_cpu = get_local_sizes()
        self.runtime_max_tokens_per_rank = (
            max(global_num_tokens_cpu)
            if global_num_tokens_cpu is not None
            else a1.shape[0]
        )

        if defer_input_quant:
            a1q, a1q_scale = a1, None
        else:
            a1q, a1q_scale = moe_kernel_quantize_input(
                a1,
                quant_config.a1_gscale,
                quant_config.quant_dtype,
                quant_config.per_act_token_quant,
                quant_config.block_shape,
                is_fp4_scale_swizzled=False,  # delay swizzle to after comm
                mx_alignment=quant_config.mx_alignment,
            )

        payloads = []
        payloads.append(a1q)
        if a1q_scale is not None:
            payloads.append(a1q_scale)
        topk_ids_payload_index = len(payloads)
        payloads.append(topk_ids)
        payloads.append(topk_weights)

        assert self.all2all_manager.moe_alltoall is not None  # type: ignore[attr-defined]
        recv_payloads = self.all2all_manager.moe_alltoall.dispatch(  # type: ignore[attr-defined]
            token_selected_experts=topk_ids,
            input_payloads=payloads,
            runtime_max_tokens_per_rank=self.runtime_max_tokens_per_rank,
            invalid_token_expert_id=-1,  # Follow TRTLLM Pattern
            expert_id_payload_index=topk_ids_payload_index,
        )
        if a1q_scale is not None:
            a1q_recv, a1q_scale_recv, topk_ids_recv, topk_weights_recv = recv_payloads
            # Apply scale interleaving only for CUTLASS (not TRT-LLM)
            if (
                quant_config.quant_dtype == "nvfp4"
                and quant_config.is_nvfp4_scale_swizzled
            ):
                a1q_scale_recv = a1q_scale_recv.view(-1, a1q_scale_recv.shape[-1])
                a1q_scale_recv = a1q_scale_recv.view(torch.uint8)
                a1q_scale_recv = nvfp4_block_scale_interleave(a1q_scale_recv)
            assert self.scale_elems_per_token > 0
            a1q_scale_recv = a1q_scale_recv.view(-1, self.scale_elems_per_token)
        else:
            a1q_recv, topk_ids_recv, topk_weights_recv = recv_payloads
            a1q_scale_recv = None
        a1q_recv = a1q_recv.view(-1, a1q_recv.shape[-1])
        topk_ids_recv = topk_ids_recv.view(-1, topk_ids_recv.shape[-1])
        topk_weights_recv = topk_weights_recv.view(-1, topk_weights_recv.shape[-1])

        return a1q_recv, a1q_scale_recv, None, topk_ids_recv, topk_weights_recv

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        assert self.all2all_manager.moe_alltoall is not None  # type: ignore[attr-defined]

        ep_size = self.all2all_manager.world_size
        hidden_size = fused_expert_output.shape[-1]
        fused_expert_output = fused_expert_output.view(
            ep_size, self.runtime_max_tokens_per_rank, hidden_size
        )

        combined_output = self.all2all_manager.moe_alltoall.combine(  # type: ignore[attr-defined]
            payload=fused_expert_output,
            runtime_max_tokens_per_rank=self.runtime_max_tokens_per_rank,
        )
        output.copy_(combined_output)
