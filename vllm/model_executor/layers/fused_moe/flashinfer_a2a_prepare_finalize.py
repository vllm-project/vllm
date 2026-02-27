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
    return get_forward_context().dp_metadata.get_chunk_sizes_across_dp_rank()


class FlashInferA2APrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """Base class for FlashInfer MoE prepare and finalize operations."""

    def __init__(
        self,
        num_dispatchers: int = 1,
    ):
        super().__init__()
        self.num_dispatchers_ = num_dispatchers
        self.all2all_manager = get_ep_group().device_communicator.all2all_manager

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        return False

    def _apply_router_weight_on_input(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> None:
        """Apply router weight on input if needed."""
        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1.mul_(topk_weights.to(a1.dtype))

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
        self._apply_router_weight_on_input(
            a1, topk_weights, topk_ids, apply_router_weight_on_input
        )
        global_num_tokens_cpu = get_local_sizes()
        top_k = topk_ids.size(1)

        (self.alltoall_info, topk_ids, topk_weights, a1q, a1q_scale) = (
            flashinfer_alltoall_dispatch(
                self.all2all_manager,
                global_num_tokens_cpu,
                a1,
                quant_config.a1_gscale,
                topk_ids,
                topk_weights,
                top_k,
                num_experts,
                quant_config,
                defer_input_quant=defer_input_quant,
            )
        )

        return a1q, a1q_scale, None, topk_ids, topk_weights

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        top_k = topk_ids.size(1)
        token_count = output.shape[0]
        fused_expert_output = flashinfer_alltoall_combine(
            self.all2all_manager,
            fused_expert_output,
            top_k=top_k,
            token_count=token_count,
            alltoall_info=self.alltoall_info,
        )
        output.copy_(fused_expert_output)


def flashinfer_alltoall_dispatch(
    all2all_manager: All2AllManagerBase,
    global_num_tokens_cpu: list[int],
    x: torch.Tensor,
    gs: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    top_k: int,
    num_experts: int,
    quant_config: FusedMoEQuantConfig,
    defer_input_quant: bool = False,
):
    from flashinfer.comm.trtllm_alltoall import MnnvlMoe

    assert all2all_manager.ensure_alltoall_workspace_initialized(), (
        "FlashInfer AllToAll workspace not available"
    )

    ep_rank = all2all_manager.rank
    ep_size = all2all_manager.world_size
    max_num_token = (
        max(global_num_tokens_cpu) if global_num_tokens_cpu is not None else x.shape[0]
    )
    orig_topk_weights_dtype = topk_weights.dtype
    alltoall_info, topk_ids, topk_weights, _ = (
        MnnvlMoe.mnnvl_moe_alltoallv_prepare_without_allgather(
            topk_ids,
            topk_weights,
            None,
            all2all_manager.prepare_workspace_tensor,
            max_num_token,
            ep_rank,
            ep_size,
            num_experts,
            num_experts,
            top_k,
        )
    )
    topk_weights = topk_weights.view(dtype=orig_topk_weights_dtype)

    if not defer_input_quant:
        x, x_sf = moe_kernel_quantize_input(
            x,
            gs,
            quant_config.quant_dtype,
            quant_config.per_act_token_quant,
            quant_config.block_shape,
            # NOTE: swizzling pads the scales to multiple of 128
            # which makes the scales tensor different shape than
            # the hidden states, breaking the A2A kernel. So, we
            # delay the swizzling until after the A2A.
            is_fp4_scale_swizzled=False,
        )

        x = MnnvlMoe.mnnvl_moe_alltoallv(
            x,
            alltoall_info,
            all2all_manager.workspace_tensor,
            ep_rank,
            ep_size,
        )

        x_sf = MnnvlMoe.mnnvl_moe_alltoallv(
            x_sf,
            alltoall_info,
            all2all_manager.workspace_tensor,
            ep_rank,
            ep_size,
        )

        # Swizzle after the A2A if MoE kernel expects swizzled scales.
        if quant_config.quant_dtype == "nvfp4" and quant_config.is_nvfp4_scale_swizzled:
            if x_sf.element_size() == 1:
                x_sf = x_sf.view(torch.uint8)
            x_sf = nvfp4_block_scale_interleave(x_sf)
    else:
        # Block-scale path: pass activations through without quantization
        x_sf = None
        x = MnnvlMoe.mnnvl_moe_alltoallv(
            x,
            alltoall_info,
            all2all_manager.workspace_tensor,
            ep_rank,
            ep_size,
        )
    return alltoall_info, topk_ids, topk_weights, x, x_sf


def flashinfer_alltoall_combine(
    all2all_manager: All2AllManagerBase,
    output: torch.Tensor,
    top_k: int,
    token_count: int,
    alltoall_info,
):
    from flashinfer.comm.trtllm_alltoall import MnnvlMoe

    assert all2all_manager.ensure_alltoall_workspace_initialized(), (
        "FlashInfer AllToAll workspace not available"
    )
    return MnnvlMoe.mnnvl_moe_alltoallv_combine(
        output,
        alltoall_info,
        all2all_manager.workspace_tensor,
        ep_rank=all2all_manager.rank,
        ep_size=all2all_manager.world_size,
        top_k=top_k,
        token_count=token_count,
    )


class FlashInferMoeA2APrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """FlashInfer implementation using the Moe AlltoAll kernel."""

    def __init__(
        self,
        max_num_tokens: int,
        top_k: int,
        num_experts: int,
        hidden_size: int,
        num_dispatchers: int = 1,
    ):
        super().__init__()
        self.max_num_tokens = max_num_tokens
        self.top_k = top_k
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.num_dispatchers_ = num_dispatchers

        self.all2all_manager = get_ep_group().device_communicator.all2all_manager
        self.all2all_manager.initialize(
            max_num_tokens=self.max_num_tokens,
            top_k=self.top_k,
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
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
        return None

    def _apply_router_weight_on_input(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> None:
        """Apply router weight on input if needed."""
        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1.mul_(topk_weights.to(a1.dtype))

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
        self._apply_router_weight_on_input(
            a1, topk_weights, topk_ids, apply_router_weight_on_input
        )

        global_num_tokens_cpu = get_local_sizes()
        self.runtime_max_tokens_per_rank = (
            max(global_num_tokens_cpu)
            if global_num_tokens_cpu is not None
            else a1.shape[0]
        )

        a1q, a1q_scale = moe_kernel_quantize_input(
            a1,
            quant_config.a1_gscale,
            quant_config.quant_dtype,
            quant_config.per_act_token_quant,
            quant_config.block_shape,
            is_fp4_scale_swizzled=False,  # delay swizzle to after comm
        )

        payloads = []
        payloads.append(a1q)
        if a1q_scale is not None:
            payloads.append(a1q_scale)
            expert_id_payload_index = 2
        else:
            expert_id_payload_index = 1
        payloads.append(topk_ids)
        payloads.append(topk_weights)

        recv_payloads = self.all2all_manager.moe_alltoall.dispatch(
            token_selected_experts=topk_ids,
            input_payloads=payloads,
            runtime_max_tokens_per_rank=self.runtime_max_tokens_per_rank,
            # invalid_token_expert_id=-1,
            # expert_id_payload_index=expert_id_payload_index,
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
            a1q_scale_recv = a1q_scale_recv.view(-1, self.hidden_size // 16)
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
        assert self.all2all_manager.moe_alltoall is not None

        ep_size = self.all2all_manager.world_size
        hidden_size = fused_expert_output.shape[-1]
        fused_expert_output = fused_expert_output.view(
            ep_size, self.runtime_max_tokens_per_rank, hidden_size
        )

        combined_output = self.all2all_manager.moe_alltoall.combine(
            payload=fused_expert_output,
            runtime_max_tokens_per_rank=self.runtime_max_tokens_per_rank,
        )
        output.copy_(combined_output)
