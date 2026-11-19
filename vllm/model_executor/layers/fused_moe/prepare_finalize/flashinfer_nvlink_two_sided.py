# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_ep_group
from vllm.distributed.device_communicators.all2all import (
    FlashInferNVLinkTwoSidedManager,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.utils.flashinfer import nvfp4_block_scale_interleave

if TYPE_CHECKING:
    from flashinfer.moe_ep import NVLinkTwoSidedAlltoAll


def get_local_sizes():
    dp_metadata = get_forward_context().dp_metadata
    assert dp_metadata is not None
    return dp_metadata.get_chunk_sizes_across_dp_rank()


class FlashInferNVLinkTwoSidedPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """Base class for FlashInfer MoE prepare and finalize operations."""

    all2all_manager: FlashInferNVLinkTwoSidedManager

    def __init__(
        self,
        max_num_tokens: int,
        top_k: int,
        num_experts: int,
        hidden_size: int,
        num_dispatchers: int = 1,
    ):
        super().__init__()
        self.num_dispatchers_ = num_dispatchers
        device_communicator = get_ep_group().device_communicator
        assert device_communicator is not None
        all2all_manager = device_communicator.all2all_manager
        assert isinstance(all2all_manager, FlashInferNVLinkTwoSidedManager)
        self.all2all_manager = all2all_manager
        self.communication = all2all_manager.get_communication(
            max_num_tokens=max_num_tokens,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
        )

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
        return True

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

        topk_ids, topk_weights, a1q, a1q_scale = flashinfer_alltoall_dispatch(
            self.communication,
            global_num_tokens_cpu,
            a1,
            quant_config.a1_gscale,
            topk_ids,
            topk_weights,
            quant_config,
            defer_input_quant=defer_input_quant,
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
        flashinfer_alltoall_combine(self.communication, fused_expert_output, output)


def flashinfer_alltoall_dispatch(
    communication: "NVLinkTwoSidedAlltoAll",
    global_num_tokens_cpu: list[int],
    x: torch.Tensor,
    gs: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    quant_config: FusedMoEQuantConfig,
    defer_input_quant: bool = False,
):
    max_num_token = (
        max(global_num_tokens_cpu) if global_num_tokens_cpu is not None else x.shape[0]
    )
    if x.shape[0] == 0:
        # FlashInfer cannot combine into an empty batch. Send one row whose
        # expert id num_experts maps to no rank; combine drops its result.
        x = x.new_zeros(1, x.shape[1])
        topk_ids = topk_ids.new_full(
            (1, topk_ids.shape[1]), communication.params.num_experts
        )
        topk_weights = topk_weights.new_zeros(1, topk_weights.shape[1])
        max_num_token = max(max_num_token, 1)

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
            is_scale_swizzled=False,
            mx_alignment=quant_config.mx_alignment,
        )
    else:
        # Block-scale path: pass activations through without quantization
        x_sf = None

    received = communication.dispatch(
        x,
        topk_ids,
        topk_weights,
        hidden_states_scale=x_sf,
        runtime_max_tokens_per_rank=max_num_token,
    )
    x_sf = received.hidden_states_scale

    # Swizzle after the A2A if MoE kernel expects swizzled scales.
    if (
        x_sf is not None
        and quant_config.quant_dtype == "nvfp4"
        and quant_config.is_scale_swizzled
    ):
        if x_sf.element_size() == 1:
            x_sf = x_sf.view(torch.uint8)
        x_sf = nvfp4_block_scale_interleave(x_sf)
    return received.topk_ids, received.topk_weights, received.hidden_states, x_sf


def flashinfer_alltoall_combine(
    communication: "NVLinkTwoSidedAlltoAll",
    fused_expert_output: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    if output.shape[0] == 0:
        # Drop the result of the row dispatch sent for an empty batch.
        communication.combine(fused_expert_output)
        return output
    return communication.combine(fused_expert_output, output=output)
