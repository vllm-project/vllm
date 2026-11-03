# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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


def get_local_sizes() -> list[int] | None:
    dp_metadata = get_forward_context().dp_metadata
    if dp_metadata is None:  # PCP with DP=1
        return None
    return dp_metadata.get_chunk_sizes_across_dp_rank()


class FlashInferNVLinkTwoSidedPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """FlashInfer two-sided NVLink all-to-all through the MoE EP communication
    wrapper (``NVLinkTwoSidedAlltoAll``)."""

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
        self.max_num_tokens = max_num_tokens
        self.top_k = top_k
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        device_communicator = get_ep_group().device_communicator
        assert device_communicator is not None
        all2all_manager = device_communicator.all2all_manager
        assert isinstance(all2all_manager, FlashInferNVLinkTwoSidedManager)
        self.all2all_manager = all2all_manager
        # Create (or grow) the communication now, during model construction:
        # creation is collective over the EP group.
        self._communication()

    def _communication(self):
        # Resolve at use time: a later layer with the same top_k can grow the
        # shared communication during model construction, replacing it.
        return self.all2all_manager.get_communication(
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
        assert topk_ids.size(1) == self.top_k, (topk_ids.shape, self.top_k)
        global_num_tokens_cpu = get_local_sizes()
        runtime_max_tokens_per_rank = (
            max(global_num_tokens_cpu)
            if global_num_tokens_cpu is not None
            else a1.shape[0]
        )

        if defer_input_quant:
            # Block-scale path: pass activations through without quantization
            x, x_sf = a1, None
        else:
            x, x_sf = moe_kernel_quantize_input(
                a1,
                quant_config.a1_gscale,
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

        communication = self._communication()
        received = communication.dispatch(
            x,
            topk_ids,
            topk_weights,
            hidden_states_scale=x_sf,
            **{
                self.all2all_manager.dispatch_max_tokens_kwarg(communication): (
                    runtime_max_tokens_per_rank
                )
            },
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

        # The wrapper exchanges routing weights in float32; hand the experts the
        # router's dtype, as the MnnvlMoe path did.
        received_topk_weights = received.topk_weights
        if received_topk_weights is not None:
            received_topk_weights = received_topk_weights.to(topk_weights.dtype)
        return (
            received.hidden_states,
            x_sf,
            None,
            received.topk_ids,
            received_topk_weights,
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
        self._communication().combine(fused_expert_output, output=output)
