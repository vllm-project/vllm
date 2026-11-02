# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

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
from vllm.v1.worker.ubatching import dbo_comm_region, dbo_current_ubatch_id


def get_local_sizes():
    dp_metadata = get_forward_context().dp_metadata
    assert dp_metadata is not None
    return dp_metadata.get_chunk_sizes_across_dp_rank()


class FlashInferNVLinkTwoSidedPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """Base class for FlashInfer MoE prepare and finalize operations."""

    all2all_manager: All2AllManagerBase

    def __init__(
        self,
        num_dispatchers: int = 1,
    ):
        super().__init__()
        self.num_dispatchers_ = num_dispatchers
        device_communicator = get_ep_group().device_communicator
        assert device_communicator is not None
        assert device_communicator.all2all_manager is not None
        self.all2all_manager = device_communicator.all2all_manager
        # finalize() needs its ubatch's dispatch routing; under DBO the peer
        # ubatch dispatches in between, so keep one slot per ubatch.
        self.alltoall_infos: list[Any] = [None, None]

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

    def supports_dbo(self) -> bool:
        # prepare()/finalize() run the all2all on the DBO comm stream (see
        # dbo_comm_region). Both ubatches' kernels are ordered on that one
        # stream and return freshly allocated tensors, so they can share the
        # manager's workspace. supports_async() stays False: it also selects
        # MK-internal shared-expert overlap.
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
        top_k = topk_ids.size(1)

        with dbo_comm_region():
            (alltoall_info, topk_ids, topk_weights, a1q, a1q_scale) = (
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
        self.alltoall_infos[dbo_current_ubatch_id()] = alltoall_info

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
        ubatch_id = dbo_current_ubatch_id()
        alltoall_info = self.alltoall_infos[ubatch_id]
        assert alltoall_info is not None, "finalize() called before prepare()"
        with dbo_comm_region():
            fused_expert_output = flashinfer_alltoall_combine(
                self.all2all_manager,
                fused_expert_output,
                top_k=top_k,
                token_count=token_count,
                alltoall_info=alltoall_info,
            )
        self.alltoall_infos[ubatch_id] = None
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

    assert all2all_manager.ensure_alltoall_workspace_initialized(), (  # type: ignore[attr-defined]
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
            all2all_manager.prepare_workspace_tensor,  # type: ignore[attr-defined]
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
            is_scale_swizzled=False,
            mx_alignment=quant_config.mx_alignment,
        )

        x = MnnvlMoe.mnnvl_moe_alltoallv(
            x,
            alltoall_info,
            all2all_manager.workspace_tensor,  # type: ignore[attr-defined]
            ep_rank,
            ep_size,
        )

        if x_sf is not None:
            x_sf = MnnvlMoe.mnnvl_moe_alltoallv(
                x_sf,
                alltoall_info,
                all2all_manager.workspace_tensor,  # type: ignore[attr-defined]
                ep_rank,
                ep_size,
            )

        # Swizzle after the A2A if MoE kernel expects swizzled scales.
        if (
            x_sf is not None
            and quant_config.quant_dtype == "nvfp4"
            and quant_config.is_scale_swizzled
        ):
            if x_sf.element_size() == 1:
                x_sf = x_sf.view(torch.uint8)
            x_sf = nvfp4_block_scale_interleave(x_sf)
    else:
        # Block-scale path: pass activations through without quantization
        x_sf = None
        x = MnnvlMoe.mnnvl_moe_alltoallv(
            x,
            alltoall_info,
            all2all_manager.workspace_tensor,  # type: ignore[attr-defined]
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

    assert all2all_manager.ensure_alltoall_workspace_initialized(), (  # type: ignore[attr-defined]
        "FlashInfer AllToAll workspace not available"
    )
    return MnnvlMoe.mnnvl_moe_alltoallv_combine(
        output,
        alltoall_info,
        all2all_manager.workspace_tensor,  # type: ignore[attr-defined]
        ep_rank=all2all_manager.rank,
        ep_size=all2all_manager.world_size,
        top_k=top_k,
        token_count=token_count,
    )
