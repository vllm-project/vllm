# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer TRT-LLM MoE prepare and finalize implementations.

TRT-LLM kernel expects linear scale layout (not interleaved), so these
implementations skip the nvfp4_block_scale_interleave step.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_dp_group, get_ep_group
from vllm.distributed.device_communicators.base_device_communicator import (
    All2AllManagerBase,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_prepare_finalize import (
    FlashInferCutlassMoEPrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input


def get_local_sizes():
    return get_forward_context().dp_metadata.get_chunk_sizes_across_dp_rank()


class FlashInferTRTLLMAllToAllMoEPrepareAndFinalize(
    FlashInferCutlassMoEPrepareAndFinalize
):
    """FlashInfer TRT-LLM implementation using AllToAll communication.

    TRT-LLM kernel expects linear scale layout, so we skip scale interleaving.
    """

    def __init__(
        self,
        use_dp: bool,
        num_dispatchers: int = 1,
    ):
        super().__init__(use_dp, num_dispatchers, use_deepseek_fp8_block_scale=False)
        self.alltoall_info = None

        # Initialize all2all_manager only for DP case
        self.all2all_manager = None
        if self.use_dp:
            self.all2all_manager = get_ep_group().device_communicator.all2all_manager

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
        self._apply_router_weight_on_input(
            a1, topk_weights, topk_ids, apply_router_weight_on_input
        )

        if not self.use_dp:
            # Non-DP case: quantize activations
            a1q, a1q_scale = moe_kernel_quantize_input(
                a1,
                quant_config.a1_gscale,
                quant_config.quant_dtype,
                quant_config.per_act_token_quant,
                quant_config.block_shape,
                is_fp4_scale_swizzled=False,  # TRT-LLM expects linear layout
            )
        else:
            # DP case: use FlashInfer AllToAll
            global_num_tokens_cpu = get_local_sizes()
            top_k = topk_ids.size(1)
            (self.alltoall_info, topk_ids, topk_weights, a1q, a1q_scale) = (
                flashinfer_trtllm_alltoall_dispatch(
                    self.all2all_manager,
                    global_num_tokens_cpu,
                    a1,
                    quant_config.a1_gscale,
                    topk_ids,
                    topk_weights,
                    top_k,
                    num_experts,
                    quant_config,
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
        if self.use_dp:
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


class FlashInferTRTLLMAllGatherMoEPrepareAndFinalize(
    FlashInferCutlassMoEPrepareAndFinalize
):
    """FlashInfer TRT-LLM implementation using AllGather communication.

    TRT-LLM kernel expects linear scale layout, so we skip scale interleaving.
    """

    def __init__(
        self,
        use_dp: bool,
        num_dispatchers: int = 1,
    ):
        super().__init__(use_dp, num_dispatchers, use_deepseek_fp8_block_scale=False)

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
        self._apply_router_weight_on_input(
            a1, topk_weights, topk_ids, apply_router_weight_on_input
        )

        # For non-DP nvfp4, return unquantized (will be quantized later)
        if not self.use_dp and quant_config.quant_dtype == "nvfp4":
            return a1, None, None, topk_ids, topk_weights

        # Quantize activations - TRT-LLM expects linear scale layout
        a1q, a1q_scale = moe_kernel_quantize_input(
            a1,
            quant_config.a1_gscale,
            quant_config.quant_dtype,
            quant_config.per_act_token_quant,
            quant_config.block_shape,
            is_fp4_scale_swizzled=False,  # TRT-LLM expects linear layout
        )

        if self.use_dp:
            # Build gather list
            gather_list = [topk_weights, topk_ids, a1q]
            if a1q_scale is not None:
                gather_list.append(a1q_scale)
                gathered = get_dp_group().all_gatherv(
                    gather_list,
                    dim=0,
                    sizes=get_local_sizes(),
                )
                topk_weights, topk_ids, a1q, a1q_scale = gathered
            else:
                gathered = get_dp_group().all_gatherv(
                    gather_list,
                    dim=0,
                    sizes=get_local_sizes(),
                )
                topk_weights, topk_ids, a1q = gathered
                a1q_scale = None

        # NOTE: We intentionally skip nvfp4_block_scale_interleave here
        # because TRT-LLM kernel expects linear scale layout

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
        assert isinstance(weight_and_reduce_impl, TopKWeightAndReduceNoOP)

        if self.use_dp:
            fused_expert_output = get_dp_group().reduce_scatterv(
                fused_expert_output, dim=0, sizes=get_local_sizes()
            )
        output.copy_(fused_expert_output)


def flashinfer_trtllm_alltoall_dispatch(
    all2all_manager: All2AllManagerBase,
    global_num_tokens_cpu: list[int],
    x: torch.Tensor,
    gs: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    top_k: int,
    num_experts: int,
    quant_config: FusedMoEQuantConfig,
):
    """Dispatch for TRT-LLM AllToAll - skips scale interleaving."""
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

    # Quantize input - TRT-LLM expects linear scale layout
    x, x_sf = moe_kernel_quantize_input(
        x,
        gs,
        quant_config.quant_dtype,
        quant_config.per_act_token_quant,
        quant_config.block_shape,
        is_fp4_scale_swizzled=False,  # TRT-LLM expects linear layout
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

    # NOTE: We intentionally skip nvfp4_block_scale_interleave here
    # because TRT-LLM kernel expects linear scale layout

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


def create_flashinfer_trtllm_prepare_finalize(
    use_dp: bool,
    enable_alltoallv: bool = False,
) -> FlashInferCutlassMoEPrepareAndFinalize:
    """Factory function to create the appropriate FlashInfer TRT-LLM implementation.

    Args:
        use_dp: Whether to use data parallelism.
        enable_alltoallv: Whether to use AllToAllV communication.
    """
    if enable_alltoallv:
        return FlashInferTRTLLMAllToAllMoEPrepareAndFinalize(use_dp)
    else:
        return FlashInferTRTLLMAllGatherMoEPrepareAndFinalize(use_dp)
