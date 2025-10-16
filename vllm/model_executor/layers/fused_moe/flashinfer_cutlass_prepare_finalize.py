# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_dp_group, get_ep_group
from vllm.distributed.device_communicators.base_device_communicator import (
    All2AllManagerBase,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.utils.flashinfer import nvfp4_block_scale_interleave


def get_local_sizes():
    return get_forward_context().dp_metadata.get_chunk_sizes_across_dp_rank()


class FlashInferCutlassMoEPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """Base class for FlashInfer MoE prepare and finalize operations."""

    def __init__(
        self,
        use_dp: bool,
        num_dispatchers: int = 1,
    ):
        super().__init__()
        self.num_dispatchers_ = num_dispatchers
        self.use_dp = use_dp
        self.local_tokens = None

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


class FlashInferAllToAllMoEPrepareAndFinalize(FlashInferCutlassMoEPrepareAndFinalize):
    """FlashInfer implementation using AllToAll communication."""

    def __init__(
        self,
        use_dp: bool,
        num_dispatchers: int = 1,
    ):
        super().__init__(use_dp, num_dispatchers)
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
            # Non-DP case: standard quantization
            a1q, a1q_scale = moe_kernel_quantize_input(
                a1,
                quant_config.a1_gscale,
                quant_config.quant_dtype,
                quant_config.per_act_token_quant,
                quant_config.block_shape,
                is_fp4_scale_swizzled=not self.use_dp,
            )
        else:
            # DP case: use FlashInfer AllToAll
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


class FlashInferAllGatherMoEPrepareAndFinalize(FlashInferCutlassMoEPrepareAndFinalize):
    def __init__(
        self,
        use_dp: bool,
        num_dispatchers: int = 1,
    ):
        super().__init__(use_dp, num_dispatchers)

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

        a1q, a1q_scale = moe_kernel_quantize_input(
            a1,
            quant_config.a1_gscale,
            quant_config.quant_dtype,
            quant_config.per_act_token_quant,
            quant_config.block_shape,
            is_fp4_scale_swizzled=not self.use_dp,
        )
        if self.use_dp:
            topk_weights, topk_ids, a1q, a1q_scale = get_dp_group().all_gatherv(
                [topk_weights, topk_ids, a1q, a1q_scale],
                dim=0,
                sizes=get_local_sizes(),
            )
            if quant_config.quant_dtype == "nvfp4":
                a1q_scale = nvfp4_block_scale_interleave(a1q_scale)

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
    alltoall_info, topk_ids, topk_weights, _ = (
        MnnvlMoe.mnnvl_moe_alltoallv_prepare_without_allgather(
            topk_ids,
            topk_weights,
            None,
            all2all_manager.prepare_workspace,
            max_num_token,
            ep_rank,
            ep_size,
            num_experts,
            num_experts,
            top_k,
        )
    )

    x, x_sf = moe_kernel_quantize_input(
        x,
        gs,
        quant_config.quant_dtype,
        quant_config.per_act_token_quant,
        quant_config.block_shape,
        is_fp4_scale_swizzled=False,  # delay swizzle to after comm
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
    x_sf = nvfp4_block_scale_interleave(x_sf)
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


def create_flashinfer_prepare_finalize(
    use_dp: bool,
    use_nvfp4: bool = False,
    enable_alltoallv: bool = False,
) -> FlashInferCutlassMoEPrepareAndFinalize:
    """Factory function to create the appropriate FlashInfer implementation."""
    if use_nvfp4:
        if enable_alltoallv:
            return FlashInferAllToAllMoEPrepareAndFinalize(use_dp)
        else:
            return FlashInferAllGatherMoEPrepareAndFinalize(use_dp)
    # Fp8 only supports AllGather
    return FlashInferAllGatherMoEPrepareAndFinalize(use_dp)
