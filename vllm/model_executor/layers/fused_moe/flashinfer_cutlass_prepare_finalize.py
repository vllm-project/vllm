# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import (get_dp_group, get_ep_group)
import vllm.envs as envs
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input)
from vllm.utils.flashinfer import nvfp4_block_scale_interleave
from flashinfer.comm.trtllm_alltoall import (MoEAlltoallInfo, MnnvlMoe)


def get_global_num_tokens_cpu():
    cu_sizes = get_forward_context().dp_metadata.cu_tokens_across_dp_cpu
    sizes = [cu_sizes[0].item()]
    for i in range(1, len(cu_sizes)):
        sizes.append((cu_sizes[i] - cu_sizes[i - 1]).item())
    return sizes

def get_local_sizes():
    return get_forward_context().dp_metadata.get_chunk_sizes_across_dp_rank()


enable_flashinfer_alltoall = envs.VLLM_ALL2ALL_BACKEND == "flashinfer"
enable_flashinfer_fp4_allgather = not enable_flashinfer_alltoall
class FlashInferCutlassMoEPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):

    def __init__(
        self,
        use_dp: bool,
        a1_gscale: Optional[torch.Tensor],
        num_dispatchers: int = 1,
    ):
        super().__init__()
        self.num_dispatchers_ = num_dispatchers
        self.use_dp = use_dp
        self.a1_gscale = a1_gscale
        self.local_tokens = None
        self.alltoall_info = None

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> Optional[int]:
        return None

    def topk_indices_dtype(self) -> Optional[torch.dtype]:
        return None

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],  # Not used
        a2_scale: Optional[torch.Tensor],  # Not used
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        # TODO(bnell): use quant_config + scales instead of ctor args
        quant_config: FusedMoEQuantConfig,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[mk.ExpertTokensMetadata], Optional[torch.Tensor],
               Optional[torch.Tensor]]:

        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            # TODO: this only works for topK=1, will need to update for topK>1
            assert topk == 1, \
                "apply_router_weight_on_input is only implemented for topk=1"
            a1.mul_(topk_weights.to(a1.dtype))
        
        if not self.use_dp:
            a1q, a1q_scale = moe_kernel_quantize_input(
                a1,
                self.a1_gscale,
                quant_config.quant_dtype,
                quant_config.per_act_token_quant,
                quant_config.block_shape,
                # Swizzling after communication
                is_fp4_scale_swizzled=not self.use_dp,
            )
        else:
            if enable_flashinfer_alltoall:
                global_num_tokens_cpu = get_local_sizes()
                top_k = topk_ids.size(1)

                # TODO(shuw): need to consider chunking for global_num_tokens_cpu
                all2all_manager = get_ep_group().device_communicator.all2all_manager
                a1, topk_ids, topk_weights, alltoall_info = all2all_manager.dispatch(
                    get_dp_group().device_communicator,
                    global_num_tokens_cpu,
                    a1,
                    topk_ids,
                    topk_weights,
                    top_k,
                    num_experts,
                )
       
                self.alltoall_info = alltoall_info

            a1q, a1q_scale = moe_kernel_quantize_input(
                a1,
                self.a1_gscale,
                quant_config.quant_dtype,
                quant_config.per_act_token_quant,
                quant_config.block_shape,
                is_fp4_scale_swizzled=not self.use_dp or enable_flashinfer_alltoall  # delay swizzle to after comm
            )

            if enable_flashinfer_fp4_allgather:
                topk_weights, topk_ids, a1q, a1q_scale = \
                    get_dp_group().all_gatherv(
                        [topk_weights, topk_ids, a1q, a1q_scale],
                        dim=0,
                        sizes=get_local_sizes(),
                    )
                a1q_scale = nvfp4_block_scale_interleave(a1q_scale)


        return a1q, a1q_scale, None, topk_ids, topk_weights

    def finalize(self, output: torch.Tensor, fused_expert_output: torch.Tensor,
                 topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                 apply_router_weight_on_input: bool,
                 weight_and_reduce_impl: mk.TopKWeightAndReduce) -> None:

        if self.use_dp:
            if enable_flashinfer_fp4_allgather:
                fused_expert_output = get_dp_group().reduce_scatterv(
                    fused_expert_output, dim=0, sizes=get_local_sizes())
                output.copy_(fused_expert_output)
            
            if enable_flashinfer_alltoall:
                all2all_manager = get_ep_group().device_communicator.all2all_manager
                top_k = topk_ids.size(1)
                token_count = output.shape[0]
                fused_expert_output = all2all_manager.flashinfer_alltoall_combine(
                    fused_expert_output,
                    # TODO(shuw): need to consider chunking for global_num_tokens_cpu
                    top_k=top_k,
                    token_count=token_count,
                    alltoall_info=self.alltoall_info,
                )
                output.copy_(fused_expert_output)
        else:
            output.copy_(fused_expert_output)
