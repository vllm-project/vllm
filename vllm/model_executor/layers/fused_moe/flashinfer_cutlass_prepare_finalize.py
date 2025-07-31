# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, Optional

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_dp_group
from vllm import envs
from vllm.forward_context import get_chunked_local_tokens
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.utils import (
    extract_required_args, moe_kernel_quantize_input)
from vllm.utils.flashinfer import nvfp4_block_scale_interleave

# def get_local_sizes(local_tokens):
#     cu_sizes = get_forward_context().dp_metadata.cu_tokens_across_dp_cpu
#     print(f"cu_sizes in get_local_sizes: {cu_sizes} from rank:{get_dp_group().rank}")
#     sizes = [cu_sizes[0].item()]
#     for i in range(1, len(cu_sizes)):
#         sizes.append((cu_sizes[i] - cu_sizes[i - 1]).item())
#     max_num_tokens = envs.VLLM_MOE_DP_CHUNK_SIZE
#     sizes_chunked = sizes#[max_num_tokens] * len(sizes)
#     if local_tokens <= max_num_tokens: # indicating this might be the last iteration, but some rank may not think so this !
#         # When the number of local tokens is less than max_num_tokens, all other
#         # ranks will also have fewer than max_num_tokens. The remaining tokens
#         # are accounted for as residual.
#         for k in range(len(sizes)):
#             old = sizes[k]
#             if old >= max_num_tokens:
#                 new = min(old, max_num_tokens)
#             else:
#                 new = old % max_num_tokens or 1
#             sizes_chunked[k] = new
#         # sizes_chunked = [x % max_num_tokens or 1 for x in sizes]

#     return sizes, sizes_chunked
# from typing import List
# def compute_iteration_schedule(num_tokens_across_dp_cpu: List[int], max_num_tokens: int) -> List[List[int]]:
#     dp_size = len(num_tokens_across_dp_cpu)
#     remaining = num_tokens_across_dp_cpu.copy()
#     schedule = []

#     while any(t > 0 for t in remaining):
#         iteration = []
#         for i in range(dp_size):
#             to_process = min(remaining[i], max_num_tokens)
#             if to_process == 0:
#                 to_process = 1  # ensure lockstep even if done
#             else:
#                 remaining[i] -= to_process
#             iteration.append(to_process)
#         schedule.append(iteration)

#     return schedule

# def get_local_sizes():
#     return get_chunked_local_tokens()
#     get_chunked_local_tokens
#     cu_sizes = get_forward_context().dp_metadata.cu_tokens_across_dp_cpu
# #     print(f"cu_sizes in get_local_sizes: {cu_sizes} from rank:{get_dp_group().rank}")
#     sizes = [cu_sizes[0].item()]
#     for i in range(1, len(cu_sizes)):
#         sizes.append((cu_sizes[i] - cu_sizes[i - 1]).item())   
#     max_num_tokens = envs.VLLM_MOE_DP_CHUNK_SIZE 
#     tmp = compute_iteration_schedule(sizes, max_num_tokens)
#     return tmp[idx]


class FlashInferCutlassMoEPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):

    def __init__(
        self,
        quant_dtype: Optional[torch.dtype] = None,
        per_channel_quant: bool = False,
        block_shape: Optional[list[int]] = None,
        num_dispatchers: int = 1,
    ):
        super().__init__()
        self.per_channel_quant = per_channel_quant
        self.block_shape = block_shape
        self.quant_dtype = quant_dtype
        self.num_dispatchers_ = num_dispatchers

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> Optional[int]:
        return None

    def topk_indices_dtype(self) -> Optional[torch.dtype]:
        return None

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    @property
    def num_tokens_across_dp_current(self):
        return self._num_tokens_across_dp_current

    def update_local_tokens(self, local_tokens: int):
        if local_tokens is None:
            raise ValueError("Local token count not set.")
        self._num_tokens_across_dp_current = DPMetadata.num_tokens_across_dp(
            local_tokens,
            get_dp_group().world_size,
            get_dp_group().rank)

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
        quant_config: FusedMoEQuantConfig,
        extra_prepare_args: Optional[dict[str, Any]]
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor], Optional[torch.Tensor]]:

        assert not apply_router_weight_on_input

        (a1_gscale, use_dp, local_tokens) = extract_required_args(
            extra_prepare_args, ['a1_gscale', 'use_dp', 'local_tokens'])

        a1q, a1q_scale = moe_kernel_quantize_input(
            a1,
            a1_gscale,
            quant_config.quant_dtype,
            self.per_channel_quant,
            self.block_shape,
            is_fp4_scale_swizzled=not use_dp,  # Swizzling after communication
        )
        if use_dp:
            # self.update_local_tokens(local_tokens)
            # assert a1q.shape[0] == local_tokens
            # chunked_sz = get_chunked_local_tokens()
            # sz, chunked_sz = get_local_sizes(local_tokens)
            # for i in range(len(self.num_tokens_across_dp_current)):
            #     assert self.num_tokens_across_dp_current[i] == chunked_sz[i], (f"not match: dp_current:{self.num_tokens_across_dp_current} vs get_local_sizes:{chunked_sz} with sz:{sz} from rank:{get_dp_group().rank} and local_tokens:{local_tokens}")
            topk_weights, topk_ids, a1q, a1q_scale = \
                get_dp_group().all_gatherv([topk_weights, topk_ids, a1q, a1q_scale], # noqa: E501
                                           dim=0,
                                           sizes=get_chunked_local_tokens())
            a1_m, a1_n = a1q.shape
            a1q_scale = nvfp4_block_scale_interleave(a1q_scale)

        return a1q, a1q_scale, None, topk_ids, topk_weights

    def finalize(self, output: torch.Tensor, fused_expert_output: torch.Tensor,
                 topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                 apply_router_weight_on_input: bool,
                 weight_and_reduce_impl: mk.TopKWeightAndReduce,
                 extra_finalize_args: Optional[dict[str, Any]]) -> None:

        (use_dp,
         local_tokens) = extract_required_args(extra_finalize_args,
                                               ['use_dp', 'local_tokens'])
        if use_dp:
            # sz, chunked_sz = get_local_sizes(local_tokens)
            # chunked_sz = get_local_sizes(extra_finalize_args['chunk_iter_idx'])
            chunked_sz = get_chunked_local_tokens()
            fused_expert_output = get_dp_group().reduce_scatterv(
                fused_expert_output,
                dim=0,
                sizes=get_chunked_local_tokens()
            )
        output.copy_(fused_expert_output)
