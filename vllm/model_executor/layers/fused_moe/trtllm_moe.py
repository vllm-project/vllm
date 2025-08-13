# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, Optional

import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig, FusedMoEConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP)
from vllm.model_executor.layers.fused_moe.utils import extract_required_args
from vllm.utils import next_power_of_2

if (envs.VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8
        or envs.VLLM_USE_FLASHINFER_MOE_MXFP4_BF16):
    # from flashinfer.fused_moe import cutlass_fused_moe
    from flashinfer import mxfp8_quantize, trtllm_fp4_block_scale_routed_moe

DEBUG_PRINTS: set[str] = set()

class TrtLlmGenExperts(mk.FusedMoEPermuteExpertsUnpermute):

    def __init__(
        self,
        moe: FusedMoEConfig 
    ):
        super().__init__(moe.quant_config)
        self.moe = moe

    @property
    def activation_formats(
        self
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        return (mk.FusedMoEActivationFormat.Standard,
                mk.FusedMoEActivationFormat.Standard)

    def supports_chunking(self) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        a: torch.Tensor,
        aq: torch.Tensor,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: Optional[mk.ExpertTokensMetadata],
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], torch.dtype]:
        self.topk = topk
        self.num_experts = local_num_experts
        self.intermediate_size = N // 2
        workspace1 = (M, topk, max(N // 2, K))
        workspace2 = (M, topk, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output, a.dtype)

    def _get_tile_tokens_dim(self, x: torch.Tensor, top_k: int):
        # Number of tokens in the input tensor.
        num_tokens = x.shape[0]
        # Factor to account for the imbalance of the experts.
        # factor equals to the
        # max_real_num_tokens_per_expert / perfect_num_tokens_per_expert
        # 1.0 means perfect expert distribution.
        # > 1.0 means some experts have more tokens than the perfect
        # distribution.
        # < 1.0 does not make sense.
        imbalance_factor = 1.3
        # Calculate the number of tokens per expert assuming perfect
        # distribution.
        num_tokens_per_expert = (num_tokens * top_k) // self.num_experts
        # Apply the imbalance factor.
        num_tokens_per_expert = int(num_tokens_per_expert * imbalance_factor)
        # And pad the number to the next power of 2.
        tile_tokens_dim = next_power_of_2(num_tokens_per_expert)
        # Cap to 8-64 tokens per CTA tile as it's the range supported by the
        #  kernel.
        tile_tokens_dim = min(max(tile_tokens_dim, 8), 64)

        return tile_tokens_dim

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: Optional[torch.Tensor],
        w1_scale: Optional[torch.Tensor],
        w2_scale: Optional[torch.Tensor],
        w1_zp: Optional[torch.Tensor],
        w2_zp: Optional[torch.Tensor],
        a1q_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: Optional[mk.ExpertTokensMetadata],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ):

        if self.moe.ep_rank == 1:
            output.copy_(hidden_states, non_blocking=True)
            return output

        topk = topk_weights.shape[-1]
        required_keys = ['gemm1_alpha', 'gemm1_beta', 'gemm1_clamp_limit', "w1_bias", "w2_bias"]

        gemm1_alpha, gemm1_beta, gemm1_clamp_limit, w1_bias, w2_bias = (extract_required_args(
            extra_expert_args, required_keys))

        # TODO (varun) : Extra work ! could we avoid this ?  or use a kernel ?
        #packed_tensor = (topk_ids.to(torch.int32) << 16) | topk_weights.view(
        #    torch.int16).to(torch.int32)

        local_num_experts = w1.size(0)
        local_expert_offset = self.moe.ep_rank * local_num_experts
        #topk_ids = torch.randint(low = local_expert_offset,
        #                         high = local_expert_offset + local_num_experts,
        #                         size = topk_ids.size(),
        #                         device=topk_ids.device)

        topk_ids = torch.randint(low = 0,
                                 high = local_num_experts - 1,
                                 size = topk_ids.size(),
                                 device=topk_ids.device)

        #topk_ids = torch.where((topk_ids < local_expert_offset) | (topk_ids >= (local_expert_offset + local_num_experts)),
        #                       global_num_experts,
        #                       topk_ids)
        #topk_ids = torch.where(topk_ids == global_num_experts,
        #                       global_num_experts,
        #                       topk_ids - local_expert_offset)



        #packed_tensor = (topk_ids.to(torch.int32) << 16) | topk_weights.to(torch.bfloat16).view(torch.int16)
        packed_tensor = torch.zeros_like(topk_ids, dtype=torch.int32, device=topk_ids.device)
        #topk_weights_d = topk_weights.to(torch.bfloat16).view(torch.int16).to(torch.int32) << 16 
        #torch.bitwise_or(topk_weights_d, topk_ids.to(torch.int32), out=packed_tensor)

        print (f"topk ids {topk_ids}")
        print(f"packed_tensor {packed_tensor}")

        x_quant = hidden_states
        x_scale = a1q_scale
        if x_scale is not None:
            x_scale = x_scale.view(torch.float8_e4m3fn).reshape(-1)

        print (f"topk ids {topk_ids}")
        print (f"intermediate {self.intermediate_size}")

        if self.moe.ep_rank == 1 and False:
            print (f"topk ids {topk_ids}")
            print (f"global num experts {global_num_experts}")
            print (f"local num experts : {local_num_experts} | local expert offset {local_expert_offset} ...")
            print (f"expert map {expert_map}")

            print (f"x_quant {x_quant.shape} {x_quant.dtype}")
            if x_scale is not None:
                print(f"x_scale {x_scale.shape} {x_scale.dtype}")
            print (f"w1 {w1.shape} {w1.dtype}")
            print (f"w2 {w2.shape} {w2.dtype}")
            print (f"w1 scale {w1_scale.shape} {w1_scale.dtype}")
            print (f"w2 scale {w2_scale.shape} {w2_scale.dtype}")

        #assert False
        #if envs.VLLM_USE_FLASHINFER_MXFP4_BF16_MOE:
        #    assert hidden_states.dtype == torch.bfloat16
        #    x_quant = hidden_states
        #    x_scale = None
        #else:
        #    x_quant, x_scale = mxfp8_quantize(hidden_states, False)  # to mxfp8
        #    x_scale = x_scale.view(torch.float8_e4m3fn).reshape(-1)

        assert w1_scale is not None
        assert w2_scale is not None

        def describe_tensor(t, name) -> str:
            if t is None:
                return f"    - {name} : None \n"
            else:
                return f"    - {name} : {t.shape} {t.dtype} {t.device} \n"

        def describe_kwargs(kwargs):
            s = "invocation: \n"
            for k, v in kwargs.items():
                if v is None or isinstance(v, torch.Tensor):
                    s += describe_tensor(v, k)
                else:
                    s += f"     - {k}: {v}\n"
            return s


        kwargs = {"topk_ids" : packed_tensor,
                  "routing_bias" : None,
                  "hidden_states": x_quant,
                  "hidden_states_scale": x_scale,
                  "gemm1_weights" : w1,
                  "gemm1_weights_scale" : w1_scale,
                  "gemm1_bias": w1_bias,
                  "gemm1_alpha" : gemm1_alpha,
                  "gemm1_beta": gemm1_beta,
                  "gemm1_clamp_limit": gemm1_clamp_limit,
                  "gemm2_weights": w2,
                  "gemm2_weights_scale": w2_scale,
                  "gemm2_bias": w2_bias,
                  "output1_scale_scalar": None,
                  "output1_scale_gate_scalar": None,
                  "output2_scale_scalar": None,
                  "num_experts": local_num_experts,
                  "top_k": topk,
                  "n_group": None,
                  "topk_group": None,
                  "intermediate_size": self.intermediate_size,
                  "local_expert_offset": 0, 
                  "local_num_experts": local_num_experts,
                  "routed_scaling_factor": None,
                  "tile_tokens_dim": 8,
                  "routing_method_type" : 1,
                  "do_finalize": True,
                  "output": output,
                  }

        print (f"CURRENT DEVICE : {torch.cuda.current_device()} ...")
        def transfer_to_current_device(kwargs):
            new_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    new_kwargs[k] = v.to(torch.cuda.current_device())
            new_kwargs = kwargs | new_kwargs
            return new_kwargs
        
        kwargs = transfer_to_current_device(kwargs)

        s = describe_kwargs(kwargs)
        if s not in DEBUG_PRINTS:
            print (s)
            DEBUG_PRINTS.add(s)

        trtllm_gen_output = trtllm_fp4_block_scale_routed_moe(**kwargs)[0]

        #trtllm_gen_output = trtllm_fp4_block_scale_routed_moe(
        #    topk_ids=packed_tensor,
        #    routing_bias=None,
        #    hidden_states=x_quant,
        #    hidden_states_scale=x_scale,
        #    gemm1_weights = w1,  # uint8 (e2m1 x 2)
        #    gemm1_weights_scale = w1_scale,  # uint8 (e4m3 x 2)
        #    gemm1_bias = w1_bias,  # fp32 per expert per channel
        #    gemm1_alpha = gemm1_alpha,  # fp32 per expert
        #    gemm1_beta = gemm1_beta,  # fp32 per expert
        #    gemm1_clamp_limit = gemm1_clamp_limit,  # fp32 per expert
        #    gemm2_weights = w2,  # uint8 (e2m1 x 2)
        #    gemm2_weights_scale = w2_scale,  # ue8m0
        #    gemm2_bias = w2_bias,  # fp32 per expert per channel
        #    output1_scale_scalar = None,
        #    output1_scale_gate_scalar = None,
        #    output2_scale_scalar = None,
        #    #num_experts = global_num_experts,
        #    num_experts = local_num_experts,
        #    top_k = topk,
        #    n_group = None,
        #    topk_group = None,
        #    intermediate_size = self.intermediate_size,
        #    #local_expert_offset = local_expert_offset,
        #    #local_num_experts = local_num_experts,
        #    local_expert_offset = 0,
        #    local_num_experts = local_num_experts,
        #    routed_scaling_factor = None,
        #    tile_tokens_dim = 8, #self._get_tile_tokens_dim(hidden_states, topk),
        #    routing_method_type = 1,  # renormalize
        #    do_finalize = True,
        #    output = output,
        #    #output = None,
        #)[0]
        return output

        print (f"trtllm_gen_output {trtllm_gen_output.shape} {trtllm_gen_output.dtype}")

        return trtllm_gen_output
