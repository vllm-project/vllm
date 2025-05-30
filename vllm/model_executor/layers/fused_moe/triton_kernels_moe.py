from typing import Callable, List, Optional
from dataclasses import dataclass, field

import torch

from vllm import _custom_ops as ops
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.fused_moe import get_config_qtype
from vllm.utils import direct_register_custom_op
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache, moe_kernel_quantize_input)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP)

from triton_kernels.matmul_ogs import matmul_ogs
from triton_kernels.routing import (routing, RoutingData, GatherIndx, ScatterIndx)

def forward_cuda_triton(hidden_states: torch.Tensor, 
                        w1: torch.Tensor, 
                        w2: torch.Tensor,
                        use_grouped_topk: bool,
                        top_k: int,
                        router_logits: torch.Tensor,
                        renormalize: bool,
                        topk_group: Optional[int] = None,
                        num_expert_group: Optional[int] = None,
                        global_num_experts: int = -1,
                        expert_map: Optional[torch.Tensor] = None,
                        # custom_routing_function: Optional[Callable] = None,
                        scoring_func: str = "softmax",
                        e_score_correction_bias: Optional[torch.Tensor] = None,
                        apply_router_weight_on_input: bool = False,
                        activation: str = "silu"
                        ) -> torch.Tensor:
    
    # feature check
    assert use_grouped_topk == False, "use_grouped_topk is not supported in new triton MoE kernel"
    assert topk_group is None, "topk_group is not supported in new triton MoE kernel"
    assert num_expert_group is None, "num_expert_group is not supported in new triton MoE kernel"
    assert scoring_func == "softmax", "scoring_func is not supported in new triton MoE kernel"
    assert e_score_correction_bias is None, "e_score_correction_bias is not supported in new triton MoE kernel"

    if not renormalize: 
        router_logits = torch.softmax(router_logits, dim=-1)
    routing_data, gather_idx, scatter_idx = routing(router_logits, top_k, renormalize)

    return fused_experts_triton_exp(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        routing_data=routing_data,
        gather_indx=gather_idx,
        scatter_indx=scatter_idx,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        global_num_experts=global_num_experts,
        expert_map=expert_map
    )

# This is a triton implementation of the fused_experts function
def fused_experts_triton_exp(hidden_states: torch.Tensor,
                  w1: torch.Tensor,
                  w2: torch.Tensor,
                  routing_data: RoutingData,
                  gather_indx: GatherIndx,
                  scatter_indx: ScatterIndx,
                  inplace: bool = False,
                  activation: str = "silu",
                  apply_router_weight_on_input: bool = False,
                  use_fp8_w8a8: bool = False,
                  use_int8_w8a8: bool = False,
                  use_int8_w8a16: bool = False,
                  use_int4_w4a16: bool = False,
                  per_channel_quant: bool = False,
                  global_num_experts: int = -1,
                  expert_map: Optional[torch.Tensor] = None,
                  w1_scale: Optional[torch.Tensor] = None,
                  w2_scale: Optional[torch.Tensor] = None,
                  w1_zp: Optional[torch.Tensor] = None,
                  w2_zp: Optional[torch.Tensor] = None,
                  a1_scale: Optional[torch.Tensor] = None,
                  a2_scale: Optional[torch.Tensor] = None,
                  block_shape: Optional[List[int]] = None,
                  allow_deep_gemm: bool = False )-> torch.Tensor:
    
    # type check
    assert hidden_states.dtype == torch.bfloat16, "hidden_states must be bfloat16"
    assert w1.dtype == torch.bfloat16, "w1 must be bfloat16"
    assert w2.dtype == torch.bfloat16, "w2 must be bfloat16"

    # Shape check
    assert hidden_states.ndim == 2, "hidden_states must be 2D"
    assert hidden_states.shape[-1] == w1.shape[-2], "hidden_states shape[-1] must be equal to w1 shape[-2]"
    assert w2.shape[-1] == w1.shape[1], "w2 shape[-1] must be equal to w1 shape[1]"

    # feature check
    # assert inplace == False, "Inplace is not supported in new triton MoE kernel"
    # assert apply_router_weight_on_input == False, "apply_router_weight_on_input is not supported in new triton MoE kernel"
    # assert use_fp8_w8a8 == False, "use_fp8_w8a8 is not supported in new triton MoE kernel"
    # assert use_int8_w8a8 == False, "use_int8_w8a8 is not supported in new triton MoE kernel"
    # assert use_int8_w8a16 == False, "use_int8_w8a16 is not supported in new triton MoE kernel"
    # assert use_int4_w4a16 == False, "use_int4_w4a16 is not supported in new triton MoE kernel"
    # assert per_channel_quant == False, "per_channel_quant is not supported in new triton MoE kernel"
    # # assert global_num_experts == -1, "global_num_experts is not supported in new triton MoE kernel"
    # assert expert_map is None, "expert_map is not supported in new triton MoE kernel"
    # assert w1_scale is None, "w1_scale is not supported in new triton MoE kernel"
    # assert w2_scale is None, "w2_scale is not supported in new triton MoE kernel"
    # assert w1_zp is None, "w1_zp is not supported in new triton MoE kernel"
    # assert w2_zp is None, "w2_zp is not supported in new triton MoE kernel"
    # assert a1_scale is None, "a1_scale is not supported in new triton MoE kernel"
    # assert a2_scale is None, "a2_scale is not supported in new triton MoE kernel"
    # assert block_shape is None, "block_shape is not supported in new triton MoE kernel"
    # assert allow_deep_gemm == False, "allow_deep_gemm is not supported in new triton MoE kernel"

    M, K = hidden_states.shape
    N = w1.shape[2]
    n_expts_tot = routing_data.n_expts_tot
    n_expts_act = routing_data.n_expts_act
    dtype = hidden_states.dtype
    
    # consistent with default implementation
    intermediate_cache2 = torch.empty((M * n_expts_act, N // 2),
                                      device="cuda",
                                      dtype=dtype)    
    
    intermediate_cache1 = matmul_ogs(hidden_states, w1, None, routing_data, gather_indx=gather_indx)
    
    if activation == "silu":
        torch.ops._C.silu_and_mul(intermediate_cache2,
                                    intermediate_cache1.view(-1, N))
    elif activation == "gelu":
        torch.ops._C.gelu_and_mul(intermediate_cache2,
                                    intermediate_cache1.view(-1, N))
    else:
        raise ValueError(f"Unsupported FusedMoe activation: {activation}")

    intermediate_cache3 = matmul_ogs(intermediate_cache2, w2, None, routing_data, scatter_indx=scatter_indx, gammas=routing_data.gate_scal)

    return intermediate_cache3


def forward_cuda_triton_fake(hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
                        use_grouped_topk: bool,
                        top_k: int,
                        router_logits: torch.Tensor,
                        renormalize: bool,
                        topk_group: Optional[int] = None,
                        num_expert_group: Optional[int] = None,
                        global_num_experts: int = -1,
                        expert_map: Optional[torch.Tensor] = None,
                        # custom_routing_function: Optional[Callable] = None,
                        scoring_func: str = "softmax",
                        e_score_correction_bias: Optional[torch.Tensor] = None,
                        apply_router_weight_on_input: bool = False,
                        activation: str = "silu"
                        ) -> torch.Tensor:
    return torch.empty_like(hidden_states)

direct_register_custom_op(
    op_name="forward_cuda_triton",
    op_func=forward_cuda_triton,
    mutates_args=[],
    fake_impl=forward_cuda_triton_fake,
    tags=(torch.Tag.needs_fixed_stride_order, ),
)

@dataclass
class ExpTritonExpertsRoutingData(mk.RoutingData):
    routing_data: RoutingData = field()
    gather_indx: GatherIndx = field()
    scatter_indx: ScatterIndx = field()

class ExpTritonExperts(mk.FusedMoEPermuteExpertsUnpermute):

    def __init__(
        self,
        use_fp8_w8a8: bool,
        use_int8_w8a8: bool,
        use_int8_w8a16: bool,
        use_int4_w4a16: bool,
        per_channel_quant: bool,
        block_shape: Optional[list[int]] = None,
        block_m: Optional[int] = None,
    ):
        super().__init__()
        self.use_fp8_w8a8 = use_fp8_w8a8
        self.use_int4_w4a16 = use_int4_w4a16
        self.use_int8_w8a8 = use_int8_w8a8
        self.use_int8_w8a16 = use_int8_w8a16
        self.block_shape = block_shape
        self.block_m = block_m
        self.qtype = get_config_qtype(use_fp8_w8a8=use_fp8_w8a8,
                                      use_int8_w8a8=use_int8_w8a8,
                                      use_int8_w8a16=use_int8_w8a16,
                                      use_int4_w4a16=use_int4_w4a16)
        self.per_channel_quant = per_channel_quant

    def workspace_shapes(
        self,
        a: torch.Tensor,
        M: int,
        N: int,
        K: int,
        topk: int,
        num_experts: int,
    ) -> tuple[int, int, torch.dtype]:
        factor = num_experts if a.dim() == 3 else 1
        workspace2 = M * topk * N * factor
        return (None, workspace2, a.dtype)

    def apply(self, 
              hidden_states: torch.Tensor, 
              w1: torch.Tensor, 
              w2: torch.Tensor, 
              routing_data_packed: mk.RoutingData,
              activation: str, 
              global_num_experts: int, 
              expert_map: Optional[torch.Tensor], 
              w1_scale: Optional[torch.Tensor], 
              w2_scale: Optional[torch.Tensor], 
              w1_zp: Optional[torch.Tensor], 
              w2_zp: Optional[torch.Tensor], 
              a1q_scale: Optional[torch.Tensor], 
              a2_scale: Optional[torch.Tensor], 
              workspace13: Optional[torch.Tensor], 
              workspace2: Optional[torch.Tensor], 
              expert_num_tokens: Optional[torch.Tensor],
              # the following is added since we may want to fuse experts weight inside the matmul
              apply_router_weight_on_input: bool,
              apply_router_weight_on_output: bool
        ) -> torch.Tensor:
        assert hidden_states.dtype == torch.bfloat16, "hidden_states must be bfloat16"
        assert w1.dtype == torch.bfloat16, "w1 must be bfloat16"
        assert w2.dtype == torch.bfloat16, "w2 must be bfloat16"

        # Shape check
        assert hidden_states.ndim == 2, "hidden_states must be 2D"
        assert hidden_states.shape[-1] == w1.shape[-2], "hidden_states shape[-1] must be equal to w1 shape[-2]"
        assert w2.shape[-1] == w1.shape[1], "w2 shape[-1] must be equal to w1 shape[1]"

        # feature check
        assert self.use_fp8_w8a8 == False, "use_fp8_w8a8 is not supported in new triton MoE kernel"
        assert self.use_int8_w8a8 == False, "use_int8_w8a8 is not supported in new triton MoE kernel"
        assert self.use_int8_w8a16 == False, "use_int8_w8a16 is not supported in new triton MoE kernel"
        assert self.use_int4_w4a16 == False, "use_int4_w4a16 is not supported in new triton MoE kernel"
        assert self.per_channel_quant == False, "per_channel_quant is not supported in new triton MoE kernel"
        # assert global_num_experts == -1, "global_num_experts is not supported in new triton MoE kernel"
        assert expert_map is None, "expert_map is not supported in new triton MoE kernel"
        assert w1_scale is None, "w1_scale is not supported in new triton MoE kernel"
        assert w2_scale is None, "w2_scale is not supported in new triton MoE kernel"
        assert w1_zp is None, "w1_zp is not supported in new triton MoE kernel"
        assert w2_zp is None, "w2_zp is not supported in new triton MoE kernel"
        assert a1q_scale is None, "a1_scale is not supported in new triton MoE kernel"
        assert a2_scale is None, "a2_scale is not supported in new triton MoE kernel"
                
        M, K = hidden_states.shape
        N = w1.shape[2]
        n_expts_tot = routing_data_packed.routing_data.n_expts_tot
        n_expts_act = routing_data_packed.routing_data.n_expts_act

        routing_data = routing_data_packed.routing_data
        gather_indx = routing_data_packed.gather_indx
        scatter_indx = routing_data_packed.scatter_indx
        
        intermediate_cache2 = _resize_cache(workspace2,
                                            (M * n_expts_act, N // 2))

        intermediate_cache1 = matmul_ogs(hidden_states, w1, None, routing_data, gather_indx=gather_indx, gammas=routing_data.gate_scal if apply_router_weight_on_input else None)

        if activation == "silu":
            torch.ops._C.silu_and_mul(intermediate_cache2,
                                        intermediate_cache1.view(-1, N))
        elif activation == "gelu":
            torch.ops._C.gelu_and_mul(intermediate_cache2,
                                        intermediate_cache1.view(-1, N))
        else:
            raise ValueError(f"Unsupported FusedMoe activation: {activation}")

        a2q_scale: Optional[torch.Tensor] = None

        qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(
            intermediate_cache2, a2_scale, self.qtype, self.per_channel_quant,
            self.block_shape)
        
        intermediate_cache3 = matmul_ogs(qintermediate_cache2, w2, None, routing_data, scatter_indx=None, gammas=routing_data.gate_scal if apply_router_weight_on_output else None)

        #TODO: change this hack
        intermediate_cache3 = intermediate_cache3[scatter_indx.src_indx].reshape(M, -1, K)

        return intermediate_cache3


class ExpTritonFusedMoEKernel(mk.FusedMoEModularKernel):
    
    def __init__(self, prepare_finalize, fused_experts):
        super().__init__(prepare_finalize, fused_experts)

    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        # custom routing
        routing_data: ExpTritonExpertsRoutingData,
        # Original Args
        inplace: bool = False,
        activation: str = "silu",
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        w1_zp: Optional[torch.Tensor] = None,
        w2_zp: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
    ):
        a1 = hidden_states
        M, K = hidden_states.shape
        E, _, N = w1.shape
        n_expts_tot = routing_data.routing_data.n_expts_tot
        n_expts_act = routing_data.routing_data.n_expts_act

        if global_num_experts == -1:
            global_num_experts = E
        
        output = a1 if inplace else torch.zeros_like(a1)

        _, workspace2_shape, workspace_dtype = (
            self.fused_experts.workspace_shapes(a1, M, N, K, n_expts_act,
                                                global_num_experts))
        
        workspace2 = torch.zeros(workspace2_shape,
                                 device=a1.device,
                                 dtype=workspace_dtype)
        
        a1q, a1q_scale, expert_num_tokens = self.prepare_finalize.prepare(
            a1, a1_scale, a2_scale, None, None, global_num_experts,
            expert_map, False)
        
        fused_out = self.fused_experts.apply(
            a1q,
            w1,
            w2,
            routing_data,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_zp=w1_zp,
            w2_zp=w2_zp,
            a1q_scale=a1q_scale,
            a2_scale=a2_scale,
            workspace13=None,
            workspace2=workspace2,
            expert_num_tokens=expert_num_tokens,
            apply_router_weight_on_input=apply_router_weight_on_input,
            apply_router_weight_on_output= not apply_router_weight_on_input,
        )

        # the last argument is a hack not to aply topk_weights since already applied in matmul
        self.prepare_finalize.finalize(output, fused_out, _, _, True)   

        return output

def modular_triton_moe_kernels_forward(
    hidden_states: torch.Tensor,
    w1: torch.Tensor, 
    w2: torch.Tensor,
    #TODO: this is a hack to aviod torch.compile bug in pytorch #154009
    gate_scal: torch.Tensor,
    expt_hist: torch.Tensor,
    n_expts_tot: int,
    n_expts_act: int,   
    topk_indx: torch.Tensor,
    gate_indx: torch.Tensor,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: Optional[list[int]] = None,
) -> torch.Tensor:
    routing_data = RoutingData(gate_scal, expt_hist, n_expts_tot, n_expts_act)
    gather_indx = GatherIndx(src_indx=topk_indx, dst_indx=gate_indx)
    scatter_indx = ScatterIndx(src_indx=gate_indx, dst_indx=topk_indx)
    exp_routing_data = ExpTritonExpertsRoutingData(routing_data, gather_indx, scatter_indx)

    qtype = get_config_qtype(
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
    )
    fn = ExpTritonFusedMoEKernel(
        MoEPrepareAndFinalizeNoEP(
            quant_dtype=qtype,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
        ),
        ExpTritonExperts(
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
        ),
    )
    return fn(hidden_states,
            w1,
            w2,
            # custom routing
            exp_routing_data,
            apply_router_weight_on_input=False)

def modular_triton_moe_kernels_forward_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor, 
    w2: torch.Tensor, 
    gate_scal: torch.Tensor,
    expt_hist: torch.Tensor,
    n_expts_tot: int,
    n_expts_act: int,   
    topk_indx: torch.Tensor,
    gate_indx: torch.Tensor,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: Optional[list[int]] = None,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)

direct_register_custom_op(
    op_name="modular_triton_moe_kernels_forward",
    op_func=modular_triton_moe_kernels_forward,
    mutates_args=[],
    fake_impl=modular_triton_moe_kernels_forward_fake,
    tags=(torch.Tag.needs_fixed_stride_order, ),
)
 