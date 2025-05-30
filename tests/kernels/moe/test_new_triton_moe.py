from typing import Callable, List, Optional
import pytest
from dataclasses import dataclass, fields

import torch
import triton
import triton.language as tl

from triton_kernels.matmul_ogs import matmul_ogs
from triton_kernels.routing import (routing, RoutingData, GatherIndx, ScatterIndx)
from triton_kernels.testing import assert_close

from vllm.model_executor.layers.fused_moe.fused_moe import ( fused_moe, fused_experts )
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.triton_kernels_moe import modular_triton_moe_kernels_forward, forward_cuda_triton
from vllm import _custom_ops as ops

def forward_modular_triton(
    x, w1, w2,
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
):
    routing_data = FusedMoE.select_experts(None, router_logits, top_k, False, renormalize)

    return torch.ops.vllm.modular_triton_moe_kernels_forward(
        x,
        w1,
        w2,
        # custom routing
        gate_scal=routing_data.routing_data.gate_scal,
        expt_hist=routing_data.routing_data.expt_hist,
        n_expts_tot=routing_data.routing_data.n_expts_tot,
        n_expts_act=routing_data.routing_data.n_expts_act, 
        topk_indx=routing_data.gather_indx.src_indx,
        gate_indx=routing_data.gather_indx.dst_indx,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
    )
    

@dataclass
class Case:
    num_token: int
    inter_size: int
    K: int
    num_expts_tot: int
    num_expts_act: int

@pytest.mark.parametrize(
    ", ".join(f.name for f in fields(Case)),
    [
        tuple(getattr(case, f.name) for f in fields(Case)) for case in [
            Case(num_token=32, inter_size=512, K=32, num_expts_tot=128, num_expts_act=4),
            Case(num_token=16, inter_size=512, K=32, num_expts_tot=128, num_expts_act=4),
            Case(num_token=1024, inter_size=2048, K=32, num_expts_tot=128, num_expts_act=4),
        ]
    ],
)
def test_equiv(num_token, inter_size, K, num_expts_tot, num_expts_act, monkeypatch: pytest.MonkeyPatch):

    # triton way to generate logits to break ties between experts
    randbits = [torch.randperm(num_expts_tot) for _ in range(num_token)]
    x = [(-1)**i * ((16384 + ((i * 512) % 4096) + bits).to(torch.int16).view(torch.bfloat16)) for i, bits in enumerate(randbits)]
    exp_data = torch.stack(x).to(device="cuda")

    # create input tensor
    x = torch.randn((num_token, K), dtype=torch.bfloat16, device="cuda")
    w1 = torch.randn((num_expts_tot, inter_size, K), dtype=torch.bfloat16, device="cuda")
    w2 = torch.randn((num_expts_tot, K, inter_size // 2), dtype=torch.bfloat16, device="cuda")
    
    exp_data_tri = exp_data.clone()
    x_tri = x.clone()
    w1_tri = w1.clone()
    w2_tri = w2.clone()
    w1_tri = w1_tri.transpose(-2, -1).contiguous()
    w2_tri = w2_tri.transpose(-2, -1).contiguous()

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_EXP_MOE", "1")
        out_triton_monolithic = forward_cuda_triton(x_tri, w1_tri, w2_tri, False, num_expts_act, exp_data_tri, True)
        out_triton = forward_modular_triton(x_tri, w1_tri, w2_tri, False, num_expts_act, exp_data_tri, True)
        out_ref = fused_moe(x, w1, w2, exp_data, num_expts_act, True)
        assert_close(ref=out_ref, tri=out_triton)
        assert_close(ref=out_ref, tri=out_triton_monolithic)
        
