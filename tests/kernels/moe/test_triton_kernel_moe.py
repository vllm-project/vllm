# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, fields

import pytest
import torch
from triton_kernels.testing import assert_close

from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe
from vllm.model_executor.layers.fused_moe.triton_kernels_moe import (
    triton_kernel_moe_forward)


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
            Case(num_token=32,
                 inter_size=512,
                 K=32,
                 num_expts_tot=128,
                 num_expts_act=4),
            Case(num_token=16,
                 inter_size=512,
                 K=32,
                 num_expts_tot=128,
                 num_expts_act=4),
            Case(num_token=1024,
                 inter_size=2048,
                 K=32,
                 num_expts_tot=128,
                 num_expts_act=4),
        ]
    ],
)
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("apply_router_weight_on_input", [False, True])
def test_equiv(num_token, inter_size, K, num_expts_tot, num_expts_act,
               renormalize, apply_router_weight_on_input):

    # triton way to generate logits to break ties between experts
    randbits = [torch.randperm(num_expts_tot) for _ in range(num_token)]
    x = [(-1)**i *
         ((16384 +
           ((i * 512) % 4096) + bits).to(torch.int16).view(torch.bfloat16))
         for i, bits in enumerate(randbits)]
    exp_data = torch.stack(x).to(device="cuda")

    # create input tensor
    x = torch.randn((num_token, K), dtype=torch.bfloat16, device="cuda")
    w1 = torch.randn((num_expts_tot, inter_size, K),
                     dtype=torch.bfloat16,
                     device="cuda")
    w2 = torch.randn((num_expts_tot, K, inter_size // 2),
                     dtype=torch.bfloat16,
                     device="cuda")

    exp_data_tri = exp_data.clone()
    x_tri = x.clone()
    w1_tri = w1.clone()
    w2_tri = w2.clone()

    # triton moe kernel use transposed shape for matmul
    w1_tri = w1_tri.transpose(-2, -1).contiguous()
    w2_tri = w2_tri.transpose(-2, -1).contiguous()

    out_triton_monolithic = triton_kernel_moe_forward(
        hidden_states=x_tri,
        w1=w1_tri,
        w2=w2_tri,
        gating_output=exp_data_tri,
        topk=num_expts_act,
        renormalize=renormalize,
        apply_router_weight_on_input=apply_router_weight_on_input)
    out_ref = fused_moe(
        hidden_states=x,
        w1=w1,
        w2=w2,
        gating_output=exp_data,
        topk=num_expts_act,
        renormalize=renormalize,
        apply_router_weight_on_input=apply_router_weight_on_input)
    assert_close(ref=out_ref, tri=out_triton_monolithic)
