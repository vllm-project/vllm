# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from copy import deepcopy
from dataclasses import dataclass, fields

import pytest
import torch
import torch.nn.functional as F
import triton_kernels.swiglu
from triton_kernels.matmul_ogs import FlexCtx, MicroscalingCtx, PrecisionConfig
from triton_kernels.numerics import InFlexData
from triton_kernels.numerics_details.mxfp import (SwizzlingType,
                                                  upcast_from_mxfp)
from triton_kernels.testing import assert_close

from vllm.model_executor.layers.fused_moe.triton_kernels_moe import (
    triton_kernel_moe_forward)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import quantize
from vllm.model_executor.layers.utils import shuffle_weight


def deshuffle(w: torch.Tensor):
    first = w[..., ::2]
    second = w[..., 1::2]

    deshuffled = torch.concat((first, second), dim=-1)
    return deshuffled


@dataclass
class ModelConfig:
    num_hidden_layers: int = 36
    num_experts: int = 128
    experts_per_token: int = 4
    vocab_size: int = 201088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    sliding_window: int = 128
    initial_context_length: int = 4096
    rope_theta: float = 150000.0
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0


def swiglu(x, alpha: float = 1.702, limit: float = 1.0):
    # Note we add an extra bias of 1 to the linear layer
    x_glu, x_linear = torch.chunk(x, 2, dim=-1)
    if limit is not None:
        x_glu = x_glu.clamp(max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    if limit is not None:
        x_linear = x_linear.clamp(min=-limit, max=limit)
    return out_glu * (x_linear + 1)


def oai_moe_forward(
        hidden_states: torch.Tensor,  # (M, K)
        w1: torch.Tensor,  # (E, 2N)
        w1_bias: torch.Tensor,  # (E, 2N, K)
        w2: torch.Tensor,  # (E, K, N)
        w2_bias: torch.Tensor,  # (E, N)
        gating_output: torch.Tensor,  # (M, E)
        topk: int):
    # model.py 309:330, assuming gating and norm
    t = hidden_states
    experts = torch.topk(gating_output, k=topk, dim=-1, sorted=True)
    expert_weights = torch.nn.functional.softmax(experts.values, dim=1)
    expert_indices = experts.indices

    # MLP #1
    mlp1_weight = w1[expert_indices, ...]
    mlp1_bias = w1_bias[expert_indices, ...]
    t = torch.einsum("beck,bk->bec", mlp1_weight, t) + mlp1_bias
    t = swiglu(t, limit=None)

    # MLP #2
    mlp2_weight = w2[expert_indices, ...]
    mlp2_bias = w2_bias[expert_indices, ...]
    t = torch.einsum("beck,bek->bec", mlp2_weight, t)
    t += mlp2_bias

    # Weighted sum of experts
    t = torch.einsum("bec,be->bc", t, expert_weights)

    return t


def smallest_even_divide_number(x, n):
    if x % n == 0:
        return x
    divisor = x // n
    return (divisor + 1) * n


@dataclass
class Case:
    a_dtype: str
    w_dtype: str


@pytest.mark.parametrize(
    ", ".join(f.name for f in fields(Case)),
    [
        tuple(getattr(case, f.name) for f in fields(Case)) for case in [
            # Case(a_dtype="bf16", w_dtype="bf16"),
            # Case(a_dtype="fp8_e4m3", w_dtype="fp8_e5m2"),
            Case(a_dtype="bf16", w_dtype="mx4")
        ]
    ],
)
@pytest.mark.parametrize("num_token", [2])
@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_equiv(num_token, a_dtype, w_dtype, tp):
    M = num_token
    E = ModelConfig.num_experts
    K = ModelConfig.hidden_size
    N = ModelConfig.intermediate_size // tp
    topk = ModelConfig.experts_per_token

    if w_dtype == "mx4":
        opt1 = dict()
        opt2 = dict()
        #TODO: improve performance by padding and turn on swizzle
        swizzle_mx_value = SwizzlingType.HOPPER
        swizzle_mx_scale = SwizzlingType.HOPPER
        opt1 = {
            "swizzle_mx_value": swizzle_mx_value,
            "swizzle_mx_scale": swizzle_mx_scale
        }
        opt2 = deepcopy(opt1)

    # fused_scatter doesn't work with fused_act, and can't worl with splitk
    # is_persistent only True on blackwell
    # constraints = {
    #     "block_m": 128,
    #     "block_k": 128,
    #     "split_k": 9,
    #     "fused_scatter": False,
    #     "is_persistent": False,
    #     "epilogue_subtile": 6,
    #     "num_stages": 4,
    #     "idle_sms": 0
    # }
    # opt_flags.update_opt_flags_constraints(constraints)

    randbits = [torch.randperm(E) for _ in range(M)]
    x = [(-1)**i *
         ((16384 +
           ((i * 512) % 4096) + bits).to(torch.int16).view(torch.bfloat16))
         for i, bits in enumerate(randbits)]
    exp_data = torch.stack(x).to(
        device="cuda")  # simulating gate_output (M, E)

    # create input tensor
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    w1 = torch.randn((E, 2 * N, K), dtype=torch.bfloat16, device="cuda")
    w1_bias = torch.randn((E, 2 * N), dtype=torch.bfloat16, device="cuda")

    w2 = torch.randn((E, K, N), dtype=torch.bfloat16, device="cuda")
    w2_bias = torch.randn((E, K), dtype=torch.bfloat16, device="cuda")

    if w_dtype == "mx4":
        w1 = w1.to(torch.float8_e5m2)
        w2 = w2.to(torch.float8_e5m2)

    exp_data_tri = exp_data.clone()
    x_tri = x.clone()
    w1_tri = w1.clone()
    w2_tri = w2.clone()

    w1_bias_tri = w1_bias.clone()
    w2_bias_tri = w2_bias.clone()
    w1_bias_tri = w1_bias_tri.to(torch.float32)
    w2_bias_tri = w2_bias_tri.to(torch.float32)

    dtype_dict = {
        "bf16": torch.bfloat16,
        "fp8_e4m3": torch.float8_e4m3fn,
        "fp8_e5m2": torch.float8_e5m2
    }

    x = x.to(dtype_dict[a_dtype]).to(torch.bfloat16)
    if w_dtype != "mx4":
        # simulate quantization support on reference impl
        w1 = w1.to(dtype_dict[w_dtype]).to(torch.bfloat16)
        w2 = w2.to(dtype_dict[w_dtype]).to(torch.bfloat16)

    # triton moe kernel use transposed shape for matmul
    w1_tri = w1_tri.transpose(-2, -1).contiguous()
    w2_tri = w2_tri.transpose(-2, -1).contiguous()

    # shuffle weights
    w1_tri = shuffle_weight(w1_tri)
    w1_bias_tri = shuffle_weight(w1_bias_tri)

    # quant triton_weights
    x_tri = x.to(dtype_dict[a_dtype])
    if w_dtype != "mx4":
        w1_tri = w1_tri.to(dtype_dict[w_dtype])
        w2_tri = w2_tri.to(dtype_dict[w_dtype])
        pc1 = PrecisionConfig(mx_ctx=MicroscalingCtx(),
                              flex_ctx=FlexCtx(rhs_data=InFlexData()))
        pc2 = PrecisionConfig(mx_ctx=MicroscalingCtx(),
                              flex_ctx=FlexCtx(rhs_data=InFlexData()))
    else:  # quantize to mx4
        # add padding to enable hbm_swizzle
        # make the tensor 64 x 256
        w1_right_pad = smallest_even_divide_number(w1_tri.shape[2],
                                                   256) - w1_tri.shape[2]

        w2_bottom_pad = w1_right_pad // 2
        w2_right_pad = smallest_even_divide_number(w2_tri.shape[2],
                                                   256) - w2_tri.shape[2]

        w1_tri = F.pad(w1_tri, (0, w1_right_pad, 0, 0, 0, 0),
                       mode="constant",
                       value=0)
        w2_tri = F.pad(w2_tri, (0, w2_right_pad, 0, w2_bottom_pad, 0, 0),
                       mode="constant",
                       value=0)

        w1_bias_tri = F.pad(w1_bias_tri, (0, w1_right_pad, 0, 0),
                            mode="constant",
                            value=0)
        w2_bias_tri = F.pad(w2_bias_tri, (0, w2_right_pad, 0, 0),
                            mode="constant",
                            value=0)

        w1_tri, w1_tri_flex, w1_tri_mx = quantize(w1_tri, "mx4", "cuda",
                                                  **opt1)
        w2_tri, w2_tri_flex, w2_tri_mx = quantize(w2_tri, "mx4", "cuda",
                                                  **opt2)
        pc1 = PrecisionConfig(mx_ctx=w1_tri_mx,
                              flex_ctx=FlexCtx(rhs_data=w1_tri_flex))
        pc2 = PrecisionConfig(mx_ctx=w2_tri_mx,
                              flex_ctx=FlexCtx(rhs_data=w2_tri_flex))

        w1 = upcast_from_mxfp(w1_tri,
                              w1_tri_mx.weight_scale,
                              torch.bfloat16,
                              axis=1,
                              swizzle_axis=2,
                              swizzle_value=swizzle_mx_value,
                              swizzle_scale=swizzle_mx_scale)
        w2 = upcast_from_mxfp(w2_tri,
                              w2_tri_mx.weight_scale,
                              torch.bfloat16,
                              axis=1,
                              swizzle_axis=2,
                              swizzle_value=swizzle_mx_value,
                              swizzle_scale=swizzle_mx_scale)

        # tucuate so the rest can run properly
        w1 = w1[..., :K, :2 * N]
        w2 = w2[..., :N, :K]

        w1 = deshuffle(w1)

        w1 = w1.transpose(-1, -2).contiguous()
        w2 = w2.transpose(-1, -2).contiguous()

    out_triton_monolithic = triton_kernel_moe_forward(
        hidden_states=x_tri,
        w1=w1_tri,
        w2=w2_tri,
        gating_output=exp_data_tri,
        topk=topk,
        renormalize=True,
        w1_bias=w1_bias_tri,
        w2_bias=w2_bias_tri,
        w1_precision=pc1,
        w2_precision=pc2)
    out_triton_monolithic = out_triton_monolithic[..., :K]

    out_ref = oai_moe_forward(hidden_states=x,
                              w1=w1,
                              w1_bias=w1_bias,
                              w2=w2,
                              w2_bias=w2_bias,
                              gating_output=exp_data,
                              topk=topk)
    assert_close(ref=out_ref,
                 tri=out_triton_monolithic,
                 maxtol=0.03,
                 rmstol=0.005)


def test_unit_shuffle():
    N = ModelConfig.intermediate_size
    K = ModelConfig.hidden_size
    m = torch.randn((K, 2 * N), dtype=torch.bfloat16, device="cuda")

    x = torch.randn(K, dtype=torch.bfloat16, device="cuda")

    m_shuffled = shuffle_weight(m)

    out_ref = x @ m
    out_ref = swiglu(out_ref, limit=1.0)

    out = x @ m_shuffled
    out = triton_kernels.swiglu.swiglu_torch(
        out,
        alpha=1.702,
        precision_config=triton_kernels.swiglu.PrecisionConfig(limit=1.0))

    assert_close(ref=out_ref, tri=out)
