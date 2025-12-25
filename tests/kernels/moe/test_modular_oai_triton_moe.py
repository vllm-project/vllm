# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test modular OAI Triton MoE
"""

import pytest
import torch

from vllm.utils.import_utils import has_triton_kernels

if not has_triton_kernels():
    pytest.skip(
        "triton_kernels not found, skipping all related tests",
        allow_module_level=True,
    )

from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig
from triton_kernels.numerics import InFlexData
from triton_kernels.numerics_details.mxfp import downcast_to_mxfp, upcast_from_mxfp
from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
from triton_kernels.tensor_details import layout
from triton_kernels.testing import assert_close

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.config import mxfp4_w4a16_moe_quant_config
from vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe import (
    OAITritonExperts,
    UnfusedOAITritonExperts,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEModularKernel
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP,
)
from vllm.model_executor.layers.utils import shuffle_weight
from vllm.platforms import current_platform

MNK = [
    (1, 512, 384),
    (1, 2880, 2880),
    (2, 512, 384),
    (2, 2880, 2880),
    (16, 2880, 2880),
]


def unshuffle_weight(w: torch.Tensor):
    first = w[..., ::2]
    second = w[..., 1::2]
    return torch.concat((first, second), dim=-1)


def make_weights(dtype, k, n, e):
    w1 = torch.randn((e, k, 2 * n), dtype=dtype, device="cuda")
    w1_bias = torch.randn((e, 2 * n), dtype=dtype, device="cuda")

    w2 = torch.randn((e, n, k), dtype=dtype, device="cuda")
    w2_bias = torch.randn((e, k), dtype=dtype, device="cuda")

    w1_tri = w1.clone()
    w2_tri = w2.clone()

    w1_bias_tri = w1_bias.clone()
    w2_bias_tri = w2_bias.clone()
    w1_bias_tri = w1_bias_tri.to(torch.float32)
    w2_bias_tri = w2_bias_tri.to(torch.float32)

    # shuffle weights
    w1_tri = shuffle_weight(w1_tri)
    w1_bias_tri = shuffle_weight(w1_bias_tri)

    # quant triton_weights
    w1_tri, w1_scale_tri = downcast_to_mxfp(w1_tri, torch.uint8, axis=1)
    w1 = upcast_from_mxfp(w1_tri, w1_scale_tri, dtype, axis=1)
    w1 = unshuffle_weight(w1)

    w2_tri, w2_scale_tri = downcast_to_mxfp(w2_tri, torch.uint8, axis=1)
    w2 = upcast_from_mxfp(w2_tri, w2_scale_tri, dtype, axis=1)

    num_warps = 8
    w_layout, w_layout_opts = layout.make_default_matmul_mxfp4_w_layout(mx_axis=1)
    w_scale_layout, w_scale_layout_opts = (
        layout.make_default_matmul_mxfp4_w_scale_layout(mx_axis=1, num_warps=num_warps)
    )

    w1_tri = convert_layout(wrap_torch_tensor(w1_tri, FP4), w_layout, **w_layout_opts)
    w1_scale_tri = convert_layout(
        wrap_torch_tensor(w1_scale_tri),
        w_scale_layout,
        **w_scale_layout_opts,
    )

    w2_tri = convert_layout(wrap_torch_tensor(w2_tri, FP4), w_layout, **w_layout_opts)
    w2_scale_tri = convert_layout(
        wrap_torch_tensor(w2_scale_tri),
        w_scale_layout,
        **w_scale_layout_opts,
    )

    w1_precision_config = PrecisionConfig(
        weight_scale=w1_scale_tri, flex_ctx=FlexCtx(rhs_data=InFlexData())
    )
    w2_precision_config = PrecisionConfig(
        weight_scale=w2_scale_tri, flex_ctx=FlexCtx(rhs_data=InFlexData())
    )

    return (
        w1,
        w2,
        w1_bias,
        w2_bias,
        w1_tri,
        w2_tri,
        w1_bias_tri,
        w2_bias_tri,
        w1_precision_config,
        w2_precision_config,
    )


def swiglu(x, alpha: float = 1.702, limit: float = 1.0):
    # Note we add an extra bias of 1 to the linear layer
    x_glu, x_linear = torch.chunk(x, 2, dim=-1)
    if limit is not None:
        x_glu = x_glu.clamp(max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    if limit is not None:
        x_linear = x_linear.clamp(min=-limit, max=limit)
    return out_glu * (x_linear + 1)


def torch_moe_impl(
    hidden_states: torch.Tensor,  # (M, K)
    w1: torch.Tensor,  # (E, K, 2N)
    w2: torch.Tensor,  # (E, N, K)
    w1_bias: torch.Tensor,  # (E, 2N)
    w2_bias: torch.Tensor,  # (E, K)
    topk_weights: torch.Tensor,  # (M, topk)
    topk_ids: torch.Tensor,  # (M, topk)
):
    w1 = w1[topk_ids, ...]
    w1_bias = w1_bias[topk_ids, ...]
    hidden_states = torch.einsum("bekc,bk->bec", w1, hidden_states) + w1_bias
    hidden_states = swiglu(hidden_states, limit=7)

    w2 = w2[topk_ids, ...]
    w2_bias = w2_bias[topk_ids, ...]
    hidden_states = torch.einsum("bekc,bek->bec", w2, hidden_states) + w2_bias

    # Weighted sum of experts
    hidden_states = torch.einsum("bec,be->bc", hidden_states, topk_weights)
    return hidden_states


def oai_triton_moe_impl(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: "PrecisionConfig",
    w2_scale: "PrecisionConfig",
    w1_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
    num_experts: int,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    unfused: bool = False,
) -> torch.Tensor:
    quant_config = mxfp4_w4a16_moe_quant_config(
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )

    if unfused:
        fused_experts = UnfusedOAITritonExperts(quant_config)
    else:
        fused_experts = OAITritonExperts(quant_config)

    mk = FusedMoEModularKernel(MoEPrepareAndFinalizeNoEP(), fused_experts)

    return mk.forward(
        hidden_states=x,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=True,
        activation="swigluoai",
        global_num_experts=num_experts,
        expert_map=None,
        apply_router_weight_on_input=False,
    )


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("m,n,k", MNK)
@pytest.mark.parametrize("num_experts", [32, 128])
@pytest.mark.parametrize("topk", [4])
@pytest.mark.parametrize("unfused", [True, False])
def test_oai_triton_moe(
    dtype: torch.dtype,
    m: int,
    n: int,
    k: int,
    num_experts: int,
    topk: int,
    unfused: bool,
    workspace_init,
):
    current_platform.seed_everything(0)
    (
        w1,
        w2,
        w1_bias,
        w2_bias,
        w1_tri,
        w2_tri,
        w1_bias_tri,
        w2_bias_tri,
        w1_precision_config,
        w2_precision_config,
    ) = make_weights(dtype, k, n, num_experts)

    x = torch.randn((m, k), dtype=dtype, device="cuda")
    router_logits = torch.randn(m, num_experts, device="cuda", dtype=dtype)
    topk_weights, topk_ids = torch.topk(router_logits, k=topk, dim=-1, sorted=True)
    topk_weights = torch.nn.functional.softmax(topk_weights, dim=-1)

    with set_current_vllm_config(VllmConfig()):
        out_ref = torch_moe_impl(x, w1, w2, w1_bias, w2_bias, topk_weights, topk_ids)

        out = oai_triton_moe_impl(
            x,
            w1_tri,
            w2_tri,
            w1_precision_config,
            w2_precision_config,
            w1_bias_tri,
            w2_bias_tri,
            num_experts,
            topk_weights,
            topk_ids,
            unfused,
        )

    assert_close(ref=out_ref, tri=out, maxtol=0.025, rmstol=0.005)
