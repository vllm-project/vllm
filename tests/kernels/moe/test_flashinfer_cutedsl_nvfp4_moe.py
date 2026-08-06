# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for FlashInfer CuTeDSL NVFP4 MoE."""

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.quantization.nvfp4_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    break_fp4_bytes,
)
from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import (
    SiluAndMul,
    SiluAndMulWithClamp,
    SwigluOAIAndMul,
)
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
    nvfp4_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_moe import (
    FlashInferCuteDSLExperts,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
    prepare_nvfp4_moe_layer_for_flashinfer_cutedsl,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_cutedsl_moe_nvfp4
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import set_random_seed

if not has_flashinfer_cutedsl_moe_nvfp4() or not (
    current_platform.is_device_capability_family(100)
):
    pytest.skip(
        "Requires FlashInfer CuTeDSL NVFP4 MoE on SM100",
        allow_module_level=True,
    )

_SWIGLU_ALPHA = 1.702
_SWIGLU_BETA = 1.0
# The GEMM1 output is O(0.5) at these tensor scales, so the OAI default of 7.0
# would never clamp anything. Pick a limit that actually bites.
_SWIGLU_LIMIT = 0.3

_ACT_CASES = [
    pytest.param(MoEActivation.SILU, None, None, None, id="silu"),
    pytest.param(MoEActivation.SILU, None, None, _SWIGLU_LIMIT, id="silu-clamp"),
    pytest.param(MoEActivation.RELU2_NO_MUL, None, None, None, id="relu2_no_mul"),
    pytest.param(
        MoEActivation.SWIGLUOAI,
        _SWIGLU_ALPHA,
        _SWIGLU_BETA,
        _SWIGLU_LIMIT,
        id="swigluoai",
    ),
    pytest.param(
        MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        _SWIGLU_ALPHA,
        _SWIGLU_BETA,
        _SWIGLU_LIMIT,
        id="swigluoai_uninterleave",
    ),
]


def _reference_activation(
    activation: MoEActivation,
    alpha: float | None,
    beta: float | None,
    limit: float | None,
):
    """vLLM's own op for this activation, so the reference is not re-derived."""
    if activation == MoEActivation.RELU2_NO_MUL:
        return lambda x: torch.square(torch.relu(x))
    if activation == MoEActivation.SWIGLUOAI:
        # SwigluOAIAndMul hardcodes beta=1 and reads gate/up interleaved.
        assert beta == 1.0
        return SwigluOAIAndMul(alpha=alpha, limit=limit)
    if activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE:
        return SiluAndMulWithClamp(limit, alpha, beta, compile_native=False)
    if limit is not None:
        return SiluAndMulWithClamp(limit, compile_native=False)
    return SiluAndMul()


def _torch_moe_reference(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    score: torch.Tensor,
    topk: int,
    act_fn,
) -> torch.Tensor:
    m = a.shape[0]
    weights, ids = torch.topk(torch.softmax(score, dim=-1, dtype=torch.float32), topk)
    x = a.view(m, 1, -1).repeat(1, topk, 1).reshape(m * topk, -1)
    flat_ids = ids.reshape(-1)
    out = torch.zeros(m * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    for expert in range(w1.shape[0]):
        mask = flat_ids == expert
        if mask.any():
            acc = act_fn(x[mask] @ w1[expert].transpose(0, 1)).to(a.dtype)
            out[mask] = acc @ w2[expert].transpose(0, 1)
    return (
        (out.view(m, topk, -1).float() * weights.view(m, topk, 1))
        .sum(dim=1)
        .to(a.dtype)
    )


def _quantize_nvfp4_linear(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weights_q = []
    scales = []
    global_scales = []
    for expert_weight in weight:
        global_scale = (
            FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / expert_weight.abs().max()
        ).to(torch.float32)
        weight_q, scale = ops.scaled_fp4_quant(
            expert_weight,
            global_scale,
            is_sf_swizzled_layout=False,
        )
        weights_q.append(weight_q)
        scales.append(scale)
        global_scales.append(global_scale)
    return torch.stack(weights_q), torch.stack(scales), torch.stack(global_scales)


def _dequantize_nvfp4_linear(
    tensor_fp4: torch.Tensor,
    tensor_sf: torch.Tensor,
    global_scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    assert tensor_fp4.dtype == torch.uint8
    m, packed_k = tensor_fp4.shape
    k = packed_k * 2
    tensor_f32 = break_fp4_bytes(tensor_fp4, torch.float32)
    tensor_f32 = tensor_f32.reshape(m, k // 16, 16)
    tensor_sf = tensor_sf.view(torch.float8_e4m3fn).to(torch.float32)
    tensor_sf = tensor_sf[:, : k // 16] / global_scale
    return (tensor_f32 * tensor_sf.unsqueeze(-1)).reshape(m, k).to(dtype)


@pytest.mark.parametrize("m,n,k,e,topk", [(16, 128, 512, 4, 2)])
@pytest.mark.parametrize("activation,alpha,beta,limit", _ACT_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.inference_mode()
def test_flashinfer_cutedsl_fp4_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    activation: MoEActivation,
    alpha: float | None,
    beta: float | None,
    limit: float | None,
    dtype: torch.dtype,
    workspace_init,
):
    set_random_seed(7)
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        hidden_states = torch.randn((m, k), device="cuda", dtype=dtype) / 10

        w1_rows = 2 * n if activation.is_gated else n
        w1 = torch.randn((e, w1_rows, k), device="cuda", dtype=dtype) / 15
        w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 15
        w1_q, w1_scale, w1_global_scale = _quantize_nvfp4_linear(w1)
        w2_q, w2_scale, w2_global_scale = _quantize_nvfp4_linear(w2)

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(
            hidden_states, score, topk, renormalize=False
        )

        fake_layer = SimpleNamespace(activation=activation)
        a1_scale = torch.ones(1, device="cuda", dtype=torch.float32)
        a2_scale = torch.ones(1, device="cuda", dtype=torch.float32)
        (
            w1_cutedsl,
            w1_scale_cutedsl,
            w1_alpha,
            a1_scale,
            w2_cutedsl,
            w2_scale_cutedsl,
            w2_alpha,
            a2_scale,
        ) = prepare_nvfp4_moe_layer_for_flashinfer_cutedsl(
            layer=fake_layer,
            w13=w1_q,
            w13_scale=w1_scale,
            w13_scale_2=(1.0 / w1_global_scale),
            a13_scale=a1_scale,
            w2=w2_q,
            w2_scale=w2_scale,
            w2_scale_2=(1.0 / w2_global_scale),
            a2_scale=a2_scale,
        )
        quant_config = nvfp4_moe_quant_config(
            g1_alphas=w1_alpha,
            g2_alphas=w2_alpha,
            a1_gscale=(1.0 / a1_scale),
            a2_gscale=(1.0 / a2_scale),
            w1_scale=w1_scale_cutedsl,
            w2_scale=w2_scale_cutedsl,
            is_scale_swizzled=False,
            # Unset params must be omitted rather than forwarded as None into
            # the kernel's float-typed SwiGLU arguments.
            gemm1_alpha=alpha,
            gemm1_beta=beta,
            gemm1_clamp_limit=limit,
        )
        moe_config = FusedMoEConfig(
            num_experts=e,
            experts_per_token=topk,
            hidden_dim=k,
            intermediate_size=n,
            num_local_experts=e,
            num_logical_experts=e,
            activation=activation,
            device="cuda",
            moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
            in_dtype=dtype,
            routing_method=RoutingMethodType.TopK,
            max_num_tokens=next_power_of_2(m),
        )

        cutedsl_experts = mk.FusedMoEKernel(
            maybe_make_prepare_finalize(
                moe=moe_config,
                quant_config=quant_config,
                allow_new_interface=True,
                use_monolithic=False,
            ),
            FlashInferCuteDSLExperts(
                moe_config=moe_config,
                quant_config=quant_config,
            ),
        )

        cutedsl_output = cutedsl_experts.apply(
            hidden_states=hidden_states,
            w1=w1_cutedsl,
            w2=w2_cutedsl,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=e,
            expert_map=None,
            apply_router_weight_on_input=False,
        )

        a_global_scale = torch.ones(1, device="cuda", dtype=torch.float32)
        a_q, a_scale = ops.scaled_fp4_quant(
            hidden_states,
            a_global_scale,
            is_sf_swizzled_layout=False,
        )
        a_in_dtype = _dequantize_nvfp4_linear(
            a_q,
            a_scale,
            a_global_scale,
            dtype=dtype,
        )

        w1_d = torch.empty((e, w1_rows, k), device="cuda", dtype=dtype)
        w2_d = torch.empty((e, k, n), device="cuda", dtype=dtype)
        for idx in range(e):
            w1_d[idx] = _dequantize_nvfp4_linear(
                w1_q[idx],
                w1_scale[idx],
                w1_global_scale[idx],
                dtype=dtype,
            )
            w2_d[idx] = _dequantize_nvfp4_linear(
                w2_q[idx],
                w2_scale[idx],
                w2_global_scale[idx],
                dtype=dtype,
            )

        torch_output = _torch_moe_reference(
            a_in_dtype,
            w1_d,
            w2_d,
            score,
            topk,
            _reference_activation(activation, alpha, beta, limit),
        )
        torch.testing.assert_close(
            torch_output,
            cutedsl_output,
            atol=3e-2,
            rtol=2e-1,
        )
        # Outputs here are O(1e-2) while NVFP4 noise is O(1e-3), so an absolute
        # tolerance loose enough for the quantization error also accepts a zero
        # tensor. Compare direction too, which dropped SwiGLU params or a wrong
        # w13 layout would break.
        cosine = torch.nn.functional.cosine_similarity(
            cutedsl_output.flatten().float(), torch_output.flatten().float(), dim=0
        )
        assert cosine > 0.99, f"cosine similarity {cosine:.4f} below 0.99"
