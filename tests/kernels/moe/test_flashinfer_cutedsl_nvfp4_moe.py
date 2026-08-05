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
from tests.kernels.utils import torch_moe
from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
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
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.inference_mode()
def test_flashinfer_cutedsl_fp4_moe_relu2_no_mul(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    workspace_init,
):
    set_random_seed(7)
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        hidden_states = torch.randn((m, k), device="cuda", dtype=dtype) / 10

        w1 = torch.randn((e, n, k), device="cuda", dtype=dtype) / 15
        w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 15
        w1_q, w1_scale, w1_global_scale = _quantize_nvfp4_linear(w1)
        w2_q, w2_scale, w2_global_scale = _quantize_nvfp4_linear(w2)

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(
            hidden_states, score, topk, renormalize=False
        )

        activation = MoEActivation.RELU2_NO_MUL
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

        w1_d = torch.empty((e, n, k), device="cuda", dtype=dtype)
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

        torch_output = torch_moe(
            a_in_dtype,
            w1_d,
            w2_d,
            score,
            topk,
            activation=activation,
        )
        torch.testing.assert_close(
            torch_output,
            cutedsl_output,
            atol=2e-1,
            rtol=2e-1,
        )
