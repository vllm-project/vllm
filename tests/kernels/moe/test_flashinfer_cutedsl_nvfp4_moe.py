# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_test_quant_config
from tests.kernels.quantization.nvfp4_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    dequantize_nvfp4_to_dtype,
)
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
)
from vllm.model_executor.layers.fused_moe.experts import (
    flashinfer_cutedsl_moe,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    convert_to_nvfp4_moe_kernel_format,
    make_nvfp4_moe_quant_config,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_cutedsl_moe_nvfp4_activation_type
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import set_random_seed

if pytest and (
    not has_flashinfer_cutedsl_moe_nvfp4_activation_type()
    or not current_platform.is_device_capability_family(100)
):
    pytest.skip(
        "Requires FlashInfer CuteDSL NvFP4 MoE activation_type support on SM100",
        allow_module_level=True,
    )


_SWIGLU_ALPHA = 1.702
_SWIGLU_BETA = 1.0
_SWIGLU_LIMIT = 7.0
_SWIGLU_KWARGS = ("swiglu_alpha", "swiglu_beta", "swiglu_limit")


def _oai_swiglu_uninterleave(x: torch.Tensor) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    gate = gate.clamp(max=_SWIGLU_LIMIT)
    up = up.clamp(min=-_SWIGLU_LIMIT, max=_SWIGLU_LIMIT)
    return gate * torch.sigmoid(_SWIGLU_ALPHA * gate) * (up + _SWIGLU_BETA)


def _apply_activation(x: torch.Tensor, activation: MoEActivation) -> torch.Tensor:
    if activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE:
        return _oai_swiglu_uninterleave(x)
    if activation == MoEActivation.RELU2_NO_MUL:
        return torch.square(torch.relu(x))
    raise AssertionError(f"Unexpected activation: {activation}")


def _torch_moe_reference(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    score: torch.Tensor,
    topk: int,
    activation: MoEActivation,
) -> torch.Tensor:
    m = a.shape[0]
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(score, topk)

    expanded = a.view(m, 1, -1).repeat(1, topk, 1).reshape(m * topk, -1)
    expert_ids = topk_ids.reshape(-1)
    out = torch.empty(m * topk, w2.shape[1], dtype=a.dtype, device=a.device)

    for expert_id in range(w1.shape[0]):
        mask = expert_ids == expert_id
        if not mask.any():
            continue
        tmp = expanded[mask] @ w1[expert_id].transpose(0, 1)
        tmp = _apply_activation(tmp, activation).to(a.dtype)
        out[mask] = tmp @ w2[expert_id].transpose(0, 1)

    return (
        (
            out.view(m, topk, w2.shape[1]).to(torch.float32)
            * topk_weights.view(m, topk, 1)
        )
        .sum(dim=1)
        .to(a.dtype)
    )


def _dequantize_nvfp4_inputs_and_weights(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    quant_config,
    activation: MoEActivation,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    block_size = 16
    a_global_scale = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / a.abs().max()).to(
        torch.float32
    )
    a_fp4, a_scale_interleaved = ops.scaled_fp4_quant(a, a_global_scale)
    a_d = dequantize_nvfp4_to_dtype(
        a_fp4,
        a_scale_interleaved,
        a_global_scale,
        dtype=a.dtype,
        device=a.device,
        block_size=block_size,
    )

    e = w1_q.shape[0]
    intermediate_size = w2_q.shape[-1] * 2
    hidden_dim = w2_q.shape[1]
    w1_rows = (2 if activation.is_gated else 1) * intermediate_size
    w1_d = torch.empty((e, w1_rows, hidden_dim), device="cuda", dtype=a.dtype)
    w2_d = torch.empty((e, hidden_dim, intermediate_size), device="cuda", dtype=a.dtype)

    assert quant_config.g1_alphas is not None
    assert quant_config.g2_alphas is not None
    assert quant_config.w1_scale is not None
    assert quant_config.w2_scale is not None
    for idx in range(e):
        w1_d[idx] = dequantize_nvfp4_to_dtype(
            w1_q[idx],
            quant_config.w1_scale[idx],
            (1 / quant_config.g1_alphas[idx]),
            dtype=a.dtype,
            device=w1_q.device,
            block_size=block_size,
        )
        w2_d[idx] = dequantize_nvfp4_to_dtype(
            w2_q[idx],
            quant_config.w2_scale[idx],
            (1 / quant_config.g2_alphas[idx]),
            dtype=a.dtype,
            device=w2_q.device,
            block_size=block_size,
        )

    return a_d, w1_d, w2_d


@pytest.mark.parametrize(
    ("gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit", "expected_kwargs"),
    [
        (None, None, None, {}),
        (1.5, None, 0.0, {"swiglu_alpha": 1.5, "swiglu_limit": 0.0}),
    ],
)
def test_flashinfer_cutedsl_nvfp4_moe_swiglu_kwargs(
    monkeypatch,
    gemm1_alpha: float | None,
    gemm1_beta: float | None,
    gemm1_clamp_limit: float | None,
    expected_kwargs: dict[str, float],
):
    scale = torch.ones(1)
    quant_config = SimpleNamespace(
        quant_dtype="nvfp4",
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
        w1_scale=scale,
        w2_scale=scale,
        g1_alphas=scale,
        g2_alphas=scale,
        a2_gscale=scale,
    )
    moe_config = SimpleNamespace(
        in_dtype=torch.bfloat16,
        hidden_dim=1,
        intermediate_size_per_partition=1,
        experts_per_token=1,
        num_local_experts=1,
        num_experts=1,
        moe_parallel_config=SimpleNamespace(ep_rank=0),
    )
    experts = flashinfer_cutedsl_moe.FlashInferCuteDSLExperts(
        moe_config=moe_config,
        quant_config=quant_config,
    )

    kernel_kwargs = {}

    def fake_flashinfer_kernel(**kwargs):
        kernel_kwargs.update(kwargs)

    monkeypatch.setattr(
        flashinfer_cutedsl_moe,
        "has_flashinfer_cutedsl_moe_nvfp4_activation_type",
        lambda: True,
    )
    monkeypatch.setattr(
        flashinfer_cutedsl_moe,
        "flashinfer_cute_dsl_fused_moe_nvfp4",
        fake_flashinfer_kernel,
    )

    experts.apply(
        output=torch.empty(1, 1),
        hidden_states=torch.empty(1, 1),
        w1=torch.empty(1),
        w2=torch.empty(1),
        topk_weights=torch.ones(1, 1),
        topk_ids=torch.zeros(1, 1, dtype=torch.int64),
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        global_num_experts=1,
        expert_map=None,
        a1q_scale=torch.ones(1, 1),
        a2_scale=None,
        workspace13=None,
        workspace2=None,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    actual_kwargs = {
        name: kernel_kwargs[name] for name in _SWIGLU_KWARGS if name in kernel_kwargs
    }
    assert actual_kwargs == expected_kwargs


@pytest.mark.parametrize(
    "activation",
    [MoEActivation.SWIGLUOAI_UNINTERLEAVE, MoEActivation.RELU2_NO_MUL],
)
@torch.inference_mode()
def test_flashinfer_cutedsl_nvfp4_moe_oai_and_relu2(
    activation: MoEActivation,
    workspace_init,
):
    set_random_seed(7)
    m, n, k, e, topk = 16, 128, 256, 8, 2
    dtype = torch.bfloat16

    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
        w1_q, w2_q, base_quant_config = make_test_quant_config(
            e,
            n,
            k,
            in_dtype=dtype,
            quant_dtype="nvfp4",
            block_shape=None,
            per_act_token_quant=False,
            make_gate=activation.is_gated,
        )

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(a, score, topk, renormalize=False)

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

        assert base_quant_config.w1_scale is not None
        assert base_quant_config.w2_scale is not None
        assert base_quant_config.g1_alphas is not None
        assert base_quant_config.g2_alphas is not None
        assert base_quant_config.a1_gscale is not None
        assert base_quant_config.a2_gscale is not None
        (
            w1_kernel,
            w1_scale,
            w1_alpha,
            a1_scale,
            w2_kernel,
            w2_scale,
            w2_alpha,
            a2_scale,
        ) = convert_to_nvfp4_moe_kernel_format(
            nvfp4_backend=NvFp4MoeBackend.FLASHINFER_CUTEDSL,
            layer=SimpleNamespace(activation=activation),
            w13=w1_q,
            w13_scale=base_quant_config.w1_scale,
            w13_scale_2=base_quant_config.g1_alphas,
            a13_scale=base_quant_config.a1_gscale,
            w2=w2_q,
            w2_scale=base_quant_config.w2_scale,
            w2_scale_2=base_quant_config.g2_alphas,
            a2_scale=base_quant_config.a2_gscale,
            is_act_and_mul=activation.is_gated,
        )

        quant_config = make_nvfp4_moe_quant_config(
            backend=NvFp4MoeBackend.FLASHINFER_CUTEDSL,
            w13_scale=w1_scale,
            w2_scale=w2_scale,
            w13_scale_2=w1_alpha,
            w2_scale_2=w2_alpha,
            a13_scale=a1_scale,
            a2_scale=a2_scale,
            swiglu_alpha=_SWIGLU_ALPHA if activation.is_gated else None,
            swiglu_beta=_SWIGLU_BETA if activation.is_gated else None,
            swiglu_limit=_SWIGLU_LIMIT if activation.is_gated else None,
        )

        experts = flashinfer_cutedsl_moe.FlashInferCuteDSLExperts(
            moe_config=moe_config,
            quant_config=quant_config,
        )
        fake_layer = torch.nn.Module()
        fake_layer.w13_weight_scale_2 = torch.nn.Parameter(
            quant_config.g1_alphas, requires_grad=False
        )
        fake_layer.w2_weight_scale_2 = torch.nn.Parameter(
            quant_config.g2_alphas, requires_grad=False
        )
        fake_layer.w13_input_scale = a1_scale
        fake_layer.w2_input_scale = a2_scale
        experts.process_weights_after_loading(fake_layer)

        kernel = mk.FusedMoEKernel(
            maybe_make_prepare_finalize(
                moe=moe_config,
                quant_config=quant_config,
                allow_new_interface=True,
                use_monolithic=False,
            ),
            experts,
        )

        cutedsl_output = kernel.apply(
            hidden_states=a,
            w1=w1_kernel,
            w2=w2_kernel,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=e,
            expert_map=None,
            apply_router_weight_on_input=False,
        )

        a_d, w1_d, w2_d = _dequantize_nvfp4_inputs_and_weights(
            a,
            w1_q,
            w2_q,
            base_quant_config,
            activation,
        )
        torch_output = _torch_moe_reference(a_d, w1_d, w2_d, score, topk, activation)

        torch.testing.assert_close(
            torch_output,
            cutedsl_output,
            atol=2e-1,
            rtol=2e-1,
        )
