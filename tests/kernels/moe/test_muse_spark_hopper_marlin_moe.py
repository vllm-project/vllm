# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hopper coverage for MuseSpark's NVFP4 Marlin and BF16 Triton experts."""

import pytest
import torch

from tests.kernels.moe.muse_spark_utils import (
    assert_cudagraph_replays,
    make_runner,
    nvfp4_quant_config,
    run_moe,
    swiglu_moe_ref,
)
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExperts
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    is_fp4_marlin_supported,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
)
from vllm.models.muse_spark.common.layers import MuseSpark1xRMSNormWithInputScale
from vllm.platforms import current_platform

if not (current_platform.is_device_capability_family(90) and is_fp4_marlin_supported()):
    pytest.skip("Requires an SM90 GPU with Marlin", allow_module_level=True)

E, N, K, TOPK, M = 8, 256, 512, 2, 8


def _build(checkpoint: dict[str, torch.Tensor], quant_config=None):
    norm = MuseSpark1xRMSNormWithInputScale(
        K,
        eps=1e-8,
        has_weight=True,
        is_post_norm=True,
        zero_centered_gamma=True,
    ).cuda()
    norm.weight.copy_(torch.linspace(-0.2, 0.2, K, device="cuda"))
    expert_bias = torch.linspace(-0.5, 0.5, E, dtype=torch.float32, device="cuda")
    router, runner, vllm_config = make_runner(
        norm,
        expert_bias,
        checkpoint,
        quant_config,
        hidden_size=K,
        intermediate_size=N,
        top_k=TOPK,
    )
    experts = runner.routed_experts.quant_method.moe_kernel.fused_experts
    return router, norm, runner, vllm_config, experts


def _nvfp4_checkpoint() -> dict[str, torch.Tensor]:
    g = torch.Generator(device="cuda").manual_seed(0)

    def packed(*shape):
        return torch.randint(
            0, 256, shape, dtype=torch.uint8, device="cuda", generator=g
        )

    def rand(*shape):
        return torch.rand(*shape, dtype=torch.float32, device="cuda", generator=g)

    return {
        "w13_weight": packed(E, 2 * N, K // 2),
        "w13_weight_scale": torch.ones(
            E, 2 * N, K // 16, dtype=torch.float8_e4m3fn, device="cuda"
        ),
        "w13_weight_scale_2": rand(E, 2) * 0.01 + 0.01,
        "w13_input_scale": rand(E, 2) + 0.5,
        "w2_weight": packed(E, K, N // 2),
        "w2_weight_scale": torch.ones(
            E, K, N // 16, dtype=torch.float8_e4m3fn, device="cuda"
        ),
        "w2_weight_scale_2": rand(E) * 0.01 + 0.01,
        "w2_input_scale": rand(E) + 0.5,
    }


def _dequant(q, scale, global_scale):
    return dequantize_to_dtype(
        q, scale, global_scale, torch.bfloat16, block_size=16, swizzle=False
    )


def _expected(router, norm, hidden_states, router_logits, w13, w2):
    topk_weights, topk_ids = router._compute_routing(hidden_states, router_logits, None)
    return swiglu_moe_ref(
        hidden_states,
        w13,
        w2,
        topk_weights,
        topk_ids,
        lambda y: norm.forward_native(y, out_dtype=torch.float32),
    )


@pytest.mark.parametrize("down_scale", [1.0, 1e-4])
@torch.inference_mode()
def test_hopper_marlin_moe_runs_with_triton_finalize(
    dist_init, workspace_init, down_scale
) -> None:
    checkpoint = _nvfp4_checkpoint()
    checkpoint["w2_weight_scale_2"].mul_(down_scale)
    router, norm, runner, vllm_config, experts = _build(
        checkpoint, nvfp4_quant_config()
    )
    assert type(experts) is MarlinExperts
    assert experts.g1_alphas is not None
    assert experts.g1_alphas.shape == (E,)

    hidden_states = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / 10
    router_logits = torch.randn(M, E, dtype=torch.float32, device="cuda")
    output = run_moe(runner, vllm_config, hidden_states, router_logits)
    assert output.shape == hidden_states.shape
    assert output.dtype == hidden_states.dtype
    assert torch.isfinite(output).all()

    w13 = torch.empty(E, 2 * N, K, dtype=torch.bfloat16, device="cuda")
    for expert in range(E):
        for shard, rows in ((0, slice(0, N)), (1, slice(N, 2 * N))):
            w13[expert, rows] = _dequant(
                checkpoint["w13_weight"][expert, rows],
                checkpoint["w13_weight_scale"][expert, rows],
                checkpoint["w13_weight_scale_2"][expert, shard],
            )
    w2 = _dequant(
        checkpoint["w2_weight"],
        checkpoint["w2_weight_scale"],
        checkpoint["w2_weight_scale_2"],
    )
    expected = _expected(router, norm, hidden_states, router_logits, w13, w2)
    assert (output.float() - expected).norm() / expected.norm() < 5e-2

    assert_cudagraph_replays(
        runner,
        vllm_config,
        hidden_states,
        router_logits,
        torch.randn_like(hidden_states),
        torch.randn_like(router_logits),
    )


@torch.inference_mode()
def test_hopper_bf16_boundary_layer_runs_with_triton_finalize(
    dist_init, workspace_init
) -> None:
    g = torch.Generator(device="cuda").manual_seed(1)
    weights = {
        "w13_weight": torch.randn(
            E, 2 * N, K, dtype=torch.bfloat16, device="cuda", generator=g
        )
        / 10,
        "w2_weight": torch.randn(
            E, K, N, dtype=torch.bfloat16, device="cuda", generator=g
        )
        / 10,
    }
    router, norm, runner, vllm_config, experts = _build(weights)
    assert type(experts) is TritonExperts

    hidden_states = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / 10
    router_logits = torch.randn(M, E, dtype=torch.float32, device="cuda")
    output = run_moe(runner, vllm_config, hidden_states, router_logits)
    assert output.shape == hidden_states.shape
    assert output.dtype == hidden_states.dtype
    assert torch.isfinite(output).all()

    expected = _expected(
        router,
        norm,
        hidden_states,
        router_logits,
        weights["w13_weight"],
        weights["w2_weight"],
    )
    assert (output.float() - expected).norm() / expected.norm() < 0.02
