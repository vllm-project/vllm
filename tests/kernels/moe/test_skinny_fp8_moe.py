# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TritonExperts on the grouped W8A16 skinny path for block-FP8 experts."""

import pytest
import torch

from tests.kernels.moe.utils import (
    make_dummy_moe_config,
    make_test_quant_config,
    modular_triton_fused_moe,
)
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.kernels.linear.cute_dsl import skinny_fp8_moe_gemm
from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm_plans import (
    SKINNY_FP8_MOE_GEMM_PLANS_BY_CAPABILITY,
)
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("CUDA only", allow_module_level=True)
if not skinny_fp8_moe_gemm.shape_dynamic_skinny_fp8_moe_gemm.is_available():
    pytest.skip("cuteDSL is not available", allow_module_level=True)
_CAPABILITY = current_platform.get_device_capability()
if _CAPABILITY not in SKINNY_FP8_MOE_GEMM_PLANS_BY_CAPABILITY:
    pytest.skip(
        f"no skinny FP8 MoE GEMM plans for {_CAPABILITY}", allow_module_level=True
    )

# One expert shape with a plan: Qwen3.5/3.6-35B-A3B (gate_up (1024, 2048),
# down (2048, 512)). Only the per-expert (N, K) matters for the plan.
E, N, K, TOPK = 16, 512, 2048, 4
BLOCK = [128, 128]


@pytest.fixture(autouse=True)
def setup_cuda():
    torch.set_default_device("cuda")


def _dequant(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return w.float() * s.repeat_interleave(BLOCK[0], 1).repeat_interleave(BLOCK[1], 2)


def _reference(a, w1, w2, w1_s, w2_s, topk_weights, topk_ids):
    """Dequantized experts in fp32 (the W8A16 path quantizes no activation)."""
    d1, d2 = _dequant(w1, w1_s), _dequant(w2, w2_s)
    out = torch.zeros(a.shape[0], K, dtype=torch.float32)
    for m in range(a.shape[0]):
        for j in range(TOPK):
            e = int(topk_ids[m, j])
            h = a[m].float() @ d1[e].t()
            h = torch.nn.functional.silu(h[:N]) * h[N:]
            out[m] += topk_weights[m, j].float() * (h @ d2[e].t())
    return out


@pytest.mark.parametrize("M", [1, 2, 4])
@torch.inference_mode()
def test_triton_experts_take_skinny_fp8_moe_path(M, monkeypatch, workspace_init):
    torch.manual_seed(0)
    a = torch.randn((M, K), dtype=torch.bfloat16) / 10
    score = torch.randn((M, E), dtype=torch.bfloat16)
    w1, w2, quant_config = make_test_quant_config(
        E, N, K, torch.bfloat16, quant_dtype=torch.float8_e4m3fn, block_shape=BLOCK
    )
    moe = modular_triton_fused_moe(
        make_dummy_moe_config(
            num_experts=E, experts_per_token=TOPK, hidden_dim=K, intermediate_size=N
        ),
        quant_config,
    )
    experts = moe.fused_experts
    plans = experts._skinny_fp8_moe_plans
    assert plans is not None and M in plans
    assert experts.expects_unquantized_inputs

    calls = []
    kernel = skinny_fp8_moe_gemm.shape_dynamic_skinny_fp8_moe_gemm
    orig = type(kernel).__call__
    monkeypatch.setattr(
        type(kernel),
        "__call__",
        lambda self, *args, **kw: (
            calls.append(kw.get("a_per_token")),
            orig(self, *args, **kw),
        )[1],
    )

    topk_weights, topk_ids, _ = fused_topk(a, score.float(), TOPK, False)
    kwargs = dict(
        activation=MoEActivation.SILU,
        apply_router_weight_on_input=False,
        expert_map=None,
        global_num_experts=E,
    )
    with set_current_vllm_config(VllmConfig()):
        out = moe.apply(a, w1, w2, topk_weights, topk_ids, **kwargs)
        # Same kernel object, plans removed: the usual Triton W8A8 path.
        experts._skinny_fp8_moe_plans = None
        assert not experts.expects_unquantized_inputs
        triton_out = moe.apply(a, w1, w2, topk_weights, topk_ids, **kwargs)

    assert calls == [True, False]  # gate_up per token, down per pair
    ref = _reference(
        a, w1, w2, quant_config.w1_scale, quant_config.w2_scale, topk_weights, topk_ids
    )
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
    # The Triton path quantizes the activations to FP8 and lands further from
    # the dequantized reference than the W8A16 path does.
    torch.testing.assert_close(triton_out.float(), ref, atol=1e-1, rtol=1e-1)


@torch.inference_mode()
def test_triton_experts_keep_triton_path_without_plan(workspace_init):
    """An expert shape without a plan never deferred the input quantization."""
    w1, w2, quant_config = make_test_quant_config(
        E, 384, 1536, torch.bfloat16, quant_dtype=torch.float8_e4m3fn, block_shape=BLOCK
    )
    moe = modular_triton_fused_moe(
        make_dummy_moe_config(
            num_experts=E,
            experts_per_token=TOPK,
            hidden_dim=1536,
            intermediate_size=384,
        ),
        quant_config,
    )
    assert moe.fused_experts._skinny_fp8_moe_plans is None
    assert not moe.fused_experts.expects_unquantized_inputs
