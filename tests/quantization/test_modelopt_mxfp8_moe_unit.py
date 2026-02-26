# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ModelOpt MXFP8 MoE method wiring.

These tests avoid depending on FlashInfer kernels by monkeypatching the
quantization op and kernel entrypoint, while still exercising the vLLM flow.

pytest --tb short tests/quantization/test_modelopt_mxfp8_moe_unit.py
"""

import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.quantization.modelopt import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE,
    MXFP8_VALUE_DTYPE,
    ModelOptMxFp8Config,
    ModelOptMxFp8FusedMoE,
    mxfp8_e4m3_quantize,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer


def _make_method(is_act_and_mul: bool = True) -> ModelOptMxFp8FusedMoE:
    quant_config = ModelOptMxFp8Config(
        is_checkpoint_mxfp8_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
    )
    moe_config = FusedMoEConfig(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=64,
        intermediate_size_per_partition=128,
        num_local_experts=8,
        num_logical_experts=8,
        activation=MoEActivation.SILU if is_act_and_mul else MoEActivation.RELU2_NO_MUL,
        device="cpu",
        routing_method=RoutingMethodType.Renormalize,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16,
        is_act_and_mul=is_act_and_mul,
    )
    return ModelOptMxFp8FusedMoE(quant_config=quant_config, moe_config=moe_config)


def test_modelopt_mxfp8_moe_process_weights_smoke(monkeypatch: pytest.MonkeyPatch):
    """Simulate create->process flow and ensure weight transforms are applied once."""
    method = _make_method(is_act_and_mul=True)
    layer = torch.nn.Module()
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    method.create_weights(
        layer=layer,
        num_experts=2,
        hidden_size=64,
        intermediate_size_per_partition=64,
        params_dtype=torch.bfloat16,
    )

    # Materialize deterministic tensors for processing.
    layer.w13_weight.data.zero_()
    layer.w2_weight.data.zero_()
    layer.w13_weight_scale.data.fill_(127)
    layer.w2_weight_scale.data.fill_(127)

    # Fake flashinfer shufflers so process_weights can run on CPU.
    fake_flashinfer: Any = types.ModuleType("flashinfer")
    fake_flashinfer.reorder_rows_for_gated_act_gemm = lambda t: t
    fake_flashinfer.shuffle_matrix_a = lambda t, _: t
    fake_flashinfer.shuffle_matrix_sf_a = lambda t, _: t
    fake_fp4_quantization: Any = types.ModuleType("flashinfer.fp4_quantization")
    fake_fp4_quantization.shuffle_matrix_sf_a = lambda t, _: t
    monkeypatch.setitem(sys.modules, "flashinfer", fake_flashinfer)
    monkeypatch.setitem(
        sys.modules, "flashinfer.fp4_quantization", fake_fp4_quantization
    )

    method.process_weights_after_loading(layer)
    assert layer._already_called_process_weights_after_loading is True
    assert layer.w13_weight.dtype == MXFP8_VALUE_DTYPE
    assert layer.w2_weight.dtype == MXFP8_VALUE_DTYPE
    assert layer.w13_weight_scale.dtype == MXFP8_SCALE_DTYPE
    assert layer.w2_weight_scale.dtype == MXFP8_SCALE_DTYPE


def test_modelopt_mxfp8_moe_apply_monolithic_normalizes_ungrouped_routing(
    monkeypatch: pytest.MonkeyPatch,
):
    """Qwen-style ungrouped routing should pass None, not 0, to TRTLLM kernel."""
    method = _make_method(is_act_and_mul=True)
    hidden_size = 64
    intermediate = 128
    x = torch.randn(4, hidden_size, dtype=torch.bfloat16)
    router_logits = torch.randn(4, 8, dtype=torch.float32)

    layer = SimpleNamespace(
        enable_eplb=False,
        activation=MoEActivation.SILU,
        e_score_correction_bias=None,
        w13_weight=torch.zeros(
            8, intermediate * 2, hidden_size, dtype=MXFP8_VALUE_DTYPE
        ),
        w13_weight_scale=torch.full(
            (8, intermediate * 2, hidden_size // MXFP8_BLOCK_SIZE),
            127,
            dtype=MXFP8_SCALE_DTYPE,
        ),
        w2_weight=torch.zeros(8, hidden_size, intermediate, dtype=MXFP8_VALUE_DTYPE),
        w2_weight_scale=torch.full(
            (8, hidden_size, intermediate // MXFP8_BLOCK_SIZE),
            127,
            dtype=MXFP8_SCALE_DTYPE,
        ),
        global_num_experts=8,
        top_k=2,
        # Simulate incoming "ungrouped" values from model config.
        num_expert_group=0,
        topk_group=0,
        intermediate_size_per_partition=intermediate,
        ep_rank=0,
        local_num_experts=8,
        routed_scaling_factor=1.0,
        routing_method_type=RoutingMethodType.Renormalize,
    )

    captured: dict[str, object] = {}

    def _fake_quantize(inp: torch.Tensor, is_sf_swizzled_layout: bool = False):
        assert is_sf_swizzled_layout is False
        scales = torch.full(
            (inp.shape[0], inp.shape[1] // MXFP8_BLOCK_SIZE),
            127,
            dtype=MXFP8_SCALE_DTYPE,
        )
        return inp.to(MXFP8_VALUE_DTYPE), scales

    def _fake_kernel(**kwargs):
        captured.update(kwargs)
        return torch.full((x.shape[0], x.shape[1]), 3.0, dtype=torch.bfloat16)

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.modelopt.mxfp8_e4m3_quantize",
        _fake_quantize,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.modelopt.flashinfer_trtllm_fp8_block_scale_moe",
        _fake_kernel,
    )
    monkeypatch.setattr(
        ModelOptMxFp8FusedMoE,
        "_get_flashinfer_mxfp8_quant_type",
        staticmethod(lambda: "MxFp8"),
    )

    out = method.apply_monolithic(layer=layer, x=x, router_logits=router_logits)
    assert out.shape == x.shape
    # Kernel expects BF16 routing logits in this path.
    assert isinstance(captured["routing_logits"], torch.Tensor)
    assert captured["routing_logits"].dtype == torch.bfloat16
    # Critical invariant: ungrouped routing should use None instead of 0.
    assert captured["n_group"] is None
    assert captured["topk_group"] is None


def _mxfp8_dequantize(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    assert x.dtype == MXFP8_VALUE_DTYPE
    x_float = x.to(torch.float32)
    scale_i32 = scale.view(torch.uint8).to(torch.int32)
    scale_fp = (scale_i32 << 23).view(torch.float32)
    scale_fp = scale_fp.reshape(*x.shape[:-1], -1)
    scale_fp = torch.stack([scale_fp] * MXFP8_BLOCK_SIZE, dim=-1).reshape(
        *x_float.shape
    )
    return x_float * scale_fp


def _check_accuracy(
    ref: torch.Tensor,
    actual: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    percent: float,
) -> None:
    assert ref.shape == actual.shape
    left = torch.abs(ref - actual)
    right = atol + rtol * torch.abs(actual)
    mismatch_percent = torch.sum(left > right) / ref.numel()
    assert mismatch_percent <= (1 - percent), (
        f"Mismatch percentage is {float(mismatch_percent):.4f} for rtol={rtol}"
    )


@pytest.mark.skipif(
    not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ),
    reason="Requires NVIDIA Blackwell (SM100+) for TRTLLM MXFP8 MoE kernel.",
)
@pytest.mark.skipif(not has_flashinfer(), reason="FlashInfer is required.")
@pytest.mark.parametrize("routed_scaling_factor", [1.0, None])
def test_modelopt_mxfp8_moe_apply_monolithic_matches_reference_path(
    monkeypatch: pytest.MonkeyPatch,
    routed_scaling_factor: float | None,
):
    """Exercise create->process->apply on real kernel and compare with reference."""
    torch.manual_seed(7)
    method = _make_method(is_act_and_mul=True)
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        lambda: 1,
    )

    # Mirror Qwen3-MoE style dimensions/routing from user repro.
    num_experts = 128
    top_k = 8
    num_tokens = 128
    hidden_size = 2048
    intermediate = 768

    layer = torch.nn.Module()
    method.create_weights(
        layer=layer,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size_per_partition=intermediate,
        params_dtype=torch.bfloat16,
    )
    layer.to("cuda")

    w13_bf16 = (
        torch.randn(
            num_experts,
            2 * intermediate,
            hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        / 20
    )
    w2_bf16 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate,
            device="cuda",
            dtype=torch.bfloat16,
        )
        / 20
    )
    w13_q, w13_s = mxfp8_e4m3_quantize(w13_bf16, is_sf_swizzled_layout=False)
    w2_q, w2_s = mxfp8_e4m3_quantize(w2_bf16, is_sf_swizzled_layout=False)
    if w13_s.ndim == 1:
        w13_s = w13_s.view(
            num_experts, 2 * intermediate, hidden_size // MXFP8_BLOCK_SIZE
        )
    if w2_s.ndim == 1:
        w2_s = w2_s.view(num_experts, hidden_size, intermediate // MXFP8_BLOCK_SIZE)

    layer.w13_weight.data.copy_(w13_q)
    layer.w2_weight.data.copy_(w2_q)
    layer.w13_weight_scale.data.copy_(w13_s)
    layer.w2_weight_scale.data.copy_(w2_s)
    method.process_weights_after_loading(layer)

    x = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    router_logits = torch.randn(
        num_tokens, num_experts, device="cuda", dtype=torch.float32
    )

    fused_layer = SimpleNamespace(
        enable_eplb=False,
        activation=MoEActivation.SILU,
        e_score_correction_bias=None,
        w13_weight=layer.w13_weight,
        w13_weight_scale=layer.w13_weight_scale,
        w2_weight=layer.w2_weight,
        w2_weight_scale=layer.w2_weight_scale,
        global_num_experts=num_experts,
        top_k=top_k,
        num_expert_group=None,
        topk_group=None,
        intermediate_size_per_partition=intermediate,
        ep_rank=0,
        local_num_experts=num_experts,
        routed_scaling_factor=routed_scaling_factor,
        routing_method_type=RoutingMethodType.Renormalize,
    )
    try:
        out = method.apply_monolithic(
            layer=fused_layer, x=x, router_logits=router_logits
        )
    except RuntimeError as e:
        if "No valid config found for the given problem shape" in str(e):
            pytest.skip(f"FlashInfer TRTLLM has no kernel config for this shape: {e}")
        raise

    # Reference path mirrors tests/kernels/moe/test_ocp_mx_moe.py.
    w13_ref = _mxfp8_dequantize(w13_q, w13_s)
    w2_ref = _mxfp8_dequantize(w2_q, w2_s)
    x_q, x_s = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=False)
    x_ref = _mxfp8_dequantize(x_q, x_s)
    # Kernel path receives BF16 logits first, then applies routing.
    router_logits_kernel = router_logits.to(torch.bfloat16).to(torch.float32)
    experts = torch.topk(router_logits_kernel, k=top_k, dim=-1, sorted=True)
    expert_weights = torch.softmax(experts.values, dim=-1)
    if routed_scaling_factor is not None:
        expert_weights = expert_weights * routed_scaling_factor
    expert_indices = experts.indices
    t = torch.einsum("beck,bk->bec", w13_ref[expert_indices], x_ref)
    x_glu, x_linear = torch.chunk(t, 2, dim=-1)
    t = (x_glu * torch.sigmoid(x_glu)) * x_linear
    t_q, t_s = mxfp8_e4m3_quantize(t.to(torch.bfloat16), is_sf_swizzled_layout=False)
    t = _mxfp8_dequantize(t_q, t_s)
    t = torch.einsum("beck,bek->bec", w2_ref[expert_indices], t)
    ref = torch.einsum("bec,be->bc", t, expert_weights).to(torch.bfloat16)

    _check_accuracy(ref, out, atol=0.1, rtol=0.85, percent=0.8)
