# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CuTe-DSL MXFP8 experts against a reference (the Triton experts are covered
by `test_moe.py`)."""

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(100),
    reason="The CuTe-DSL MXFP8 experts need SM10x",
)


def _quantize(value):
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )

    flat = value.reshape(-1, value.shape[-1]).contiguous()
    quant, scales = mxfp8_e4m3_quantize(flat, is_sf_swizzled_layout=False)
    return quant.reshape(value.shape), scales.reshape(*value.shape[:-1], -1)


def _make_experts(experts, topk, hidden, intermediate, clamp):
    from torch import nn

    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEParallelConfig,
        FusedMoEQuantConfig,
        RoutingMethodType,
    )
    from vllm.models.nemotron_shortcut.nvidia.moe.config import (
        NemotronShortcutMoEConfig,
    )
    from vllm.models.nemotron_shortcut.nvidia.moe.flashinfer import (
        NemotronShortcutCuteDslMxfp8Experts,
        has_cute_dsl_mxfp8,
    )

    if not has_cute_dsl_mxfp8():
        pytest.skip("FlashInfer without CuTe-DSL MXFP8 ClampedRelu2")
    device = torch.device("cuda")
    weights = nn.Module()
    for name, shape in (
        ("w13", (experts, intermediate, hidden)),
        ("w2", (experts, hidden, intermediate)),
    ):
        quant, scales = _quantize(
            torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.1
        )
        weights.register_buffer(f"{name}_weight", quant)
        weights.register_buffer(f"{name}_weight_scale", scales)
    config = NemotronShortcutMoEConfig(
        num_experts=experts,
        experts_per_token=topk,
        hidden_dim=hidden,
        intermediate_size=intermediate,
        num_local_experts=experts,
        num_logical_experts=experts,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.RELU2_NO_MUL,
        activation_clamp_value=clamp,
        in_dtype=torch.bfloat16,
        device=device,
        routing_method=RoutingMethodType.TopK,
        moe_backend="flashinfer_cutedsl",
    )
    quant_config = FusedMoEQuantConfig.make(
        "mxfp8",
        w1_scale=weights.w13_weight_scale,
        w2_scale=weights.w2_weight_scale,
        block_shape=[1, 32],
        is_scale_swizzled=False,
    )
    kernel = NemotronShortcutCuteDslMxfp8Experts(config, quant_config)
    kernel.process_weights_after_loading(weights)
    return weights, kernel


@torch.inference_mode()
def test_refit_updates_captured_mxfp8_scales():
    weights, kernel = _make_experts(8, 2, 128, 256, 0.25)
    up, down, alpha = kernel._w1_sf, kernel._w2_sf, kernel._alpha
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_up = up.clone()
        captured_down = down.clone()

    for a, b in ((123, 124), (125, 122)):
        weights.w13_weight_scale.fill_(a)
        weights.w2_weight_scale.fill_(b)
        kernel.process_weights_after_loading(weights)
        graph.replay()
        assert torch.all(captured_up == a)
        assert torch.all(captured_down == b)
        assert kernel._alpha is alpha


@pytest.mark.parametrize("tokens", [7, 32])
@pytest.mark.parametrize("hidden,intermediate", [(128, 256), (256, 128)])
@pytest.mark.parametrize("ordered", [False, True])
@torch.inference_mode()
def test_cute_dsl_mxfp8_experts_apply_the_clamp(
    tokens, hidden, intermediate, ordered, monkeypatch
):
    """Compare the clamped experts against a quantized reference. Ordered
    reduction must also produce bit-identical outputs across repeated calls."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        dequant_mxfp8_to_bf16,
    )

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1" if ordered else "0")
    torch.manual_seed(7)
    device = torch.device("cuda")
    experts, topk, clamp = 8, 8 if ordered else 2, 0.25
    weights, kernel = _make_experts(experts, topk, hidden, intermediate, clamp)
    x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    ids = torch.rand(tokens, experts, device=device).topk(topk, dim=-1).indices
    routing = torch.randn(tokens, topk, device=device).softmax(-1)

    def apply(x, ids, routing):
        output = torch.empty_like(x)
        kernel.apply(
            output,
            x,
            weights.w13_weight,
            weights.w2_weight,
            routing,
            ids,
            MoEActivation.RELU2_NO_MUL,
            experts,
            None,
            None,
            None,
            None,
            None,
            None,
            False,
        )
        return output

    output = apply(x, ids, routing)
    if ordered:
        for rows in (torch.arange(tokens, device=device).flip(0), [0]):
            actual = apply(x[rows], ids[rows], routing[rows])
            torch.testing.assert_close(actual, output[rows], atol=0, rtol=0)
        torch.testing.assert_close(apply(x, ids, routing), output, atol=0, rtol=0)
    xref = dequant_mxfp8_to_bf16(*_quantize(x)).float()
    w1 = dequant_mxfp8_to_bf16(weights.w13_weight, weights.w13_weight_scale).float()
    w2 = dequant_mxfp8_to_bf16(weights.w2_weight, weights.w2_weight_scale).float()
    expected = torch.zeros_like(output, dtype=torch.float32)
    for expert in range(experts):
        rows, slots = torch.where(ids == expert)
        if rows.numel() == 0:
            continue
        up = F.linear(xref[rows], w1[expert])
        activated = (clamp * torch.tanh(up / clamp)).clamp_min(0).square()
        activated = dequant_mxfp8_to_bf16(*_quantize(activated.bfloat16())).float()
        down = F.linear(activated, w2[expert])
        expected.index_add_(0, rows, down * routing[rows, slots, None])
    assert torch.isfinite(output).all()
    error = (output.float() - expected).norm() / expected.norm()
    assert error < 0.05, error.item()
