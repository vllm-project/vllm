# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
from vllm.model_executor.layers.fused_moe.runner import moe_runner  # noqa: F401
from vllm.model_executor.layers.fused_moe.runner.moe_runner_interface import (
    MoERunnerInterface,
)
from vllm.model_executor.layers.quantization.utils import flashinfer_fp4_moe
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
    prepare_nvfp4_moe_layer_for_fi_or_cutlass,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    align_fp4_moe_hidden_dim_for_fi,
    align_fp4_moe_weights_for_fi,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import _USE_LAYERNAME, LayerName


def test_shared_nvfp4_input_scales_have_writable_storage(monkeypatch):
    monkeypatch.setattr(flashinfer_fp4_moe, "swizzle_blockscale", lambda x: x)

    num_experts = 3
    layer = SimpleNamespace(
        activation=SimpleNamespace(is_gated=False),
        moe_config=SimpleNamespace(
            moe_parallel_config=SimpleNamespace(enable_eplb=False)
        ),
    )
    w13 = torch.zeros((num_experts, 2, 1), dtype=torch.uint8)
    w2 = torch.zeros((num_experts, 2, 1), dtype=torch.uint8)
    w13_scale = torch.zeros((num_experts, 2, 1), dtype=torch.float8_e4m3fn)
    w2_scale = torch.zeros((num_experts, 2, 1), dtype=torch.float8_e4m3fn)
    weight_scale = torch.ones(num_experts)

    outputs = prepare_nvfp4_moe_layer_for_fi_or_cutlass(
        backend=NvFp4MoeBackend.FLASHINFER_CUTLASS,
        layer=layer,
        w13=w13,
        w13_scale=w13_scale,
        w13_scale_2=weight_scale,
        a13_scale=torch.tensor([1.0, 2.0, 3.0]),
        w2=w2,
        w2_scale=w2_scale,
        w2_scale_2=weight_scale,
        a2_scale=torch.tensor([4.0, 5.0, 6.0]),
        is_act_and_mul=False,
    )
    a13_scale, a2_scale = outputs[3], outputs[7]

    torch.testing.assert_close(a13_scale, torch.full((num_experts,), 3.0))
    torch.testing.assert_close(a2_scale, torch.full((num_experts,), 6.0))
    distinct_values = torch.arange(num_experts, dtype=torch.float32)
    a13_scale.copy_(distinct_values)
    a2_scale.copy_(distinct_values)
    torch.testing.assert_close(a13_scale, distinct_values)
    torch.testing.assert_close(a2_scale, distinct_values)


def test_align_trtllm_fp4_moe_hidden_dim_noop():
    w13 = torch.arange(2 * 8 * 256, dtype=torch.uint8).reshape(2, 8, 256)
    w13_scale = torch.arange(2 * 8 * 32, dtype=torch.uint8).reshape(2, 8, 32)
    w2 = torch.arange(2 * 512 * 4, dtype=torch.uint8).reshape(2, 512, 4)
    w2_scale = torch.arange(2 * 512 * 1, dtype=torch.uint8).reshape(2, 512, 1)

    out_w13, out_w13_scale, out_w2, out_w2_scale, padded_hidden = (
        align_fp4_moe_hidden_dim_for_fi(w13, w13_scale, w2, w2_scale)
    )

    assert padded_hidden == 512
    assert out_w13 is w13
    assert out_w13_scale is w13_scale
    assert out_w2 is w2
    assert out_w2_scale is w2_scale


def test_align_trtllm_fp4_moe_hidden_dim_pads_to_256_multiple():
    hidden_dim = 2688
    padded_hidden_dim = 2816

    w13 = torch.arange(2 * 12 * (hidden_dim // 2), dtype=torch.uint8).reshape(
        2, 12, hidden_dim // 2
    )
    w13_scale = torch.arange(2 * 12 * (hidden_dim // 16), dtype=torch.uint8).reshape(
        2, 12, hidden_dim // 16
    )

    w2 = torch.arange(2 * hidden_dim * 6, dtype=torch.uint8).reshape(2, hidden_dim, 6)
    w2_scale = torch.arange(2 * hidden_dim * 2, dtype=torch.uint8).reshape(
        2, hidden_dim, 2
    )

    out_w13, out_w13_scale, out_w2, out_w2_scale, out_hidden_dim = (
        align_fp4_moe_hidden_dim_for_fi(w13, w13_scale, w2, w2_scale)
    )

    assert out_hidden_dim == padded_hidden_dim
    assert out_w13.shape == (2, 12, padded_hidden_dim // 2)
    assert out_w13_scale.shape == (2, 12, padded_hidden_dim // 16)
    assert out_w2.shape == (2, padded_hidden_dim, 6)
    assert out_w2_scale.shape == (2, padded_hidden_dim, 2)

    torch.testing.assert_close(out_w13[:, :, : hidden_dim // 2], w13)
    torch.testing.assert_close(out_w13_scale[:, :, : hidden_dim // 16], w13_scale)
    torch.testing.assert_close(out_w2[:, :hidden_dim, :], w2)
    torch.testing.assert_close(out_w2_scale[:, :hidden_dim, :], w2_scale)

    assert torch.count_nonzero(out_w13[:, :, hidden_dim // 2 :]) == 0
    assert torch.count_nonzero(out_w13_scale[:, :, hidden_dim // 16 :]) == 0
    assert torch.count_nonzero(out_w2[:, hidden_dim:, :]) == 0
    assert torch.count_nonzero(out_w2_scale[:, hidden_dim:, :]) == 0


def test_prepare_gated_unaligned_pads_to_64_and_updates_config(monkeypatch):
    # Qwen3.8-Flash-Next at TP=4: intermediate 640/4 = 160 per rank. The
    # swizzled block scales follow a 128-row SF atom on the stacked gate+up
    # dim, so 160 pads to 192 (not 256) and moe_config must follow.
    monkeypatch.setattr(flashinfer_fp4_moe, "swizzle_blockscale", lambda x: x)

    num_experts, intermediate, hidden_dim = 2, 160, 32
    layer = SimpleNamespace(
        activation=SimpleNamespace(is_gated=True),
        moe_config=SimpleNamespace(
            moe_parallel_config=SimpleNamespace(enable_eplb=False),
            intermediate_size_per_partition=intermediate,
        ),
    )
    gate = torch.ones((num_experts, intermediate, hidden_dim // 2), dtype=torch.uint8)
    up = torch.full_like(gate, 2)
    w13 = torch.cat((gate, up), dim=1)
    w13_scale = torch.full(
        (num_experts, 2 * intermediate, hidden_dim // 16), 3, dtype=torch.float8_e4m3fn
    )
    w2 = torch.full((num_experts, hidden_dim, intermediate // 2), 5, dtype=torch.uint8)
    w2_scale = torch.full(
        (num_experts, hidden_dim, intermediate // 16), 6, dtype=torch.float8_e4m3fn
    )
    weight_scale = torch.ones(num_experts)

    outputs = prepare_nvfp4_moe_layer_for_fi_or_cutlass(
        backend=NvFp4MoeBackend.FLASHINFER_CUTLASS,
        layer=layer,
        w13=w13,
        w13_scale=w13_scale,
        w13_scale_2=weight_scale,
        a13_scale=torch.ones(num_experts),
        w2=w2,
        w2_scale=w2_scale,
        w2_scale_2=weight_scale,
        a2_scale=torch.ones(num_experts),
        is_act_and_mul=True,
    )
    out_w13, out_w13_scale, out_w2, out_w2_scale = (
        outputs[0],
        outputs[1],
        outputs[4],
        outputs[5],
    )

    assert layer.moe_config.intermediate_size_per_partition == 192
    assert out_w13.shape == (num_experts, 2 * 192, hidden_dim // 2)
    assert out_w13_scale.shape == (num_experts, 2 * 192, hidden_dim // 16)
    assert out_w2.shape == (num_experts, hidden_dim, 192 // 2)
    assert out_w2_scale.shape == (num_experts, hidden_dim, 192 // 16)
    # Gate and up stay in separate halves, each zero-padded to 192 rows.
    torch.testing.assert_close(out_w13[:, :intermediate], up)
    torch.testing.assert_close(out_w13[:, 192 : 192 + intermediate], gate)
    assert torch.count_nonzero(out_w13[:, intermediate:192]) == 0
    assert torch.count_nonzero(out_w13[:, 192 + intermediate :]) == 0


def test_prepare_gated_aligned_is_noop(monkeypatch):
    monkeypatch.setattr(flashinfer_fp4_moe, "swizzle_blockscale", lambda x: x)

    num_experts, intermediate, hidden_dim = 2, 192, 32
    layer = SimpleNamespace(
        activation=SimpleNamespace(is_gated=True),
        moe_config=SimpleNamespace(
            moe_parallel_config=SimpleNamespace(enable_eplb=False),
            intermediate_size_per_partition=intermediate,
        ),
    )
    w13 = torch.ones(
        (num_experts, 2 * intermediate, hidden_dim // 2), dtype=torch.uint8
    )
    w13_scale = torch.full(
        (num_experts, 2 * intermediate, hidden_dim // 16), 3, dtype=torch.float8_e4m3fn
    )
    w2 = torch.ones((num_experts, hidden_dim, intermediate // 2), dtype=torch.uint8)
    w2_scale = torch.full(
        (num_experts, hidden_dim, intermediate // 16), 6, dtype=torch.float8_e4m3fn
    )
    weight_scale = torch.ones(num_experts)

    outputs = prepare_nvfp4_moe_layer_for_fi_or_cutlass(
        backend=NvFp4MoeBackend.FLASHINFER_CUTLASS,
        layer=layer,
        w13=w13,
        w13_scale=w13_scale,
        w13_scale_2=weight_scale,
        a13_scale=torch.ones(num_experts),
        w2=w2,
        w2_scale=w2_scale,
        w2_scale_2=weight_scale,
        a2_scale=torch.ones(num_experts),
        is_act_and_mul=True,
    )

    assert layer.moe_config.intermediate_size_per_partition == 192
    assert outputs[0] is w13
    assert outputs[1] is w13_scale
    assert outputs[4] is w2
    assert outputs[5] is w2_scale


def test_prepare_non_gated_pads_to_128(monkeypatch):
    # Non-gated weights stack to I rows, so the SF atom forces I % 128.
    monkeypatch.setattr(flashinfer_fp4_moe, "swizzle_blockscale", lambda x: x)

    num_experts, intermediate, hidden_dim = 2, 160, 32
    layer = SimpleNamespace(
        activation=SimpleNamespace(is_gated=False),
        moe_config=SimpleNamespace(
            moe_parallel_config=SimpleNamespace(enable_eplb=False),
            intermediate_size_per_partition=intermediate,
        ),
    )
    w13 = torch.ones((num_experts, intermediate, hidden_dim // 2), dtype=torch.uint8)
    w13_scale = torch.full(
        (num_experts, intermediate, hidden_dim // 16), 3, dtype=torch.float8_e4m3fn
    )
    w2 = torch.ones((num_experts, hidden_dim, intermediate // 2), dtype=torch.uint8)
    w2_scale = torch.full(
        (num_experts, hidden_dim, intermediate // 16), 6, dtype=torch.float8_e4m3fn
    )
    weight_scale = torch.ones(num_experts)

    outputs = prepare_nvfp4_moe_layer_for_fi_or_cutlass(
        backend=NvFp4MoeBackend.FLASHINFER_CUTLASS,
        layer=layer,
        w13=w13,
        w13_scale=w13_scale,
        w13_scale_2=weight_scale,
        a13_scale=torch.ones(num_experts),
        w2=w2,
        w2_scale=w2_scale,
        w2_scale_2=weight_scale,
        a2_scale=torch.ones(num_experts),
        is_act_and_mul=False,
    )

    assert layer.moe_config.intermediate_size_per_partition == 256
    assert outputs[0].shape == (num_experts, 256, hidden_dim // 2)
    assert outputs[1].shape == (num_experts, 256, hidden_dim // 16)
    assert outputs[4].shape == (num_experts, hidden_dim, 256 // 2)
    assert outputs[5].shape == (num_experts, hidden_dim, 256 // 16)


def test_align_trtllm_fp4_moe_intermediate_pads_gate_and_up_separately():
    # Qwen3.8-Flash-Next has intermediate_size=640. At TP4 each rank owns
    # 160 values, which TRTLLM pads to the next 64-value boundary (192).
    intermediate = 160
    padded_intermediate = 192
    hidden_dim = 32

    gate = torch.ones((1, intermediate, hidden_dim // 2), dtype=torch.uint8)
    up = torch.full_like(gate, 2)
    w13 = torch.cat((gate, up), dim=1)
    gate_scale = torch.full((1, intermediate, hidden_dim // 16), 3, dtype=torch.uint8)
    up_scale = torch.full_like(gate_scale, 4)
    w13_scale = torch.cat((gate_scale, up_scale), dim=1)
    w2 = torch.full((1, hidden_dim, intermediate // 2), 5, dtype=torch.uint8)
    w2_scale = torch.full((1, hidden_dim, intermediate // 16), 6, dtype=torch.uint8)

    out_w13, out_w13_scale, out_w2, out_w2_scale, out_intermediate = (
        align_fp4_moe_weights_for_fi(
            w13,
            w13_scale,
            w2,
            w2_scale,
            is_act_and_mul=True,
            min_alignment=64,
        )
    )

    assert out_intermediate == padded_intermediate
    torch.testing.assert_close(out_w13[:, :intermediate], gate)
    torch.testing.assert_close(
        out_w13[:, padded_intermediate : padded_intermediate + intermediate], up
    )
    torch.testing.assert_close(out_w13_scale[:, :intermediate], gate_scale)
    torch.testing.assert_close(
        out_w13_scale[:, padded_intermediate : padded_intermediate + intermediate],
        up_scale,
    )

    assert torch.count_nonzero(out_w13[:, intermediate:padded_intermediate]) == 0
    assert torch.count_nonzero(out_w13[:, padded_intermediate + intermediate :]) == 0
    assert torch.count_nonzero(out_w13_scale[:, intermediate:padded_intermediate]) == 0
    assert (
        torch.count_nonzero(out_w13_scale[:, padded_intermediate + intermediate :]) == 0
    )
    torch.testing.assert_close(out_w2[:, :, : intermediate // 2], w2)
    torch.testing.assert_close(out_w2_scale[:, :, : intermediate // 16], w2_scale)
    assert torch.count_nonzero(out_w2[:, :, intermediate // 2 :]) == 0
    assert torch.count_nonzero(out_w2_scale[:, :, intermediate // 16 :]) == 0


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_moe_forward_shared_gets_packed_shared_input_under_inductor():
    """Inductor must not pass the shared experts a strided alias of the padded
    routed input: their NVFP4 quant kernel assumes packed rows."""
    hidden, padded_hidden, num_tokens = 2688, 2816, 64
    name = "model.layers.0.mixer.experts"
    shared_input_strides = []

    def forward_impl(hidden_states, router_logits, shared_experts_input, input_ids):
        shared_input_strides.append(shared_experts_input.stride())
        return torch.zeros_like(shared_experts_input), torch.zeros_like(hidden_states)

    layer = Mock(spec=MoERunnerInterface)
    layer._forward_impl = forward_impl
    layer_name = LayerName(name) if _USE_LAYERNAME else name

    def moe_block(x, residual, norm_weight, gate_weight):
        x = x + residual
        x = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5)
        x = x.to(torch.bfloat16) * norm_weight
        router_logits = torch.mm(x, gate_weight.t(), out_dtype=torch.float32)
        padded = F.pad(x, (0, padded_hidden - hidden))
        shared_out, fused_out = torch.ops.vllm.moe_forward_shared(
            padded, router_logits, x, None, layer_name, 0
        )
        return shared_out + fused_out[..., :hidden]

    x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    norm_weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    gate_weight = torch.randn(128, hidden, device="cuda", dtype=torch.bfloat16)
    torch._dynamo.mark_dynamic(x, 0)
    torch._dynamo.mark_dynamic(residual, 0)

    vllm_config = VllmConfig()
    vllm_config.compilation_config.static_forward_context = {name: layer}
    with (
        torch.inference_mode(),
        torch._inductor.config.patch(force_disable_caches=True),
        set_current_vllm_config(vllm_config),
        set_forward_context(None, vllm_config),
    ):
        torch.compile(moe_block, backend="inductor", fullgraph=True)(
            x, residual, norm_weight, gate_weight
        )

    assert shared_input_strides == [(hidden, 1)]
