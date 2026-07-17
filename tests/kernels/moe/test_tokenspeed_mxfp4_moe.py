# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_dummy_moe_config
from tests.kernels.utils import torch_experts
from tests.quantization.reference_mxfp4 import dq_mxfp4_torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.experts.tokenspeed_mxfp4_moe import (
    TokenSpeedMxfp4Experts,
)
from vllm.model_executor.layers.fused_moe.oracle import mxfp4
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,
    make_mxfp4_moe_quant_config,
    select_mxfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticTensorSym,
    kMxfp4Static,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed


def _make_config(in_dtype: torch.dtype = torch.bfloat16, use_ep: bool = False):
    config = make_dummy_moe_config(
        num_experts=4,
        experts_per_token=2,
        hidden_dim=256,
        intermediate_size=256,
        in_dtype=in_dtype,
        activation=MoEActivation.SWIGLUOAI,
    )
    config.routing_method = RoutingMethodType.Renormalize
    config.moe_backend = "tokenspeed"
    config.moe_parallel_config.use_ep = use_ep
    config.moe_parallel_config.ep_size = 2 if use_ep else 1
    return config


def _tokenspeed_mxfp4_available() -> bool:
    try:
        return all(
            importlib.util.find_spec(module) is not None
            for module in (
                "tokenspeed_kernel_amd.ops.moe.fused_mxfp_gfx950",
                "tokenspeed_kernel_amd.ops.moe.mxfp4_gfx950_preprocess",
            )
        )
    except ModuleNotFoundError:
        return False


def _is_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


def _make_layer(
    num_experts: int, hidden_size: int, intermediate_size: int
) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(
        torch.randint(
            256,
            (num_experts, 2 * intermediate_size, hidden_size // 2),
            device="cuda",
            dtype=torch.uint8,
        ),
        requires_grad=False,
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.randint(
            256,
            (num_experts, hidden_size, intermediate_size // 2),
            device="cuda",
            dtype=torch.uint8,
        ),
        requires_grad=False,
    )
    layer.w13_weight_scale = torch.nn.Parameter(
        torch.randint(
            118,
            123,
            (num_experts, 2 * intermediate_size, hidden_size // 32),
            device="cuda",
            dtype=torch.uint8,
        ),
        requires_grad=False,
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        torch.randint(
            118,
            123,
            (num_experts, hidden_size, intermediate_size // 32),
            device="cuda",
            dtype=torch.uint8,
        ),
        requires_grad=False,
    )
    layer.w13_input_scale = torch.nn.Parameter(
        torch.ones(num_experts, device="cuda"), requires_grad=False
    )
    layer.w2_input_scale = torch.nn.Parameter(
        torch.ones(num_experts, device="cuda"), requires_grad=False
    )
    return layer


def test_tokenspeed_backend_selection(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        TokenSpeedMxfp4Experts, "_supports_current_device", lambda: True
    )
    monkeypatch.setattr(mxfp4, "_user_moe_activation_override", lambda: None)

    backend, experts_cls = select_mxfp4_moe_backend(
        _make_config(), activation_key=kFp8StaticTensorSym
    )

    assert backend == Mxfp4MoeBackend.TOKENSPEED
    assert experts_cls is TokenSpeedMxfp4Experts


@pytest.mark.parametrize(
    "in_dtype,use_ep,supported",
    [
        (torch.bfloat16, False, True),
        (torch.float16, False, True),
        (torch.float32, False, False),
        (torch.bfloat16, True, False),
    ],
)
def test_tokenspeed_supported_config(
    monkeypatch: pytest.MonkeyPatch,
    in_dtype: torch.dtype,
    use_ep: bool,
    supported: bool,
):
    monkeypatch.setattr(
        TokenSpeedMxfp4Experts, "_supports_current_device", lambda: True
    )

    result, _ = TokenSpeedMxfp4Experts.is_supported_config(
        TokenSpeedMxfp4Experts,
        _make_config(in_dtype, use_ep),
        kMxfp4Static,
        kFp8StaticTensorSym,
        mk.FusedMoEActivationFormat.Standard,
    )

    assert result is supported


@pytest.mark.skipif(
    not (_is_gfx950() and _tokenspeed_mxfp4_available()),
    reason="Requires GFX950 and tokenspeed-kernel-amd MXFP4 MoE support",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_tokenspeed_mxfp4_moe_matches_reference(default_vllm_config, dtype):
    set_random_seed(7)
    num_tokens, num_experts, hidden_size, intermediate_size, topk = 4, 4, 256, 256, 2

    layer = _make_layer(num_experts, hidden_size, intermediate_size)
    reference_w13 = dq_mxfp4_torch(layer.w13_weight, layer.w13_weight_scale, dtype)
    reference_w2 = dq_mxfp4_torch(layer.w2_weight, layer.w2_weight_scale, dtype)
    w13_kernel, w2_kernel, w13_kernel_scale, w2_kernel_scale, _, _ = (
        convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(
            Mxfp4MoeBackend.TOKENSPEED,
            layer,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
        )
    )
    quant_config = make_mxfp4_moe_quant_config(
        Mxfp4MoeBackend.TOKENSPEED,
        w13_kernel_scale,
        w2_kernel_scale,
        layer=layer,
    )
    assert quant_config is not None

    hidden_states = (
        torch.randn(num_tokens, hidden_size, device="cuda", dtype=dtype) / 10
    )
    router_logits = torch.randn(num_tokens, num_experts, device="cuda")
    output = TokenSpeedMxfp4Experts(_make_config(dtype), quant_config).apply(
        hidden_states,
        w13_kernel,
        w2_kernel,
        router_logits,
        MoEActivation.SWIGLUOAI,
        num_experts,
        None,
        None,
        False,
    )
    topk_logits, topk_ids = router_logits.topk(topk, dim=-1)

    reference = torch_experts(
        a=hidden_states,
        w1=reference_w13,
        w2=reference_w2,
        topk_weight=topk_logits.softmax(dim=-1),
        topk_ids=topk_ids,
        global_num_experts=num_experts,
        b_bias1=layer.w13_weight_bias,
        b_bias2=layer.w2_weight_bias,
        w1_scale=torch.ones((num_experts, 1, 1), device="cuda"),
        w2_scale=torch.ones((num_experts, 1, 1), device="cuda"),
        a1_scale=layer.w13_act_scale,
        a2_scale=layer.w2_act_scale,
        quant_dtype=torch.float8_e4m3fn,
        activation=MoEActivation.SWIGLUOAI,
    )

    torch.testing.assert_close(output.float(), reference.float(), rtol=0.1, atol=0.1)
