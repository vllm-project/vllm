# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for FP8 per-channel online quantization.

Per-output-channel weight scale + dynamic per-token activation scale.
bf16/fp16 checkpoints are quantized at load time with one fp32 scale per
output channel for weights and one fp32 scale per token for activations
(computed dynamically inside the kernel). Run via
`pytest tests/quantization/test_fp8_per_channel.py --forked`.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tests.quantization.utils import is_quant_method_supported
from vllm import _custom_ops as ops
from vllm.config.quantization import (
    _ONLINE_SHORTHANDS,
    QUANT_KEY_NAMES,
    QuantizationConfigArgs,
)
from vllm.model_executor.layers.quantization.online.base import (
    _ONLINE_LINEAR_METHODS,
    _ONLINE_MOE_METHODS,
)
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PtpcOnlineLinearMethod,
    Fp8PtpcOnlineMoEMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
    kFp8StaticChannelSym,
)
from vllm.platforms import current_platform

if current_platform.is_rocm():
    from vllm.platforms.rocm import on_gfx942
else:

    def on_gfx942() -> bool:
        return False


def test_fp8_per_channel_shorthand_registered() -> None:
    """The `fp8_per_channel` CLI shorthand must resolve to a config that
    dispatches the per-channel methods. Guards against regressions in
    `_ONLINE_SHORTHANDS` / `_ONLINE_LINEAR_METHODS` / `_ONLINE_MOE_METHODS`
    drifting out of sync.
    """
    args = _ONLINE_SHORTHANDS["fp8_per_channel"]
    assert isinstance(args, QuantizationConfigArgs)
    assert args.linear is not None
    assert args.moe is not None
    assert args.linear.weight is kFp8StaticChannelSym
    assert args.moe.weight is kFp8StaticChannelSym

    assert _ONLINE_LINEAR_METHODS[kFp8StaticChannelSym] is Fp8PtpcOnlineLinearMethod
    assert _ONLINE_MOE_METHODS[kFp8StaticChannelSym] is Fp8PtpcOnlineMoEMethod

    assert QUANT_KEY_NAMES["fp8_per_channel_static"] is kFp8StaticChannelSym


@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="FP8 is not supported on this GPU type.",
)
@pytest.mark.skipif(
    on_gfx942(),
    reason="The dynamic per-token FP8 kernel can segfault on gfx942.",
)
def test_scaled_fp8_quant_per_channel_shape() -> None:
    """Verify the kernel call per-channel quant depends on: passing a 2D
    weight to `ops.scaled_fp8_quant` with `use_per_token_if_dynamic=True`
    yields one scale per output row -- a [N, 1] fp32 tensor.
    """
    x = (torch.randn(size=(96, 256), device=current_platform.device_type) * 13).to(
        torch.bfloat16
    )
    y, s = ops.scaled_fp8_quant(x, scale=None, use_per_token_if_dynamic=True)
    assert y.shape == (96, 256)
    assert y.dtype == current_platform.fp8_dtype()
    assert s.shape == (96, 1)
    assert s.dtype == torch.float32


@pytest.mark.skipif(not on_gfx942(), reason="Requires gfx942.")
def test_gfx942_moe_weight_quant_avoids_dynamic_per_token_op(monkeypatch) -> None:
    """gfx942 load-time MoE quantization uses the torch per-channel path."""
    device = current_platform.device_type
    torch.manual_seed(0)
    w13_weight = torch.randn(2, 96, 128, device=device, dtype=torch.bfloat16)
    w2_weight = torch.randn(2, 128, 48, device=device, dtype=torch.bfloat16)

    method = object.__new__(Fp8PtpcOnlineMoEMethod)
    method.moe = SimpleNamespace(tp_size=1)
    layer = SimpleNamespace(local_num_experts=2)

    def fail_dynamic_quant(*args, **kwargs):
        raise AssertionError("gfx942 must not call dynamic per-token FP8 quant")

    monkeypatch.setattr(ops, "scaled_fp8_quant", fail_dynamic_quant)
    with (
        mock.patch.object(
            method,
            "get_weights_for_quantization",
            return_value=(w13_weight, w2_weight),
        ) as get_weights,
        mock.patch.object(method, "_setup_kernel") as setup_kernel,
        mock.patch.object(
            method, "release_requantization_source_weights"
        ) as release_weights,
    ):
        method.process_weights_after_loading(layer)

    get_weights.assert_called_once_with(layer)
    setup_kernel.assert_called_once()
    release_weights.assert_called_once_with(layer)
    _, got_w13, got_w2, got_w13_scale, got_w2_scale = setup_kernel.call_args.args
    assert setup_kernel.call_args.kwargs == {
        "w13_input_scale": None,
        "w2_input_scale": None,
    }

    fp8_min, fp8_max = get_fp8_min_max()

    def reference(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fp8_max_tensor = torch.tensor(
            fp8_max, device=weight.device, dtype=torch.float32
        )
        scale = weight.abs().amax(dim=-1, keepdim=True).float() / fp8_max_tensor
        scale.clamp_min_(1.0 / (fp8_max * 512))
        quantized = (
            (weight / scale).clamp(fp8_min, fp8_max).to(current_platform.fp8_dtype())
        )
        return quantized, scale

    expected_w13, expected_w13_scale = reference(w13_weight)
    expected_w2, expected_w2_scale = reference(w2_weight)
    assert torch.equal(got_w13_scale, expected_w13_scale)
    assert torch.equal(got_w13, expected_w13)
    assert torch.equal(got_w2_scale, expected_w2_scale)
    assert torch.equal(got_w2, expected_w2)
    assert layer._already_called_process_weights_after_loading
