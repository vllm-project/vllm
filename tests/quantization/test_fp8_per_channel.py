# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for FP8 per-channel online quantization.

Per-output-channel weight scale + dynamic per-token activation scale.
bf16/fp16 checkpoints are quantized at load time with one fp32 scale per
output channel for weights and one fp32 scale per token for activations
(computed dynamically inside the kernel). Run via
`pytest tests/quantization/test_fp8_per_channel.py --forked`.
"""

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
    kFp8StaticChannelSym,
)
from vllm.platforms import current_platform


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
def test_scaled_fp8_quant_per_channel_shape() -> None:
    """Verify the kernel call per-channel quant depends on: passing a 2D
    weight to `ops.scaled_fp8_quant` with `use_per_token_if_dynamic=True`
    yields one scale per output row -- a [N, 1] fp32 tensor.
    """
    x = (torch.randn(size=(96, 256), device="cuda") * 13).to(torch.bfloat16)
    y, s = ops.scaled_fp8_quant(x, scale=None, use_per_token_if_dynamic=True)
    assert y.shape == (96, 256)
    assert y.dtype == current_platform.fp8_dtype()
    assert s.shape == (96, 1)
    assert s.dtype == torch.float32
