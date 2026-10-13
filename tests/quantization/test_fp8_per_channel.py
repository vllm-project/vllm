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
    x = (torch.randn(size=(96, 256), device=current_platform.device_type) * 13).to(
        torch.bfloat16
    )
    y, s = ops.scaled_fp8_quant(x, scale=None, use_per_token_if_dynamic=True)
    assert y.shape == (96, 256)
    assert y.dtype == current_platform.fp8_dtype()
    assert s.shape == (96, 1)
    assert s.dtype == torch.float32


def test_fp8_ptpc_create_weights_surfaces_kernel_rejection_reasons(
    monkeypatch,
) -> None:
    """PTPC create_weights must fail before weight allocation and include the
    chooser's per-kernel rejection reasons (issue #61316).
    """
    from unittest.mock import MagicMock

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.quantization.online import fp8 as fp8_mod

    layer = MagicMock()
    layer._parameters = {}

    def _raise_chooser_error(**kwargs):
        raise ValueError(
            "Failed to find a kernel that can implement the "
            "ScaledMM linear layer. Reasons: \n"
            "CutlassFP8ScaledMMLinearKernel CUTLASS FP8 kernels not "
            "available.\n"
            "MarlinFP8ScaledMMLinearKernel is W8A16 weight-only and "
            "cannot honor per-token activation quantization."
        )

    monkeypatch.setattr(fp8_mod, "init_fp8_linear_kernel", _raise_chooser_error)

    model_config = MagicMock()
    model_config.dtype = torch.bfloat16
    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.model_config = model_config

    with set_current_vllm_config(vllm_config):
        method = Fp8PtpcOnlineLinearMethod()
        with pytest.raises(ValueError, match='quantization="fp8_per_channel"') as ei:
            method.create_weights(
                layer,
                input_size_per_partition=128,
                output_partition_sizes=[256],
                input_size=128,
                output_size=256,
                params_dtype=torch.bfloat16,
            )

    message = str(ei.value)
    assert "Reasons:" in message
    assert "MarlinFP8ScaledMMLinearKernel" in message
    assert "per-token" in message
    # Must fail before OnlineLinearBase allocates the meta weight.
    layer.register_parameter.assert_not_called()
