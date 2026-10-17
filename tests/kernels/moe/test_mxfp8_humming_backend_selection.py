# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 MoE backend selection and kernel format conversion tests."""

import dataclasses
from unittest.mock import MagicMock, patch

import torch

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend
from vllm.model_executor.layers.fused_moe.oracle.mxfp8 import (
    convert_to_mxfp8_moe_kernel_format,
    select_mxfp8_moe_backend,
)


def test_humming_selected_for_mxfp8():
    cfg = make_dummy_moe_config(num_experts=8, hidden_dim=256)
    cfg = dataclasses.replace(cfg, moe_backend="humming")

    sentinel_cls = object()
    with patch(
        "vllm.model_executor.layers.fused_moe.oracle.mxfp8._select_kernel_cls",
        return_value=sentinel_cls,
    ):
        backend, cls = select_mxfp8_moe_backend(cfg)
        assert backend == Fp8MoeBackend.HUMMING
        assert cls is sentinel_cls


def test_convert_to_mxfp8_moe_kernel_format_humming():
    layer = MagicMock()
    w13 = torch.empty((8, 512, 256), dtype=torch.float8_e4m3fn)
    w2 = torch.empty((8, 256, 256), dtype=torch.float8_e4m3fn)
    w13_scale = torch.empty((8, 512, 8), dtype=torch.uint8)
    w2_scale = torch.empty((8, 256, 8), dtype=torch.uint8)

    layer.w13_weight = w13
    layer.w2_weight = w2
    layer.w13_weight_scale = w13_scale
    layer.w2_weight_scale = w2_scale

    with patch(
        "vllm.model_executor.layers.quantization.utils.humming.convert_to_humming_moe_kernel_format"
    ) as mock_convert:
        out_w13, out_w2, out_w13_scale, out_w2_scale = (
            convert_to_mxfp8_moe_kernel_format(
                backend=Fp8MoeBackend.HUMMING,
                layer=layer,
                w13=w13,
                w2=w2,
                w13_scale=w13_scale,
                w2_scale=w2_scale,
            )
        )
        assert mock_convert.called
        assert out_w13 is w13
        assert out_w2 is w2
        assert out_w13_scale is w13_scale
        assert out_w2_scale is w2_scale


def test_convert_to_mxfp8_moe_kernel_format_delegates():
    layer = MagicMock()
    t = torch.empty(1)
    with patch(
        "vllm.model_executor.layers.fused_moe.oracle.fp8.convert_to_fp8_moe_kernel_format",
        return_value=(t, t, t, t),
    ) as mock_fp8_convert:
        res = convert_to_mxfp8_moe_kernel_format(
            backend=Fp8MoeBackend.TRITON_MXFP8,
            layer=layer,
            w13=t,
            w2=t,
            w13_scale=t,
            w2_scale=t,
        )
        assert mock_fp8_convert.called
        assert res == (t, t, t, t)
