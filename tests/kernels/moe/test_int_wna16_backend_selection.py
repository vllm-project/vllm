# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import patch

import pytest
import torch

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.experts.xpu_moe import XPUExpertsInt4
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    WNA16MoEBackend,
    convert_to_wna16_moe_kernel_format,
    select_wna16_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kInt4Static,
    kInt8Static,
)


def test_select_xpu_wna16_int4_backend():
    with (
        patch(
            "vllm.model_executor.layers.fused_moe.oracle.int_wna16.current_platform"
        ) as oracle_platform,
        patch(
            "vllm.model_executor.layers.fused_moe.experts.xpu_moe.current_platform"
        ) as xpu_platform,
    ):
        oracle_platform.is_xpu.return_value = True
        xpu_platform.is_xpu.return_value = True

        backend, experts_cls = select_wna16_moe_backend(
            make_dummy_moe_config(),
            kInt4Static,
            weight_bits=4,
        )

    assert backend == WNA16MoEBackend.XPU
    assert experts_cls is XPUExpertsInt4


def test_select_xpu_wna16_rejects_int8():
    with (
        patch(
            "vllm.model_executor.layers.fused_moe.oracle.int_wna16.current_platform"
        ) as oracle_platform,
        patch(
            "vllm.model_executor.layers.fused_moe.experts.xpu_moe.current_platform"
        ) as xpu_platform,
    ):
        oracle_platform.is_xpu.return_value = True
        xpu_platform.is_xpu.return_value = True

        with pytest.raises(NotImplementedError):
            select_wna16_moe_backend(
                make_dummy_moe_config(),
                kInt8Static,
                weight_bits=8,
            )


def test_convert_xpu_wna16_moe_kernel_format():
    num_experts = 2
    hidden_size = 16
    intermediate_size = 8
    group_size = 4
    pack_factor = 8

    w13 = torch.arange(
        num_experts * (hidden_size // pack_factor) * (2 * intermediate_size),
        dtype=torch.int32,
    ).reshape(num_experts, hidden_size // pack_factor, 2 * intermediate_size)
    w2 = torch.arange(
        num_experts * (intermediate_size // pack_factor) * hidden_size,
        dtype=torch.int32,
    ).reshape(num_experts, intermediate_size // pack_factor, hidden_size)
    w13_scale = torch.randn(
        num_experts,
        hidden_size // group_size,
        2 * intermediate_size,
        dtype=torch.float16,
    )
    w2_scale = torch.randn(
        num_experts,
        intermediate_size // group_size,
        hidden_size,
        dtype=torch.float16,
    )

    (
        xpu_w13,
        xpu_w2,
        xpu_w13_scale,
        xpu_w2_scale,
        w13_g_idx,
        w2_g_idx,
        w13_g_idx_sort_indices,
        w2_g_idx_sort_indices,
        w13_input_global_scale,
        w2_input_global_scale,
        _w13_bias,
        _w2_bias,
    ) = convert_to_wna16_moe_kernel_format(
        backend=WNA16MoEBackend.XPU,
        layer=torch.nn.Module(),
        quant_config=None,
        input_dtype=None,
        w13=w13,
        w2=w2,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        w13_g_idx=torch.empty(0),
        w2_g_idx=torch.empty(0),
    )

    assert torch.equal(xpu_w13, w13.transpose(1, 2).contiguous().view(torch.uint8))
    assert torch.equal(xpu_w2, w2.transpose(1, 2).contiguous().view(torch.uint8))
    assert torch.equal(xpu_w13_scale, w13_scale.transpose(1, 2).contiguous())
    assert torch.equal(xpu_w2_scale, w2_scale.transpose(1, 2).contiguous())
    assert xpu_w13.shape == (num_experts, 2 * intermediate_size, hidden_size // 2)
    assert xpu_w2.shape == (num_experts, hidden_size, intermediate_size // 2)
    assert xpu_w13_scale.shape == (
        num_experts,
        2 * intermediate_size,
        hidden_size // group_size,
    )
    assert xpu_w2_scale.shape == (
        num_experts,
        hidden_size,
        intermediate_size // group_size,
    )
    assert w13_g_idx is None
    assert w2_g_idx is None
    assert w13_g_idx_sort_indices is None
    assert w2_g_idx_sort_indices is None
    assert w13_input_global_scale is None
    assert w2_input_global_scale is None
