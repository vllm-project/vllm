# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    convert_weight_to_mxfp4_moe_kernel_format,
    mxfp4_round_up_hidden_size_and_intermediate_size,
)
from vllm.model_executor.layers.quantization.quark import quark_moe


@pytest.mark.parametrize(
    ("scheme", "aiter_enabled", "expected"),
    [
        ("w_mxfp4_a_mxfp4", True, True),
        ("w_mxfp4_a_mxfp4", False, False),
        ("w_mxfp4", True, False),
        ("w_mxfp4_a_fp8", True, False),
    ],
)
def test_use_k3_situ_aiter_a8w4(monkeypatch, scheme, aiter_enabled, expected):
    monkeypatch.setattr(
        quark_moe.rocm_aiter_ops,
        "is_fused_moe_situv2_a8w4_enabled",
        lambda: aiter_enabled,
    )

    assert quark_moe._use_k3_situ_aiter_a8w4(MagicMock(), scheme) is expected


def test_situ_aiter_keeps_tp8_intermediate_size():
    hidden_size, intermediate_size = mxfp4_round_up_hidden_size_and_intermediate_size(
        Mxfp4MoeBackend.AITER_MXFP4_BF16,
        3584,
        384,
        activation=MoEActivation.SITU,
    )

    assert hidden_size == 3584
    assert intermediate_size == 384


@pytest.mark.parametrize("guinterleave", [False, True])
def test_convert_situ_weight_to_mxfp4_kernel_format(monkeypatch, guinterleave):
    layer = MagicMock()
    w13_weight = torch.nn.Parameter(
        torch.zeros((2, 4, 4), dtype=torch.uint8), requires_grad=False
    )
    w2_weight = torch.nn.Parameter(
        torch.zeros((2, 4, 4), dtype=torch.uint8), requires_grad=False
    )
    w13_weight_scale = torch.nn.Parameter(
        torch.zeros((2, 4, 2), dtype=torch.uint8), requires_grad=False
    )
    w2_weight_scale = torch.nn.Parameter(
        torch.zeros((2, 4, 2), dtype=torch.uint8), requires_grad=False
    )

    shuffled_w13 = torch.empty(1)
    shuffled_w2 = torch.empty(1)
    shuffled_w13_scale = torch.empty(1)
    shuffled_w2_scale = torch.empty(1)
    e8m0_shuffle = MagicMock(return_value=shuffled_w2_scale)
    aiter_module = ModuleType("aiter")
    aiter_utility_module = ModuleType("aiter.utility")
    fp4_utils_module = ModuleType("aiter.utility.fp4_utils")
    fp4_utils_module.e8m0_shuffle = e8m0_shuffle

    with (
        patch.dict(
            sys.modules,
            {
                "aiter": aiter_module,
                "aiter.utility": aiter_utility_module,
                "aiter.utility.fp4_utils": fp4_utils_module,
            },
        ),
        patch(
            "vllm._aiter_ops.rocm_aiter_ops.is_fused_moe_situv2_a8w4_enabled",
            return_value=guinterleave,
        ),
        patch(
            "vllm._aiter_ops.rocm_aiter_ops.shuffle_weight_a16w4",
            side_effect=[shuffled_w13, shuffled_w2],
        ) as shuffle_weight,
        patch(
            "vllm._aiter_ops.rocm_aiter_ops.shuffle_scale_a16w4",
            return_value=shuffled_w13_scale,
        ) as shuffle_scale,
    ):
        monkeypatch.delenv("AITER_BF16_FP8_MOE_BOUND", raising=False)
        converted = convert_weight_to_mxfp4_moe_kernel_format(
            mxfp4_backend=Mxfp4MoeBackend.AITER_MXFP4_BF16,
            layer=layer,
            w13_weight=w13_weight,
            w2_weight=w2_weight,
            w13_weight_scale=w13_weight_scale,
            w2_weight_scale=w2_weight_scale,
            activation=MoEActivation.SITU,
        )

    assert converted[:4] == (
        shuffled_w13,
        shuffled_w2,
        shuffled_w13_scale,
        shuffled_w2_scale,
    )
    assert shuffled_w13.is_shuffled
    assert shuffled_w2.is_shuffled
    assert shuffle_weight.call_args_list[0].args[2] is guinterleave
    assert shuffle_weight.call_args_list[1].args[2] is False
    assert shuffle_scale.call_args.args[2] is guinterleave
    e8m0_shuffle.assert_called_once()
    assert os.environ["AITER_BF16_FP8_MOE_BOUND"] == "0"
