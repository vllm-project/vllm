# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.kernels.linear.scaled_mm.xpu import XPUW8A8FP8LinearKernel


def _make_xpu_w8a8_kernel(out_features: int, in_features: int):
    kernel = object.__new__(XPUW8A8FP8LinearKernel)
    kernel.config = SimpleNamespace(weight_shape=(out_features, in_features))
    return kernel


def _make_layer(weight: torch.Tensor, out_features: int, in_features: int):
    layer = torch.nn.Module()
    layer.input_size_per_partition = in_features
    layer.output_size_per_partition = out_features
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(
        torch.ones(out_features, dtype=torch.float32), requires_grad=False
    )
    return layer


@pytest.mark.parametrize(
    ("weight", "expected"),
    [
        pytest.param(
            torch.arange(15, dtype=torch.float32).reshape(3, 5),
            torch.arange(15, dtype=torch.float32).reshape(3, 5).t().contiguous(),
            id="checkpoint_nk",
        ),
        pytest.param(
            torch.arange(15, dtype=torch.float32).reshape(3, 5).t(),
            torch.arange(15, dtype=torch.float32).reshape(3, 5).t().contiguous(),
            id="already_transposed_view",
        ),
        pytest.param(
            torch.arange(15, dtype=torch.float32).reshape(3, 5).t().contiguous(),
            torch.arange(15, dtype=torch.float32).reshape(3, 5).t().contiguous(),
            id="already_kn_contiguous",
        ),
    ],
)
def test_xpu_w8a8_process_weights_makes_kn_weight_contiguous(
    weight: torch.Tensor, expected: torch.Tensor
):
    kernel = _make_xpu_w8a8_kernel(out_features=3, in_features=5)
    layer = _make_layer(weight, out_features=3, in_features=5)

    kernel.process_weights_after_loading(layer)

    assert layer.weight.shape == (5, 3)
    assert layer.weight.is_contiguous()
    assert torch.equal(layer.weight, expected)


@pytest.mark.parametrize(
    ("weight", "expected"),
    [
        pytest.param(
            torch.arange(16, dtype=torch.float32).reshape(4, 4),
            torch.arange(16, dtype=torch.float32).reshape(4, 4).t().contiguous(),
            id="square_checkpoint_contiguous",
        ),
        pytest.param(
            torch.arange(16, dtype=torch.float32).reshape(4, 4).t(),
            torch.arange(16, dtype=torch.float32).reshape(4, 4).t().contiguous(),
            id="square_transposed_view",
        ),
    ],
)
def test_xpu_w8a8_process_weights_disambiguates_square_weight_layout(
    weight: torch.Tensor, expected: torch.Tensor
):
    kernel = _make_xpu_w8a8_kernel(out_features=4, in_features=4)
    layer = _make_layer(weight, out_features=4, in_features=4)

    kernel.process_weights_after_loading(layer)

    assert layer.weight.shape == (4, 4)
    assert layer.weight.is_contiguous()
    assert torch.equal(layer.weight, expected)
