# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.linear import MergedColumnParallelLinear


def test_load_weights_reports_missing_parameter() -> None:
    layer = object.__new__(MergedColumnParallelLinear)
    nn.Module.__init__(layer)
    layer.add_module("projection", nn.Module())

    weights = [("projection.weight_scale", torch.ones(1))]

    with pytest.raises(
        ValueError,
        match=(
            r"Cannot load weight 'projection\.weight_scale': expected a "
            r"torch\.nn\.Parameter, but found MergedColumnParallelLinear\."
        ),
    ):
        list(layer.load_weights(weights))
