# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.layers.linear import MergedColumnParallelLinear


class DummyLayer(nn.Module):
    """A dummy layer to test weight loading without distributed GPU state."""

    def __init__(self):
        super().__init__()
        self.prefix = "dummy_layer"

    def validate_shard_id(self, shard_id: int | None) -> None:
        pass


def test_linear_missing_parameter_raises_clear_error():
    """Verify loading an undeclared weight triggers ValueError."""
    layer = DummyLayer()

    # Bind the actual vLLM method to our dummy layer to test its logic safely
    layer.load_weights = MergedColumnParallelLinear.load_weights.__get__(layer)

    # Simulate a checkpoint carrying an unexpected tensor
    fake_weights = [("unexpected_scale", torch.tensor([1.0, 2.0]))]

    with pytest.raises(ValueError, match="no such parameter, got DummyLayer instead"):
        list(layer.load_weights(fake_weights))
