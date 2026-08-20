import pytest
import torch

from vllm.model_executor.layers.linear import ColumnParallelLinear


def test_load_weights_missing_param_raises():
    layer = ColumnParallelLinear(4, 4, bias=True, params_dtype=torch.float32)
    weights = [("nonexistent", torch.tensor(1.0))]
    with pytest.raises(ValueError) as excinfo:
        list(layer.load_weights(weights))
    assert "cannot load 'nonexistent'" in str(excinfo.value)
