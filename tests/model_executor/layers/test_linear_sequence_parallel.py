# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
import torch

from vllm.lora.layers.column_parallel_linear import ColumnParallelLinearWithLoRA
from vllm.lora.layers.row_parallel_linear import RowParallelLinearWithLoRA
from vllm.model_executor import parameter as parameter_module
from vllm.model_executor.layers import linear as linear_module
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)


def _fill_weights(layer: torch.nn.Module) -> None:
    with torch.no_grad():
        for param in layer.parameters():
            param.copy_(torch.arange(param.numel(), dtype=torch.float32).view_as(param))


@pytest.fixture(autouse=True)
def _tp_size_two(monkeypatch):
    """Fake a two-rank TP group.

    ``RowParallelLinear`` reads the TP size/rank from the distributed state
    at construction time, so the layer constructors below would otherwise
    fall back to tp_size == 1 and never exercise the sequence-parallel paths.
    """
    monkeypatch.setattr(
        linear_module,
        "get_tensor_model_parallel_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(
        linear_module,
        "get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        parameter_module,
        "get_tensor_model_parallel_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(
        parameter_module,
        "get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        linear_module,
        "dispatch_unquantized_gemm",
        lambda: lambda layer, x, weight, bias: torch.nn.functional.linear(
            x, weight, bias
        ),
    )


def _column_layer(*, sequence_parallel: bool = True) -> ColumnParallelLinear:
    return ColumnParallelLinear(
        input_size=4,
        output_size=8,
        bias=False,
        quant_config=None,
        sequence_parallel=sequence_parallel,
        tp_size=2,
        tp_rank=0,
        return_bias=False,
    )


def _row_layer(*, sequence_parallel: bool = True) -> RowParallelLinear:
    return RowParallelLinear(
        input_size=4,
        output_size=8,
        bias=False,
        quant_config=None,
        sequence_parallel=sequence_parallel,
        return_bias=False,
    )


def test_column_parallel_linear_gathers_sequence_shards(monkeypatch) -> None:
    local_input = torch.arange(8, dtype=torch.float32).view(2, 4)
    gathered_input = torch.cat([local_input, local_input + 8], dim=0)
    all_gather = Mock(return_value=gathered_input)
    monkeypatch.setattr(linear_module, "tensor_model_parallel_all_gather", all_gather)

    layer = _column_layer()
    _fill_weights(layer)

    output = layer(local_input)

    all_gather.assert_called_once_with(local_input, dim=0)
    torch.testing.assert_close(output, gathered_input @ layer.weight.T)


def test_column_parallel_linear_removes_sequence_padding(monkeypatch) -> None:
    local_input = torch.arange(8, dtype=torch.float32).view(2, 4)
    gathered_input = torch.cat([local_input, local_input + 8], dim=0)
    monkeypatch.setattr(
        linear_module,
        "tensor_model_parallel_all_gather",
        Mock(return_value=gathered_input),
    )
    layer = _column_layer()
    _fill_weights(layer)

    output = layer(local_input, sequence_parallel_unpadded_size=3)

    torch.testing.assert_close(output, gathered_input[:3] @ layer.weight.T)


def test_column_parallel_linear_keeps_shards_without_sequence_parallel(
    monkeypatch,
) -> None:
    all_gather = Mock(side_effect=AssertionError("unexpected all-gather"))
    monkeypatch.setattr(linear_module, "tensor_model_parallel_all_gather", all_gather)

    layer = _column_layer(sequence_parallel=False)
    _fill_weights(layer)
    local_input = torch.arange(8, dtype=torch.float32).view(2, 4)

    output = layer(local_input)

    all_gather.assert_not_called()
    torch.testing.assert_close(output, local_input @ layer.weight.T)


def test_row_parallel_linear_reduce_scatters_sequence_shards(monkeypatch) -> None:
    input_parallel = torch.arange(6, dtype=torch.float32).view(3, 2)
    layer = _row_layer()
    _fill_weights(layer)
    output_parallel = input_parallel @ layer.weight.T
    padded_output = torch.nn.functional.pad(output_parallel, (0, 0, 0, 1))
    local_output = padded_output[:2]
    reduce_scatter = Mock(return_value=local_output)
    all_reduce = Mock(side_effect=AssertionError("unexpected all-reduce"))
    monkeypatch.setattr(
        linear_module,
        "tensor_model_parallel_reduce_scatter",
        reduce_scatter,
    )
    monkeypatch.setattr(
        linear_module,
        "tensor_model_parallel_all_reduce",
        all_reduce,
    )

    output = layer(input_parallel)

    reduce_scatter.assert_called_once()
    reduce_scatter_input = reduce_scatter.call_args.args[0]
    assert reduce_scatter.call_args.kwargs == {"dim": 0}
    torch.testing.assert_close(reduce_scatter_input, padded_output)
    all_reduce.assert_not_called()
    torch.testing.assert_close(output, local_output)


def test_row_parallel_linear_keeps_all_reduce_by_default(monkeypatch) -> None:
    output_parallel = torch.arange(16, dtype=torch.float32).view(2, 8)
    reduced_output = output_parallel + 1
    all_reduce = Mock(return_value=reduced_output)
    reduce_scatter = Mock(side_effect=AssertionError("unexpected reduce-scatter"))
    monkeypatch.setattr(
        linear_module,
        "tensor_model_parallel_all_reduce",
        all_reduce,
    )
    monkeypatch.setattr(
        linear_module,
        "tensor_model_parallel_reduce_scatter",
        reduce_scatter,
    )

    layer = _row_layer(sequence_parallel=False)

    output = layer.reduce_output(output_parallel)

    all_reduce.assert_called_once_with(output_parallel)
    reduce_scatter.assert_not_called()
    torch.testing.assert_close(output, reduced_output)


def test_column_parallel_lora_reuses_base_input_preparation() -> None:
    local_input = torch.arange(8, dtype=torch.float32).view(2, 4)
    gathered_input = torch.cat([local_input, local_input + 8], dim=0)
    base_layer = _column_layer()
    base_layer.prepare_input = Mock(return_value=gathered_input)
    lora_layer = object.__new__(ColumnParallelLinearWithLoRA)
    torch.nn.Module.__init__(lora_layer)
    lora_layer.base_layer = base_layer
    lora_layer.tp_size = 2
    object.__setattr__(lora_layer, "apply", Mock(return_value=torch.zeros(4, 4)))

    lora_layer(local_input, sequence_parallel_unpadded_size=3)

    base_layer.prepare_input.assert_called_once_with(local_input, 3)


def test_row_parallel_lora_reuses_base_output_reduction() -> None:
    input_parallel = torch.arange(8, dtype=torch.float32).view(4, 2)
    output_parallel = torch.cat([input_parallel, input_parallel], dim=1)
    local_output = output_parallel[:2]
    base_layer = _row_layer()
    base_layer.reduce_output = Mock(return_value=local_output)
    lora_layer = object.__new__(RowParallelLinearWithLoRA)
    torch.nn.Module.__init__(lora_layer)
    lora_layer.base_layer = base_layer
    lora_layer.tp_rank = 0
    lora_layer.tp_size = 2
    object.__setattr__(lora_layer, "apply", Mock(return_value=output_parallel))

    output = lora_layer(input_parallel)

    base_layer.reduce_output.assert_called_once_with(output_parallel)
    torch.testing.assert_close(output, local_output)
