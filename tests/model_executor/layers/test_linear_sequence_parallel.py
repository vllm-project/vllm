# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm.distributed import communication_op
from vllm.lora.layers.column_parallel_linear import ColumnParallelLinearWithLoRA
from vllm.lora.layers.row_parallel_linear import RowParallelLinearWithLoRA
from vllm.model_executor.layers import linear as linear_module
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.models import utils as model_utils


def _column_layer() -> ColumnParallelLinear:
    layer = object.__new__(ColumnParallelLinear)
    nn.Module.__init__(layer)
    layer.bias = None
    layer.gather_output = False
    layer.quant_method = Mock()
    layer.return_bias = False
    layer.sequence_parallel = True
    layer.skip_bias_add = False
    layer.tp_size = 2
    return layer


def _row_layer() -> RowParallelLinear:
    layer = object.__new__(RowParallelLinear)
    nn.Module.__init__(layer)
    layer.bias = None
    layer.input_is_parallel = True
    layer.quant_method = Mock()
    layer.reduce_results = False
    layer.return_bias = False
    layer.sequence_parallel = True
    layer.skip_bias_add = False
    layer.tp_rank = 0
    layer.tp_size = 2
    return layer


def test_column_parallel_linear_gathers_sequence_shards(monkeypatch):
    local_input = torch.arange(4, dtype=torch.float32).view(2, 2)
    gathered_input = torch.cat([local_input, local_input + 4])
    output_parallel = gathered_input[:, :1]
    all_gather = Mock(return_value=gathered_input)
    monkeypatch.setattr(linear_module, "sequence_parallel_all_gather", all_gather)
    monkeypatch.setattr(linear_module, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        linear_module,
        "get_forward_context",
        lambda: SimpleNamespace(batch_descriptor=SimpleNamespace(num_tokens=4)),
    )
    layer = _column_layer()
    layer.quant_method.apply.return_value = output_parallel

    output = layer(local_input)

    all_gather.assert_called_once_with(local_input)
    layer.quant_method.apply.assert_called_once_with(layer, gathered_input, None)
    torch.testing.assert_close(output, output_parallel)


def test_column_parallel_linear_skips_gather_for_replicated_input(monkeypatch):
    replicated_input = torch.arange(8, dtype=torch.float32).view(4, 2)
    output_parallel = replicated_input[:, :1]
    all_gather = Mock(side_effect=AssertionError("unexpected all-gather"))
    monkeypatch.setattr(linear_module, "sequence_parallel_all_gather", all_gather)
    monkeypatch.setattr(linear_module, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        linear_module,
        "get_forward_context",
        lambda: SimpleNamespace(batch_descriptor=SimpleNamespace(num_tokens=4)),
    )
    layer = _column_layer()
    layer.quant_method.apply.return_value = output_parallel

    output = layer(replicated_input)

    all_gather.assert_not_called()
    layer.quant_method.apply.assert_called_once_with(layer, replicated_input, None)
    torch.testing.assert_close(output, output_parallel)


def test_row_parallel_linear_reduce_scatters_sequence_shards(monkeypatch):
    input_parallel = torch.arange(4, dtype=torch.float32).view(4, 1)
    output_parallel = torch.cat([input_parallel, input_parallel], dim=1)
    local_output = output_parallel[:2]
    reduce_scatter = Mock(return_value=local_output)
    all_reduce = Mock(side_effect=AssertionError("unexpected all-reduce"))
    monkeypatch.setattr(
        linear_module,
        "sequence_parallel_reduce_scatter",
        reduce_scatter,
    )
    monkeypatch.setattr(
        linear_module,
        "tensor_model_parallel_all_reduce",
        all_reduce,
    )
    layer = _row_layer()
    layer.quant_method.apply.return_value = output_parallel

    output = layer(input_parallel)

    reduce_scatter.assert_called_once_with(output_parallel)
    all_reduce.assert_not_called()
    torch.testing.assert_close(output, local_output)


def test_row_parallel_linear_requires_runner_padding(monkeypatch):
    output_parallel = torch.arange(6, dtype=torch.float32).view(3, 2)
    reduce_scatter = Mock()
    monkeypatch.setattr(
        linear_module,
        "sequence_parallel_reduce_scatter",
        reduce_scatter,
    )
    layer = _row_layer()

    with pytest.raises(AssertionError, match="padded by the model runner"):
        layer.reduce_output(output_parallel)

    reduce_scatter.assert_not_called()


def test_sequence_parallel_chunk_requires_runner_padding(monkeypatch):
    monkeypatch.setattr(model_utils, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(model_utils, "get_tensor_model_parallel_rank", lambda: 0)

    with pytest.raises(AssertionError, match="padded by the model runner"):
        model_utils.sequence_parallel_chunk_impl(torch.zeros(3, 2))


def test_sequence_parallel_reduce_scatter_does_not_pad(monkeypatch):
    output_parallel = torch.arange(6, dtype=torch.float32).view(3, 2)
    local_output = output_parallel[:1]
    custom_collective = Mock(return_value=None)
    reduce_scatter = Mock(return_value=local_output)
    monkeypatch.setattr(
        communication_op,
        "_custom_sequence_parallel_collective",
        custom_collective,
    )
    monkeypatch.setattr(
        communication_op,
        "tensor_model_parallel_reduce_scatter",
        reduce_scatter,
    )

    output = communication_op.sequence_parallel_reduce_scatter(output_parallel)

    custom_collective.assert_called_once_with("custom_reduce_scatter", output_parallel)
    reduce_scatter.assert_called_once_with(output_parallel, dim=0)
    torch.testing.assert_close(output, local_output)


def test_row_parallel_linear_keeps_all_reduce_by_default(monkeypatch):
    output_parallel = torch.arange(8, dtype=torch.float32).view(4, 2)
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
        "sequence_parallel_reduce_scatter",
        reduce_scatter,
    )
    layer = _row_layer()
    layer.sequence_parallel = False
    layer.reduce_results = True

    output = layer.reduce_output(output_parallel)

    all_reduce.assert_called_once_with(output_parallel)
    reduce_scatter.assert_not_called()
    torch.testing.assert_close(output, reduced_output)


def test_column_parallel_lora_reuses_base_input_preparation():
    local_input = torch.arange(4, dtype=torch.float32).view(2, 2)
    gathered_input = torch.cat([local_input, local_input + 4])
    output_parallel = gathered_input[:, :1]
    base_layer = _column_layer()
    base_layer.prepare_input = Mock(return_value=gathered_input)
    lora_layer = object.__new__(ColumnParallelLinearWithLoRA)
    nn.Module.__init__(lora_layer)
    lora_layer.base_layer = base_layer
    lora_layer.tp_size = 2
    object.__setattr__(lora_layer, "apply", Mock(return_value=output_parallel))

    output = lora_layer(local_input)

    base_layer.prepare_input.assert_called_once_with(local_input)
    lora_layer.apply.assert_called_once_with(gathered_input, None)
    torch.testing.assert_close(output, output_parallel)


def test_row_parallel_lora_reuses_base_output_reduction():
    input_parallel = torch.arange(4, dtype=torch.float32).view(4, 1)
    output_parallel = torch.cat([input_parallel, input_parallel], dim=1)
    local_output = output_parallel[:2]
    base_layer = _row_layer()
    base_layer.reduce_output = Mock(return_value=local_output)
    lora_layer = object.__new__(RowParallelLinearWithLoRA)
    nn.Module.__init__(lora_layer)
    lora_layer.base_layer = base_layer
    lora_layer.tp_rank = 0
    lora_layer.tp_size = 2
    object.__setattr__(lora_layer, "apply", Mock(return_value=output_parallel))

    output = lora_layer(input_parallel)

    base_layer.reduce_output.assert_called_once_with(output_parallel)
    torch.testing.assert_close(output, local_output)
