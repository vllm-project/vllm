# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import direct_register_custom_op


def _custom_collective(name: str, x: torch.Tensor) -> torch.Tensor | None:
    device_communicator = get_tp_group().device_communicator
    if device_communicator is None:
        return None
    collective = getattr(device_communicator, name, None)
    return None if collective is None else collective(x)


def sp_all_gather(x: torch.Tensor) -> torch.Tensor:
    output = _custom_collective("custom_all_gather", x)
    if output is not None:
        return output
    return tensor_model_parallel_all_gather(x, 0)


def _sp_reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2
    tp_size = get_tensor_model_parallel_world_size()
    sp_pad = (-x.shape[0]) % tp_size
    if sp_pad > 0:
        x = torch.nn.functional.pad(x, (0, 0, 0, sp_pad))
    output = _custom_collective("custom_reduce_scatter", x)
    if output is not None:
        return output
    return tensor_model_parallel_reduce_scatter(x, 0)


def _sp_shard(x: torch.Tensor) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    sp_pad = (-x.shape[0]) % tp_size
    if sp_pad > 0:
        pad = (0, 0) * (x.ndim - 1) + (0, sp_pad)
        x = torch.nn.functional.pad(x, pad)
    chunk = x.shape[0] // tp_size
    return x[tp_rank * chunk : (tp_rank + 1) * chunk]


def _sp_shard_op(x: torch.Tensor) -> torch.Tensor:
    output = _sp_shard(x)
    # A functional custom op must not return a view of its input.
    unpadded = x.shape[0] % get_tensor_model_parallel_world_size() == 0
    return output.clone() if unpadded else output


def _sp_local_fake(x: torch.Tensor) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    return x.new_empty((cdiv(x.shape[0], tp_size), *x.shape[1:]))


# Dynamo would specialize the padding branch on the traced token count, and
# vLLM drops the resulting guards. Keep the branch opaque when compiling.
direct_register_custom_op(
    op_name="sp_reduce_scatter",
    op_func=_sp_reduce_scatter,
    fake_impl=_sp_local_fake,
)
direct_register_custom_op(
    op_name="sp_shard",
    op_func=_sp_shard_op,
    fake_impl=_sp_local_fake,
)


def sp_reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return torch.ops.vllm.sp_reduce_scatter(x)
    return _sp_reduce_scatter(x)


def sp_shard(x: torch.Tensor) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return torch.ops.vllm.sp_shard(x)
    return _sp_shard(x)


def sp_padding_mask(
    is_padding: torch.Tensor | None,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    if is_padding is None:
        is_padding = hidden_states.new_zeros(num_tokens, dtype=torch.bool)
    assert is_padding.shape[0] == num_tokens

    tp_size = get_tensor_model_parallel_world_size()
    sp_pad = (-num_tokens) % tp_size
    if sp_pad > 0:
        is_padding = torch.nn.functional.pad(is_padding, (0, sp_pad), value=True)
    chunk = is_padding.shape[0] // tp_size
    tp_rank = get_tensor_model_parallel_rank()
    return is_padding[tp_rank * chunk : (tp_rank + 1) * chunk]
