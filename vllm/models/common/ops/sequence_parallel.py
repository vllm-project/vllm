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
from vllm.utils.torch_utils import direct_register_custom_op


def _sp_pad_compiled(x: torch.Tensor, tp_size: int, value: float) -> torch.Tensor:
    sp_pad = (-x.shape[0]) % tp_size
    pad = (0, 0) * (x.ndim - 1) + (0, sp_pad)
    return torch.nn.functional.pad(x, pad, value=value)


def _sp_pad_compiled_fake(x: torch.Tensor, tp_size: int, value: float) -> torch.Tensor:
    new_shape = list(x.shape)
    new_shape[0] += (-new_shape[0]) % tp_size
    return torch.empty(new_shape, dtype=x.dtype, device=x.device)


direct_register_custom_op(
    op_name="sp_pad_compiled",
    op_func=_sp_pad_compiled,
    fake_impl=_sp_pad_compiled_fake,
)


def _sp_pad(x: torch.Tensor, tp_size: int, value: float = 0.0) -> torch.Tensor:
    sp_pad = (-x.shape[0]) % tp_size
    if torch.compiler.is_compiling():
        # Keep dynamic padding opaque to Inductor. If aten.constant_pad_nd is
        # compiled inline, its output can incorrectly reuse the unpadded input
        # buffer when the tracing shape needs no padding.
        return torch.ops.vllm.sp_pad_compiled(x, tp_size, value)
    if sp_pad == 0:
        return x
    pad = (0, 0) * (x.ndim - 1) + (0, sp_pad)
    return torch.nn.functional.pad(x, pad, value=value)


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


def sp_restore_outputs(
    tensors: list[torch.Tensor],
    sequence_parallel: list[bool],
    full_num_tokens: int,
) -> list[torch.Tensor]:
    """Restore sequence-sharded outputs with a single packed all-gather."""
    assert len(tensors) == len(sequence_parallel)
    sharded_indices = [
        idx for idx, is_sharded in enumerate(sequence_parallel) if is_sharded
    ]
    if not sharded_indices:
        return tensors

    sharded_tensors = [tensors[idx] for idx in sharded_indices]
    split_sizes = [tensor.shape[-1] for tensor in sharded_tensors]
    packed = sp_all_gather(torch.cat(sharded_tensors, dim=-1))[:full_num_tokens]
    for idx, tensor in zip(
        sharded_indices, packed.split(split_sizes, dim=-1), strict=True
    ):
        tensors[idx] = tensor
    return tensors


def sp_reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2
    tp_size = get_tensor_model_parallel_world_size()
    x = _sp_pad(x, tp_size)
    output = _custom_collective("custom_reduce_scatter", x)
    if output is not None:
        return output
    return tensor_model_parallel_reduce_scatter(x, 0)


def sp_shard(x: torch.Tensor) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    x = _sp_pad(x, tp_size)
    chunk = x.shape[0] // tp_size
    return x[tp_rank * chunk : (tp_rank + 1) * chunk]


def sp_padding_mask(
    is_padding: torch.Tensor | None,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    if is_padding is None:
        is_padding = hidden_states.new_zeros(num_tokens, dtype=torch.bool)
    assert is_padding.shape[0] == num_tokens

    tp_size = get_tensor_model_parallel_world_size()
    is_padding = _sp_pad(is_padding, tp_size, value=1.0)
    chunk = is_padding.shape[0] // tp_size
    tp_rank = get_tensor_model_parallel_rank()
    return is_padding[tp_rank * chunk : (tp_rank + 1) * chunk]
