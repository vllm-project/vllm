# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import lru_cache

import torch

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)


def _custom_collective(name: str, x: torch.Tensor) -> torch.Tensor | None:
    device_communicator = get_tp_group().device_communicator
    if device_communicator is None:
        return None
    collective = getattr(device_communicator, name, None)
    return None if collective is None else collective(x)


@lru_cache
def is_sequence_parallel_active(
    is_enabled: bool, num_tokens: int, min_tokens: int = 0
) -> bool:
    """Whether a batch of ``num_tokens`` shards the stream along the sequence.

    Below one token per rank the shards are mostly padding, so the batch
    stays replicated. A model may raise this lower bound with ``min_tokens``.
    """
    return is_enabled and num_tokens >= max(
        get_tensor_model_parallel_world_size(), min_tokens
    )


def sp_all_gather(x: torch.Tensor) -> torch.Tensor:
    output = _custom_collective("custom_all_gather", x)
    if output is not None:
        return output
    return tensor_model_parallel_all_gather(x, 0)


# Below this many tokens per rank the pipelined copy-engine all-gather costs
# more than it hides behind the GEMM (B300 TP4, K=8192).
_FUSED_AG_MATMUL_MIN_TOKENS_PER_RANK = 1024


@lru_cache
def _has_fused_all_gather_matmul() -> bool:
    try:
        import torch.distributed._symmetric_memory  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch.ops.symm_mem, "fused_all_gather_matmul")


def can_sp_all_gather_matmul(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether ``sp_all_gather_matmul`` applies to this shard and weight."""
    return (
        x.shape[0] >= _FUSED_AG_MATMUL_MIN_TOKENS_PER_RANK
        and x.dtype == weight.dtype == torch.bfloat16
        and weight.ndim == 2
        and not torch.cuda.is_current_stream_capturing()
        and _has_fused_all_gather_matmul()
    )


def sp_all_gather_matmul(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``sp_all_gather(x) @ weight.T``, overlapping the gather with the GEMM."""
    _, outputs = torch.ops.symm_mem.fused_all_gather_matmul(
        x.contiguous(),
        [weight.t()],
        gather_dim=0,
        group_name=get_tp_group().device_group.group_name,
    )
    return outputs[0]


def sp_reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2
    tp_size = get_tensor_model_parallel_world_size()
    sp_pad = (-x.shape[0]) % tp_size
    if sp_pad > 0:
        x = torch.nn.functional.pad(x, (0, 0, 0, sp_pad))
    output = _custom_collective("custom_reduce_scatter", x)
    if output is not None:
        return output
    return tensor_model_parallel_reduce_scatter(x, 0)


def sp_shard(x: torch.Tensor) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    sp_pad = (-x.shape[0]) % tp_size
    if sp_pad > 0:
        pad = (0, 0) * (x.ndim - 1) + (0, sp_pad)
        x = torch.nn.functional.pad(x, pad)
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
    sp_pad = (-num_tokens) % tp_size
    if sp_pad > 0:
        is_padding = torch.nn.functional.pad(is_padding, (0, sp_pad), value=True)
    chunk = is_padding.shape[0] // tp_size
    tp_rank = get_tensor_model_parallel_rank()
    return is_padding[tp_rank * chunk : (tp_rank + 1) * chunk]
