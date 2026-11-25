# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import lru_cache
from typing import NamedTuple

import torch

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.utils.math_utils import cdiv


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
    """Whether a batch of `num_tokens` shards the stream along the sequence.

    Below one token per rank the shards are mostly padding, so the batch
    stays replicated. A model may raise this lower bound with `min_tokens`.
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
    """Whether `sp_all_gather_matmul` applies to this shard and weight."""
    return (
        x.shape[0] >= _FUSED_AG_MATMUL_MIN_TOKENS_PER_RANK
        and x.dtype == weight.dtype == torch.bfloat16
        and weight.ndim == 2
        and not torch.cuda.is_current_stream_capturing()
        and _has_fused_all_gather_matmul()
    )


def sp_all_gather_matmul(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """`sp_all_gather(x) @ weight.T`, overlapping the gather with the GEMM."""
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


def zigzag_chunk_size(num_tokens: int, world_size: int) -> int:
    """Rows per chunk when `num_tokens` rows split into `2 * world_size`
    zigzag chunks."""
    return cdiv(num_tokens, 2 * world_size)


class ZigzagShards(NamedTuple):
    """A batch's rows split zigzag over the TP group.

    The rows, padded to `2 * world` equal chunks, go two chunks per rank: rank
    `r` owns chunks `r` and `2 * world - 1 - r`, so every rank holds as many
    rows and about as much causal attention work:

        chunks:  | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
        rank 0:    0                           7
        rank 1:        1                   6
        rank 2:            2           5
        rank 3:                3   4

    An all-gather of the ranks' rows is rank-major: rank 0's rows, then rank
    1's, and so on.
    """

    local_rows: torch.Tensor
    """This rank's rows, early chunk first, with padding clamped to the last
    row."""
    gather_index: torch.Tensor
    """Each row's index in the rank-major all-gather."""
    rank_major_rows: torch.Tensor
    """The row each slot of the rank-major all-gather holds, padding
    included."""


def zigzag_shards(num_tokens: int, device: torch.device) -> ZigzagShards:
    world = get_tensor_model_parallel_world_size()
    rank = get_tensor_model_parallel_rank()
    chunk = zigzag_chunk_size(num_tokens, world)
    local_size = 2 * chunk
    # Index arithmetic only (no host copies), so CUDA graphs can capture it.
    slots = torch.arange(world * local_size, device=device)
    owner = slots // local_size
    offset = slots % local_size
    chunk_ids = torch.where(offset >= chunk, 2 * world - 1 - owner, owner)
    rank_major_rows = chunk_ids * chunk + offset % chunk
    rows = torch.arange(num_tokens, device=device)
    chunk_ids = rows // chunk
    late = chunk_ids >= world
    owner = torch.where(late, 2 * world - 1 - chunk_ids, chunk_ids)
    gather_index = owner * local_size + late * chunk + rows % chunk
    local_rows = rank_major_rows[rank * local_size : (rank + 1) * local_size]
    return ZigzagShards(
        local_rows.clamp(max=num_tokens - 1), gather_index, rank_major_rows
    )


def zigzag_shard(x: torch.Tensor, shards: ZigzagShards) -> torch.Tensor:
    return x.index_select(0, shards.local_rows)


def zigzag_all_gather(x: torch.Tensor, shards: ZigzagShards) -> torch.Tensor:
    """Every rank's rows, back in batch order."""
    return sp_all_gather(x).index_select(0, shards.gather_index)


def zigzag_padding_mask(
    is_padding: torch.Tensor | None, num_tokens: int, shards: ZigzagShards
) -> torch.Tensor:
    """`is_padding` over the rank-major all-gather; the rows that pad the
    batch to whole chunks are padding too."""
    if is_padding is None:
        is_padding = torch.zeros(
            num_tokens, dtype=torch.bool, device=shards.gather_index.device
        )
    assert is_padding.shape[0] == num_tokens
    sp_pad = shards.rank_major_rows.shape[0] - num_tokens
    is_padding = torch.nn.functional.pad(is_padding, (0, sp_pad), value=True)
    return is_padding[shards.rank_major_rows]
