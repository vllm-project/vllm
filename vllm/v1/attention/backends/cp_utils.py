# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch
from torch import distributed as dist

from vllm.distributed.parallel_state import (get_context_parallel_rank,
                                             get_context_parallel_world_size,
                                             get_cp_group)
from vllm.logger import init_logger
from vllm.v1.worker.block_table import MultiGroupBlockTable
from vllm.v1.worker.gpu_input_batch import CachedRequestState

logger = init_logger(__name__)


def cp_get_shard_size(num_tokens: int) -> tuple[int, int]:
    # Get the number of tokens in a given CP shard.
    cp_num_shards = 2 * get_context_parallel_world_size()
    num_pad_tokens = -num_tokens % cp_num_shards
    num_padded_tokens = num_tokens + num_pad_tokens
    cp_shard_size = num_padded_tokens // cp_num_shards
    return cp_shard_size, num_pad_tokens


def _cp_shard_positions_for_prefill(
    cp_size: int,
    cp_rank: int,
    positions_np: np.ndarray,
    arange_np: np.ndarray,
    num_prefill_tokens: int,
    seq_offset: int,
    padding_position: int = -1,
) -> list[int]:
    """
    Compute token positions and seq lengths for context parallel (CP) shards.

    Args:
        cp_size (int): CP world size.
        cp_rank (int): This CP rank.
        positions_np (np.ndarray): Output positions.
        arange_np (np.ndarray): Sequential indices.
        num_prefill_tokens (int): Tokens to prefill.
        seq_offset (int): Position offset.
        padding_position (int): Padding value (default: -1).

    Returns:
        list[int]: Sequence lengths per shard.
    """
    cp_shard_size, num_pad_tokens_all = cp_get_shard_size(num_prefill_tokens)
    # Compute the token index ranges for the two shards handled by this rank
    chunk0_start = cp_rank * cp_shard_size
    chunk1_start = (2 * cp_size - cp_rank - 1) * cp_shard_size
    chunk0_arange = arange_np[chunk0_start:chunk0_start + cp_shard_size]
    chunk1_arange = arange_np[chunk1_start:chunk1_start + cp_shard_size]

    if num_pad_tokens_all == 0:
        positions_np[:cp_shard_size] = chunk0_arange + seq_offset
        positions_np[cp_shard_size:2 *
                     cp_shard_size] = chunk1_arange + seq_offset
        return [cp_shard_size, cp_shard_size]

    last_token_idx = num_prefill_tokens - 1

    if cp_shard_size >= num_pad_tokens_all and cp_rank == 0:
        # Special case: padding only in second shard of rank 0
        num_real_tokens = cp_shard_size - num_pad_tokens_all
        positions_np[:cp_shard_size] = chunk0_arange + seq_offset
        positions_np[cp_shard_size:cp_shard_size +
                     num_real_tokens] = (chunk1_arange[:num_real_tokens] +
                                         seq_offset)
        positions_np[cp_shard_size + num_real_tokens:2 *
                     cp_shard_size] = padding_position
    else:
        # General padding case
        _fill_chunk_positions(positions_np, chunk0_arange, 0, cp_shard_size,
                              seq_offset, last_token_idx, padding_position)
        _fill_chunk_positions(positions_np, chunk1_arange, cp_shard_size,
                              cp_shard_size, seq_offset, last_token_idx,
                              padding_position)

    return [cp_shard_size, cp_shard_size]


def _fill_chunk_positions(positions_np: np.ndarray, chunk_arange: np.ndarray,
                          start_idx: int, chunk_size: int, seq_offset: int,
                          last_token_idx: int, padding_position: int) -> None:
    if len(chunk_arange) == 0 or chunk_arange[0] > last_token_idx:
        # All positions should be padding
        positions_np[start_idx:start_idx + chunk_size] = padding_position
    elif chunk_arange[-1] <= last_token_idx:
        # All positions are valid
        positions_np[start_idx:start_idx +
                     chunk_size] = chunk_arange + seq_offset
    else:
        # Partial chunk - some valid, some padding
        num_valid = last_token_idx - int(chunk_arange[0]) + 1
        positions_np[start_idx:start_idx +
                     num_valid] = chunk_arange[:num_valid] + seq_offset
        positions_np[start_idx + num_valid:start_idx +
                     chunk_size] = padding_position


def _cp_get_computed_positions(cp_size, cp_rank,
                               computed_positions_np: np.ndarray,
                               arange_np: np.ndarray,
                               num_computed_tokens: list[int],
                               padding_position: int) -> int:
    """
    Get computed token positions for a CP rank.

    Example:
      If CP world size=2, num_computed_tokens=[0,4,8,9,10,11]:
        - CP rank 0: [0,3,4,7,8,10]
        - CP rank 1: [1,2,5,6,9]
    """
    computed_chunk_sizes = np.diff(num_computed_tokens)
    if computed_chunk_sizes.size == 0:
        return 0

    num_local_computed_tokens = 0
    for idx, chunk_size in enumerate(computed_chunk_sizes):
        if chunk_size > 1:
            # For prefill chunks
            seqlens = _cp_shard_positions_for_prefill(
                cp_size,
                cp_rank,
                computed_positions_np[num_local_computed_tokens:],
                arange_np,
                chunk_size,
                num_computed_tokens[idx],
                padding_position,
            )
            # Update the count of local tokens processed
            num_local_computed_tokens += sum(seqlens)
        else:
            # For single token decode, use simple round-robin distribution
            # Only process tokens assigned to this rank
            if num_computed_tokens[idx] % cp_size == cp_rank:
                computed_positions_np[
                    num_local_computed_tokens] = num_computed_tokens[idx]
                num_local_computed_tokens += 1

    return num_local_computed_tokens


def prepare_inputs_for_cp(
    num_scheduled_tokens: dict[str, int],
    requests: dict[str, CachedRequestState],
    req_ids: list[str],
    block_table: MultiGroupBlockTable,
    positions_np: np.ndarray,
    computed_positions_np: np.ndarray,
    arange_np: np.ndarray,
    padding_loc: int,
) -> tuple[list[int], list[int], list[list[int]]]:
    """
    Prepare inputs for context parallelism (CP).

    This method handles the distribution of tokens across context
    parallel ranks, computing local token counts and positions for
    both scheduled and computed tokens. It processes each request to
    determine how many tokens each CP rank should handle and calculates
    the appropriate sequence lengths for attention computation.

    Args:
        num_scheduled_tokens: Dictionary mapping request IDs to number
            of scheduled tokens per request
        requests: Dictionary mapping request IDs to their cached
            request states
        req_ids: List of request IDs to process
        block_table: Multi-group block table for managing KV cache
            slot mappings
        positions_np: NumPy array to store position indices for
            scheduled tokens
        computed_positions_np: NumPy array to store position indices
            for computed tokens
        arange_np: NumPy array containing sequential indices used for
            token positioning
        padding_loc: Integer value used for padding positions when
            sharding tokens

    Returns:
        tuple containing:
        - num_local_scheduled_tokens: Number of scheduled tokens per
            request on this CP rank
        - num_local_computed_tokens: Number of computed tokens per
            request on this CP rank
        - q_seqlens_sharded: Query sequence lengths for each request
            (list of [1st_shard_size, 2nd_shard_size] for prefill
            requests, or [1] for decode requests)
    """
    cp_size = get_context_parallel_world_size()
    cp_rank = get_context_parallel_rank()

    q_seqlens_sharded = []
    num_scheduled_tokens_local = [0] * len(req_ids)
    num_computed_tokens_local = [0] * len(req_ids)
    total_num_local_scheduled_tokens = 0

    for idx, req_id in enumerate(req_ids):
        req_state = requests[req_id]

        # Calculate how many computed tokens this CP rank should handle
        # for this request
        num_computed_tokens_local[idx] = _cp_get_computed_positions(
            cp_size,
            cp_rank,
            computed_positions_np[sum(num_computed_tokens_local[:idx]):],
            arange_np,
            req_state.num_computed_tokens,
            padding_loc,
        )

        # Set up slot mapping for computed tokens if any exist. For
        # context parallel, we do not need to track the absolute
        # position of each token in the block table; preserving the
        # correct relative ordering is sufficient for correct mapping.
        # It also saves KV cache space by avoiding unnecessary
        # allocation for absolute positions.

        if num_computed_tokens_local[idx] != 0:
            start_offset = sum(num_computed_tokens_local[:idx])
            computed_req_indices = np.full(num_computed_tokens_local[idx],
                                           idx,
                                           dtype=np.int32)
            cp_computed_positions = arange_np[start_offset:start_offset +
                                              num_computed_tokens_local[idx]]
            block_table.compute_slot_mapping(computed_req_indices,
                                             cp_computed_positions,
                                             offset=start_offset)

        num_scheduled_tokens_per_req = num_scheduled_tokens[req_id]
        if num_scheduled_tokens_per_req > 1:
            # Prefill case: shard the prefill tokens across CP ranks
            seqlens = _cp_shard_positions_for_prefill(
                cp_size,
                cp_rank,
                positions_np[total_num_local_scheduled_tokens:],
                arange_np,
                num_scheduled_tokens_per_req,
                req_state.num_computed_tokens[-1],
                padding_loc,
            )
            assert len(seqlens) == 2
            q_seqlens_sharded.append(seqlens)
            num_scheduled_tokens_local[idx] = sum(seqlens)
        else:
            # Decode case: each rank processes 1 token
            positions_np[total_num_local_scheduled_tokens] = (
                req_state.num_computed_tokens[-1])
            num_scheduled_tokens_local[idx] = 1
            q_seqlens_sharded.append([1])

        total_num_local_scheduled_tokens += num_scheduled_tokens_local[idx]

    return num_scheduled_tokens_local, num_computed_tokens_local, q_seqlens_sharded


def cp_get_neighbor_ranks() -> tuple[int, int]:
    return (get_cp_group().prev_rank, get_cp_group().next_rank)


def cp_pass_around(
        tensors: list[torch.Tensor], to_rank: int, from_rank: int
) -> tuple[list[torch.Tensor], list[torch.distributed.Work]]:
    """
    Passes a list of tensors to designated to_rank, and receives the same
    number of tensors with the same sizes from designated from_rank.
    Note: to_rank and from_rank are the ranks in default PG rather than
    context parallel pg. All ranks in a CP group must call this function
    together, which results in passing the same tensors in a circular way
    across all ranks in a CP group.

    Args:
        tensors: list of tensors to be passed around in the CP group
        to_rank: rank to pass my tensors to
        from_rank: rank to receive tensors from

    Returns:
        dests: list of tensors received from from_rank
        reqs: list of P2POp requests to wait for to complete receiving
            from from_rank
    """
    dests = []
    p2p_ops = []
    for x in tensors:
        x = x.contiguous()
        dest = torch.empty_like(x)
        dests.append(dest)
        p2p_ops += [
            dist.P2POp(dist.isend,
                       x,
                       to_rank,
                       group=get_cp_group().device_group),
            dist.P2POp(dist.irecv,
                       dest,
                       from_rank,
                       group=get_cp_group().device_group),
        ]
    return dests, dist.batch_isend_irecv(p2p_ops)
