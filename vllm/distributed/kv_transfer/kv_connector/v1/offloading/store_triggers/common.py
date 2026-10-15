# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, NamedTuple

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import SlidingWindowSpec
from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        GroupOffloadConfig,
        RequestOffloadState,
        SchedulerOffloadConfig,
    )


class StoreCandidates(NamedTuple):
    """The computed chunks of a request that a trigger can store."""

    num_offloadable_tokens: int
    keys: list[OffloadKey]
    # (config_idx, chunk_idx) of each key
    chunk_positions: list[tuple[int, int]]
    # (start_chunk_idx, end_chunk_idx) of each group
    group_ranges: list[tuple[int, int]]


def select_store_candidates(
    config: "SchedulerOffloadConfig",
    scheduler_output: SchedulerOutput,
    req_status: "RequestOffloadState",
    start_chunk_idxs: Sequence[int],
) -> StoreCandidates:
    """Select the computed chunks of a request, from the given chunk of each group.

    Args:
        config: the scheduler offloading config.
        scheduler_output: the output of the current scheduler step.
        req_status: the request.
        start_chunk_idxs: for each group, the first chunk that the trigger did
            not store yet.

    Returns:
        The keys and positions of the reachable chunks, and the chunk range of
        each group.

    """
    blocks_per_chunk = config.blocks_per_chunk
    req = req_status.req

    if req.status is RequestStatus.FINISHED_ABORTED:
        num_tokens_after_batch = req.num_computed_tokens
    elif req.is_finished():
        # Clamp to the GPU prefix cache's commit point. The final sampled
        # token's slot is never committed (under spec decode it holds a
        # rejected draft's KV), so a block ending there must not be stored.
        num_tokens_after_batch = max(req.num_prompt_tokens, req.num_tokens - 1)
    else:
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req.request_id]
        num_tokens_after_batch = req.num_computed_tokens + num_scheduled_tokens

    num_offloadable_tokens = req_status.num_offloadable_tokens(num_tokens_after_batch)
    prompt_offloadable_tokens = req_status.num_offloadable_tokens(req.num_prompt_tokens)

    # Filter out chunks skipped due to sliding window attention / SSM
    # or unreachable by the load path's alignment constraints.
    new_offload_keys: list[OffloadKey] = []
    chunk_positions: list[tuple[int, int]] = []
    group_store_ranges: list[tuple[int, int]] = []

    reachable_boundaries: tuple[int, ...] = ()
    if config.retention_interval is not None:
        reachable_boundaries = (req.num_prompt_tokens - 1,)
        if req.shared_prefix_boundary:
            reachable_boundaries += (req.shared_prefix_boundary,)

    for config_idx, (group_config, group_state, stored_chunk_idx) in enumerate(
        zip(config.kv_group_configs, req_status.group_states, start_chunk_idxs)
    ):
        num_chunks = req_status.storable_chunks(
            group_config, group_state, num_offloadable_tokens
        )

        start_chunk_idx = stored_chunk_idx
        prompt_horizon_chunks = (
            prompt_offloadable_tokens // group_config.tokens_per_chunk
        )
        final_swa_alignment_blocks = _final_swa_alignment_blocks(config, group_config)
        reconsider_final_swa_tail = (
            req.is_finished()
            and num_chunks != prompt_horizon_chunks
            and config.retention_interval is None
            and final_swa_alignment_blocks is not None
        )
        if config.retention_interval is not None or group_config.is_eagle_group:
            store_horizon_chunks = None
        elif req.is_finished():
            store_horizon_chunks = num_chunks
        elif num_chunks <= prompt_horizon_chunks:
            store_horizon_chunks = prompt_horizon_chunks
        else:
            # An active decode frontier is not a final request
            # boundary. Only fixed alignment tails are reachable.
            store_horizon_chunks = None
        if reconsider_final_swa_tail:
            assert final_swa_alignment_blocks is not None
            horizon_blocks = num_chunks * blocks_per_chunk
            partial_segment_start_block = (
                horizon_blocks - horizon_blocks % final_swa_alignment_blocks
            )
            partial_segment_start_chunk = (
                partial_segment_start_block // blocks_per_chunk
            )
            start_chunk_idx = min(start_chunk_idx, partial_segment_start_chunk)
        group_store_ranges.append((start_chunk_idx, num_chunks))

        if group_config.requires_cow_source:
            continue

        if num_chunks <= start_chunk_idx:
            continue
        offload_keys = group_state.offload_keys[start_chunk_idx:num_chunks]
        # For each chunk, take the last corresponding GPU block. For
        # blocks_per_chunk=3 and GPU block IDs 1 5 6 7 2 4 9 3 8,
        # this selects GPU blocks 6 4 8.
        # A block_id of 0 means either a sliding window / SSM skip
        # or a stale entry that was zeroed out — skip it either way.
        offload_block_ids = group_state.block_ids[
            start_chunk_idx * blocks_per_chunk + blocks_per_chunk - 1 : num_chunks
            * blocks_per_chunk : blocks_per_chunk
        ]
        assert len(offload_keys) == len(offload_block_ids)

        # Use reachable_block_mask to filter unreachable chunks
        # (SWA/Mamba sparsity + retention interval).
        # reachable_block_mask operates in KV-block coordinates,
        # so convert chunk indices to block indices.
        block_mask = _reachable_store_block_mask(
            config=config,
            group_config=group_config,
            start_chunk_idx=start_chunk_idx,
            end_chunk_idx=num_chunks,
            final_segment_end_chunk_idx=store_horizon_chunks,
            reachable_boundaries=reachable_boundaries,
        )

        prompt_horizon_block_mask: list[bool] | None = None
        if reconsider_final_swa_tail:
            prompt_horizon_block_mask = _reachable_store_block_mask(
                config=config,
                group_config=group_config,
                start_chunk_idx=start_chunk_idx,
                end_chunk_idx=num_chunks,
                final_segment_end_chunk_idx=prompt_horizon_chunks,
                reachable_boundaries=reachable_boundaries,
            )

        for key_idx, (offload_key, block_id) in enumerate(
            zip(offload_keys, offload_block_ids)
        ):
            if block_id == 0:
                continue
            abs_chunk_idx = start_chunk_idx + key_idx
            if (
                reconsider_final_swa_tail
                and abs_chunk_idx < stored_chunk_idx
                and (
                    prompt_horizon_block_mask is None
                    or any(
                        prompt_horizon_block_mask[
                            key_idx * blocks_per_chunk + block_idx
                        ]
                        for block_idx in range(blocks_per_chunk)
                    )
                )
            ):
                continue
            # A chunk is reachable if any of its constituent
            # blocks is reachable.
            if block_mask is not None and not any(
                block_mask[key_idx * blocks_per_chunk + b]
                for b in range(blocks_per_chunk)
            ):
                continue
            new_offload_keys.append(offload_key)
            chunk_positions.append((config_idx, abs_chunk_idx))

    return StoreCandidates(
        num_offloadable_tokens, new_offload_keys, chunk_positions, group_store_ranges
    )


def _reachable_store_block_mask(
    config: "SchedulerOffloadConfig",
    group_config: "GroupOffloadConfig",
    start_chunk_idx: int,
    end_chunk_idx: int,
    final_segment_end_chunk_idx: int | None,
    reachable_boundaries: tuple[int, ...],
) -> list[bool] | None:
    """Build the block mask for a range of candidate offload chunks."""
    blocks_per_chunk = config.blocks_per_chunk
    kv_cache_spec = group_config.kv_cache_spec
    if isinstance(kv_cache_spec, SlidingWindowSpec) and (
        kv_cache_spec.extra_retained_tokens
    ):
        # Offload restores all allocated history, including MTP re-prefill
        # tokens. Widen only the store mask, not the model's attention window.
        kv_cache_spec = replace(
            kv_cache_spec,
            sliding_window=(
                kv_cache_spec.sliding_window + kv_cache_spec.extra_retained_tokens
            ),
            extra_retained_tokens=0,
        )
    return group_config.manager_cls.reachable_block_mask(
        start_block=start_chunk_idx * blocks_per_chunk,
        end_block=end_chunk_idx * blocks_per_chunk,
        alignment_tokens=config.alignment_tokens,
        kv_cache_spec=kv_cache_spec,
        use_eagle=group_config.is_eagle_group,
        retention_interval=config.retention_interval,
        reachable_boundaries=reachable_boundaries,
        dcp_world_size=config.dcp_world_size,
        final_segment_end_block=(
            final_segment_end_chunk_idx * blocks_per_chunk
            if final_segment_end_chunk_idx is not None
            else None
        ),
    )


def _final_swa_alignment_blocks(
    config: "SchedulerOffloadConfig", group_config: "GroupOffloadConfig"
) -> int | None:
    """Return representable final-tail alignment for a non-EAGLE SWA group."""
    alignment_tokens = config.alignment_tokens
    if (
        alignment_tokens is None
        or group_config.is_eagle_group
        or not isinstance(group_config.kv_cache_spec, SlidingWindowSpec)
        or alignment_tokens % group_config.tokens_per_block != 0
    ):
        return None
    return alignment_tokens // group_config.tokens_per_block
