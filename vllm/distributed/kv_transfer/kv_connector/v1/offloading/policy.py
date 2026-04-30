# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import ReqId
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import GPULoadStoreSpec, OffloadingManager, OffloadKey

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        RequestKVState,
        SchedulerOffloadConfig,
    )
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_offload.base import LoadStoreSpec

logger = init_logger(__name__)


@dataclass
class StorePlanEntry:
    """Store decision for one request returned by OffloadPolicy."""

    src_spec: GPULoadStoreSpec
    dst_spec: LoadStoreSpec
    keys: set[OffloadKey]
    gpu_block_ids: list[int]


class OffloadPolicy(ABC):
    """
    Decides which KV cache blocks to offload each scheduler step.

    Implementations may store per-request state but must clean it up
    via ``request_finished``.
    """

    @abstractmethod
    def get_blocks_to_store(
        self,
        req_kv_states: dict[str, RequestKVState],
        scheduler_output: SchedulerOutput,
        config: SchedulerOffloadConfig,
        manager: OffloadingManager,
    ) -> dict[ReqId, StorePlanEntry]:
        """
        Decide which blocks to store this scheduler step.

        Called after the scheduler has applied block-ID updates and fence
        checks for the current step.  Implementations read the already-updated
        ``req_kv_states`` and ``scheduler_output.num_scheduled_tokens`` to
        determine which blocks are newly computable and eligible for transfer.

        Args:
            req_kv_states: per-request KV tracking state (block IDs already
                updated by the caller for this step).
            scheduler_output: the current scheduler output.
            config: offloading configuration.
            manager: the offloading manager to call prepare_store on.

        Returns:
            A dict mapping request ID to a StorePlanEntry describing the
            transfer to submit.
        """
        pass

    @abstractmethod
    def request_finished(self, req_id: str) -> None:
        """Clean up per-request policy state on request completion."""
        pass

    @abstractmethod
    def notify_load_scheduled(
        self, req_id: str, next_block_idx_per_group: list[int]
    ) -> None:
        """
        Advance the store watermark when blocks are scheduled for loading,
        preventing the policy from re-storing blocks already being loaded.

        Args:
            req_id: the request whose watermark to advance.
            next_block_idx_per_group: per-group block count up to which a
                load has been scheduled.
        """
        pass


class StoreOnComputePolicy(OffloadPolicy):
    """
    Store blocks immediately as they are computed.

    This is the default policy: each scheduler step it identifies newly
    computed full offload-blocks and queues them for transfer to the
    offload medium.
    """

    def __init__(self) -> None:
        # req_id -> per-group index of the next block that needs to be stored
        self._next_stored_block_idx: dict[str, list[int]] = {}

    def get_blocks_to_store(
        self,
        req_kv_states: dict[str, RequestKVState],
        scheduler_output: SchedulerOutput,
        config: SchedulerOffloadConfig,
        manager: OffloadingManager,
    ) -> dict[ReqId, StorePlanEntry]:
        block_size_factor = config.block_size_factor
        reqs_to_store: dict[ReqId, StorePlanEntry] = {}

        for req_id in scheduler_output.num_scheduled_tokens:
            req_kv_state = req_kv_states.get(req_id)
            if req_kv_state is None:
                continue
            req_kv_state.update_offload_keys()
            req = req_kv_state.req

            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_tokens_after_batch = req.num_computed_tokens + num_scheduled_tokens
            num_offloadable_tokens = min(num_tokens_after_batch, req.num_tokens)

            if req_id not in self._next_stored_block_idx:
                self._next_stored_block_idx[req_id] = [
                    0 for _ in config.kv_group_configs
                ]
            watermark = self._next_stored_block_idx[req_id]

            # Collect eligible offload keys across all groups, filtering out
            # blocks skipped due to sliding window attention or SSM.
            new_offload_keys: list[OffloadKey] = []
            for group_idx, (group_config, group_state) in enumerate(
                zip(config.kv_group_configs, req_kv_state.group_states)
            ):
                num_blocks = num_offloadable_tokens // group_config.offloaded_block_size
                start_block_idx = watermark[group_idx]
                if num_blocks <= start_block_idx:
                    continue
                offload_keys = group_state.offload_keys[start_block_idx:num_blocks]
                # Take the last GPU block of each offloaded block to determine
                # whether the block was skipped (block_id == 0).
                offload_block_ids = group_state.block_ids[
                    start_block_idx * block_size_factor
                    + block_size_factor
                    - 1 : num_blocks * block_size_factor : block_size_factor
                ]
                assert len(offload_keys) == len(offload_block_ids)
                for offload_key, block_id in zip(offload_keys, offload_block_ids):
                    if block_id != 0:
                        new_offload_keys.append(offload_key)

            if not new_offload_keys:
                # No new blocks to store; advance the watermark.
                for group_idx, group_config in enumerate(config.kv_group_configs):
                    num_blocks = (
                        num_offloadable_tokens // group_config.offloaded_block_size
                    )
                    watermark[group_idx] = max(watermark[group_idx], num_blocks)
                continue

            store_output = manager.prepare_store(
                new_offload_keys, req_kv_state.req_context
            )
            if store_output is None:
                logger.warning("Request %s: cannot store blocks", req_id)
                continue

            if not store_output.keys_to_store:
                # Manager declined; advance the watermark.
                for group_idx, group_config in enumerate(config.kv_group_configs):
                    num_blocks = (
                        num_offloadable_tokens // group_config.offloaded_block_size
                    )
                    watermark[group_idx] = max(watermark[group_idx], num_blocks)
                continue

            for group_state in req_kv_state.group_states:
                manager.touch(group_state.offload_keys)

            keys_to_store = set(store_output.keys_to_store)

            group_sizes: list[int] = []
            block_indices: list[int] = []
            src_block_ids: list[int] = []
            for group_idx, (group_config, group_state) in enumerate(
                zip(config.kv_group_configs, req_kv_state.group_states)
            ):
                num_blocks = num_offloadable_tokens // group_config.offloaded_block_size
                start_block_idx = watermark[group_idx]
                block_ids = group_state.block_ids
                num_group_blocks = 0
                start_gpu_block_idx: int | None = None
                for idx, offload_key in enumerate(
                    group_state.offload_keys[start_block_idx:num_blocks]
                ):
                    if offload_key not in keys_to_store:
                        continue
                    offloaded_block_idx = start_block_idx + idx
                    gpu_block_idx = offloaded_block_idx * block_size_factor
                    num_group_blocks += block_size_factor
                    for i in range(block_size_factor):
                        block_id = block_ids[gpu_block_idx + i]
                        if block_id == 0:
                            # Skipped blocks cannot appear after non-skipped blocks.
                            assert start_gpu_block_idx is None
                            continue
                        elif start_gpu_block_idx is None:
                            start_gpu_block_idx = gpu_block_idx + i
                        src_block_ids.append(block_id)
                group_sizes.append(num_group_blocks)
                block_indices.append(start_gpu_block_idx or 0)
                watermark[group_idx] = num_blocks

            src_spec = GPULoadStoreSpec(
                src_block_ids,
                group_sizes=tuple(group_sizes),
                block_indices=tuple(block_indices),
            )
            dst_spec = store_output.store_spec

            reqs_to_store[req_id] = StorePlanEntry(
                src_spec=src_spec,
                dst_spec=dst_spec,
                keys=keys_to_store,
                gpu_block_ids=src_block_ids,
            )

            logger.debug(
                "Request %s: queuing store for %s blocks",
                req_id,
                len(keys_to_store),
            )

        return reqs_to_store

    def request_finished(self, req_id: str) -> None:
        self._next_stored_block_idx.pop(req_id, None)

    def notify_load_scheduled(
        self, req_id: str, next_block_idx_per_group: list[int]
    ) -> None:
        state = self._next_stored_block_idx.setdefault(
            req_id, [0] * len(next_block_idx_per_group)
        )
        for i, val in enumerate(next_block_idx_per_group):
            state[i] = max(state[i], val)
