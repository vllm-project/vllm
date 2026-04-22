# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import ReqId
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import GPULoadStoreSpec, OffloadingManager, OffloadKey
from vllm.v1.kv_offload.worker.worker import TransferSpec

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        RequestKVState,
        SchedulerOffloadConfig,
    )
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


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
        reqs_being_stored: dict[ReqId, set[OffloadKey]],
    ) -> dict[ReqId, TransferSpec]:
        """
        Decide which blocks to store this scheduler step.

        Args:
            req_kv_states: per-request KV tracking state.
            scheduler_output: the current scheduler output.
            config: offloading configuration.
            manager: the offloading manager to call prepare_store on.
            reqs_being_stored: scheduler-owned dict of in-flight store keys,
                updated in-place for each request that gets a store queued.

        Returns:
            A dict mapping request ID to the TransferSpec to submit.
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
        reqs_being_stored: dict[ReqId, set[OffloadKey]],
    ) -> dict[ReqId, TransferSpec]:
        # Below assertion will be removed once this function supports HMA
        assert len(config.kv_group_configs) == 1
        group_config = config.kv_group_configs[0]

        reqs_to_store: dict[ReqId, TransferSpec] = {}
        for req_id, new_block_id_groups, preempted in yield_req_data(scheduler_output):
            req_kv_state = req_kv_states[req_id]
            req_kv_state.update_offload_keys()

            if preempted:
                for group_state in req_kv_state.group_states:
                    group_state.block_ids.clear()

            if new_block_id_groups:
                req_kv_state.update_block_id_groups(new_block_id_groups)

            # Below assertion will be removed once this function supports HMA
            assert len(req_kv_state.group_states) == 1
            group_state = req_kv_state.group_states[0]

            block_ids = group_state.block_ids

            req = req_kv_state.req
            new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            expected_tokens = req.num_computed_tokens + new_tokens
            total_tokens = min(expected_tokens, req.num_tokens)
            num_blocks = total_tokens // group_config.offloaded_block_size

            if req_id not in self._next_stored_block_idx:
                self._next_stored_block_idx[req_id] = [0] * len(
                    req_kv_state.group_states
                )
            start_block_idx = self._next_stored_block_idx[req_id][0]
            num_new_blocks = num_blocks - start_block_idx

            if num_new_blocks <= 0:
                continue

            num_gpu_blocks = num_blocks * config.block_size_factor
            assert len(req.block_hashes) >= num_gpu_blocks

            new_offload_keys = group_state.offload_keys[start_block_idx:num_blocks]
            store_output = manager.prepare_store(
                new_offload_keys, req_kv_state.req_context
            )
            if store_output is None:
                logger.warning(
                    "Request %s: cannot store %s blocks", req_id, num_new_blocks
                )
                continue

            self._next_stored_block_idx[req_id][0] = num_blocks

            if not store_output.keys_to_store:
                continue
            keys_to_store = set(store_output.keys_to_store)

            manager.touch(group_state.offload_keys[:num_blocks])

            dst_spec = store_output.store_spec
            src_block_ids: list[int] = []
            for idx, key in enumerate(new_offload_keys):
                if key not in keys_to_store:
                    continue
                offloaded_block_idx = start_block_idx + idx
                gpu_block_idx = offloaded_block_idx * config.block_size_factor
                for i in range(config.block_size_factor):
                    src_block_ids.append(block_ids[gpu_block_idx + i])
            src_spec = GPULoadStoreSpec(
                src_block_ids,
                group_sizes=(len(src_block_ids),),
                block_indices=(0,),
            )

            reqs_to_store[req_id] = (src_spec, dst_spec)
            reqs_being_stored[req_id] |= keys_to_store

            logger.debug(
                "Request %s offloading %s blocks starting from block #%d",
                req_id,
                len(keys_to_store),
                start_block_idx,
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
