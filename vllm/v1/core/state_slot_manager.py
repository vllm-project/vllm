# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-block persistent state and scheduler-owned retirement."""

from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from vllm.config import VllmConfig
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashList, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    PersistentStateSpec,
    StateSlotHandle,
)
from vllm.v1.request import Request


def validate_state_groups(groups: list[KVCacheGroupSpec]) -> None:
    state_groups = [
        g for g in groups if isinstance(g.kv_cache_spec, PersistentStateSpec)
    ]
    if not state_groups:
        return
    if len(state_groups) != len(groups):
        raise ValueError("Persistent state requires a state-only cache layout")
    if len(groups) != 1 or len(groups[0].layer_names) != 1:
        raise ValueError("Persistent state requires one aggregate state spec")


def validate_state_config(config: VllmConfig) -> None:
    unsupported = []
    for field in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
    ):
        if getattr(config.parallel_config, field) != 1:
            unsupported.append(field)
    for enabled, label in (
        (config.scheduler_config.async_scheduling, "async scheduling"),
        (config.speculative_config is not None, "speculative decoding"),
        (config.cache_config.enable_prefix_caching, "prefix caching"),
        (config.kv_transfer_config is not None, "KV transfer"),
        (config.parallel_config.enable_dbo, "dual batch overlap"),
        (config.model_config.enable_sleep_mode, "sleep mode"),
        (config.model_config.enable_cumem_allocator, "CuMem"),
        (config.attention_config.hisparse_config is not None, "HiSparse"),
    ):
        if enabled:
            unsupported.append(label)
    if unsupported:
        raise ValueError("Persistent state does not support " + ", ".join(unsupported))


class PersistentStateManager(SingleTypeKVCacheManager):
    """One resident block per request, independent of token positions."""

    def __init__(
        self,
        kv_cache_spec: PersistentStateSpec,
        block_pool: BlockPool,
        enable_caching: bool,
        kv_cache_group_id: int,
        scheduler_block_size: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        needs_kv_cache_zeroing: bool = False,
        max_admission_blocks_per_request: int | None = None,
    ) -> None:
        if enable_caching or dcp_world_size != 1 or pcp_world_size != 1:
            raise ValueError("Persistent state requires no prefix caching and CP=1")
        if max_admission_blocks_per_request not in (None, 1):
            raise ValueError("Persistent state admits exactly one block per request")
        super().__init__(
            kv_cache_spec,
            block_pool,
            enable_caching,
            kv_cache_group_id,
            scheduler_block_size,
            dcp_world_size,
            pcp_world_size,
            needs_kv_cache_zeroing,
            max_admission_blocks_per_request=1,
        )
        self.engine_epoch = uuid4().hex
        self.last_scheduled: dict[str, int] = {}
        self._requests: dict[str, StateSlotHandle] = {}
        self._handles: dict[int, StateSlotHandle] = {}
        self._generations = [0] * block_pool.num_gpu_blocks
        self._completed = [0] * block_pool.num_gpu_blocks
        self._retired: dict[int, tuple[int, list[KVCacheBlock]]] = {}
        self._quarantined: dict[StateSlotHandle, int] = {}

    @property
    def num_free_slots(self) -> int:
        return self.block_pool.get_num_free_blocks()

    def get(self, request_id: str) -> StateSlotHandle | None:
        return self._requests.get(request_id)

    def allocate(self, request_id: str) -> StateSlotHandle | None:
        if handle := self.get(request_id):
            return handle
        if not self.num_free_slots:
            return None
        self.allocate_new_blocks(request_id, 0, 0)
        return self.get(request_id)

    def allocate_for_admission(
        self, request_id: str, undo: ExitStack
    ) -> StateSlotHandle | None:
        """Journal the one free block before any allocation mutation."""
        if not self.num_free_slots:
            return None
        assert self.get(request_id) is None
        pool = self.block_pool
        block = pool.free_block_queue.fake_free_list_head.next_free_block
        assert block is not None and not block.is_null
        assert block.block_id not in pool._reuse_watchers
        slot = block.block_id
        generation, completed = self._generations[slot], self._completed[slot]
        metrics = pool.metrics_collector
        old_metrics = metrics.block_metrics.get(slot) if metrics else None

        def rollback() -> None:
            self._requests.pop(request_id, None)
            self._handles.pop(slot, None)
            self.req_to_blocks.pop(request_id, None)
            self._generations[slot], self._completed[slot] = generation, completed
            if block.prev_free_block is None:
                block.ref_cnt = 0
                pool.free_block_queue.prepend_n([block])
            if metrics is not None:
                if old_metrics is None:
                    metrics.block_metrics.pop(slot, None)
                else:
                    metrics.block_metrics[slot] = old_metrics

        undo.callback(rollback)
        return self.allocate(request_id)

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
        prefill_end: int = 0,
    ) -> int:
        if new_computed_blocks:
            raise ValueError("Persistent state cannot import computed blocks")
        return int(request_id not in self._requests)

    def allocate_new_blocks(
        self,
        request_id: str,
        num_tokens: int,
        num_tokens_main_model: int,
    ) -> list[KVCacheBlock]:
        if request_id in self._requests:
            return []
        blocks = self.block_pool.get_new_blocks(1)
        slot = blocks[0].block_id
        self._generations[slot] += 1
        handle = StateSlotHandle(self.engine_epoch, slot, self._generations[slot])
        self._requests[request_id] = handle
        self._handles[slot] = handle
        self._completed[slot] = 0
        self.req_to_blocks[request_id] = blocks
        return blocks

    def validate(self, handle: StateSlotHandle) -> None:
        if self._handles.get(handle.slot) != handle:
            raise ValueError("Stale state slot handle")

    def retire(self, request_id: str, fence_step: int) -> StateSlotHandle | None:
        handle = self._requests.pop(request_id, None)
        if handle is not None:
            blocks = super().pop_blocks_for_free(request_id)
            self._retired[handle.slot] = (fence_step, blocks)
        return handle

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        if request_id in self._requests:
            raise RuntimeError("Persistent state requires scheduler retirement")
        return []

    def complete(self, handle: StateSlotHandle, step_seq: int) -> None:
        self.validate(handle)
        self._completed[handle.slot] = max(self._completed[handle.slot], step_seq)
        if step_seq >= self._quarantined.get(handle, step_seq + 1):
            del self._quarantined[handle]

    def quarantine(self, handle: StateSlotHandle, step_seq: int) -> None:
        self.validate(handle)
        self._quarantined[handle] = max(self._quarantined.get(handle, 0), step_seq)

    def release(self, handle: StateSlotHandle, fence_step: int) -> bool:
        self.validate(handle)
        retired = self._retired.get(handle.slot)
        if retired is None or retired[0] != fence_step:
            raise ValueError("State slot is not retired at this fence")
        if handle in self._quarantined or self._completed[handle.slot] < fence_step:
            return False
        del self._retired[handle.slot]
        del self._handles[handle.slot]
        self.block_pool.free_blocks(retired[1])
        return True

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: Any,
        drop_eagle_block: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        del (
            block_hashes,
            max_length,
            block_pool,
            kv_cache_spec,
            drop_eagle_block,
            alignment_tokens,
            dcp_world_size,
            pcp_world_size,
        )
        return tuple([] for _ in kv_cache_group_ids), 0

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        del running_request_id
        return 0

    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
        *,
        replay_boundaries: Sequence[int] = (),
    ) -> None:
        del request, num_tokens, retention_interval, replay_boundaries

    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        return None

    def add_local_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        del request_id, num_local_computed_tokens, num_external_computed_tokens
        if new_computed_blocks:
            raise ValueError("unsupported persistent-state computed-block import")

    def allocate_external_computed_blocks(
        self,
        request_id: str,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
        record_for_zeroing: bool = True,
    ) -> None:
        del request_id, num_local_computed_tokens
        if not num_external_computed_tokens:
            return
        raise ValueError("unsupported persistent-state external computed blocks")


@dataclass(frozen=True)
class StateSlotRetirement:
    allocator: PersistentStateManager
    handle: StateSlotHandle
    fence_step: int
    request_id: str
    layer_name: str
