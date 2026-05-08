# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from abc import ABC, abstractmethod

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.state import (
    RequestKVState,
    SchedulerOffloadConfig,
)
from vllm.v1.kv_offload.base import OffloadKey


class OffloadPolicy(ABC):
    """Abstraction for deciding which KV blocks to store each scheduler step."""

    @abstractmethod
    def get_blocks_to_store(
        self,
        req_kv_state: RequestKVState,
        num_offloadable_tokens: int,
    ) -> tuple[list[OffloadKey], list[int]]:
        """Return (keys_to_store, per_group_start_idx) for this scheduler step.

        The implementation is responsible for tracking per-request progress
        and advancing it on every call so the same blocks are not returned
        twice.

        Args:
            req_kv_state: current KV state of the request. Read-only from
                the policy's perspective.
            num_offloadable_tokens: token count available for offloading
                after this scheduler step.

        Returns:
            A 2-tuple of:
              - Possibly-empty list of OffloadKey values to store.
              - Per-group starting block index (one entry per KV cache group),
                indicating where the new keys begin in each group's offload_keys
                list. Callers may use this to skip already-processed blocks.
        """
        ...

    def on_blocks_loaded(
        self,
        req_id: str,
        num_offloadable_tokens: int,
    ) -> None:
        """Called when blocks are being loaded so the policy can advance past them.

        Args:
            req_id: the request being loaded.
            num_offloadable_tokens: token count up to which blocks are loaded.
        """
        return

    def request_finished(self, req_id: str) -> None:
        """Release any per-request state held by the policy."""
        return


class StoreOnComputePolicy(OffloadPolicy):
    """Store blocks as soon as they are computed (the default policy).

    Tracks per-request, per-group progress so that each block is submitted
    for offloading exactly once, in order.
    """

    def __init__(self, config: SchedulerOffloadConfig) -> None:
        self._config = config
        self._block_size_factor: int = config.block_size_factor
        # req_id -> per-group next stored block index
        self._stored_idx: dict[str, list[int]] = {}

    def get_blocks_to_store(
        self,
        req_kv_state: RequestKVState,
        num_offloadable_tokens: int,
    ) -> tuple[list[OffloadKey], list[int]]:
        req_id = req_kv_state.req.request_id
        stored = self._stored_idx.get(req_id)
        if stored is None:
            stored = [0] * len(self._config.kv_group_configs)
            self._stored_idx[req_id] = stored
        new_offload_keys: list[OffloadKey] = []
        per_group_start: list[int] = []
        for group_idx, group_config in enumerate(self._config.kv_group_configs):
            group_state = req_kv_state.group_states[group_idx]
            num_blocks = num_offloadable_tokens // group_config.offloaded_block_size
            start_block_idx = stored[group_idx]
            per_group_start.append(start_block_idx)
            if num_blocks <= start_block_idx:
                continue
            offload_keys = group_state.offload_keys[start_block_idx:num_blocks]
            # For each offloaded block, inspect the last corresponding GPU block.
            # A block_id of 0 indicates a sliding-window / SSM padding slot that
            # should be skipped; we know all earlier blocks are skipped too.
            offload_block_ids = group_state.block_ids[
                start_block_idx * self._block_size_factor
                + self._block_size_factor
                - 1 : num_blocks * self._block_size_factor : self._block_size_factor
            ]
            assert len(offload_keys) == len(offload_block_ids)
            for offload_key, block_id in zip(offload_keys, offload_block_ids):
                if block_id != 0:
                    new_offload_keys.append(offload_key)
            # Always advance regardless of prepare_store filtering later.
            stored[group_idx] = num_blocks
        return new_offload_keys, per_group_start

    def on_blocks_loaded(
        self,
        req_id: str,
        num_offloadable_tokens: int,
    ) -> None:
        # Use setdefault so that a load preceding the first store call still
        # advances the index, preventing already-loaded blocks from being
        # returned by a subsequent get_blocks_to_store call.
        stored = self._stored_idx.setdefault(
            req_id, [0] * len(self._config.kv_group_configs)
        )
        for group_idx, group_config in enumerate(self._config.kv_group_configs):
            num_blocks = num_offloadable_tokens // group_config.offloaded_block_size
            stored[group_idx] = num_blocks

    def request_finished(self, req_id: str) -> None:
        self._stored_idx.pop(req_id, None)
