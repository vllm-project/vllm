# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from collections import defaultdict
from collections.abc import Iterable
from itertools import islice
from typing import Any

from vllm.distributed.kv_events import BlockRemoved, BlockStored, KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    ReqId,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import BlockHash, BlockHashListWithBlockSize
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_offload.abstract import OffloadingManager
from vllm.v1.kv_offload.hashing import HybridChunkBlockHashList
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from vllm.v1.kv_offload.planner import HybridOffloadPlanner
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.worker.worker import TransferSpec
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

logger = init_logger(__name__)


class OffloadingConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, spec: OffloadingSpec):
        self.hybrid_offload_enabled = spec.hybrid_offload_enabled
        self.hybrid_planner: HybridOffloadPlanner | None = spec.hybrid_planner
        self.requires_partial_group_offload = spec.requires_partial_group_offload
        self.gpu_block_sizes = tuple(spec.gpu_block_size)
        self.group_hash_block_sizes = tuple(spec.group_hash_block_size)
        self.hash_function = spec.hash_function
        self.num_kv_groups = len(self.gpu_block_sizes)
        self.hash_block_size = spec.hash_block_size
        self.offloaded_block_size = spec.offloaded_block_size
        self.max_concurrent_loads = spec.vllm_config.scheduler_config.max_num_seqs
        self.hash_block_size_factor: int | None = None
        if not self.hybrid_offload_enabled:
            assert self.offloaded_block_size % self.hash_block_size == 0
            self.hash_block_size_factor = (
                self.offloaded_block_size // self.hash_block_size
            )
        self.block_size_factors = tuple(spec.block_size_factors)
        if self.hybrid_offload_enabled:
            for gpu_block_size, unit_size in zip(
                self.gpu_block_sizes, self.group_hash_block_sizes
            ):
                assert gpu_block_size % unit_size == 0, (
                    "Hybrid GPU block size must be divisible by group offload unit size"
                )
        self.manager: OffloadingManager = spec.get_manager()

        self._requests: dict[ReqId, Request] = {}
        # list of GPU block IDs per request
        self._request_block_ids: dict[ReqId, tuple[list[int], ...]] = {}
        # requests to load for the current scheduler step
        self._reqs_to_load: dict[ReqId, TransferSpec] = {}
        # request blocks are stored in order
        # index of next block (of size offloaded_block_size) to offload
        self._next_stored_block_idx: dict[ReqId, int] = {}
        # if GPU prefix caching is enabled,
        # track loaded blocks to avoid redundant loads
        self._blocks_being_loaded: set[BlockHash] | None = (
            set() if spec.vllm_config.cache_config.enable_prefix_caching else None
        )

        # request ID -> set(block hashes being stored/load)
        self._reqs_being_stored = defaultdict[ReqId, set[BlockHash]](set)
        self._reqs_being_loaded = defaultdict[ReqId, set[BlockHash]](set)

        # Hybrid mode: one HybridChunkBlockHashList per active request, reused
        # across scheduler steps so RequestBlockHashList lazily caches computed
        # group-level hashes instead of recomputing them from scratch each step.
        self._hybrid_hash_lists: dict[ReqId, HybridChunkBlockHashList] = {}

        # Load timeout: if a load takes longer than this, cancel it and
        # fall back to recompute.  Prevents requests from stalling
        # indefinitely on slow NFS or hung storage.
        self._load_timeout_seconds = float(
            spec.extra_config.get("load_timeout_seconds", 30.0)
        )
        self._load_start_times: dict[ReqId, float] = {}

    def _chunk_prefix_tokens(self, chunk_count: int) -> int:
        if not self.hybrid_offload_enabled:
            return chunk_count * self.offloaded_block_size
        return self._get_hybrid_planner().chunk_prefix_tokens(chunk_count)

    def _chunk_count_for_tokens(self, tokens: int) -> int:
        if not self.hybrid_offload_enabled:
            return tokens // self.offloaded_block_size
        planner = self._get_hybrid_planner()
        return planner.chunk_count_for_tokens(tokens)

    def _get_hybrid_planner(self):
        planner = self.hybrid_planner
        if planner is None:
            raise RuntimeError("Hybrid offload planner is not configured")
        return planner

    def _empty_block_groups(self) -> tuple[list[int], ...]:
        return tuple([] for _ in range(self.num_kv_groups))

    @staticmethod
    def _flatten_block_groups(
        block_groups: tuple[list[int], ...] | list[list[int]],
    ) -> tuple[list[int], tuple[int, ...]]:
        flat_block_ids: list[int] = []
        group_sizes: list[int] = []
        for group_block_ids in block_groups:
            group_sizes.append(len(group_block_ids))
            flat_block_ids.extend(group_block_ids)
        return flat_block_ids, tuple(group_sizes)

    def _append_block_groups(
        self,
        req_id: ReqId,
        new_block_id_groups: tuple[list[int], ...],
    ) -> None:
        existing = self._request_block_ids[req_id]
        assert len(existing) == len(new_block_id_groups) == self.num_kv_groups
        for group_index, group_block_ids in enumerate(new_block_id_groups):
            existing[group_index].extend(group_block_ids)

    def _build_gpu_transfer_spec_from_chunk_range(
        self,
        block_groups: tuple[list[int], ...] | list[list[int]],
        start_chunk_idx: int,
        end_chunk_idx: int,
        include_block_indices: bool = False,
    ) -> GPULoadStoreSpec:
        if not self.hybrid_offload_enabled:
            block_groups_list: list[list[int]] = []
            block_indices: list[int] = []
            for group_index, group_block_ids in enumerate(block_groups):
                group_factor = self.block_size_factors[group_index]
                start_gpu_block_idx = start_chunk_idx * group_factor
                end_gpu_block_idx = end_chunk_idx * group_factor
                if include_block_indices:
                    block_indices.append(start_gpu_block_idx)
                block_groups_list.append(
                    group_block_ids[start_gpu_block_idx:end_gpu_block_idx]
                )
            flat_block_ids, group_sizes = self._flatten_block_groups(block_groups_list)
            return GPULoadStoreSpec(
                flat_block_ids,
                group_sizes=group_sizes,
                block_indices=tuple(block_indices) if include_block_indices else None,
            )

        flat_block_ids: list[int] = []  # type: ignore[no-redef]
        flat_block_offsets: list[int] = []
        flat_block_counts: list[int] = []
        group_sizes: list[int] = []  # type: ignore[no-redef]
        block_indices: list[int] = []  # type: ignore[no-redef]
        planner = self._get_hybrid_planner()
        if start_chunk_idx == 0:
            group_start_tokens = tuple(0 for _ in range(self.num_kv_groups))
        else:
            group_start_tokens = planner.group_covered_tokens_for_chunk_count(
                start_chunk_idx
            )
        group_end_tokens = planner.group_covered_tokens_for_chunk_count(end_chunk_idx)

        for group_index, group_block_ids in enumerate(block_groups):
            unit_size = self.group_hash_block_sizes[group_index]
            gpu_block_size = self.gpu_block_sizes[group_index]
            sub_blocks_per_gpu_block = gpu_block_size // unit_size

            start_unit_idx = group_start_tokens[group_index] // unit_size
            end_unit_idx = group_end_tokens[group_index] // unit_size
            assert end_unit_idx >= start_unit_idx

            group_entry_count = 0
            if include_block_indices:
                block_indices.append(start_unit_idx // sub_blocks_per_gpu_block)

            unit_idx = start_unit_idx
            while unit_idx < end_unit_idx:
                gpu_block_idx = unit_idx // sub_blocks_per_gpu_block
                sub_block_offset = unit_idx % sub_blocks_per_gpu_block
                sub_block_count = min(
                    sub_blocks_per_gpu_block - sub_block_offset,
                    end_unit_idx - unit_idx,
                )
                flat_block_ids.append(group_block_ids[gpu_block_idx])
                flat_block_offsets.append(sub_block_offset)
                flat_block_counts.append(sub_block_count)
                group_entry_count += 1
                unit_idx += sub_block_count

            group_sizes.append(group_entry_count)  # type: ignore[attr-defined]

        return GPULoadStoreSpec(
            flat_block_ids,
            group_sizes=tuple(group_sizes),
            block_indices=tuple(block_indices) if include_block_indices else None,
            block_offsets=flat_block_offsets,
            block_counts=flat_block_counts,
        )

    def _get_block_hashes(
        self,
        req: Request,
        start_idx: int = 0,
        end_idx: int | None = None,
    ) -> Iterable[BlockHash]:
        if self.hybrid_offload_enabled:
            # Reuse a cached HybridChunkBlockHashList so that
            # RequestBlockHashList's lazily-computed per-group hashes survive
            # across multiple calls within a step and across scheduler steps.
            # Without caching, each call rebuilds the list and recomputes all
            # previously-seen group-level hashes from scratch.
            req_id = req.request_id
            offloaded_hashes = self._hybrid_hash_lists.get(req_id)
            if offloaded_hashes is None:
                offloaded_hashes = HybridChunkBlockHashList(
                    req,
                    self.group_hash_block_sizes,
                    self.offloaded_block_size,
                    self.hash_function,
                )
                self._hybrid_hash_lists[req_id] = offloaded_hashes
            return islice(offloaded_hashes, start_idx, end_idx)

        simple_hashes = BlockHashListWithBlockSize(
            req.block_hashes,
            self.hash_block_size,
            self.offloaded_block_size,
        )
        return islice(simple_hashes, start_idx, end_idx)

    def _get_num_offloaded_blocks(self, request: Request) -> int:
        if self.hybrid_offload_enabled:
            return self._chunk_count_for_tokens(request.num_tokens)

        assert self.hash_block_size_factor is not None
        return len(request.block_hashes) // self.hash_block_size_factor

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """
        Get number of new tokens that can be loaded beyond the
        num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - The number of tokens that can be loaded beyond what is
                  already computed.
                  If None, it means that the connector needs more time to
                  determine the number of matched tokens, and the scheduler
                  should query for this request again later.
                - `True` if tokens will be loaded asynchronously
                  (between scheduler steps).
        """
        # Backpressure: if too many requests are already loading from
        # external storage, defer this one to the next scheduler step.
        # Without this cap, a burst of concurrent loads can queue
        # hundreds of I/O tasks and stall the EngineCore.
        if len(self._reqs_being_loaded) >= self.max_concurrent_loads:
            return None, False

        num_blocks = self._get_num_offloaded_blocks(request)
        block_hashes = self._get_block_hashes(request)

        self.manager.touch(block_hashes)

        full_block_tokens = self._chunk_prefix_tokens(num_blocks)
        if self.hybrid_offload_enabled:
            if full_block_tokens <= num_computed_tokens:
                return 0, False
            start_block_idx = self._chunk_count_for_tokens(num_computed_tokens)
        else:
            if full_block_tokens - num_computed_tokens < self.offloaded_block_size:
                # we can load less than a block, skip
                return 0, False
            start_block_idx = num_computed_tokens // self.offloaded_block_size
        hits = self.manager.lookup(
            self._get_block_hashes(request, start_idx=start_block_idx)
        )
        if hits is None:
            # indicates a lookup that should be tried later
            return None, False
        if hits == 0:
            return 0, False

        num_hit_tokens = (
            self._chunk_prefix_tokens(start_block_idx + hits) - num_computed_tokens
        )
        logger.debug(
            "Request %s hit %s offloaded tokens after %s GPU hit tokens",
            request.request_id,
            num_hit_tokens,
            num_computed_tokens,
        )
        min_hit_tokens = (
            self.hash_block_size
            if self.hybrid_offload_enabled
            else self.offloaded_block_size
        )
        if num_hit_tokens < min_hit_tokens:
            return 0, False

        if self._blocks_being_loaded:
            block_hashes = self._get_block_hashes(
                request, start_idx=start_block_idx, end_idx=start_block_idx + hits
            )

            if any(
                block_hash in self._blocks_being_loaded for block_hash in block_hashes
            ):
                # hit blocks are being loaded, delay request
                logger.debug(
                    "Delaying request %s since some of its blocks are already"
                    " being loaded",
                    request.request_id,
                )
                return None, False

        return num_hit_tokens, True

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        self._requests[request.request_id] = request
        self._request_block_ids[request.request_id] = self._empty_block_groups()

        if num_external_tokens == 0:
            return

        block_groups = blocks.get_block_ids()
        computed_tokens_per_group: list[int] = []
        for group_index, group_blocks in enumerate(blocks.blocks):
            num_computed_gpu_blocks = sum(
                block.block_hash is not None
                for block in group_blocks
                if not block.is_null
            )
            computed_tokens_per_group.append(
                num_computed_gpu_blocks * self.gpu_block_sizes[group_index]
            )

        num_computed_tokens = min(computed_tokens_per_group, default=0)
        groups_agree = all(
            group_tokens == num_computed_tokens
            for group_tokens in computed_tokens_per_group
        )
        if not groups_agree:
            # Some groups loaded more blocks than others (e.g., stale
            # cache files rejected for one group but not others, or
            # kernel block size mismatch on the attention group).
            # Fall back to recompute by reporting 0 external tokens.
            logger.warning(
                "KV groups disagree on computed prefix length: %s. "
                "Falling back to full recompute.",
                computed_tokens_per_group,
            )
            num_computed_tokens = 0
            num_external_tokens = 0

        full_block_tokens = num_computed_tokens + num_external_tokens
        start_block_idx = self._chunk_count_for_tokens(num_computed_tokens)
        num_blocks = self._chunk_count_for_tokens(full_block_tokens)
        assert self._chunk_prefix_tokens(num_blocks) == full_block_tokens

        assert self._get_num_offloaded_blocks(request) >= num_blocks
        # Materialise into a list so the same hashes can be passed to
        # prepare_load (which consumes the iterable) and also used to
        # update _reqs_being_loaded without a second HybridChunkBlockHashList.
        block_hashes = list(
            self._get_block_hashes(
                request, start_idx=start_block_idx, end_idx=num_blocks
            )
        )

        src_spec = self.manager.prepare_load(block_hashes)
        dst_spec = self._build_gpu_transfer_spec_from_chunk_range(
            block_groups,
            start_chunk_idx=start_block_idx,
            end_chunk_idx=num_blocks,
            include_block_indices=True,
        )

        self._reqs_to_load[request.request_id] = (src_spec, dst_spec)
        req_blocks_being_loaded = self._reqs_being_loaded[request.request_id]
        req_blocks_being_loaded.update(block_hashes)
        self._load_start_times[request.request_id] = time.monotonic()
        self._next_stored_block_idx[request.request_id] = num_blocks

        if self._blocks_being_loaded is not None:
            self._blocks_being_loaded.update(req_blocks_being_loaded)

    def _get_reqs_to_store(self, scheduler_output: SchedulerOutput):
        reqs_to_store: dict[ReqId, TransferSpec] = {}
        # iterate over both new and cached requests
        for req_id, new_block_id_groups, preempted in yield_req_data(scheduler_output):
            if preempted:
                self._request_block_ids[req_id] = self._empty_block_groups()

            if new_block_id_groups:
                self._append_block_groups(req_id, new_block_id_groups)

            block_groups = self._request_block_ids[req_id]

            req = self._requests[req_id]
            new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            expected_tokens = req.num_computed_tokens + new_tokens
            # with async scheduling, some tokens may be missing
            total_tokens = min(expected_tokens, req.num_tokens)
            num_blocks = self._chunk_count_for_tokens(total_tokens)
            start_block_idx = self._next_stored_block_idx.get(req_id, 0)
            num_new_blocks = num_blocks - start_block_idx

            if num_new_blocks <= 0:
                continue

            new_block_hashes = list(
                self._get_block_hashes(
                    req, start_idx=start_block_idx, end_idx=num_blocks
                )
            )
            store_output = self.manager.prepare_store(new_block_hashes)
            if store_output is None:
                logger.warning(
                    "Request %s: cannot store %s blocks", req_id, num_new_blocks
                )
                continue

            self._next_stored_block_idx[req_id] = num_blocks

            if not store_output.block_hashes_to_store:
                continue
            block_hashes_to_store = set(store_output.block_hashes_to_store)

            block_hashes = self._get_block_hashes(req, end_idx=num_blocks)
            self.manager.touch(block_hashes)

            dst_spec = store_output.store_spec
            if self.hybrid_offload_enabled:
                block_hash_to_chunk_idx = {
                    block_hash: start_block_idx + idx
                    for idx, block_hash in enumerate(new_block_hashes)
                }
                src_specs: list[GPULoadStoreSpec] = []
                for block_hash in new_block_hashes:
                    if block_hash not in block_hashes_to_store:
                        continue
                    chunk_idx = block_hash_to_chunk_idx[block_hash]
                    src_specs.append(
                        self._build_gpu_transfer_spec_from_chunk_range(
                            block_groups,
                            start_chunk_idx=chunk_idx,
                            end_chunk_idx=chunk_idx + 1,
                        )
                    )

                # Accumulate per-group, then flatten.  Each
                # src_spec_part's flat arrays are ordered by group
                # (group_0 entries, group_1 entries, ...), but when
                # we combine *multiple* spec parts we must keep all
                # of group_0's entries contiguous, then group_1's,
                # etc.  The old code appended chunk-by-chunk which
                # interleaved groups and caused mamba sub-block
                # offsets to bleed into attention-group entries.
                per_group_ids: list[list[int]] = [[] for _ in range(self.num_kv_groups)]
                per_group_offsets: list[list[int]] = [
                    [] for _ in range(self.num_kv_groups)
                ]
                per_group_counts: list[list[int]] = [
                    [] for _ in range(self.num_kv_groups)
                ]
                group_sizes = [0] * self.num_kv_groups
                for src_spec_part in src_specs:
                    start = 0
                    for group_index, group_size in enumerate(src_spec_part.group_sizes):
                        end = start + group_size
                        per_group_ids[group_index].extend(
                            src_spec_part.block_ids[start:end].tolist()
                        )
                        assert src_spec_part.block_offsets is not None
                        assert src_spec_part.block_counts is not None
                        per_group_offsets[group_index].extend(
                            src_spec_part.block_offsets[start:end].tolist()
                        )
                        per_group_counts[group_index].extend(
                            src_spec_part.block_counts[start:end].tolist()
                        )
                        group_sizes[group_index] += group_size
                        start = end
                flat_src_block_ids: list[int] = []
                flat_block_offsets: list[int] = []
                flat_block_counts: list[int] = []
                for gi in range(self.num_kv_groups):
                    flat_src_block_ids.extend(per_group_ids[gi])
                    flat_block_offsets.extend(per_group_offsets[gi])
                    flat_block_counts.extend(per_group_counts[gi])
                src_spec = GPULoadStoreSpec(
                    flat_src_block_ids,
                    group_sizes=tuple(group_sizes),
                    block_offsets=flat_block_offsets,
                    block_counts=flat_block_counts,
                )
            else:
                src_block_groups: list[list[int]] = []
                for group_index, group_block_ids in enumerate(block_groups):
                    group_factor = self.block_size_factors[group_index]
                    src_group_block_ids: list[int] = []
                    for idx, blk_hash in enumerate(new_block_hashes):
                        if blk_hash not in block_hashes_to_store:
                            continue
                        offloaded_block_idx = start_block_idx + idx
                        gpu_block_idx = offloaded_block_idx * group_factor
                        for i in range(group_factor):
                            src_group_block_ids.append(
                                group_block_ids[gpu_block_idx + i]
                            )
                    src_block_groups.append(src_group_block_ids)
                flat_src_block_ids, src_group_sizes = self._flatten_block_groups(
                    src_block_groups
                )
                src_spec = GPULoadStoreSpec(
                    flat_src_block_ids, group_sizes=src_group_sizes
                )

            reqs_to_store[req_id] = (src_spec, dst_spec)
            self._reqs_being_stored[req_id] |= block_hashes_to_store

            logger.debug(
                "Request %s offloading %s blocks starting from block #%d",
                req_id,
                len(block_hashes_to_store),
                start_block_idx,
            )

        return reqs_to_store

    def get_timed_out_loads(self) -> set[ReqId]:
        """Return request IDs whose loads have exceeded the timeout.

        Timed-out loads are removed from ``_reqs_being_loaded`` and
        their block hashes are released, so the scheduler can treat
        them as failed and fall back to recompute.
        """
        if not self._load_start_times:
            return set()

        now = time.monotonic()
        timed_out: set[ReqId] = set()
        for req_id, start in list(self._load_start_times.items()):
            if now - start > self._load_timeout_seconds:
                elapsed = now - start
                logger.warning(
                    "Load timeout: req_id=%s exceeded %.0fs "
                    "(elapsed %.1fs). Falling back to recompute.",
                    req_id,
                    self._load_timeout_seconds,
                    elapsed,
                )
                timed_out.add(req_id)
                self._load_start_times.pop(req_id)
                block_hashes = self._reqs_being_loaded.pop(req_id, None)
                if block_hashes and self._blocks_being_loaded:
                    self._blocks_being_loaded.difference_update(block_hashes)

        return timed_out

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        meta = OffloadingConnectorMetadata(
            reqs_to_load=self._reqs_to_load,
            reqs_to_store=self._get_reqs_to_store(scheduler_output),
            reqs_to_flush=scheduler_output.preempted_req_ids,
        )
        self._reqs_to_load = {}

        # NOTE (orozery): we should move this logic to update_connector_output
        # once KVConnectorOutput allows us to report completed transfers
        for req_id in scheduler_output.preempted_req_ids or ():
            block_hashes = self._reqs_being_stored.get(req_id)
            if block_hashes:
                self.manager.complete_store(block_hashes)
                block_hashes.clear()

        return meta

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """
        for req_id in connector_output.finished_sending or []:
            block_hashes = self._reqs_being_stored.pop(req_id, None)
            if block_hashes:
                self.manager.complete_store(block_hashes)

        for req_id in connector_output.finished_recving or []:
            self._load_start_times.pop(req_id, None)
            block_hashes = self._reqs_being_loaded.pop(req_id, None)
            if block_hashes:
                if self._blocks_being_loaded:
                    self._blocks_being_loaded.difference_update(block_hashes)
                self.manager.complete_load(block_hashes)

    def request_finished(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        req_id = request.request_id
        self._requests.pop(req_id, None)
        self._request_block_ids.pop(req_id, None)
        self._hybrid_hash_lists.pop(req_id, None)
        self._load_start_times.pop(req_id, None)

        # TODO(orozery): possibly kickoff offload for last block
        # which may have been deferred due to async scheduling
        self._next_stored_block_idx.pop(req_id, None)

        request_being_stored = req_id in self._reqs_being_stored
        return request_being_stored, None

    def take_events(self) -> Iterable[KVCacheEvent]:
        """Take the KV cache events from the connector.

        Returns:
            A list of KV cache events.
        """
        for event in self.manager.take_events():
            if event.removed:
                yield BlockRemoved(block_hashes=event.block_hashes, medium=event.medium)
            else:
                yield BlockStored(
                    block_hashes=event.block_hashes,
                    parent_block_hash=None,
                    token_ids=[],
                    lora_id=None,
                    block_size=event.block_size,
                    medium=event.medium,
                    lora_name=None,
                )
