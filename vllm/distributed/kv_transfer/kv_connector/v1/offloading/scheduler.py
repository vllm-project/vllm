# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import islice
from typing import Any

from vllm.distributed.kv_events import BlockRemoved, BlockStored, KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    ReqId,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.policy import (
    OffloadingPolicy,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_offload.abstract import OffloadingManager
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.worker.worker import TransferSpec
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class RequestStatus:
    req: Request
    # total of expected tokens
    num_computed_tokens: int = 0
    # list of GPU block ids
    block_ids: list[int] = field(default_factory=list)
    # index of next block (of size offloaded_block_size) to offload
    next_stored_block_idx: int = 0


class OffloadingConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, spec: OffloadingSpec, policy: OffloadingPolicy):
        assert len(spec.gpu_block_size) == 1
        self.gpu_block_size = spec.gpu_block_size[0]
        self.offloaded_block_size = self.gpu_block_size * spec.block_size_factor
        self.block_size_factor = spec.block_size_factor
        self.manager: OffloadingManager = spec.get_manager()
        self.policy: OffloadingPolicy = policy

        self._req_status: dict[ReqId, RequestStatus] = {}
        self._reqs_to_load: dict[ReqId, TransferSpec] = {}

        # if GPU prefix caching is enabled and policy supports it,
        # track loaded blocks to avoid scheduling redundant loads
        self._blocks_being_loaded: set[BlockHash] | None = (
            set()
            if (
                spec.vllm_config.cache_config.enable_prefix_caching
                and self.policy.allow_redundant_load_prevention()
            )
            else None
        )

        # request ID -> set(block hashes being stored/load)
        self._reqs_being_stored = defaultdict[ReqId, set[BlockHash]](set)
        self._reqs_being_loaded = defaultdict[ReqId, set[BlockHash]](set)

    def _get_block_hashes(
        self,
        req: Request,
        start_idx: int = 0,
        end_idx: int | None = None,
    ) -> Iterable[BlockHash]:
        return islice(
            req.block_hashes,
            self.block_size_factor * start_idx + self.block_size_factor - 1,
            self.block_size_factor * end_idx if end_idx else None,
            self.block_size_factor,
        )

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
        if request.request_id not in self._req_status:
            self._req_status[request.request_id] = RequestStatus(request)

        if not self.policy.should_lookup(request):
            return 0, False

        num_blocks = request.num_tokens // self.offloaded_block_size

        assert len(request.block_hashes) // self.block_size_factor == num_blocks
        block_hashes = self._get_block_hashes(request)

        self.manager.touch(block_hashes)

        full_block_tokens = self.offloaded_block_size * num_blocks
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
            self.offloaded_block_size * (start_block_idx + hits) - num_computed_tokens
        )
        logger.debug(
            "Request %s hit %s offloaded tokens after %s GPU hit tokens",
            request.request_id,
            num_hit_tokens,
            num_computed_tokens,
        )
        if num_hit_tokens < self.offloaded_block_size:
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
        if num_external_tokens == 0:
            return

        block_groups = blocks.get_block_ids()
        block_ids = block_groups[0]

        num_computed_gpu_blocks = sum(
            block.block_hash is not None for block in blocks.blocks[0]
        )
        num_computed_tokens = num_computed_gpu_blocks * self.gpu_block_size
        full_block_tokens = num_computed_tokens + num_external_tokens
        assert full_block_tokens % self.offloaded_block_size == 0

        num_pending_gpu_blocks = len(block_ids) - num_computed_gpu_blocks
        assert num_external_tokens == num_pending_gpu_blocks * self.gpu_block_size

        start_block_idx = num_computed_tokens // self.offloaded_block_size
        num_blocks = full_block_tokens // self.offloaded_block_size

        assert len(request.block_hashes) // self.block_size_factor >= num_blocks
        block_hashes = self._get_block_hashes(
            request, start_idx=start_block_idx, end_idx=num_blocks
        )

        src_spec = self.manager.prepare_load(block_hashes)
        dst_spec = GPULoadStoreSpec(
            block_ids[num_computed_gpu_blocks:],
            group_sizes=(num_pending_gpu_blocks,),
            block_indices=(num_computed_gpu_blocks,),
        )

        block_hashes = self._get_block_hashes(
            request, start_idx=start_block_idx, end_idx=num_blocks
        )

        self._reqs_to_load[request.request_id] = (src_spec, dst_spec)
        req_blocks_being_loaded = self._reqs_being_loaded[request.request_id]
        req_blocks_being_loaded.update(block_hashes)
        self._req_status[request.request_id].next_stored_block_idx = num_blocks

        if self._blocks_being_loaded is not None:
            self._blocks_being_loaded.update(req_blocks_being_loaded)

    def _get_reqs_to_store(
        self, scheduler_output: SchedulerOutput
    ) -> dict[ReqId, TransferSpec]:
        reqs_to_store: dict[ReqId, TransferSpec] = {}

        all_req_ids: list[ReqId] = []
        for req_id, new_block_id_groups, preempted in yield_req_data(scheduler_output):
            req_status = self._req_status[req_id]

            if preempted:
                req_status.block_ids = []

            if new_block_id_groups:
                req_status.block_ids += new_block_id_groups[0]

            req_status.num_computed_tokens = (
                req_status.req.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[req_id]
            )

            all_req_ids.append(req_id)

        # iterate over requests that should be stored
        for req_id in self.policy.get_store_candidate_req_ids(
            scheduler_output, all_req_ids
        ):
            transfer_spec = self._maybe_store_request(self._req_status[req_id])
            if transfer_spec is not None:
                reqs_to_store[req_id] = transfer_spec

        return reqs_to_store

    def _maybe_store_request(self, req_status: RequestStatus) -> TransferSpec | None:
        req = req_status.req
        req_id = req.request_id
        # with async scheduling, some tokens may be missing
        total_tokens = min(req_status.num_computed_tokens, req.num_tokens)
        num_blocks = total_tokens // self.offloaded_block_size
        start_block_idx = req_status.next_stored_block_idx
        num_new_blocks = num_blocks - start_block_idx

        if num_new_blocks <= 0:
            return None

        num_gpu_blocks = num_blocks * self.block_size_factor
        assert len(req.block_hashes) >= num_gpu_blocks

        new_block_hashes = self._get_block_hashes(
            req, start_idx=start_block_idx, end_idx=num_blocks
        )
        store_output = self.manager.prepare_store(new_block_hashes)
        if store_output is None:
            logger.warning("Request %s: cannot store %s blocks", req_id, num_new_blocks)
            return None

        req_status.next_stored_block_idx = num_blocks

        if not store_output.block_hashes_to_store:
            return None
        block_hashes_to_store = set(store_output.block_hashes_to_store)

        block_hashes = self._get_block_hashes(req, end_idx=num_blocks)
        self.manager.touch(block_hashes)

        new_block_hashes = self._get_block_hashes(
            req, start_idx=start_block_idx, end_idx=num_blocks
        )
        dst_spec = store_output.store_spec
        block_ids = req_status.block_ids
        src_block_ids: list[int] = []
        for idx, blk_hash in enumerate(new_block_hashes):
            if blk_hash not in block_hashes_to_store:
                continue
            offloaded_block_idx = start_block_idx + idx
            gpu_block_idx = offloaded_block_idx * self.block_size_factor
            for i in range(self.block_size_factor):
                src_block_ids.append(block_ids[gpu_block_idx + i])
        src_spec = GPULoadStoreSpec(src_block_ids, group_sizes=(len(src_block_ids),))

        self._reqs_being_stored[req_id] |= block_hashes_to_store

        logger.debug(
            "Request %s offloading %s blocks starting from block #%d",
            req_id,
            len(block_hashes_to_store),
            start_block_idx,
        )

        return src_spec, dst_spec

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
            block_hashes = self._reqs_being_loaded.pop(req_id, None)
            if block_hashes:
                if self._blocks_being_loaded:
                    self._blocks_being_loaded.difference_update(block_hashes)
                self.manager.complete_load(block_hashes)

    def request_finished(
        self,
        request: Request,
        block_ids: list[int],
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

        # TODO(orozery): possibly kickoff offload for last block
        # which may have been deferred due to async scheduling
        self._req_status.pop(req_id, None)

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
