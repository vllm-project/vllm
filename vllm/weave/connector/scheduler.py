# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from collections.abc import Iterable
from itertools import islice
from typing import Any

from vllm.distributed.kv_events import BlockRemoved, BlockStored, KVCacheEvent
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_offload.abstract import OffloadingManager
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.worker.worker import TransferSpec
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

from .metadata import ReqId, RequestPhase, WeaveConnectorMetadata
from .policy import WeavePolicy
from ..weave_logger import get_weave_logger

logger = get_weave_logger(__name__)


class WeaveConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, spec: OffloadingSpec):
        self.gpu_block_size = spec.gpu_block_size
        self.offloaded_block_size = spec.offloaded_block_size
        self.block_size_factor = self.offloaded_block_size // self.gpu_block_size
        self.manager: OffloadingManager = spec.get_manager()

        self.policy = WeavePolicy(
            offloaded_block_size=self.offloaded_block_size,
            block_size_factor=self.block_size_factor,
            manager=self.manager,
        )

        self._requests: dict[ReqId, Request] = {}
        # list of GPU block IDs per request
        self._request_block_ids: dict[ReqId, list[int]] = {}
        # requests to load for the current scheduler step
        self._reqs_to_load: dict[ReqId, TransferSpec] = {}
        # request blocks are stored in order
        # index of next block (of size offloaded_block_size) to offload
        self._next_stored_block_idx: dict[ReqId, int] = {}

        # request ID -> set(block hashes being stored/load)
        self._reqs_being_stored = defaultdict[ReqId, set[BlockHash]](set)
        self._reqs_being_loaded = defaultdict[ReqId, set[BlockHash]](set)

        self._request_phases: dict[ReqId, RequestPhase] = {}

    def _refresh_request_phases(self) -> None:
        for req_id, req in self._requests.items():
            new_phase = (
                RequestPhase.PREFILL
                if req.num_computed_tokens < req.num_prompt_tokens
                else RequestPhase.DECODE
            )
            old_phase = self._request_phases.get(req_id)
            if old_phase is not None and old_phase != new_phase:
                logger.debug(
                    "Request %s phase %s -> %s (num_computed_tokens=%d, num_prompt_tokens=%d)",
                    req_id,
                    old_phase,
                    new_phase,
                    req.num_computed_tokens,
                    req.num_prompt_tokens,
                )
            self._request_phases[req_id] = new_phase

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
    ) -> tuple[int, bool]:
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
                - `True` if tokens will be loaded asynchronously
                  (between scheduler steps).
        """
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

        return num_hit_tokens, True

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        self._requests[request.request_id] = request
        self._request_phases[request.request_id] = RequestPhase.PREFILL
        # the block ids are updated in _get_reqs_to_store
        self._request_block_ids[request.request_id] = []

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
        dst_spec = GPULoadStoreSpec(block_ids[num_computed_gpu_blocks:])

        block_hashes = self._get_block_hashes(
            request, start_idx=start_block_idx, end_idx=num_blocks
        )

        self._reqs_to_load[request.request_id] = (src_spec, dst_spec)
        self._reqs_being_loaded[request.request_id].update(block_hashes)
        self._next_stored_block_idx[request.request_id] = num_blocks

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> WeaveConnectorMetadata:
        self._refresh_request_phases()
        num_prefill = 0
        num_decode = 0
        for req_id in scheduler_output.num_scheduled_tokens:
            phase = self._request_phases.get(req_id)
            if phase == RequestPhase.DECODE:
                num_decode += 1
            else:
                num_prefill += 1
        if scheduler_output.num_scheduled_tokens:
            logger.debug(
                "Weave request phase stats: prefill=%d decode=%d total=%d",
                num_prefill,
                num_decode,
                num_prefill + num_decode,
            )

        meta = WeaveConnectorMetadata(
            reqs_to_load=self._reqs_to_load,
            reqs_to_store=self.policy.get_reqs_to_store(
                scheduler_output,
                requests=self._requests,
                request_block_ids=self._request_block_ids,
                next_stored_block_idx=self._next_stored_block_idx,
                reqs_being_stored=self._reqs_being_stored,
                get_block_hashes=self._get_block_hashes,
                request_phases=self._request_phases,
            ),
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
        self._requests.pop(req_id, None)
        self._request_phases.pop(req_id, None)
        self._request_block_ids.pop(req_id, None)
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
