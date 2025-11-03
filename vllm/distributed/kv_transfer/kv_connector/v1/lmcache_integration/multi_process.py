# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import enum
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, Literal

import torch
import zmq
from lmcache.utils import init_logger
from lmcache.v1.multiprocess.custom_types import (
    CudaIPCWrapper,
    IPCCacheEngineKey,
    KVCache,
)
from lmcache.v1.multiprocess.mq import MessageQueueClient, MessagingFuture
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.request import Request

# Type checking only
from vllm.v1.utils import ConstantList

logger = init_logger(__name__)


def wrap_kv_caches(kv_caches: dict[str, KVCache]) -> KVCache:
    logger.info("KV caches keys are %s", list(kv_caches.keys()))
    return [CudaIPCWrapper(tensor) for tensor in kv_caches.values()]


def send_lmcache_request(
    mq_client: MessageQueueClient,
    request_type: RequestType,
    payloads: list[Any],
) -> MessagingFuture[Any]:
    future = mq_client.submit_request(
        request_type, payloads, get_response_class(request_type)
    )
    return future


def striding_block_hashes(
    block_hashes: list[bytes],
    blocks_in_chunk,
) -> Iterable[bytes]:
    """Striding the block hashes to get the block hashes for each chunk.
    For example, if blocks_in_chunk is 16, then we will get the block hashes
    for the 16th, 32nd, 48th, ... blocks.
    """
    return islice(block_hashes, blocks_in_chunk - 1, None, blocks_in_chunk)


class LMCacheMPRequestState(enum.Enum):
    """
    State machine:
    PREFETCHING -- update_state_after_alloc --> WAITING_FOR_LOAD
    WAITING_FOR_LOAD -- process_loading_requests --> READY
    """

    PREFETCHING = enum.auto()
    WAITING_FOR_LOAD = enum.auto()
    READY = enum.auto()


@dataclass
class LoadStoreOp:
    block_hashes: list[bytes]
    block_ids: list[int]

    def __len__(self) -> int:
        return len(self.block_hashes)

    def __post_init__(self):
        assert len(self.block_hashes) == len(self.block_ids), (
            "The number of block hashes should be equal to the number of block ids "
            f"But got {len(self.block_hashes)} and {len(self.block_ids)}"
        )


@dataclass
class LMCacheMPRequestTracker:
    # NOTE: this class used vLLM data structures, should be part of
    # vLLM integration code

    request_id: str

    # Read-only lists to track the token ids and block hashes
    all_token_ids: ConstantList[int]
    block_hashes: ConstantList[BlockHash]

    # Block ids and hashes will be updated at update_states_after_alloc and
    # during the generation
    allocated_block_ids: list[int] = field(default_factory=list)

    # Number of blocks stored will be initialized when lookup the external
    # hit tokens and will be updated when processing new requests and cached
    # requests.
    num_stored_blocks: int = 0

    # Staging load operation -- save vllm and lmcache hit tokens during lookup
    num_vllm_hit_blocks: int = 0
    num_lmcache_hit_blocks: int = 0

    # Main state
    state: LMCacheMPRequestState = LMCacheMPRequestState.PREFETCHING

    def __init__(self, request: Request):
        self.request_id = request.request_id
        self.all_token_ids = request.all_token_ids
        self.block_hashes = ConstantList(request.block_hashes)
        self.allocated_block_ids = []
        self.num_stored_blocks = 0
        self.num_vllm_hit_blocks = 0
        self.num_lmcache_hit_blocks = 0
        self.state = LMCacheMPRequestState.PREFETCHING

    ####
    # Check the state of the request
    ####
    def needs_retrieve(self) -> bool:
        """Check whether the current request needs retrieve, will be used
        update_stage_after_alloc"""
        return (
            self.num_lmcache_hit_blocks > self.num_vllm_hit_blocks
            and self.state != LMCacheMPRequestState.READY
        )

    def is_ready_for_retrieving(self) -> bool:
        """Check whether the current request is ready for retrieving,
        will be used in process_loading_requests"""
        return (
            self.state == LMCacheMPRequestState.WAITING_FOR_LOAD
            and self.needs_retrieve()
        )

    ####
    # Update internal states
    ####
    def increase_num_stored_blocks(self, num_new_blocks: int):
        """Increase the number of stored blocks for the current request
        This function will be called when processing the cached requests.
        """
        self.num_stored_blocks += num_new_blocks
        # NOTE: when having a new request, we will first see the num computed
        # tokens before seeing the allocated block ids.
        # assert self.num_stored_blocks <= len(self.allocated_block_ids), \
        #        "The number of stored blocks should not exceed the number "\
        #        "of allocated blocks"

    def update_block_ids(
        self,
        new_block_ids: list[int],
    ):
        """Update the block ids for the current request
        This function will be called when processing the cached requests.
        """
        self.allocated_block_ids.extend(new_block_ids)

    ####
    # For debugging
    ####
    def __repr__(self) -> str:
        return (
            f"LMCacheMPRequestTracker(request_id={self.request_id}, "
            f"num_tokens={len(self.all_token_ids)}, "
            f"num_block_hashes={len(self.block_hashes)}, "
            f"num_allocated_blocks={len(self.allocated_block_ids)}, "
            f"num_stored_blocks={self.num_stored_blocks}, "
            f"vllm_hit_blocks={self.num_vllm_hit_blocks}, "
            f"lmcache_hit_blocks={self.num_lmcache_hit_blocks}, "
            f"state={self.state})"
        )

    def __str__(self) -> str:
        return self.__repr__()


@dataclass
class LMCacheMPRequestMetadata:
    request_id: str
    direction: Literal["STORE", "RETRIEVE"]
    op: LoadStoreOp

    @staticmethod
    def GetStoreMetadata(
        tracker: LMCacheMPRequestTracker,
        blocks_in_chunk: int,
    ) -> "LMCacheMPRequestMetadata | None":
        """
        Generate the store metadata for the current request tracker.

        Args:
            tracker: The request tracker to generate the metadata from.
            blocks_in_chunk: the number of blocks in a LMCache data chunk
        """
        # Store the blocks that has block hashes
        # NOTE: the invariant here is that `num_stored_blocks` should
        # always be a multiple of `blocks_in_chunk`
        # TODO: This should be checked everytime we update the num_stored_blocks
        min_available_blocks = min(
            len(tracker.block_hashes), len(tracker.allocated_block_ids)
        )
        num_staging_blocks = min_available_blocks - tracker.num_stored_blocks
        num_chunks = num_staging_blocks // blocks_in_chunk

        if num_chunks >= 1:
            start = tracker.num_stored_blocks
            end = start + num_chunks * blocks_in_chunk
            block_hashes = tracker.block_hashes[start:end]
            block_ids = tracker.allocated_block_ids[start:end]

            ret = LMCacheMPRequestMetadata(
                request_id=tracker.request_id,
                direction="STORE",
                op=LoadStoreOp(block_hashes=block_hashes, block_ids=block_ids),
            )

            # Update the request tracker
            tracker.increase_num_stored_blocks(end - start)
            return ret

        return None

    @staticmethod
    def GetRetrieveMetadata(
        tracker: LMCacheMPRequestTracker,
        blocks_in_chunk: int,
    ) -> "LMCacheMPRequestMetadata | None":
        """
        Generate the retrieve metadata for the current request tracker.

        Args:
            tracker: The request tracker to generate the metadata from.
            blocks_in_chunk: the number of blocks in a LMCache data chunk
        """
        if not tracker.is_ready_for_retrieving():
            return None

        # |---------------------|-----------------|----------------|
        # | num_vllm_hit_blocks |
        # | lmcache chunk 1   | lmcache chunk 2   |
        #                     |  need to retrieve |

        start = tracker.num_vllm_hit_blocks // blocks_in_chunk * blocks_in_chunk
        end = tracker.num_lmcache_hit_blocks
        assert end % blocks_in_chunk == 0, (
            "The number of LMCache hit blocks should be a multiple of the "
            "number of blocks in a lmcache chunk. "
        )
        assert len(tracker.block_hashes) >= end, (
            "The number of block hashes should be greater than or equal to the "
            "number of LMCache hit blocks. "
        )
        if end > start:
            block_hashes = tracker.block_hashes[start:end]
            block_ids = tracker.allocated_block_ids[start:end]

            ret = LMCacheMPRequestMetadata(
                request_id=tracker.request_id,
                direction="RETRIEVE",
                op=LoadStoreOp(block_hashes=block_hashes, block_ids=block_ids),
            )
            return ret

        return None


StoreResult = bool
RetrieveResult = list[bool]
LookupResult = list[bool]


class LMCacheMPSchedulerAdapter:
    def __init__(self, server_url: str, context: zmq.Context):
        self.mq_client = MessageQueueClient(server_url, context)

        # Request futures
        self.lookup_futures: dict[str, MessagingFuture[LookupResult]] = {}

        # TODO: metadata is hard-coded for now, please remove
        self.model_name = "Qwen/Qwen3-0.6B"
        self.world_size = 1
        self.worker_id = 0

        self.blocks_in_chunk = 16
        self.chunk_size = 256  # Chunk size in tokens

        pass

    def maybe_submit_lookup_request(self, request_id: str, block_hashes: list[bytes]):
        if request_id in self.lookup_futures:
            # Skip if there is already a lookup request
            return

        s = striding_block_hashes(block_hashes, self.blocks_in_chunk)
        keys = [self._create_key(block_hash) for block_hash in s]
        future = send_lmcache_request(
            self.mq_client,
            RequestType.LOOKUP,
            [keys, True],
        )
        self.lookup_futures[request_id] = future

    def check_lookup_result(self, request_id: str) -> int | None:
        assert request_id in self.lookup_futures, (
            f"Lookup request for request_id={request_id} has not been submitted"
        )

        future = self.lookup_futures[request_id]
        if not future.query():
            return None

        result = future.result()
        num_chunks = sum(result)
        return num_chunks * self.chunk_size

    # Helper functions
    def _create_key(self, block_hash: bytes) -> IPCCacheEngineKey:
        """Convert a block hash to an IPC cache engine key"""
        return IPCCacheEngineKey(
            model_name=self.model_name,
            world_size=self.world_size,
            worker_id=self.worker_id,
            chunk_hash=block_hash,
        )


class LMCacheMPWorkerAdapter:
    def __init__(self, server_url: str, context: zmq.Context):
        self.mq_client = MessageQueueClient(server_url, context)

        # Instance id for GPU worker
        self.instance_id = os.getpid()

        # Registered kv caches from vLLM
        self.kv_caches: dict[str, torch.Tensor] = {}

        # Request futures
        self.store_futures: dict[str, MessagingFuture[StoreResult]] = {}
        self.retrieve_futures: dict[str, MessagingFuture[RetrieveResult]] = {}

        self.finished_stores: set[str] = set()
        self.previously_finished: set[str] = set()

        # TODO: metadata is hard-coded for now, please remove
        self.model_name = "Qwen/Qwen3-0.6B"
        self.world_size = 1
        self.worker_id = 0

        self.blocks_in_chunk = 16

    def register_kv_caches(self, kv_caches: dict[str, KVCache]):
        # Register kv cache and send the request
        self.kv_caches = kv_caches
        logger.info("Registering kv caches")
        future = send_lmcache_request(
            self.mq_client,
            RequestType.REGISTER_KV_CACHE,
            [self.instance_id, wrap_kv_caches(kv_caches)],
        )
        future.result()

    def submit_store_request(self, request_id: str, op: LoadStoreOp):
        keys = self._block_hashes_to_keys(op.block_hashes)
        future = send_lmcache_request(
            self.mq_client, RequestType.STORE, [keys, self.instance_id, op.block_ids]
        )
        self.store_futures[request_id] = future

    def submit_retrieve_request(self, request_id: str, op: LoadStoreOp):
        # TODO: submit the retrieve task
        keys = self._block_hashes_to_keys(op.block_hashes)
        future = send_lmcache_request(
            self.mq_client, RequestType.RETRIEVE, [keys, self.instance_id, op.block_ids]
        )
        self.retrieve_futures[request_id] = future

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        finished_stores = set()
        finished_retrieves = set()
        for request_id, future in self.store_futures.items():
            if not future.query():
                continue

            result = future.result()
            finished_stores.add(request_id)
            if not result:
                # TODO: add error handling here
                logger.error(
                    "Something went wrong when processing the "
                    "store request for request_id=%s",
                    request_id,
                )

        for request_id, future in self.retrieve_futures.items():
            if not future.query():
                continue

            result = future.result()
            finished_retrieves.add(request_id)
            if not all(result):
                # TODO: add error handing here
                logger.error(
                    "Something went wrong when processing the "
                    "retrieve request for request_id=%s, result=%s",
                    request_id,
                    result,
                )

        # Remove the finished requests from the tracking dicts
        for request_id in finished_stores:
            del self.store_futures[request_id]
        for request_id in finished_retrieves:
            del self.retrieve_futures[request_id]

        # Update the internal states
        self.finished_stores.update(finished_stores)

        ret_stores = set()
        for req_id in finished_req_ids:
            if req_id in self.finished_stores or req_id in self.store_futures:
                self.previously_finished.add(req_id)
            else:
                ret_stores.add(req_id)

        # Calculate the final finished stores
        ret_stores.update(self._update_and_get_finished_store())

        return ret_stores, finished_retrieves

    def shutdown(self):
        # Unregister kv cache
        logger.info("Unregistering kv caches")
        send_lmcache_request(
            self.mq_client, RequestType.UNREGISTER_KV_CACHE, [self.instance_id]
        ).result()

        self.mq_client.close()

    # Helper functions
    def _update_and_get_finished_store(
        self,
    ) -> set[str]:
        """Converge the internal states about finished stores
        and returns the 'safe finished store request ids' back
        """
        safe_finished_s = self.finished_stores.intersection(self.previously_finished)
        self.finished_stores.difference_update(self.previously_finished)
        self.previously_finished.difference_update(safe_finished_s)

        return safe_finished_s

    def _create_key(self, block_hash: bytes) -> IPCCacheEngineKey:
        """Convert a block hash to an IPC cache engine key"""
        return IPCCacheEngineKey(
            model_name=self.model_name,
            world_size=self.world_size,
            worker_id=self.worker_id,
            chunk_hash=block_hash,
        )

    def _block_hashes_to_keys(
        self, block_hashes: list[bytes]
    ) -> list[IPCCacheEngineKey]:
        """Convert block hashes to IPC cache engine keys"""
        s = striding_block_hashes(block_hashes, self.blocks_in_chunk)
        return [self._create_key(block_hash) for block_hash in s]
