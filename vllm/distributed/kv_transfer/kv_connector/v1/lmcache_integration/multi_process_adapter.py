# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from typing import Any

import torch
import zmq
from lmcache.utils import _lmcache_nvtx_annotate, init_logger
from lmcache.v1.multiprocess.custom_types import (
    CudaIPCWrapper,
    IPCCacheEngineKey,
    KVCache,
)
from lmcache.v1.multiprocess.mq import MessageQueueClient, MessagingFuture
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class

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


def get_lmcache_chunk_size(
    mq_client: MessageQueueClient,
) -> int:
    future = send_lmcache_request(mq_client, RequestType.GET_CHUNK_SIZE, [])
    chunk_size = future.result()
    return chunk_size


def striding_block_hashes(
    block_hashes: list[bytes],
    blocks_in_chunk,
) -> Iterable[bytes]:
    """Striding the block hashes to get the block hashes for each chunk.
    For example, if blocks_in_chunk is 16, then we will get the block hashes
    for the 16th, 32nd, 48th, ... blocks.
    """
    return islice(block_hashes, blocks_in_chunk - 1, None, blocks_in_chunk)


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


StoreResult = bool
RetrieveResult = list[bool]
LookupResult = list[bool]


class LMCacheMPSchedulerAdapter:
    def __init__(
        self,
        server_url: str,
        context: zmq.Context,
        model_name: str,
        world_size: int,
        kv_rank: int,
        vllm_block_size: int,
    ):
        """
        Args:
            server_url: The server URL for the LMCache message queue
            context: The ZMQ context

            model_name: The model name used for LMCache keys
            world_size: The world size used for LMCache keys
            kv_rank: The kv rank used for LMCache keys
            vllm_block_size: The block size used in vLLM
        """
        self.mq_client = MessageQueueClient(server_url, context)

        # Request futures
        self.lookup_futures: dict[str, MessagingFuture[LookupResult]] = {}

        self.model_name = model_name
        self.world_size = world_size
        self.worker_id = kv_rank

        # Read chunk size from lmcache
        self.chunk_size = get_lmcache_chunk_size(self.mq_client)
        assert self.chunk_size % vllm_block_size == 0, (
            "LMCache chunk size should be a multiple of vLLM block size"
        )
        self.blocks_in_chunk = self.chunk_size // vllm_block_size

    @_lmcache_nvtx_annotate
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

    @_lmcache_nvtx_annotate
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

    def num_blocks_per_chunk(self) -> int:
        """
        Returns:
            The number of vllm blocks in a LMCache data chunk
        """
        return self.blocks_in_chunk

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
    def __init__(
        self,
        server_url: str,
        context: zmq.Context,
        model_name: str,
        world_size: int,
        kv_rank: int,
        vllm_block_size: int,
    ):
        self.mq_client = MessageQueueClient(server_url, context)

        # Instance id for GPU worker
        self.instance_id = os.getpid()

        # Registered kv caches from vLLM
        self.kv_caches: dict[str, torch.Tensor] = {}

        # Request futures
        # request_id -> (future, other merged requests)
        self.store_futures: dict[
            str, tuple[MessagingFuture[StoreResult], list[str]]
        ] = {}
        self.retrieve_futures: dict[
            str, tuple[MessagingFuture[RetrieveResult], list[str]]
        ] = {}

        self.finished_stores: set[str] = set()
        self.previously_finished: set[str] = set()

        self.model_name = model_name
        self.world_size = world_size
        self.worker_id = kv_rank

        # Read chunk size from lmcache
        chunk_size = get_lmcache_chunk_size(self.mq_client)
        assert chunk_size % vllm_block_size == 0, (
            "LMCache chunk size should be a multiple of vLLM block size"
        )
        self.blocks_in_chunk = chunk_size // vllm_block_size

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

    @_lmcache_nvtx_annotate
    def submit_store_request(
        self, request_id: str, op: LoadStoreOp, event: torch.cuda.Event
    ):
        keys = self._block_hashes_to_keys(op.block_hashes)
        future = send_lmcache_request(
            self.mq_client,
            RequestType.STORE,
            [keys, self.instance_id, op.block_ids, event.ipc_handle()],
        ).to_cuda_future()
        self.store_futures[request_id] = (future, [])

    @_lmcache_nvtx_annotate
    def submit_retrieve_request(
        self, request_id: str, op: LoadStoreOp, event: torch.cuda.Event
    ):
        keys = self._block_hashes_to_keys(op.block_hashes)
        future = send_lmcache_request(
            self.mq_client,
            RequestType.RETRIEVE,
            [keys, self.instance_id, op.block_ids, event.ipc_handle()],
        ).to_cuda_future()
        self.retrieve_futures[request_id] = (future, [])

    @_lmcache_nvtx_annotate
    def batched_submit_store_requests(
        self,
        request_ids: list[str],
        ops: list[LoadStoreOp],
        event: torch.cuda.Event,
    ):
        keys = []
        block_ids = []
        for op in ops:
            keys.extend(self._block_hashes_to_keys(op.block_hashes))
            block_ids.extend(op.block_ids)
        future = send_lmcache_request(
            self.mq_client,
            RequestType.STORE,
            [keys, self.instance_id, block_ids, event.ipc_handle()],
        ).to_cuda_future()
        self.store_futures[request_ids[0]] = (future, request_ids[1:])

    @_lmcache_nvtx_annotate
    def batched_submit_retrieve_requests(
        self,
        request_ids: list[str],
        ops: list[LoadStoreOp],
        event: torch.cuda.Event,
    ):
        keys = []
        block_ids = []
        for op in ops:
            keys.extend(self._block_hashes_to_keys(op.block_hashes))
            block_ids.extend(op.block_ids)
        future = send_lmcache_request(
            self.mq_client,
            RequestType.RETRIEVE,
            [keys, self.instance_id, block_ids, event.ipc_handle()],
        ).to_cuda_future()
        self.retrieve_futures[request_ids[0]] = (future, request_ids[1:])

    @_lmcache_nvtx_annotate
    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        finished_stores = set()
        finished_retrieves = set()
        for request_id, (future, other_reqs) in self.store_futures.items():
            if not future.query():
                continue

            result = future.result()
            finished_stores.add(request_id)
            finished_stores.update(other_reqs)

            if not result:
                # TODO: add error handling here
                logger.error(
                    "Something went wrong when processing the "
                    "store request for request_id=%s",
                    request_id,
                )

        for request_id, (future, other_reqs) in self.retrieve_futures.items():
            if not future.query():
                continue

            result = future.result()
            finished_retrieves.add(request_id)
            finished_retrieves.update(other_reqs)

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
            self.store_futures.pop(request_id, None)
        for request_id in finished_retrieves:
            self.retrieve_futures.pop(request_id, None)

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

    def num_blocks_per_chunk(self) -> int:
        """
        Returns:
            The number of vllm blocks in a LMCache data chunk
        """
        return self.blocks_in_chunk

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
