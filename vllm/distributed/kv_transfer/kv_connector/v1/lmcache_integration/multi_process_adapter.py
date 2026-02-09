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


def wrap_kv_caches(kv_caches: dict[str, torch.Tensor]) -> KVCache:
    logger.info("KV caches keys are %s", list(kv_caches.keys()))
    return [CudaIPCWrapper(tensor) for tensor in kv_caches.values()]


def striding_block_hashes(
    block_hashes: list[bytes], blocks_in_chunk: int
) -> Iterable[bytes]:
    """Extract chunk-level hashes from block hashes by striding.

    In hash-based vLLM, each vLLM block has its own hash.  LMCache chunks
    span ``blocks_in_chunk`` consecutive blocks.  The representative hash
    for a chunk is the hash of the **last** block in that chunk (because
    each block hash already encodes its prefix).  So we start at index
    ``blocks_in_chunk - 1`` and stride by ``blocks_in_chunk``.
    """
    return islice(block_hashes, blocks_in_chunk - 1, None, blocks_in_chunk)


def send_lmcache_request(
    mq_client: MessageQueueClient,
    request_type: RequestType,
    payloads: list[Any],
) -> MessagingFuture[Any]:
    """
    Helper function to send the request to the LMCache multiprocess server

    Args:
        mq_client: The LMCache multiprocess mode message queue client
        request_type: The request type
        payloads: The request payloads

    Returns:
        A messaging future for the request
    """

    future = mq_client.submit_request(
        request_type, payloads, get_response_class(request_type)
    )
    return future


def get_lmcache_chunk_size(
    mq_client: MessageQueueClient,
) -> int:
    """
    Helper function to get the LMCache chunk size from the server

    Args:
        mq_client: The LMCache multiprocess mode message queue client

    Returns:
        An integer representing the LMCache chunk size
    """
    future = send_lmcache_request(mq_client, RequestType.GET_CHUNK_SIZE, [])
    chunk_size = future.result()
    return chunk_size


@dataclass
class LoadStoreOp:
    block_ids: list[int]
    """Block ids for the load/store operation"""

    token_ids: list[int] | None = None
    """Token IDs for the load/store operation (token mode)"""

    block_hashes: list[bytes] | None = None
    """Block hashes for the load/store operation (hash mode)"""

    start: int = 0
    """Start token index (token mode only)"""

    end: int = 0
    """End token index (token mode only)"""

    def __len__(self) -> int:
        return len(self.block_ids)


StoreResult = bool
RetrieveResult = list[bool]
LookupResult = int


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
    def maybe_submit_lookup_request(
        self,
        request_id: str,
        block_hashes: list[bytes] | None = None,
        token_ids: list[int] | None = None,
    ):
        """
        Submit a new lookup request to LMCache if there is no ongoing request.

        Supports both token-based and hash-based vLLM:
        - token_ids: token IDs (token-based vLLM) -> single token-mode key
        - block_hashes: block hashes (hash-based vLLM) -> strided hash-mode keys

        Exactly one of block_hashes or token_ids must be provided.

        Args:
            request_id: The ID of the lookup request. The same ID indicates it's
                from the same request
            block_hashes: Block hashes to lookup from LMCache (hash mode)
            token_ids: Token IDs to lookup from LMCache (token mode)

        Returns:
            None

        Notes:
            This function will have a side-effect: submitting a look up request to
            LMCache, which will essentially 'lock' the KV cache chunks in the LMCache
            for later retrieve operations.
            In the meantime, this function will record the lookup request, and the
            status of the look up request can be checked by `check_lookup_result`.
        """
        if request_id in self.lookup_futures:
            # Skip if there is already a lookup request
            return

        assert (block_hashes is None) != (token_ids is None), (
            "Exactly one of block_hashes or token_ids must be provided"
        )

        if block_hashes is not None:
            # Hash mode: stride block hashes -> N hash-mode keys
            chunk_hashes = list(
                striding_block_hashes(block_hashes, self.blocks_in_chunk)
            )
            keys = [
                self._create_hash_key(ch, request_id=request_id) for ch in chunk_hashes
            ]
        else:
            # Token mode: truncate to chunk-aligned length
            assert token_ids is not None
            aligned_end = (len(token_ids) // self.chunk_size) * self.chunk_size
            if aligned_end == 0:
                return
            keys = [
                self._create_key(
                    token_ids,
                    start=0,
                    end=aligned_end,
                    request_id=request_id,
                ).no_worker_id_version()
            ]

        future = send_lmcache_request(
            self.mq_client,
            RequestType.LOOKUP,
            [keys],
        )
        self.lookup_futures[request_id] = future

    @_lmcache_nvtx_annotate
    def check_lookup_result(self, request_id: str) -> int | None:
        """
        Check the result of a previously submitted lookup request.

        Args:
            request_id: The ID of the lookup request submitted in
                `maybe_submit_lookup_request`

        Returns:
            An integer representing the total number of tokens matched
            in LMCache (prefix matching), or
            None if the lookup request is not finished yet.
        """
        assert request_id in self.lookup_futures, (
            f"Lookup request for request_id={request_id} has not been submitted"
        )

        future = self.lookup_futures[request_id]
        if not future.query():
            return None

        result = future.result()
        num_chunks = result
        return num_chunks * self.chunk_size

    def num_blocks_per_chunk(self) -> int:
        """
        Returns:
            The number of vllm blocks in a LMCache data chunk
        """
        return self.blocks_in_chunk

    def cleanup_lookup_result(self, request_id: str) -> None:
        """
        Clean up lookup future for a finished request to prevent memory leak.
        Args:
            request_id: The ID of the finished request.
        """
        self.lookup_futures.pop(request_id, None)

    def end_session(self, request_id: str) -> None:
        """
        Notify LMCache server to remove the session for a finished request.
        Args:
            request_id: The ID of the finished request.
        """
        send_lmcache_request(
            self.mq_client,
            RequestType.END_SESSION,
            [request_id],
        )

    # Helper functions
    def _create_key(
        self,
        token_ids: list[int],
        start: int = 0,
        end: int = 0,
        request_id: str | None = None,
    ) -> IPCCacheEngineKey:
        """Convert token IDs to an IPC cache engine key"""
        return IPCCacheEngineKey(
            model_name=self.model_name,
            world_size=self.world_size,
            worker_id=self.worker_id,
            token_ids=tuple(token_ids),
            start=start,
            end=end,
            request_id=request_id,
        )

    def _create_hash_key(
        self, chunk_hash: bytes, request_id: str | None = None
    ) -> IPCCacheEngineKey:
        """Create a hash-mode IPC cache engine key"""
        return IPCCacheEngineKey(
            model_name=self.model_name,
            world_size=self.world_size,
            worker_id=None,
            chunk_hash=chunk_hash,
            request_id=request_id,
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

        # The store requests that have finished execution in LMCache
        self.finished_stores: set[str] = set()
        # The finished request ids that are passed via vLLM and also
        # have corresponding store requests submitted to LMCache before
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

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """
        Register the kv caches with LMCache server

        Args:
            kv_caches: A dict of kv caches to register. The keys are the
                layer names and the values are the corresponding tensors.
        """
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
        """
        Submit a KV cache store request to LMCache

        Args:
            request_id: The ID of the request
            op: The LoadStoreOp describing the store operation.
            event: The CUDA event that is recorded after the current
                model inference step
        """
        if op.block_hashes is not None:
            # Hash mode
            chunk_hashes = list(
                striding_block_hashes(op.block_hashes, self.blocks_in_chunk)
            )
            keys = [
                self._create_hash_key(ch, request_id=request_id) for ch in chunk_hashes
            ]
        else:
            # Token mode
            assert op.token_ids is not None
            keys = [
                self._create_key(op.token_ids, op.start, op.end, request_id=request_id)
            ]
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
        """
        Submit a KV cache retrieve request to LMCache

        Args:
            request_id: The ID of the request
            op: The LoadStoreOp describing the retrieve operation.
            event: The CUDA event that is recorded after the current
                model inference step
        """
        if op.block_hashes is not None:
            # Hash mode
            chunk_hashes = list(
                striding_block_hashes(op.block_hashes, self.blocks_in_chunk)
            )
            keys = [
                self._create_hash_key(ch, request_id=request_id) for ch in chunk_hashes
            ]
        else:
            # Token mode
            assert op.token_ids is not None
            keys = [
                self._create_key(op.token_ids, op.start, op.end, request_id=request_id)
            ]
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
        """
        Submit a batched store request to LMCache

        Args:
            request_ids: The IDs of the requests
            ops: The LoadStoreOps describing the store operations. Should have
                the same length as request_ids
            event: The CUDA event that is recorded after the current
                model inference step
        """
        all_keys: list[IPCCacheEngineKey] = []
        block_ids: list[int] = []
        for request_id, op in zip(request_ids, ops, strict=False):
            if op.block_hashes is not None:
                chunk_hashes = list(
                    striding_block_hashes(op.block_hashes, self.blocks_in_chunk)
                )
                keys = [
                    self._create_hash_key(ch, request_id=request_id)
                    for ch in chunk_hashes
                ]
                all_keys.extend(keys)
            else:
                assert op.token_ids is not None
                all_keys.append(
                    self._create_key(
                        op.token_ids, op.start, op.end, request_id=request_id
                    )
                )
            block_ids.extend(op.block_ids)
        future = send_lmcache_request(
            self.mq_client,
            RequestType.STORE,
            [
                all_keys,
                self.instance_id,
                block_ids,
                event.ipc_handle(),
            ],
        ).to_cuda_future()
        self.store_futures[request_ids[0]] = (future, list(request_ids[1:]))

    @_lmcache_nvtx_annotate
    def batched_submit_retrieve_requests(
        self,
        request_ids: list[str],
        ops: list[LoadStoreOp],
        event: torch.cuda.Event,
    ):
        """
        Submit a batched retrieve request to LMCache

        Args:
            request_ids: The IDs of the requests
            ops: The LoadStoreOps describing the retrieve operations. Should have
                the same length as request_ids
            event: The CUDA event that is recorded after the current
                model inference step
        """
        all_keys: list[IPCCacheEngineKey] = []
        block_ids: list[int] = []
        for request_id, op in zip(request_ids, ops, strict=False):
            if op.block_hashes is not None:
                chunk_hashes = list(
                    striding_block_hashes(op.block_hashes, self.blocks_in_chunk)
                )
                keys = [
                    self._create_hash_key(ch, request_id=request_id)
                    for ch in chunk_hashes
                ]
                all_keys.extend(keys)
            else:
                assert op.token_ids is not None
                all_keys.append(
                    self._create_key(
                        op.token_ids, op.start, op.end, request_id=request_id
                    )
                )
            block_ids.extend(op.block_ids)
        future = send_lmcache_request(
            self.mq_client,
            RequestType.RETRIEVE,
            [
                all_keys,
                self.instance_id,
                block_ids,
                event.ipc_handle(),
            ],
        ).to_cuda_future()
        self.retrieve_futures[request_ids[0]] = (future, list(request_ids[1:]))

    @_lmcache_nvtx_annotate
    def get_finished(
        self, finished_req_ids_from_engine: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Check and get the finished store and retrieve requests.

        Args:
            finished_req_ids_from_engine: the set of request ids that are
                reported as finished from the vLLM engine side.

        Returns:
            A tuple of two sets:
            - The first set contains the finished store request ids. The returned
                store request ids MUST be seen before in the
                `finished_req_ids_from_engine`.
            - The second set contains the finished retrieve request ids.

        Notes:
            When enabling async scheduling in vLLM, the same request ID may appear
            multiple times in `finished_req_ids_from_engine`. The adapter should
            take care of deduplicating the request IDs and only return the request
            IDs that have not been returned before.
        """
        finished_stores = set()
        finished_retrieves = set()
        for request_id, (s_future, other_reqs) in self.store_futures.items():
            if not s_future.query():
                continue

            s_result = s_future.result()
            finished_stores.add(request_id)
            finished_stores.update(other_reqs)

            if not s_result:
                # TODO: add error handling here
                logger.error(
                    "Something went wrong when processing the "
                    "store request for request_id=%s",
                    request_id,
                )

        for request_id, (r_future, other_reqs) in self.retrieve_futures.items():
            if not r_future.query():
                continue

            r_result = r_future.result()
            finished_retrieves.add(request_id)
            finished_retrieves.update(other_reqs)

            if not all(r_result):
                # TODO: add error handing here
                logger.error(
                    "Something went wrong when processing the "
                    "retrieve request for request_id=%s, result=%s",
                    request_id,
                    r_result,
                )

        # Remove the finished requests from the tracking dicts
        for request_id in finished_stores:
            self.store_futures.pop(request_id, None)
        for request_id in finished_retrieves:
            self.retrieve_futures.pop(request_id, None)

        # Update the internal states
        self.finished_stores.update(finished_stores)

        ret_stores = set()
        for req_id in finished_req_ids_from_engine:
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
        """
        Shutdown the LMCache MP worker adapter
        """
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

    def _create_key(
        self,
        token_ids: list[int],
        start: int = 0,
        end: int = 0,
        request_id: str | None = None,
    ) -> IPCCacheEngineKey:
        """Convert token IDs to an IPC cache engine key"""
        return IPCCacheEngineKey(
            model_name=self.model_name,
            world_size=self.world_size,
            worker_id=self.worker_id,
            token_ids=tuple(token_ids),
            start=start,
            end=end,
            request_id=request_id,
        )

    def _create_hash_key(
        self, chunk_hash: bytes, request_id: str | None = None
    ) -> IPCCacheEngineKey:
        """Create a hash-mode IPC cache engine key"""
        return IPCCacheEngineKey(
            model_name=self.model_name,
            world_size=self.world_size,
            worker_id=self.worker_id,
            chunk_hash=chunk_hash,
            request_id=request_id,
        )
