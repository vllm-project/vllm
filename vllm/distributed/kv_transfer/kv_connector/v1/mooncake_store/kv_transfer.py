import queue
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
from vllm.logger import logger

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.backend.backend import Backend

# isort: off
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.config_data import (
    ChunkedTokenDatabase,
    LayerMultiBlockReqMeta,
    ReqMeta,
)
# isort: on


class KVTransferThread(threading.Thread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        tp_rank: int,
        dcp_size: int,
        ready_event: threading.Event,
        name: str,
    ):
        super().__init__(daemon=True, name=name)
        self.m_store = m_store
        self.ready_event = ready_event
        self.block_size = block_size
        self.tp_rank = tp_rank
        self.dcp_size = dcp_size
        self.token_database = token_database
        self.done_task_lock = threading.Lock()
        self.request_queue: queue.Queue[Any] = queue.Queue()
        # TODO(jianzs): make this configurable
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.finished_requests: set[str] = set()

    def add_request(
        self,
        request: ReqMeta | LayerMultiBlockReqMeta,
    ) -> torch.Tensor:
        self.request_queue.put(request)

    def get_and_clear_finished_requests(self) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        with self.done_task_lock:
            finished_requests = self.finished_requests.copy()
            self.finished_requests.clear()
        return finished_requests

    def set_finished_request(self, req_id):
        with self.done_task_lock:
            self.finished_requests.add(req_id)

    def run(self):
        """Run the thread to handle KV cache transfer requests."""
        self.ready_event.set()
        while True:
            try:
                request_data = self.request_queue.get()
                if request_data is None:
                    logger.warning("Received a None request!")
                    self.request_queue.task_done()
                    continue
                self._handle_request(request_data)
            except Exception as e:
                logger.error(f"Error in KVCacheTransferThread: {e}")

    def _handle_request(self, req_meta: Any):
        pass

    def lookup(
        self,
        keys: list[str],
    ) -> int:
        """
        Checks the existence of KV cache of the tokens from the cache engine.
        :param tokens: the input tokens, with shape [seq_len]
        :return: An int indicating how many prefix tokens are cached.
        """
        try:
            res = self.m_store.exists(keys)  # type: ignore[assignment]
            for index, value in enumerate(res):  # type: ignore[arg-type]
                if value != 1:
                    return index
            # all tokens where found, return the maximal end
        except Exception as e:
            logger.error(f"Remote connection failed in contains: {e}")
            return 0
        return len(keys)


class KVCacheStoreSendingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        tp_rank: int,
        dcp_size: int,
        put_step: int,
        kv_role: str,
        ready_event: threading.Event,
    ):
        super().__init__(
            m_store, token_database, block_size, tp_rank, dcp_size, ready_event, name="KVCacheSendingThread"
        )
        self.put_step = put_step
        self.kv_role = kv_role
        self.stored_requests = defaultdict[str, int](int)

    def add_stored_request(self, req_id: str):
        with self.done_task_lock:
            self.stored_requests[req_id] += 1

    def dec_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                self.stored_requests[req_id] -= 1

    def delete_finished_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                del self.stored_requests[req_id]

    def _handle_request(self, req_meta: ReqMeta):
        token_len = req_meta.token_len_chunk
        block_ids = req_meta.block_ids
        req_id = req_meta.req_id
        current_event = req_meta.current_event
        starts = []
        ends = []
        keys = []
        if req_id not in self.stored_requests:
            self.request_queue.task_done()
            return

        for start, end, key in self.token_database.process_tokens(token_len, req_meta.block_hashes):
            starts.append(start)
            ends.append(end)
            keys.append(key.to_string())

        if not self.dcp_size > 1:
            starts = starts[self.tp_rank % self.put_step :: self.put_step]
            ends = ends[self.tp_rank % self.put_step :: self.put_step]
            keys = keys[self.tp_rank % self.put_step :: self.put_step]

        if not keys:
            self.dec_stored_request(req_id)
            return

        skip_block_num = self.lookup(keys)

        if skip_block_num == len(keys):
            self.dec_stored_request(req_id)
            return

        starts = starts[skip_block_num:]
        ends = ends[skip_block_num:]
        keys = keys[skip_block_num:]

        logger.info(
            "Storing KV cache for %d out of %d blocks (skip_block_num=%d) for request %s",
            len(keys),
            token_len // self.block_size,
            skip_block_num,
            req_id,
        )

        if keys:
            """
            Note: Due to a bug in ADXL, calling current_event.synchronize() may occasionally hang.
            This issue will be fixed in CANN version 8.5.rc1.
            You can manually build the master branch of the project at https://gitcode.com/cann/hixl
            to resolve this issue before the 8.5.RC1 release.
            """
            addrs = []
            sizes = []
            for index, start in enumerate(starts):
                addr, size, _ = self.token_database.prepare_value(start, ends[index], block_ids)
                addrs.append(addr)
                sizes.append(size)

            if self.kv_role == "kv_consumer":
                keys, addrs, sizes = self.token_database.decode_adaptor_prefill_pp(keys, addrs, sizes)

            if current_event is not None:
                current_event.synchronize()
            self.m_store.put(keys, addrs, sizes)

        self.dec_stored_request(req_id)
        self.request_queue.task_done()


class KVCacheStoreRecvingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        tp_rank: int,
        dcp_size: int,
        ready_event: threading.Event,
    ):
        super().__init__(
            m_store, token_database, block_size, tp_rank, dcp_size, ready_event, name="KVCacheStoreRecvingThread"
        )

    def _handle_request(self, req_meta: ReqMeta):
        token_len = req_meta.load_spec.token_len  # type: ignore[union-attr]
        req_id = req_meta.req_id
        mask_num = (
            req_meta.load_spec.vllm_cached_tokens  # type: ignore[union-attr]
            // self.block_size
            * self.block_size
        )
        addr_list = []
        size_list = []
        key_list = []
        for start, end, key in self.token_database.process_tokens(token_len, req_meta.block_hashes, mask_num):
            addr, size, _ = self.token_database.prepare_value(start, end, req_meta.block_ids)
            key_list.append(key.to_string())
            addr_list.append(addr)
            size_list.append(size)
        key_list_c = key_list[self.tp_rank % len(key_list) :] + key_list[: self.tp_rank % len(key_list)]
        addr_list_c = addr_list[self.tp_rank % len(addr_list) :] + addr_list[: self.tp_rank % len(addr_list)]
        size_list_c = size_list[self.tp_rank % len(size_list) :] + size_list[: self.tp_rank % len(size_list)]
        self.m_store.get(key_list_c, addr_list_c, size_list_c)
        self.set_finished_request(req_id)
        self.request_queue.task_done()


class KVCacheStoreLayerSendingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        tp_rank: int,
        dcp_size: int,
        put_step: int,
        ready_event: threading.Event,
        num_layers: int,
    ):
        super().__init__(
            m_store, token_database, block_size, tp_rank, dcp_size, ready_event, name="KVCacheStoreLayerSendingThread"
        )
        self.final_layer_id = num_layers - 1
        self.put_step = put_step

    def add_request(  # type: ignore[override]
        self, req_meta: ReqMeta
    ) -> torch.Tensor:
        self.request_queue.put(req_meta)

    def _handle_request(  # type: ignore[override]
        self, req_meta: LayerMultiBlockReqMeta
    ):
        starts = req_meta.starts
        ends = req_meta.ends
        keys = req_meta.keys
        layer_id = req_meta.layer_id
        current_event = req_meta.current_event
        total_block = len(keys)
        is_last_chunk = req_meta.is_last_chunk
        if not self.dcp_size > 1:
            starts = starts[self.tp_rank % self.put_step :: self.put_step]
            ends = ends[self.tp_rank % self.put_step :: self.put_step]
            keys = keys[self.tp_rank % self.put_step :: self.put_step]

        if not keys:
            if is_last_chunk:
                self.set_finished_request(req_meta.req_id)
            return

        key_list = []
        for key in keys:
            key_list.append(key.to_string())

        skip_block_num = self.lookup(key_list)

        if skip_block_num == len(key_list):
            if is_last_chunk and layer_id == self.final_layer_id:
                self.set_finished_request(req_meta.req_id)
            return

        starts = starts[skip_block_num:]
        ends = ends[skip_block_num:]
        key_list = key_list[skip_block_num:]

        addr_list = []
        size_list = []
        for index, key in enumerate(key_list):
            addr, size = self.token_database.prepare_value_layer(
                starts[index], ends[index], req_meta.block_ids, layer_id
            )
            addr_list.append(addr)
            size_list.append(size)

        if current_event is not None:
            current_event.synchronize()
        self.m_store.put(key_list, addr_list, size_list)

        if layer_id == self.final_layer_id and is_last_chunk:
            self.set_finished_request(req_meta.req_id)
        self.request_queue.task_done()

        logger.info(
            "Storing KV cache for %d out of %d blocks (skip_block_num=%d) for request %s",
            len(keys),
            total_block,
            skip_block_num,
            req_meta.req_id,
        )


class KVCacheStoreLayerRecvingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        tp_rank: int,
        dcp_size: int,
        ready_event: threading.Event,
        get_event: threading.Event,
    ):
        super().__init__(
            m_store, token_database, block_size, tp_rank, dcp_size, ready_event, name="KVCacheStoreLayerRecvingThread"
        )
        self.get_event = get_event

    def add_request(  # type: ignore[override]
        self, req_meta: LayerMultiBlockReqMeta
    ) -> torch.Tensor:
        self.request_queue.put(req_meta)

    def _handle_request(  # type: ignore[override]
        self, req_meta: LayerMultiBlockReqMeta
    ):
        addr_list = []
        size_list = []
        key_list = []
        for index, key in enumerate(req_meta.keys):
            addr, size = self.token_database.prepare_value_layer(
                req_meta.starts[index], req_meta.ends[index], req_meta.block_ids, req_meta.layer_id
            )
            key_list.append(key.to_string())
            addr_list.append(addr)
            size_list.append(size)
        key_list_c = key_list[self.tp_rank % len(key_list) :] + key_list[: self.tp_rank % len(key_list)]
        addr_list_c = addr_list[self.tp_rank % len(addr_list) :] + addr_list[: self.tp_rank % len(addr_list)]
        size_list_c = size_list[self.tp_rank % len(size_list) :] + size_list[: self.tp_rank % len(size_list)]
        self.m_store.get(key_list_c, addr_list_c, size_list_c)

        self.request_queue.task_done()
        self.get_event.set()
