
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations
from abc import ABC, abstractmethod
import queue
import threading
from typing import Callable
from collections import defaultdict
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.separated_encode.ec_transfer.connector.redis import (
    RedisECConnector)
from vllm.v1.request import Request

logger = init_logger(__name__)


class EncoderCachePreallocatorTemplate(ABC):

    def __init__(self, vllm_config: VllmConfig):
        self.ec_connector = RedisECConnector(
            vllm_config,
            "scheduler",
            self._receive_encoder_cache_metadata,
            None
        )

    @abstractmethod
    def is_empty(self,):
        pass 
    
    @abstractmethod
    def add_request(self, request: Request):
        pass

    @abstractmethod
    def finish_request(self, request: Request):
        pass

    @abstractmethod
    def update_mm_inputs_done(self, request: Request):
        pass

    @abstractmethod
    def _receive_encoder_cache_metadata(
        self, req_id: str, input_id: int, size: int
    ):     
        pass

class SyncEncoderCachePreallocator(EncoderCachePreallocatorTemplate):
    def __init__(
        self,
        vllm_config: VllmConfig,
        perform_allocation: Callable,
    ):
        super().__init__(vllm_config)

        self.active_requests: set[str] = set()

        self.mm_inputs_done = defaultdict(int)
        self.mm_inputs_total = defaultdict(int)
        
        self.preallocs_queue = queue.Queue()
        self.prealloc_candidate = None
        self.pending_preallocs: dict[str, set[int]] = {}
        
        self.waiting_preallocs: dict[str, list[int]] = {}
        
        self.ignored_preallocs = set()
        self.received_metas_reqs = set()

        self.recv_lock = threading.Lock()
        self.scheduling_lock = threading.Lock()

    def is_empty(self, ):
        return (self.preallocs_queue.qsize() == 0)

    def finish_request(self, request: Request):
        with self.recv_lock:
            if request.request_id in self.waiting_preallocs:
                self.waiting_preallocs.pop(request.request_id)
            if request.request_id in self.pending_preallocs:
                self.pending_preallocs.pop(request.request_id)
            for _ in range(self.mm_inputs_total[request.request_id]):
                # Clean ignored_preallocs later, currently we assume that
                # all mm_inputs will come to the instance at some moment
                if (request.request_id, _) in self.received_metas_reqs:
                    self.received_metas_reqs.remove((request.request_id, _))
                    continue
                self.ignored_preallocs.add((request.request_id, _))
            self.mm_inputs_done.pop(request.request_id)
            self.mm_inputs_total.pop(request.request_id)
            self.active_requests.remove(request.request_id)


    def _schedule_prealloc_request(self, req_id: str, input_id: int, size: int):
        if req_id not in self.pending_preallocs:
            self.pending_preallocs[req_id] = set()  
        self.pending_preallocs[req_id].add(input_id)
        self.preallocs_queue.put_nowait((req_id, input_id, size))

    def _receive_encoder_cache_metadata(self, req_id: str, input_id: int, size: int):
        # callback function
        with self.scheduling_lock:
            with self.recv_lock:
                if (req_id, input_id) in self.ignored_preallocs:
                    # if request is not active/data is obtained from KV cache
                    self.ignored_preallocs.remove((req_id, input_id))
                    self.ec_connector.schedule_send_prealloc_notification(
                        req_id, input_id, False
                    )
                    return            
                self.received_metas_reqs.add((req_id, input_id))
                if req_id not in self.active_requests:
                    if req_id not in self.waiting_preallocs:
                        self.waiting_preallocs[req_id] = []
                    self.waiting_preallocs[req_id].append((input_id, size))
                    return

                self._schedule_prealloc_request(req_id, input_id, size)

    def add_request(self, request: Request):
        with self.recv_lock:
            req_id = request.request_id
            self.active_requests.add(req_id)
            self.mm_inputs_done[req_id] = 0
            self.mm_inputs_total[req_id] = len(request.mm_inputs) 
            if req_id not in self.waiting_preallocs:
                return
            for (input_id, size) in self.waiting_preallocs[req_id]:
                self._schedule_prealloc_request(req_id, input_id, size)
            self.waiting_preallocs.pop(req_id)

    def update_mm_inputs_done(self, request: Request):
        if not request.has_encoder_inputs:
            return
        
        with self.scheduling_lock:
            req_id = request.request_id
            mm_inputs_done_local = self.mm_inputs_done[req_id] 

            while mm_inputs_done_local < self.mm_inputs_total[req_id]:
                pos_info = request.mm_positions[mm_inputs_done_local]
                mm_inputs_end = pos_info.offset + pos_info.length 
                if mm_inputs_end > request.num_computed_tokens:
                    break
                
                if (req_id in self.pending_preallocs 
                    and mm_inputs_done_local in self.pending_preallocs[req_id]
                ):
                    self.pending_preallocs[req_id].remove(mm_inputs_done_local)
                    self.ec_connector.schedule_send_prealloc_notification(
                        req_id, mm_inputs_done_local, False
                    )
                    self.ignored_preallocs.add((req_id, mm_inputs_done_local))
                mm_inputs_done_local += 1
            
            self.mm_inputs_done[req_id] = mm_inputs_done_local

    def get_prealloc_candidate(self, free_space: int, fill_next: bool):
        #-> tuple[bool, tuple[str, int, int]]:
        with self.scheduling_lock:
            if self.prealloc_candidate is None:
                if fill_next is True:
                    self.prealloc_candidate = self.preallocs_queue.get()
                return (True, None) # No candidate get next
                
            (request_id, input_id, encoder_cache_size) = self.prealloc_candidate
            if encoder_cache_size > free_space:
                return (False, None)

            if fill_next is True:
                self.prealloc_candidate = self.preallocs_queue.get()
            else:
                self.prealloc_candidate = None

            if (request_id, input_id) in self.ignored_preallocs:
                self.ignored_preallocs.remove((request_id, input_id))
                return (True, None) # Skip and get next
            
            self.pending_preallocs[request_id].remove(input_id)

            self.ec_connector.schedule_send_prealloc_notification(
                request_id, input_id, True
            )
            return (True, (request_id, input_id, encoder_cache_size))

class AsyncEncoderCachePreallocator(EncoderCachePreallocatorTemplate):
    def __init__(
        self,
        vllm_config: VllmConfig,
        perform_preallocations: Callable,
    ):
        super().__init__(vllm_config)
        self.perform_preallocations = perform_preallocations
        self.mm_inputs_done = defaultdict(int)
        self.mm_inputs_total = defaultdict(int)

        self.preallocs_queue = queue.Queue()
        self.prealloc_candidate = None
        self.recv_lock = threading.Lock()

    def finish_request(self, request: Request):
        pass

    def is_empty(self, ):
        return (self.preallocs_queue.qsize() == 0)

    def _receive_encoder_cache_metadata(self, req_id: str, input_id: int, size: int):
        self.preallocs_queue.put_nowait((req_id, input_id, size))
        self.perform_preallocations()

    def add_request(self, request: Request):
        self.mm_inputs_done[request.request_id] = 0
        self.mm_inputs_total[request.request_id] = len(request.mm_inputs)

    def update_mm_inputs_done(self, request: Request):
        if not request.has_encoder_inputs:
            return
        req_id = request.request_id
        mm_inputs_done_local = self.mm_inputs_done[req_id] 
        while mm_inputs_done_local < self.mm_inputs_total[req_id]:
            pos_info = request.mm_positions[mm_inputs_done_local]
            mm_inputs_end = pos_info.offset + pos_info.length 
            if mm_inputs_end > request.num_computed_tokens:
                break
            mm_inputs_done_local += 1
        
        self.mm_inputs_done[req_id] = mm_inputs_done_local

    def get_prealloc_candidate(self, free_space: int, fill_next: bool):
        if self.prealloc_candidate is None:
            if fill_next is True:
                self.prealloc_candidate = self.preallocs_queue.get()
            return (True, None) 
            
        (request_id, input_id, encoder_cache_size) = self.prealloc_candidate
        if encoder_cache_size > free_space:
            return (False, None)

        if fill_next is True:
            self.prealloc_candidate = self.preallocs_queue.get()
        else:
            self.prealloc_candidate = None
        self.ec_connector.schedule_send_prealloc_notification(
            request_id, input_id, True
        )
        return (True, (request_id, input_id, encoder_cache_size))