
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations
from abc import ABC, abstractmethod
import os
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
    """Abstract base class for encoder cache preallocation strategies.

    Defines the interface for managing encoder cache preallocation in 
    disaggregated deployments. Concrete implementations handle the coordination
    between encoding and prefill instances, ensuring cache space is reserved
    before encoder outputs are transferred.

    This template provides the connection infrastructure through RedisECConnector
    and defines the required methods that subclasses must implement to handle
    request lifecycle, preallocation scheduling, and metadata reception.

    Attributes:
        ec_connector: Redis-based connector for communication between instances.
            Handles metadata reception and preallocation notifications.
    """
    def __init__(self, vllm_config: VllmConfig):
        """Initialize the preallocator with ECConnector."""
        self.ec_connector = RedisECConnector(
            vllm_config = vllm_config,
            device = None,
            # no need to pass device if intra_instance_type scheduler
            intra_instance_type = "scheduler",
            preallocate_callback = self._receive_encoder_cache_metadata,
            injection_callback = None,
            redis_host=os.getenv("REDIS_HOST"),
            redis_port=os.getenv("REDIS_PORT"),
        )

    @abstractmethod
    def is_empty(self,) -> bool:
        """Check if there are pending preallocations to process.
        """
        pass 
    
    @abstractmethod
    def add_request(self, request: Request):
        """Register a new request for preallocation tracking.

        Called when a request arrives at the prefill instance. Implementations
        should initialize tracking structures and process any metadata that
        arrived before the request.
        """
        pass

    @abstractmethod
    def finish_request(self, request: Request):
        """Clean up resources for a finished or cancelled request.

        Called when a request completes, is cancelled, or is aborted.
        Implementations should remove all tracking information and handle
        any pending preallocations for this request.
        """
        pass

    @abstractmethod
    def update_mm_inputs_done(self, request: Request):
        """Update multimodal input processing progress for a request.
        """
        pass

    @abstractmethod
    def _receive_encoder_cache_metadata(
        self, req_id: str, input_id: int, size: int, mm_hash: str
    ):     
        """Handle incoming encoder cache metadata from encoding instance.

        Callback method invoked by ec_connector when metadata arrives.
        Implementations should handle both cases where the request is
        already active and where metadata arrives before the request.
        """
        pass

class SyncEncoderCachePreallocator(EncoderCachePreallocatorTemplate):
    """Synchronous preallocation manager for encoder cache in disaggregated systems.

    This class coordinates the preallocations of encoder cache space between
    encoding and prefill instances in a disaggregated deployment. It ensures
    that cache space is reserved before encoder outputs are transferred between
    instances, manages the lifecycle of multimodal inputs across distributed 
    processing stages, handles out-of-order arrival of encoder metadata and
    ensures proper synchronization between request arrival and encoder cache
    metadata receiving.

    Key responsibilities:
    - Queue and schedule preallocation requests
    - Track multimodal input processing progress for each request
    - Handle metadata that arrives before or after request initialization
    - Send notification through EC Connector
    - Provide requests with inputs that are ready for preallocations
    - Clean up resources for finished & cancelled requests

    Attributes:
        active_requests: Set of request IDs currently being processed.
        mm_inputs_done: Maps request ID to number of processed multimodal inputs.
        mm_inputs_total: Maps request ID to total number of multimodal inputs.
        preallocs_queue: Queue of pending preallocation requests.
        prealloc_candidate: Current preallocation being considered for processing.
        pending_preallocs: Maps request ID to set of input IDs awaiting preallocation.
        waiting_preallocs: Stores preallocation data for requests not yet received on instance.
        ignored_preallocs: Set of (request_id, input_id) pairs to skip.
        received_metas_reqs: Set of (request_id, input_id) pairs with received metadata.
    """
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

    def _schedule_prealloc_request(self, req_id: str, input_id: int, 
                                   size: int, mm_hash: str):
        """Schedule a preallocation request for processing.

        Internal method that adds a preallocation request to the queue and
        tracks it in pending_preallocs.
        """
        if req_id not in self.pending_preallocs:
            self.pending_preallocs[req_id] = set()  
        self.pending_preallocs[req_id].add(input_id)
        self.preallocs_queue.put_nowait((req_id, input_id, size, mm_hash))

    def _receive_encoder_cache_metadata(self, req_id: str, input_id: int,
                                        size: int, mm_hash: str):
        """Handle incoming encoder cache metadata from encoding instance.

        This callback processes metadata about encoder outputs that need to be
        transferred. If the request is active, it schedules preallocation. If not,
        it stores the metadata for later processing when the request arrives.
        """

        with self.scheduling_lock:
            with self.recv_lock:
                if (req_id, input_id) in self.ignored_preallocs:
                    # if request is not active/data is obtained from KV cache
                    self.ignored_preallocs.remove((req_id, input_id))
                    self.ec_connector.schedule_send_prealloc_notification(
                        req_id, input_id, False, mm_hash
                    )
                    return
                self.received_metas_reqs.add((req_id, input_id))
                if req_id not in self.active_requests:
                    if req_id not in self.waiting_preallocs:
                        self.waiting_preallocs[req_id] = []
                    self.waiting_preallocs[req_id].append((input_id, size))
                    return

                self._schedule_prealloc_request(req_id, input_id, size, mm_hash)

    def add_request(self, request: Request):
        """Register a new request and process any waiting preallocations.

        When a request arrives, this method initializes tracking structures and
        processes any encoder metadata that arrived before the request.
        """
        with self.recv_lock:
            req_id = request.request_id
            self.active_requests.add(req_id)
            self.mm_inputs_done[req_id] = 0
            self.mm_inputs_total[req_id] = len(request.mm_hashes) 
            if req_id not in self.waiting_preallocs:
                return
            for (input_id, size) in self.waiting_preallocs[req_id]:
                mm_hash = request.mm_hashes[input_id]
                self._schedule_prealloc_request(req_id, input_id, size, mm_hash)
            self.waiting_preallocs.pop(req_id)

    def update_mm_inputs_done(self, request: Request):
        """Update the progress of multimodal input processing for a request.

        Tracks which multimodal inputs have been fully processed based on the
        number of computed tokens. For inputs that were prealloc candidates but
        are now obtained from cache, sends notifications to cancel transfers.
        """

        if not request.has_encoder_inputs:
            return
        
        with self.scheduling_lock:
            req_id = request.request_id
            mm_inputs_done_local = self.mm_inputs_done[req_id] 

            while mm_inputs_done_local < self.mm_inputs_total[req_id]:
                mm_hash = request.mm_hashes[mm_inputs_done_local]
                pos_info = request.mm_positions[mm_inputs_done_local]
                mm_inputs_end = pos_info.offset + pos_info.length 
                if mm_inputs_end > request.num_computed_tokens:
                    break
                
                if (req_id in self.pending_preallocs 
                    and mm_inputs_done_local in self.pending_preallocs[req_id]
                ):
                    self.pending_preallocs[req_id].remove(mm_inputs_done_local)
                    self.ec_connector.schedule_send_prealloc_notification(
                        req_id, mm_inputs_done_local, False, mm_hash 
                    )
                    self.ignored_preallocs.add((req_id, mm_inputs_done_local))
                mm_inputs_done_local += 1
            
            self.mm_inputs_done[req_id] = mm_inputs_done_local

    def get_prealloc_candidate(
        self, free_space: int, fill_next: bool
    ) -> tuple[bool, tuple[str, int, int, str] | None]:
        """Validate the preallocation candidate, fill the next preallocation
        candidate

        Validate current preallocation candidate, retrieves the next 
        preallocation request from the queue. Skips ignored preallocations
        and checks whether prellocated data will fit in space constraints.

        Args:
            free_space: Available cache space in encoder tokens.
            fill_next: Whether to fetch the next candidate after processing.

        Returns:
            Tuple of (should_continue, candidate_data) where:
            - should_continue: True if caller should continue preallocations, 
                               False if caller should stop.
            - candidate_data: None or tuple of (request_id, input_id, 
              num_encoder_tokens, mm_hash)
        """
        with self.scheduling_lock:
            if self.prealloc_candidate is None:
                if fill_next is True:
                    self.prealloc_candidate = self.preallocs_queue.get()
                return (True, None) # No candidate, just get next candidate
                
            (request_id, input_id, num_encoder_tokens, mm_hash) = \
                self.prealloc_candidate
            if num_encoder_tokens > free_space:
                return (False, None)

            if fill_next is True:
                self.prealloc_candidate = self.preallocs_queue.get()
            else:
                self.prealloc_candidate = None

            if (request_id, input_id) in self.ignored_preallocs:
                self.ignored_preallocs.remove((request_id, input_id))
                return (True, None) # Skip and get next
            
            self.pending_preallocs[request_id].remove(input_id)

            return (True, (request_id, input_id, num_encoder_tokens, mm_hash))
        
    def send_prealloc_notification(
        self, 
        req_id: str, 
        input_id: int, 
        is_receiving_required: bool, 
        mm_hash: str
    ):
        """Send a preallocation notification to the encoding instance."""
        self.ec_connector.schedule_send_prealloc_notification(
            req_id, input_id, is_receiving_required, mm_hash
        )