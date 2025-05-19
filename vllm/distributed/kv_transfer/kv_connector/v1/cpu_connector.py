# SPDX-License-Identifier: Apache-2.0
import contextlib
import math
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple

import msgspec
import torch
import zmq

from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size,
    get_tp_group)
from vllm.logger import init_logger
from vllm.utils import make_zmq_path, make_zmq_socket, round_down, cdiv
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus
from vllm import _custom_ops as ops

from lmcache.utils import _lmcache_nvtx_annotate

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import CachedRequestData, NewRequestData
    from vllm.v1.request import Request

logger = init_logger(__name__)

def d2h_page_copy(
        src_layer: torch.Tensor,
        dst_buffer: torch.Tensor,
        block_ids: list[int]
    ) -> None:
    """Copy data from device to host.

    Args:
        src_layer (torch.Tensor): The source layer on device, shape is 
            (2, num_vllm_blocks, page_size, ...remaining dims...) 
        dst_buffer (torch.Tensor): The destination buffer on host, shape is
            (2, len(block_ids), page_size, ...remaining dims...)
    """
    # debug copy:
    block_mapping = torch.stack([torch.tensor(block_ids, dtype = torch.long), 
                                 torch.arange(len(block_ids), dtype = torch.long)], dim = 1)
    ops.swap_blocks(src_layer[0], dst_buffer[0], block_mapping)
    ops.swap_blocks(src_layer[1], dst_buffer[1], block_mapping)
    #for dst_idx, block_id in enumerate(block_ids):
    #    src_k, src_v = src_layer[:, block_id, :, :]
    #    dst_k, dst_v = dst_buffer[:, dst_idx, :, :]
    #    # Copy the data from device to host
    #    dst_k.copy_(src_k, non_blocking=True)
    #    dst_v.copy_(src_v, non_blocking=True)

def h2d_page_copy(
        src_buffer: torch.Tensor,
        dst_layer: torch.Tensor,
        block_ids: list[int]
    ) -> None:
    """Copy data from host to device.

    Args:
        src_buffer (torch.Tensor): The source buffer on host, shape is 
            (2, len(block_ids), page_size, ...remaining dims...) 
        dst_layer (torch.Tensor): The destination layer on device, shape is
            (2, num_vllm_pages, page_size, ...remaining dims...)
    """
    for src_idx, block_id in enumerate(block_ids):
        dst_k, dst_v = dst_layer[:, block_id, :, :]
        src_k, src_v = src_buffer[:, src_idx, :, :]
        # Copy the data from host to device
        dst_k.copy_(src_k, non_blocking=True)
        dst_v.copy_(src_v, non_blocking=True)

@dataclass
class DestinationSpec:
    """DestinationSpec is used to specify the destination of kv sending task.

    Attributes:
        rank (int): The rank of the destination.
        host (str): The path of the destination.
        base_port (int): The base port of the destination.
    """
    rank: int
    host: str
    base_port: int

    def __str__(self) -> str:
        return f"DestinationSpec(rank={self.rank}, host={self.host}, base_port={self.base_port})"

@dataclass
class SourceSpec:
    """SourceSpec is used to specify the source of kv sending task.
    """
    # The request id of the kv cache
    request_id: str

    # The layer id of the kv cache
    layer_id: int

    # The range of tokens to be offloaded
    token_range: slice

    # The shape of the offloaded KV cache tensor
    shape: torch.Size 

    # The dtype of the offloaded KV cache tensor
    dtype: torch.dtype

    def get_size(self) -> int:
        """Get the size in bytes of the cooresponding kv cache.
        """
        return self.shape.numel() * self.dtype.itemsize

    def __str__(self) -> str:
        return f"SourceSpec(request_id={self.request_id}, " + \
                f"layer_id={self.layer_id}, " + \
                f"token_range={self.token_range}, shape={self.shape})"

@dataclass
class SendTaskState:
    """SendTaskState is used to track the state of a send task.
    """
    sender_ready: bool = False
    receiver_ready: bool = False
    is_sending: bool = False
    send_done: bool = False

    def __str__(self) -> str:
        return (f"SendTaskState(sender_ready={self.sender_ready}, "
                f"receiver_ready={self.receiver_ready}, "
                f"is_sending={self.is_sending}, "
                f"send_done={self.send_done})")

    def is_ready(self) -> bool:
        """Check if the send task is ready to be sent.

        Returns:
            bool: True if the send task is ready, False otherwise.
        """
        return self.sender_ready and self.receiver_ready

    def is_done(self) -> bool:
        """Check if the send task is done.

        Returns:
            bool: True if the send task is done, False otherwise.
        """
        return self.send_done

@dataclass
class SendTask:
    """Wraps a KV Cache sending task
    """

    # A flat buffer holding the tensor data
    buffer: torch.Tensor
    source_spec: SourceSpec
    destination_spec: DestinationSpec
    state: SendTaskState

    @property
    def tensor(self) -> torch.Tensor:
        """Get the tensor of the send task.

        Returns:
            torch.Tensor: The tensor of the send task.
        """
        num_elements = self.source_spec.shape.numel()
        return self.buffer.view(
                self.source_spec.dtype)[:num_elements].view(
                        self.source_spec.shape)

    def update_states(self) -> None:
        """Update the states of the send task. This needs to be OVERWRITTEN in
        subclasses to handle different types of send tasks.
        
        This function should be called periodically to ensure that the send
        task is being processed.
        """
        raise NotImplementedError

    def is_ready(self) -> bool:
        """Check if the send task is ready to be sent.

        Returns:
            bool: True if the send task is ready, False otherwise.
        """
        return self.state.is_ready()

    def is_sending(self) -> bool:
        """Check if the send task is currently sending.

        Returns:
            bool: True if the send task is sending, False otherwise.
        """
        return self.state.is_sending

    def is_done(self) -> bool:
        """Check if the send task is done.

        Returns:
            bool: True if the send task is done, False otherwise.
        """
        return self.state.is_done()

class KVSenderInterface(ABC):
    """KVSenderInterface is an interface for sending KV cache data.
    """

    def __init__(self) -> None:
        self._send_tasks: list[SendTask] = []


    def add_send_task(self, task: SendTask) -> None:
        """Add a send task to the list of send tasks.

        Args:
            task (SendTask): The send task to be added.
        """
        self._send_tasks.append(task)

    def get_send_tasks(self) -> list[SendTask]:
        """Get the list of send tasks.

        Returns:
            list[SendTask]: The list of send tasks.
        """
        return self._send_tasks

    @_lmcache_nvtx_annotate
    def progress(self) -> None:
        """A fast, non-blocking function to check and update the states of all
        send tasks. This function should be called periodically to ensure that
        the send tasks are being processed.
        """
        # Update before going through all send tasks
        self.pre_progress_hook()

        new_task_list = []

        for task in self._send_tasks:
            should_add = True

            if task.is_ready() and not task.is_sending():
                self._send(task)

            if task.is_done():
                self.free_task(task)
                should_add = False

            if should_add:
                new_task_list.append(task)

        self._send_tasks = new_task_list

        # Update after going through all send tasks
        self.post_progress_hook()

    ######################################################
    # Abstract methods (to be implemented by subclasses) #
    ######################################################

    @abstractmethod
    def create_send_task(
            self,
            source_spec: SourceSpec,
            destination_spec: DestinationSpec,
        ) -> SendTask:
        """Create a non-ready send task with a CPU buffer allocated.

        Args:
            source_spec (SourceSpec): The source specification of the send 
                task.
            destination_spec (DestinationSpec): The destination 
                specification of the send task.
        """
        raise NotImplementedError("create_send_task() not implemented")

    @abstractmethod
    def free_task(self, task: SendTask) -> None:
        """Free the send task.

        Args:
            task (SendTask): The send task to be freed.
        """
        raise NotImplementedError("free_task() not implemented")

    @abstractmethod
    def send_task(self, task: SendTask) -> None:
        """Send the send task after it is ready.

        Args:
            task (SendTask): The send task to be sent.
        """
        raise NotImplementedError("send_task() not implemented")

    @abstractmethod
    def pre_progress_hook(self, task: SendTask) -> None:
        """Hook to be called before processing the send task.

        Args:
            task (SendTask): The send task to be processed.
        """
        raise NotImplementedError("pre_progress_hook() not implemented")

    @abstractmethod
    def post_progress_hook(self, task: SendTask) -> None:
        """Hook to be called after processing the send task.

        Args:
            task (SendTask): The send task to be processed.
        """
        raise NotImplementedError("post_progress_hook() not implemented")


# DEBUG IMPLEMENTATION: NO REAL SEND BUT HAVE MEMORY MANAGEMENT AND D2H COPY
class RingBufferAllocator:
    """RingBufferAllocator is a simple ring buffer allocator for managing
    memory allocation and deallocation.
    """

    def __init__(self, size: int, align_to: int = 256) -> None:
        """Initialize the ring buffer allocator with the given size.

        Args:
            size (int): The size of the ring buffer (in bytes).
            align_to (int): The alignment size (in bytes). Default is 8.
        """
        self._size = size
        self._buffer = torch.empty(size, dtype=torch.uint8)
        self._high_watermark = 0
        self._low_watermark = 0
        self._align_to = align_to

        self._allocated = OrderedDict()  # Track allocated buffers

        # Register pin memory
        cudart = torch.cuda.cudart()
        cudart.cudaHostRegister(self._buffer.data_ptr(), size, 0)

    def _align_size(self, base: int) -> int:
        """Align the given size to the nearest multiple of the alignment size.

        Args:
            base (int): The size to be aligned.

        Returns:
            int: The aligned size.
        """
        return ((base - 1) // self._align_to + 1) * self._align_to

    def allocate(self, size: int) -> Tuple[int, Optional[torch.Tensor]]:
        """Allocate a buffer of the given size.

        Args:
            size (int): The size of the buffer to be allocated.

        Returns:
            Optional[Tuple[int, torch.Tensor]]: A tuple containing the address
                of the allocated buffer and the buffer itself. If allocation
                fails, returns None.
        """
        # During allocation, we always make sure that high watermark and
        # low watermark are aligned to the alignment size
        aligned_size = self._align_size(size)   # Align the requested size
        turnaround_size = (self._high_watermark // self._size + 1) * self._size

        local_high = self._high_watermark % self._size
        local_low = self._low_watermark % self._size

        if local_high >= local_low:
            if local_high == local_low and \
                    self._high_watermark > self._low_watermark:
                # No space available
                return -1, None

            # If high watermark + requested size is okay, directly allocate
            if local_high + size < self._size:
                address = self._high_watermark
                self._allocated[address] = aligned_size
                start = local_high
                end = start + size
                self._high_watermark += aligned_size
                return address, self._buffer[start:end]
            else:
                # If high watermark + requested size is not okay, we need to
                # wrap around and allocate again
                self._high_watermark = turnaround_size
                return self.allocate(size)
        else:
            # High watermark is below low watermark, check if we can allocate
            if local_high + size < local_low:
                address = self._high_watermark
                self._allocated[address] = aligned_size
                start = local_high
                end = start + size
                self._high_watermark += aligned_size
                return address, self._buffer[start:end]
            else:
                # No space available
                return -1, None

    def free(self, address: int) -> None:
        """Free the buffer at the given address.

        Args:
            address (int): The address of the buffer to be freed, which
                is returned by the allocate() method.
        """
        assert address in self._allocated, \
                "Address not found in allocated buffers"

        # Pop the address from the allocated dict, and update the 
        # low watermark
        self._allocated.pop(address)

        # If there is nothing allocated, set low_watermark to high watermark
        new_low_watermark = self._high_watermark

        # Else, set the low_watermark to the first address in the allocated
        # dict
        for addr in self._allocated.keys():
            new_low_watermark = addr
            break
        self._low_watermark = new_low_watermark

    @property
    def high_watermark(self) -> int:
        return self._high_watermark

    @property
    def low_watermark(self) -> int:
        return self._low_watermark


@dataclass
class CPUSendTask(SendTask):
    """CPUSendTask is a send task that uses CPU memory for the buffer.
    """
    buffer_addr: int
    creation_time: float = 0.0
    cuda_event: Optional[torch.cuda.Event] = None

    dbg_send_time: Optional[float] = None

    def __post_init__(self) -> None:
        self.creation_time = time.time()

    @_lmcache_nvtx_annotate
    def update_states(self) -> None:
        """Update the states of the send task.
        """
        # Check the cuda event
        if not self.state.sender_ready and self.cuda_event is not None \
                and self.cuda_event.query():
            self.state.sender_ready = True

        curr_time = time.time()
        if curr_time - self.creation_time > 0.5:
            self.state.receiver_ready = True

        if self.dbg_send_time is not None and \
                curr_time - self.dbg_send_time > 1:
            self.state.send_done = True

    def dbg_mark_sending(self) -> None:
        """Mark the send task as sending.
        """
        self.state.is_sending = True
        self.dbg_send_time = time.time()

class CPUKVSender(KVSenderInterface):
    """CPUKVSender is an implementation of KVSenderInterface that provides a
    ring buffer allocator for managing pin memory allocation and deallocation.
    """

    def __init__(self, buffer_size: int) -> None: 
        super().__init__()
        self._buffer_size = buffer_size
        self._allocator = RingBufferAllocator(self._buffer_size)

    def create_send_task(
            self,
            source_spec: SourceSpec,
            destination_spec: DestinationSpec,
        ) -> SendTask:
        """Create a non-ready send task with a CPU buffer allocated.

        Args:
            source_spec (SourceSpec): The source specification of the send 
                task.
            destination_spec (DestinationSpec): The destination 
                specification of the send task.
        """
        # Allocate a buffer for the send task
        size = source_spec.get_size()
        address, buffer = self._allocator.allocate(size)
        while address == -1:
            # If allocation fails, wait for a while to process 
            # and try again
            time.sleep(0.001)
            self.progress()
            address, buffer = self._allocator.allocate(size)
        assert buffer is not None, "Buffer allocation failed"

        # Create a send task with the allocated buffer
        task = CPUSendTask(
            buffer=buffer,
            source_spec=source_spec,
            destination_spec=destination_spec,
            state=SendTaskState(),
            buffer_addr=address,
        )
        self.add_send_task(task)
        return task

    def free_task(self, task: SendTask) -> None:
        """Free the send task.

        Args:
            task (SendTask): The send task to be freed.
        """
        # Free the buffer in the ring buffer allocator
        self._allocator.free(task.buffer_addr)

    def send_task(self, task: SendTask) -> None:
        """Send the send task after it is ready.

        Args:
            task (SendTask): The send task to be sent.
        """
        # DEBUG IMPLEMENTATION
        logger.error("CPUKVSender.send_task() not implemented, running a debug implementation!")
        task.dbg_mark_sending()

    def pre_progress_hook(self) -> None:
        for task in self.get_send_tasks():
            task.update_states()

    def post_progress_hook(self) -> None:
        pass

    def _send(self, task: SendTask) -> None:
        # NO IMPLEMENTATION YET
        pass


#####################################################################
# Connector related code
#####################################################################

@dataclass
class PrefillRequestTracker:
    """RequestTracker is used to track the state of a request.

    Attributes:
        req_id (str): The id of the request.
        num_saved_tokens (int): The number of tokens saved.
        num_loaded_tokens (int): The number of tokens loaded.
        num_computed_tokens (int): The number of tokens computed.
        allocated_block_ids (list[int]): The list of allocated block ids.
    """
    # Request id
    req_id: str

    # Total number of tokens that are in this request
    num_total_tokens: int = 0

    # Number of tokens that are already saved
    num_saved_tokens: int = 0

    # Block ids that are already allocated for this request
    allocated_block_ids: list[int] = None

    @staticmethod
    def from_new_request(
            new_request: "NewRequestData",
            num_tokens_to_compute: int,
    ) -> "PrefillRequestTracker":
        """Create the request tracker from a new request.

        Args:
            new_request (NewRequestData): the new request data.
            num_tokens_to_compute (int): the number of tokens that will 
                be 'computed', including the `num_computed_tokens` (vLLM's
                local cache hit) and new tokens that will be scheduled.
        """
        unfolded_block_ids = []
        for block_ids in new_request.block_ids:
            unfolded_block_ids.extend(block_ids)

        return PrefillRequestTracker(
            req_id=new_request.req_id,
            num_total_tokens = num_tokens_to_compute,
            num_saved_tokens=0,
            allocated_block_ids=unfolded_block_ids,
        )

    def update(self, cached_request: "CachedRequestData") -> None:
        """Update the request tracker with the cached request data.

        Args:
            cached_request (CachedRequestData): the cached request data.
        """
        new_block_ids = []
        for nb in cached_request.new_block_ids:
            new_block_ids.extend(nb)
        self.allocated_block_ids.extend(new_block_ids)
        self.num_total_tokens += len(cached_request.new_token_ids)

    def update_num_saved_tokens(self, num_saved_tokens: int) -> None:
        """Update the number of saved tokens.

        Args:
            num_saved_tokens (int): the number of saved tokens.
        """
        self.num_saved_tokens = num_saved_tokens

@dataclass
class PrefillReqMeta:
    # Request id
    req_id: str
    # Blocks to save
    blocks_to_save: list[int]
    # The range of tokens to save
    token_range: slice
    # Skip first N tokens
    skip_leading_tokens: int
    # Skip last N tokens
    skip_trailing_tokens: int

    @staticmethod
    def from_request_tracker(
        request_tracker: PrefillRequestTracker,
        block_size: int,
    ) -> "PrefillReqMeta":
        """Create the request meta from the request tracker. Determine which 
        blocks to save and the number of leading/trailing tokens to skip for
        the worker connector.
        It also updates the request tracker's num_saved_tokens.

        Args:
            request_tracker (PrefillRequestTracker): the request tracker.
            block_size (int): the block size in vLLM.

        Returns:
            PrefillReqMeta: the request meta.
        """
        assert request_tracker.num_total_tokens <= \
                len(request_tracker.allocated_block_ids) * block_size, \
                f"Request {req_id} has more tokens than allocated blocks"

        token_range = slice(request_tracker.num_saved_tokens,
                request_tracker.num_total_tokens)

        num_saved_full_blocks = request_tracker.num_saved_tokens // block_size
        num_active_blocks = cdiv(request_tracker.num_total_tokens, block_size)

        blocks_to_save = request_tracker.allocated_block_ids[\
                num_saved_full_blocks:num_active_blocks]
        skip_leading_tokens = request_tracker.num_saved_tokens % block_size
        skip_trailing_tokens = num_active_blocks * block_size - \
                request_tracker.num_total_tokens
        logger.debug(
            "Request %s: num_saved_full_blocks=%d, num_active_blocks=%d, "
            "blocks_to_save=%s, skip_leading_tokens=%d, "
            "skip_trailing_tokens=%d", 
            request_tracker.req_id,
            num_saved_full_blocks, num_active_blocks,
            blocks_to_save, skip_leading_tokens, skip_trailing_tokens)

        # Update the request tracker with the number of saved tokens
        request_tracker.update_num_saved_tokens(
            request_tracker.num_total_tokens)
        return PrefillReqMeta(
            req_id=request_tracker.req_id,
            blocks_to_save=blocks_to_save,
            token_range=token_range,
            skip_leading_tokens=skip_leading_tokens,
            skip_trailing_tokens=skip_trailing_tokens,
        )


@dataclass
class DecodeReqMeta:
    pass

@dataclass
class CPUConnectorMetadata(KVConnectorMetadata):
    prefill_meta: list[PrefillReqMeta] 
    decode_meta: list[DecodeReqMeta]

    def __init__(self) -> None:
        super().__init__()
        self.prefill_meta = []
        self.decode_meta = []

    def add_prefill(self, prefill_meta: PrefillReqMeta) -> None:
        """Add a prefill request metadata to the metadata.

        Args:
            prefill_meta (PrefillReqMeta): The prefill request metadata to be
                added.
        """
        self.prefill_meta.append(prefill_meta)

    def add_decode(self, decode_meta: DecodeReqMeta) -> None:
        """Add a decode request metadata to the metadata.

        Args:
            decode_meta (DecodeReqMeta): The decode request metadata to be
                added.
        """
        self.decode_meta.append(decode_meta)


class CPUConnector(KVConnectorBase_V1):
    """CPUKVConnector is an implementation of KVConnectorBase_V1 that
    provides a CPU-based KV cache sending mechanism.
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole) -> None:
        super().__init__(vllm_config, role)

        self.kv_role = vllm_config.kv_transfer_config.kv_role

        self._block_size = vllm_config.cache_config.block_size

        if role == KVConnectorRole.SCHEDULER:
            pass
        elif role == KVConnectorRole.WORKER:
            # Prefiller side sender
            self._cpu_kv_sender = CPUKVSender(1024 * 1024 * 1024) # 1GB for debug

        # request_id -> prefill request trackers
        self._prefill_reqs: dict[str, PrefillRequestTracker] = {}

        # gpu kv caches
        self._gpu_kv_caches: dict[str, torch.Tensor] = {}
        self._layer_name_to_id: dict[str, int] = {}
        self._layer_id_to_name: dict[int, str] = {}
        self._kv_page_shape: torch.Size = torch.Size([0])

        # separate cuda streams
        self._cuda_stream = torch.cuda.Stream()

        # prefill offload tasks
        self._inflight_copy_tasks: list[CPUSendTask] = []

    
    ############################################################
    # Scheduler Side Methods
    ############################################################
    def _build_prefiller_meta(
            self, 
            scheduler_output: SchedulerOutput,
            output_meta: CPUConnectorMetadata) -> None:
        """Build the prefill request metadata from the scheduler output.

        Args:
            scheduler_output (SchedulerOutput): The scheduler output.
            output_meta (CPUConnectorMetadata): The output metadata.
        """
        for finished_req_id in scheduler_output.finished_req_ids:
            self._prefill_reqs.pop(finished_req_id, None)

        for request in scheduler_output.scheduled_new_reqs:
            num_tokens_to_compute = request.num_computed_tokens + \
                    scheduler_output.num_scheduled_tokens[request.req_id]
            request_tracker = PrefillRequestTracker.from_new_request(
                request, num_tokens_to_compute)
            self._prefill_reqs[request.req_id] = request_tracker

            req_meta = PrefillReqMeta.from_request_tracker(
                request_tracker,
                self._block_size)
            output_meta.add_prefill(req_meta)

        for request in scheduler_output.scheduled_cached_reqs:
            request_tracker = self._prefill_reqs[request.req_id]
            request_tracker.update(request)

            req_meta = PrefillReqMeta.from_request_tracker(
                request_tracker,
                self._block_size)
            output_meta.add_prefill(req_meta)

    def build_decode_meta(
            self,
            scheduler_output: SchedulerOutput,
            output_meta: CPUConnectorMetadata) -> None:
        """Build the decode request metadata from the scheduler output.

        Args:
            scheduler_output (SchedulerOutput): The scheduler output.
            output_meta (CPUConnectorMetadata): The output metadata.
        """
        logger.error("build_decode_meta() not implemented, running a debug implementation!")
        pass


    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        return 0, False

    def update_state_after_alloc(
            self,
            request: "Request",
            blocks: "KVCacheBlocks",
            num_external_tokens: int) -> None:
        print("In update_state_after_alloc")
        pass

    def build_connector_meta(
            self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        meta = CPUConnectorMetadata()

        if self.kv_role == "kv_producer":
            self._build_prefiller_meta(scheduler_output, meta)
        elif self.kv_role == "kv_consumer":
            self.build_decode_meta(scheduler_output, meta)
        else:
            raise ValueError(f"Unknown kv_role: {self.kv_role}")
        
        return meta

    def request_finished(
            self,
            request: "Request",
            block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        print("In request_finished")
        return False, None

    #############################################################
    # Worker Side Methods
    #############################################################
    def _get_layer_id(self, layer_name: str) -> int:
        assert layer_name in self._layer_name_to_id, \
                f"Layer {layer_name} not found in layer name to id map"
        return self._layer_name_to_id[layer_name]

    def _get_layer_name(self, layer_id: int) -> str:
        assert layer_id in self._layer_id_to_name, \
                f"Layer id {layer_id} not found in layer id to name map"
        return self._layer_id_to_name[layer_id]

    def _get_kv_shape(self, num_blocks: int) -> torch.Size:
        return torch.Size((2, num_blocks, ) + self._kv_page_shape)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self._gpu_kv_caches = kv_caches
        idx = 0
        for layer_name in kv_caches.keys():
            self._layer_name_to_id[layer_name] = idx
            self._layer_id_to_name[idx] = layer_name
            idx += 1

        self._kv_page_shape = kv_caches[list(kv_caches.keys())[0]].shape[2:]


    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be 
            the same.
            
        """
        pass


    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.
        
        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        pass

    @_lmcache_nvtx_annotate
    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """
        Start saving a layer of KV cache from vLLM's paged buffer 
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current 
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        meta = self._get_connector_metadata()
        assert isinstance(meta, CPUConnectorMetadata), \
                "Connector metadata is not of type CPUConnectorMetadata"

        assert self._cpu_kv_sender is not None

        for prefill_req in meta.prefill_meta:
            # TODO: add skip leading/trailing tokens into source_spec
            # or maybe recompute it at the receiver side based on the 
            # token_range
            source_spec = SourceSpec(
                request_id = prefill_req.req_id,
                layer_id = self._get_layer_id(layer_name),
                token_range = prefill_req.token_range,
                shape = self._get_kv_shape(
                    len(prefill_req.blocks_to_save)),
                dtype = kv_layer.dtype
            )

            # Create a destination spec
            # TODO: remove the hard-code here
            dest_spec = DestinationSpec(
                rank = get_tensor_model_parallel_rank(),
                host = "localhost",
                base_port = "54321", 
            )

            # Create the send task
            task = self._cpu_kv_sender.create_send_task(
                source_spec=source_spec,
                destination_spec=dest_spec,
            )
            assert isinstance(task, CPUSendTask), \
                    "Send task is not of type CPUSendTask"

            # Start copying the data to the CPU buffer
            buffer = task.tensor
            self._cuda_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._cuda_stream):
                # Copy the data from the GPU to the CPU buffer page by page
                d2h_page_copy(
                    src_layer=kv_layer,
                    dst_buffer=buffer,
                    block_ids=prefill_req.blocks_to_save
                )

            # record the cuda stream
            task.cuda_event = torch.cuda.Event()
            task.cuda_event.record(self._cuda_stream)

            self._inflight_copy_tasks.append(task)

        # Check the task states and send the tasks
        self._cpu_kv_sender.progress()


    @_lmcache_nvtx_annotate
    def wait_for_save(self):
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        for task in self._inflight_copy_tasks:
            if task.cuda_event is not None:
                task.cuda_event.synchronize()
        self._inflight_copy_tasks.clear()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer,
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """
        return None, None
