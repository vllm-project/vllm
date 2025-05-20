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
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size,
    get_tp_group)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_cpu_utils import (
        DestinationSpec, SourceSpec, RingBufferAllocator)
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
        num_elements = self.source_spec.tensor_shape.numel()
        return self.buffer.view(
                self.source_spec.dtype)[:num_elements].view(
                        self.source_spec.tensor_shape)

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



@dataclass
class CPUSendTask(SendTask):
    """CPUSendTask is a send task that uses CPU memory for the buffer.
    """
    buffer_addr: int
    cuda_event: Optional[torch.cuda.Event] = None

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
