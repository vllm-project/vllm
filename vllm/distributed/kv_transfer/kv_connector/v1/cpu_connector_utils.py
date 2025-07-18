# SPDX-License-Identifier: Apache-2.0
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import msgspec
import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)


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
        return f"DestinationSpec(rank={self.rank}, " + \
               f"host={self.host}, base_port={self.base_port})"

    def get_id(self) -> str:
        """Get the id of the destination spec.

        Returns:
            str: The id of the destination spec.
        """
        return f"{self.rank}_{self.host}_{self.base_port}"


class SourceSpec(msgspec.Struct):
    """SourceSpec is used to specify the source of kv sending task.
    """
    # The request id of the kv cache
    request_id: str

    # The layer id of the kv cache
    layer_id: int

    # The range of tokens to be offloaded
    start: int  # For token_range slice
    stop: int  # For token_range slice

    # The shape of the offloaded KV cache tensor as a tuple
    shape: tuple[int, ...]

    # The dtype of the offloaded KV cache tensor as a string
    dtype_str: str

    # The total number of tokens in the "full request"
    num_all_tokens: int

    @property
    def token_range(self) -> slice:
        """Get the token range as a slice object."""
        return slice(self.start, self.stop)

    @property
    def tensor_shape(self) -> torch.Size:
        """Get the shape as a torch.Size object."""
        return torch.Size(self.shape)

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype as a torch.dtype object."""
        return getattr(torch, self.dtype_str)

    def get_size(self) -> int:
        """Get the size in bytes of the corresponding kv cache."""
        return math.prod(self.shape) * self.dtype.itemsize

    def __str__(self) -> str:
        return (f"SourceSpec(request_id={self.request_id}, "
                f"layer_id={self.layer_id}, "
                f"token_range={self.token_range}, shape={self.tensor_shape})")


@dataclass
class DecoderKVSpec:
    # Start index of the KV cache (inclusive)
    start: int
    # Stop index of the KV cache (exclusive)
    stop: int
    # The shape of the KV cache
    buffer: torch.Tensor


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
        return self.buffer.view(self.source_spec.dtype)[:num_elements].view(
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

    def mark_sending(self) -> None:
        """Mark the send task as sending.
        """
        self.state.is_sending = True


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

    def progress(self) -> None:
        """A fast, non-blocking function to check and update the states of all
        send tasks. This function should be called periodically to ensure that
        the send tasks are being processed.
        """
        # Update before going through all send tasks
        self.pre_progress_hook()

        new_task_list = []

        num_sent = 0
        num_freed = 0
        for task in self._send_tasks:
            should_add = True

            if task.is_ready() and not task.is_sending():
                self.send_task(task)
                task.mark_sending()
                num_sent += 1

            if task.is_done():
                self.free_task(task)
                should_add = False
                num_freed += 1

            if should_add:
                new_task_list.append(task)

        self._send_tasks = new_task_list

        # Update after going through all send tasks
        self.post_progress_hook()

        if num_sent > 0 or num_freed > 0:
            logger.debug("KVSender progress: sent %d, freed %d", num_sent,
                         num_freed)

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
        Will be called in the pre-implemented progress() method.

        Args:
            task (SendTask): The send task to be freed.
        """
        raise NotImplementedError("free_task() not implemented")

    @abstractmethod
    def send_task(self, task: SendTask) -> None:
        """Send the send task after it is ready.
        Will be called in the pre-implemented progress() method.

        Args:
            task (SendTask): The send task to be sent.
        """
        raise NotImplementedError("send_task() not implemented")

    @abstractmethod
    def pre_progress_hook(self) -> None:
        """Hook to be called before processing the send task.
        """
        raise NotImplementedError("pre_progress_hook() not implemented")

    @abstractmethod
    def post_progress_hook(self) -> None:
        """Hook to be called after processing the send task.
        """
        raise NotImplementedError("post_progress_hook() not implemented")
