# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tensor IPC transport via torch.multiprocessing.Queue.

This module contains the queue-based transport logic for sharing tensors
between processes (e.g., API server -> engine core). The msgpack layer
emits/consumes lightweight :class:`TensorIpcHandle` values, while transport
state such as request association, handle generation, queue routing, buffering,
and cleanup lives here.
"""

import contextlib
import dataclasses
import threading
from multiprocessing.queues import Queue as MPQueue
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.serial_utils import TensorIpcHandle

logger = init_logger(__name__)

TensorIpcQueue = MPQueue


@dataclasses.dataclass
class TensorIpcData:
    """
    Data sent via torch.multiprocessing.Queue for zero-copy IPC.

    Contains the request_id, tensor_id and the actual tensor. The tensor is
    shared in memory (GPU or CPU) for efficient inter-process communication.
    """

    request_id: str | None
    tensor_id: str
    tensor: torch.Tensor


class TensorIpcSender:
    """Send-side logic for tensor IPC via torch.multiprocessing.Queue.

    Uses a single queue targeting rank 0 (the only rank that consumes
    multimodal tensors during TP>1 / PP>1. Note: DP>1 not supported).
    """

    def __init__(self, queue: TensorIpcQueue):
        self.queue = queue
        self._tensor_id_counter = 0
        self._current_request_id: str | None = None

    @property
    def current_request_id(self) -> str | None:
        return self._current_request_id

    def set_request_context(self, request_id: str | None) -> None:
        self._current_request_id = request_id

    def set_target_engine(self, target_engine: int) -> None:
        if target_engine != 0:
            raise IndexError(
                "TensorIpcSender only supports a single queue; "
                f"got target engine {target_engine}"
            )

    def send_tensor(
        self,
        tensor: torch.Tensor,
        request_id: str | None = None,
        tensor_id: str | None = None,
    ) -> TensorIpcHandle | None:
        """Send tensor via queue, return its handle. Returns None if failed."""
        try:
            if request_id is None:
                request_id = self._current_request_id
            if tensor_id is None:
                tensor_id = f"{id(self)}_{self._tensor_id_counter}"
                self._tensor_id_counter += 1

            # Move tensor to shared memory for IPC
            # This is required for proper inter-process communication
            if not tensor.is_shared():
                tensor = tensor.share_memory_()

            ipc_data = TensorIpcData(
                request_id=request_id,
                tensor_id=tensor_id,
                tensor=tensor,
            )
            # Use a timeout to avoid blocking indefinitely
            self.queue.put(ipc_data, timeout=10.0)

            logger.debug(
                "Sent tensor %s for request %s (shape=%s, device=%s) "
                "via IPC queue (shared memory)",
                tensor_id,
                request_id,
                tensor.shape,
                tensor.device,
            )

            return TensorIpcHandle(
                request_id=request_id,
                tensor_id=tensor_id,
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype).removeprefix("torch."),
                device=str(tensor.device),
            )
        except Exception as e:
            logger.warning(
                "Failed to send tensor via IPC queue: %s. "
                "Falling back to standard serialization.",
                e,
            )
            return None


class TensorIpcReceiver:
    """Receive-side logic for tensor IPC via torch.multiprocessing.Queue.

    Wraps the queue receive logic previously embedded in MsgpackDecoder.
    """

    def __init__(self, queue: TensorIpcQueue):
        self.queue = queue
        self._tensor_buffer: dict[tuple[str | None, str], torch.Tensor] = {}
        self._request_to_tensors: dict[str, list[tuple[str | None, str]]] = {}
        self._buffer_lock = threading.Lock()

    @staticmethod
    def is_handle_like(obj: Any) -> bool:
        return isinstance(obj, (list, tuple)) and len(obj) == 5

    @staticmethod
    def parse_handle(obj: Any) -> TensorIpcHandle:
        if isinstance(obj, (list, tuple)) and len(obj) == 5:
            return TensorIpcHandle(*obj)
        raise TypeError(f"Object is not a TensorIpcHandle: {type(obj)}")

    def recv_tensor(self, handle: TensorIpcHandle | Any) -> torch.Tensor:
        """Retrieve a tensor from torch.multiprocessing.Queue.

        Uses a drain-and-buffer pattern: drains all available tensors from
        the queue, buffering them, until the requested tensor is found.
        Works for CUDA and CPU.
        """
        handle = self.parse_handle(handle)
        # Create lookup key from handle
        lookup_key = (handle.request_id, handle.tensor_id)

        # Drain all available tensors. We save them regardless if this is
        # the one we're waiting for as they may arrive out of order from
        # multiple producers.
        while True:
            # Check if tensor is already in buffer (with lock)
            with self._buffer_lock:
                if lookup_key in self._tensor_buffer:
                    # Retrieve and remove tensor from buffer
                    tensor = self._tensor_buffer.pop(lookup_key)

                    # Remove from request tracking when consumed
                    if (
                        handle.request_id is not None
                        and handle.request_id in self._request_to_tensors
                    ):
                        tensors = self._request_to_tensors.get(handle.request_id)
                        if tensors:
                            tensors.remove(lookup_key)
                            # Clean up if this is the last tensor for
                            # the request
                            if not tensors:
                                del self._request_to_tensors[handle.request_id]

                    logger.debug(
                        "Received tensor %s for request %s "
                        "(shape=%s, device=%s) via IPC queue (shared memory)",
                        handle.tensor_id,
                        handle.request_id,
                        tensor.shape,
                        tensor.device,
                    )
                    return tensor

            # Release lock while waiting on queue (important to avoid
            # blocking cleanup)
            ipc_data: TensorIpcData = self.queue.get(timeout=10.0)

            # Store the received tensor (with lock)
            with self._buffer_lock:
                # Store tensor with tuple key (request_id, tensor_id)
                tensor_key = (ipc_data.request_id, ipc_data.tensor_id)
                self._tensor_buffer[tensor_key] = ipc_data.tensor

                # Track which request this tensor belongs to for cleanup
                if ipc_data.request_id is not None:
                    if ipc_data.request_id not in self._request_to_tensors:
                        self._request_to_tensors[ipc_data.request_id] = []
                    self._request_to_tensors[ipc_data.request_id].append(tensor_key)

    def cleanup_request_tensors(self, request_id: str) -> int:
        """Remove all orphaned tensors associated with a request.

        This should be called when a request is aborted, times out, or fails
        to ensure tensors in the buffer don't accumulate indefinitely.

        Args:
            request_id: The request ID whose tensors should be cleaned up.

        Returns:
            The number of tensors that were removed from the buffer.
        """
        with self._buffer_lock:
            if request_id not in self._request_to_tensors:
                return 0

            tensor_keys = self._request_to_tensors.pop(request_id)
            removed_count = 0

            for tensor_key in tensor_keys:
                if tensor_key in self._tensor_buffer:
                    del self._tensor_buffer[tensor_key]
                    removed_count += 1
                    logger.debug(
                        "Cleaned up orphaned tensor %s for request %s",
                        tensor_key[1],  # Just log the tensor_id part
                        request_id,
                    )

            return removed_count


@contextlib.contextmanager
def encoder_request_context(
    sender: TensorIpcSender | None,
    request_type: EngineCoreRequestType,
    request: Any,
):
    """Context manager for setting request state during request encoding.

    When tensor IPC is in use, sets the request context on entry and
    clears it on exit.
    """
    if sender is None:
        yield
        return

    # Set request context if this is an ADD request with a request_id
    if request_type == EngineCoreRequestType.ADD and hasattr(request, "request_id"):
        sender.set_request_context(request.request_id)

    try:
        yield
    finally:
        sender.set_request_context(None)
