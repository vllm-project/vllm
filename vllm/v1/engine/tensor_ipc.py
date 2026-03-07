# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tensor IPC transport via torch.multiprocessing.Queue.

This module contains the queue-based transport logic for sharing tensors
between processes (e.g., API server -> engine core). The MsgpackEncoder/Decoder
in serial_utils.py handle emitting/consuming tensor references
(TensorIpcHandle), while this module handles the actual transport.
"""

import dataclasses
import threading
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.v1.serial_utils import TensorIpcHandle

logger = init_logger(__name__)


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

    def __init__(self, queue: Any):
        self.queue = queue

    def send_tensor(
        self,
        tensor: torch.Tensor,
        request_id: str | None,
        tensor_id: str,
    ) -> TensorIpcHandle:
        """Send tensor via queue, return its handle."""

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


class TensorIpcReceiver:
    """Receive-side logic for tensor IPC via torch.multiprocessing.Queue.

    Wraps the queue receive logic previously embedded in MsgpackDecoder.
    """

    def __init__(self, queue: Any):
        self.queue = queue
        self._tensor_buffer: dict[tuple[str | None, str], torch.Tensor] = {}
        self._request_to_tensors: dict[str, list[tuple[str | None, str]]] = {}
        self._buffer_lock = threading.Lock()

    def recv_tensor(self, handle: TensorIpcHandle) -> torch.Tensor:
        """Retrieve a tensor from torch.multiprocessing.Queue.

        Uses a drain-and-buffer pattern: drains all available tensors from
        the queue, buffering them, until the requested tensor is found.
        Works for CUDA and CPU.
        """
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
