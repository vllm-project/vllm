# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tensor IPC transport via torch.multiprocessing.Queue.

This module contains the queue-based transport logic for sharing tensors
between processes (e.g., API server -> engine core). The msgpack layer
emits/consumes lightweight :class:`TensorIpcHandle` values, while transport
state such as request association, handle generation, queue routing, buffering,
and cleanup lives here.
"""

import dataclasses
import uuid
from multiprocessing.queues import Queue as MPQueue
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

TensorIpcQueue = MPQueue


@dataclasses.dataclass
class TensorIpcData:
    """
    Data sent via torch.multiprocessing.Queue for zero-copy IPC.

    Contains the tensor_id and the actual tensor. The tensor is
    shared in memory (GPU or CPU) for efficient inter-process communication.
    """

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
        self._sender_id = uuid.uuid4().hex

    def set_target_engine(self, target_engine: int) -> None:
        if target_engine != 0:
            raise IndexError(
                "TensorIpcSender only supports a single queue; "
                f"got target engine {target_engine}"
            )

    def send_tensor(self, tensor: torch.Tensor) -> dict[str, Any] | None:
        """Send tensor via queue, return its handle. Returns None if failed."""
        try:
            if not tensor.is_cuda:
                return None

            tensor_id = f"{self._sender_id}_{self._tensor_id_counter}"
            self._tensor_id_counter += 1

            # Move tensor to shared memory for IPC
            # This is required for proper inter-process communication
            if not tensor.is_shared():
                tensor = tensor.share_memory_()

            ipc_data = TensorIpcData(tensor_id=tensor_id, tensor=tensor)
            # Use a timeout to avoid blocking indefinitely
            self.queue.put(ipc_data, timeout=10.0)

            logger.debug(
                "Sent tensor %s for (shape=%s, device=%s) "
                "via IPC queue (shared memory)",
                tensor_id,
                tensor.shape,
                tensor.device,
            )

            return {"tensor_id": tensor_id, "device": str(tensor.device)}
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

    def __init__(self, queue: TensorIpcQueue, max_senders: int = 1):
        self.queue = queue
        self.max_senders = max_senders
        self._tensor_buffer: dict[str, tuple[int, torch.Tensor]] = {}
        self._counter = 0

    def recv_tensor(
        self, dtype: str, shape: tuple[int, ...], meta: dict[str, Any]
    ) -> torch.Tensor:
        """Retrieve a tensor from torch.multiprocessing.Queue.

        Uses a drain-and-buffer pattern: drains all available tensors from
        the queue, buffering them, until the requested tensor is found.
        Works for CUDA and CPU.
        """

        # Create lookup key from handle
        lookup_key = meta["tensor_id"]

        # Drain all available tensors. We save them regardless if this is
        # the one we're waiting for as they may arrive out of order from
        # multiple producers.
        while True:
            _, tensor = self._tensor_buffer.pop(lookup_key, (None, None))
            if tensor is not None:
                logger.debug(
                    "Received tensor %s for (shape=%s, device=%s) "
                    "via IPC queue (shared memory)",
                    meta["tensor_id"],
                    tensor.shape,
                    tensor.device,
                )
                return tensor

            # Clear out any stale tensors from the buffer. This should only occur if
            # there was some error sending the main message.
            while self._tensor_buffer:
                next_id, (next_count, _) = next(iter(self._tensor_buffer.items()))
                if self._counter - next_count < self.max_senders:
                    break
                logger.warning(
                    "Discarding stale tensor from buffer with id %s", next_id
                )
                self._tensor_buffer.pop(next_id)

            # Release lock while waiting on queue (important to avoid
            # blocking cleanup)
            ipc_data: TensorIpcData = self.queue.get(timeout=10.0)

            # Store tensor
            self._tensor_buffer[ipc_data.tensor_id] = (self._counter, ipc_data.tensor)
            self._counter += 1
