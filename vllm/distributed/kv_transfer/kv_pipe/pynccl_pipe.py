# SPDX-License-Identifier: Apache-2.0
"""
    This module implements a PyNccl pipe for sending and receiving 
    Optional[torch.Tensor] between distributed ranks with advanced 
    communication features.

    Key Features:
    - Supports sending and receiving tensors with metadata
    - Handles both CUDA and CPU device communications
    - Implements a non-blocking tensor transfer mechanism
    - Manages buffer size and provides backpressure control
    - Supports distributed process groups with configurable parameters
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Optional, Tuple

import torch

from vllm.config import KVTransferConfig
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.kv_transfer.kv_pipe.base import KVPipeBase
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger

logger = init_logger(__name__)


class BrokenPipeException(Exception):

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


Metadata = Dict[str, Optional[torch.Tensor]]


class PyNcclPipe(KVPipeBase):

    METADATA_LENGTH = 16
    MAX_TENSOR_DIMENSIONS = 14
    METADATA_DTYPE = torch.int64

    def __init__(self,
                 local_rank: int,
                 config: KVTransferConfig,
                 device: Optional[str] = None,
                 port_offset: int = 0):
        self.config = config
        self.local_rank = local_rank
        self.kv_rank = self.config.kv_rank
        self.kv_parallel_size = self.config.kv_parallel_size
        if device is None:
            self.device = self._select_device(self.config.kv_buffer_device)
        else:
            self.device = self._select_device(device)

        # build distributed connection and send/recv implementation
        self.group = StatelessProcessGroup.create(
            host=self.config.kv_ip,
            port=self.config.kv_port + port_offset,
            rank=self.kv_rank,
            world_size=self.kv_parallel_size,
        )
        # add a barrier to make sure the connection is initiated properly
        self.group.barrier()
        impl = self._get_device_send_recv_impl(self.group)
        self.device_send_func, self.device_recv_func = impl
        # set target rank
        self.target_rank_for_send = (self.kv_rank + 1) % self.kv_parallel_size
        self.target_rank_for_recv = (self.kv_rank - 1) % self.kv_parallel_size

        # transportation-related variables
        self.transport_thread: Optional[ThreadPoolExecutor] = None
        self.buffer_size = 0
        self.buffer_size_lock = threading.Lock()
        self.buffer_size_thresh = self.config.kv_buffer_size

    def _get_device_send_recv_impl(
        self, group: StatelessProcessGroup
    ) -> Tuple[Callable[[torch.Tensor, int], None], Callable[
        [torch.Tensor, int], None]]:

        send: Callable[[torch.Tensor, int], None]
        recv: Callable[[torch.Tensor, int], None]
        if self.device.type == "cuda":
            # use PyNCCL for send / recv
            comm = PyNcclCommunicator(group, device=self.local_rank)
            comm.disabled = False
            send, recv = comm.send, comm.recv  # type: ignore
        else:
            # This send / recv implementation here is NOT intended to transfer
            # KV caches (and should NOT be repurposed to transfer KV caches).
            # Currently it is only used to transmit control-plane messages
            # for PyNcclBuffer.
            send = group.send_obj

            def my_recv(x, src):
                x[...] = group.recv_obj(src)

            recv = my_recv

        return send, recv

    def _select_device(self, device: str):
        logger.info("Selecting device: %s", device)
        if device == "cuda":
            return torch.device(f"cuda:{self.local_rank}")
        else:
            return torch.device("cpu")

    def _make_metadata(self, tensor: Optional[torch.Tensor]) -> Metadata:
        """
        Create the metadata as a dictionary based on the input tensor.

        Parameters:
            - tensor: The input tensor or None if no tensor is provided.

        Returns:
            - metadata: A dictionary with the following keys:
                - "dtype": The data type of the tensor or None.
                - "shape": The shape of the tensor or None.
        """
        if tensor is None:
            return {"dtype": None, "shape": None}
        else:
            return {"dtype": tensor.dtype, "shape": tensor.shape}

    def _prepare_recv_buffer(self, metadata: Metadata) -> torch.Tensor:
        """
        Create a buffer to receive the tensor based on the provided metadata.

        Parameters:
            - metadata: A dictionary with keys "dtype" and "shape", describing 
              the tensor's data type and shape.

        Returns:
            - buffer: A tensor of the specified type and shape, allocated on 
              self.device.
        """
        return torch.empty(metadata["shape"],
                           dtype=metadata["dtype"],
                           device=self.device)

    def _send_metadata(self, metadata: Metadata):
        """
        Send the metadata dictionary to the target rank.

        Parameters:
            - metadata: A dictionary with keys "dtype" and "shape".
        """
        self.group.send_obj(metadata, self.target_rank_for_send)

    def _recv_metadata(self) -> Metadata:
        """
        Receive the metadata dictionary from the target rank.

        Returns:
            - metadata: A dictionary with keys "dtype" and "shape" describing 
              the tensor.
        """
        return self.group.recv_obj(self.target_rank_for_recv)

    def _send_impl(self, tensor: Optional[torch.Tensor]) -> None:
        """
        The actual implementation of sending the tensor and its metadata to the 
        target rank.

        Parameters:
            - tensor: The input tensor to be sent, or None if no tensor is 
              being sent.
        """
        metadata = self._make_metadata(tensor)
        self._send_metadata(metadata)
        if tensor is not None:
            self.device_send_func(tensor.to(self.device),
                                  self.target_rank_for_send)

    def _recv_impl(self) -> Optional[torch.Tensor]:
        """
        The actual implementation of receiving a tensor and its metadata from 
        the target rank.

        Returns:
            - buffer: The received tensor, or None if no tensor is received.
        """
        metadata = self._recv_metadata()
        if metadata["dtype"] is None:
            return None
        buffer = self._prepare_recv_buffer(metadata)
        self.device_recv_func(buffer, self.target_rank_for_recv)

        return buffer

    def send_tensor_wrapper(self, tensor: Optional[torch.Tensor],
                            tensor_size: int) -> None:
        """
        Wrapper for _send_impl to handle exceptions and update buffer size.
        """
        try:
            self._send_impl(tensor)

            with self.buffer_size_lock:
                self.buffer_size -= tensor_size
        except Exception as e:
            logger.error("[rank%d]: Exception when trying to send %s, msg: %s",
                         torch.distributed.get_rank(), str(tensor), str(e))
            import traceback
            traceback.print_exc()

    def block_if_full(self):
        """
        Block the current thread if the buffer size is larger than the 
        threshold.
        """
        while self.buffer_size > self.buffer_size_thresh:
            logger.debug("KV cache transfer pipe is full. Waiting...")
            time.sleep(0.05)

    def send_tensor(self, tensor: Optional[torch.Tensor]) -> None:
        """
        Sends a tensor and its metadata to the destination rank in a 
        non-blocking way.

        Parameters:
            - tensor: The tensor to send, or None if no tensor is being sent.
        """
        if self.transport_thread is None:
            self.transport_thread = ThreadPoolExecutor(max_workers=1)

        if tensor is not None:
            tensor_size = tensor.element_size() * tensor.numel()
        else:
            tensor_size = 0

        self.block_if_full()

        with self.buffer_size_lock:
            self.buffer_size += tensor_size

        self.transport_thread.submit(self.send_tensor_wrapper, tensor,
                                     tensor_size)

    def recv_tensor(self) -> Optional[torch.Tensor]:
        """
        Receives a tensor and its metadata from the source rank. Blocking call.

        Returns:
            - tensor: The received tensor, or None if no tensor is received.
        """
        if self.transport_thread is None:
            self.transport_thread = ThreadPoolExecutor(max_workers=1)

        future = self.transport_thread.submit(self._recv_impl)

        try:
            tensor = future.result()
        except Exception as e:
            logger.error("Encountering exception in KV receiving thread")
            logger.error("%s", e)
            logger.error("My device: %s", self.device)
            import traceback
            traceback.print_exc()
            raise e

        return tensor

    def close(self):
        """
        Close the pipe and release associated resources.
        """
        if hasattr(self,
                   "transport_thread") and self.transport_thread is not None:
            self.transport_thread.shutdown()
