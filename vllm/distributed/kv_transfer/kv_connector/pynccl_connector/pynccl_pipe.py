"""
    This file implements a simple torch distributed connector by 3 classes:
    - `TorchDistributedPipe`: a tensor transmission pipe between vllm instances,
        using `torch.distributed`
    - `TorchDistributedBuffer`: a buffer to store tensors, implemented on top 
        of `TorchDistributedPipe`
    - `TorchDistributedConnector`: a torch distributed connector between P/D 
      instance, implemented on top of `TorchDistributedBuffer`
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union
from copy import deepcopy

import torch
from torch.distributed import Backend

from vllm.distributed.utils import StatelessProcessGroup
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.logger import init_logger
from vllm.config import ParallelConfig



logger = init_logger(__name__)


# if the tensor is only one-element and only contains NONE_INT
# this means that the sended object is None.
NONE_INT = -150886311

# Mapping tensor dtype to INT64, used for tensor metadata transmission
FLOAT16_INT = -543205003776624
INT64_INT = -375623078607432
BOOL_INT = -28035262008646
BFLOAT16_INT = -452084912267662
FLOAT32_INT = -1049557997456592
FLOAT64_INT = -452201007054137
FLOAT8_E4M3FN_INT = -1066697177659525
FLOAT8_E5M2_INT = -618182574682355

DTYPE2INT = {
    torch.float16: FLOAT16_INT,
    torch.int64: INT64_INT,
    torch.bool: BOOL_INT,
    torch.bfloat16: BFLOAT16_INT,
    torch.float32: FLOAT32_INT,
    torch.float64: FLOAT64_INT,
    torch.float8_e4m3fn: FLOAT8_E4M3FN_INT,
    torch.float8_e5m2: FLOAT8_E5M2_INT,
}

INT2DTYPE = {
    FLOAT16_INT: torch.float16,
    INT64_INT: torch.int64,
    BOOL_INT: torch.bool,
    BFLOAT16_INT: torch.bfloat16,
    FLOAT32_INT: torch.float32,
    FLOAT64_INT: torch.float64,
    FLOAT8_E4M3FN_INT: torch.float8_e4m3fn,
    FLOAT8_E5M2_INT: torch.float8_e5m2,
}


class BrokenPipeException(Exception):

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
    
    
Metadata = Dict[str, Optional[torch.Tensor]]


class PyNcclPipe:
    
    METADATA_LENGTH = 16
    MAX_TENSOR_DIMENSIONS = 14
    METADATA_DTYPE = torch.int64

    def __init__(
        self,
        local_rank: int,
        config: ParallelConfig
    ):
        self.config = config
        
        self.local_rank = local_rank
        self.kv_rank = self.config.kv_rank
        self.kv_parallel_size = self.config.kv_parallel_size
        self.device = self._select_device()
        

        # build distributed connection and send/recv implementation
        self.group = StatelessProcessGroup.create(
            host = self.config.kv_ip,
            port = self.config.kv_port,
            rank = self.kv_rank,
            world_size = self.kv_parallel_size
        )
        # add a barrier to make sure the connection is initiated properly
        self.group.barrier()
        self.device_send_func, self.device_recv_func = \
            self._get_device_send_recv_impl(self.group)
        # set target rank
        self.target_rank_for_send = (self.kv_rank+ 1) % self.kv_parallel_size
        self.target_rank_for_recv = (self.kv_rank - 1) % self.kv_parallel_size


        # transportation-related variables
        self.transport_thread: Optional[ThreadPoolExecutor] = None
        self.buffer_size = 0
        self.buffer_size_lock = threading.Lock()
        self.buffer_size_thresh = self.config.kv_buffer_size

        self.none_tensor = torch.tensor([NONE_INT], device=self.device)

    
    def _get_device_send_recv_impl(self, group: StatelessProcessGroup):
        if self.config.kv_buffer_device == "cuda":
            # use PyNCCL for send / recv
            comm = PyNcclCommunicator(
                group,
                device=self.local_rank,
            )
            comm.disabled = False
            send, recv = comm.send, comm.recv
        else:
            # use cpu communication
            send = group.send
            recv = group.recv

        return send, recv

    def _select_device(self):
        if self.config.kv_buffer_device == "cuda":
            return torch.device(f"cuda:{self.local_rank}")
        else:
            return "cpu"

    def _make_metadata(self, tensor: Optional[torch.Tensor]) -> Metadata:
        """Create the metadata on based on the input tensor, and move it to GPU.
        The metadata's length is `TorchDistributedPipe.METADATA_LENGTH`.

        Currently, the metadata is a int64 tensor and it includes dtype, number
        of dimensions, and the shape information of the input tensor.


        The information follows the layout below:
        - metadata[0] -- dtype
        - metadata[1] -- number of dimensions
        - metadata[2 : 2+ndims] -- the shape of the input tensor

        Parameters:
            - tensor: the input tensor

        Returns:
            - metadata: the metadata tensor, on self.device
        """
        if tensor is None:
            return {"dtype": None, "shape": None}
        else:
            return {
                "dtype": tensor.dtype,
                "shape": tensor.shape
            }

    def _prepare_recv_buffer(self,
                             metadata: Metadata) -> torch.Tensor:
        """Create a buffer to receive the tensor based on the metadata.

        Parameters:
            - d_metadata_buffer: the metadata tensor on self.device

        Returns:
            - buffer: the buffer tensor to receive the tensor, on self.device
        """
        return torch.empty(metadata["shape"], 
                           dtype=metadata["dtype"], device=self.device)

    def _send_metadata(self, metadata):
        """Send the metadata buffer to the target rank.
        """
        self.group.send_obj(metadata, self.target_rank_for_send)

    def _recv_metadata(self) -> torch.Tensor:
        """Receive the metadata buffer from the target rank.

        Returns:
            - metadata_buffer: the metadata buffer tensor, on self.device

        Note:
            The current implementation uses the assumption that there is no
            race conditions during sending/receiving. Therefore, the metadata
            buffer can be reused
        """
        return self.group.recv_obj(
            self.target_rank_for_recv
        )

    def _send_impl(self, tensor: Optional[torch.Tensor]) -> None:
        """
        The actual implementation of sending the tensor to the target rank.
        This function will first send the metadata, and then send the tensor.

        Parameters:
            - tensor: the input tensor to be sent
        """

        metadata = self._make_metadata(tensor)
        self._send_metadata(metadata)
        if tensor is not None:
            self.device_send_func(tensor.to(self.device), 
                                  self.target_rank_for_send)

    def _recv_impl(self) -> Optional[torch.Tensor]:
        """
        The actual implementation of receiving the tensor from the target rank.
        This function will first receive the metadata, then receive the tensor.

        This function will block if there is no tensor to receive.

        Returns:
            - buffer: the received tensor, on self.device
        """
        print('recv_metadata...')
        metadata = self._recv_metadata()
        print('recv metadata done')
        if metadata["dtype"] is None:
            return None
        print('receiving tensor ...')
        buffer = self._prepare_recv_buffer(metadata)
        self.device_recv_func(buffer, self.target_rank_for_recv)
        print('recv tensor done.')

        return buffer

    def send_tensor_wrapper(
        self, 
        tensor: Optional[torch.Tensor],
        tensor_size: int
    ) -> None:
        try:
            """Wrapper for send_tensor_dict"""
            self._send_impl(tensor)

            with self.buffer_size_lock:
                self.buffer_size = self.buffer_size - tensor_size
        except Exception as e:
            logger.error("[rank%d]: Exception when trying to send %s, msg: %s",
                         torch.distributed.get_rank(), str(tensor), str(e))
            import traceback
            traceback.print_exc()

    def block_if_full(self):
        """Block the current thread if the buffer size is larger than 1e9."""
        # TODO: replace this 1e9 with a configurable parameter or a constant
        while self.buffer_size > self.buffer_size_thresh:
            logger.debug("KV cache transfer pipe is full. Waiting...")
            time.sleep(0.05)

    def send_tensor(self, tensor: Optional[torch.Tensor]) -> None:
        """Sends a tensor to the destination rank in a non-blocking way.
        Flow: send tensor dim -- send tensor shape -- send tensor data
        """

        if self.transport_thread is None:
            self.transport_thread = ThreadPoolExecutor(max_workers=1)

        if tensor is None:
            tensor_size = 0
        else:
            tensor_size = tensor.element_size() * tensor.numel()

        self.block_if_full()

        with self.buffer_size_lock:
            self.buffer_size = self.buffer_size + tensor_size

        self.transport_thread.submit(
            self.send_tensor_wrapper,
            tensor,
            tensor_size,
        )

    def recv_tensor(self) -> Optional[torch.Tensor]:
        """Receives a tensor from the src rank. Blocking."""
        if self.transport_thread is None:
            self.transport_thread = ThreadPoolExecutor(max_workers=1)

        future = self.transport_thread.submit(self._recv_impl)

        try:
            tensor = future.result()
        except Exception as e:
            # the underlying pipe is likely broken
            logger.error("Encountering exception in KV receiving thread")
            logger.error("%s", e)
            import traceback
            traceback.print_exc()
            raise e
        
        return tensor

    def close(self):
        """Close the pipe and release the resources."""
        if (hasattr(self, "transport_thread")
                and self.transport_thread is not None):
            self.transport_thread.shutdown()

