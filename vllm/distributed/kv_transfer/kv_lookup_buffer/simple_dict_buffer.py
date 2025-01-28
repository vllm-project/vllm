"""
    Implements a distributed key-value (KV) cache transfer mechanism.

    Key Features:
    - Distributed KV cache transmission using PyNccl pipes.
    - Non-blocking `insert`, blocking `drop_select`.
    - Use CPU signal pipe to avoid racing condition
    - Handles buffer size constraints and provide backpressure mechanism to 
      stop the prefill instance when the decode instance is slow.
"""
import threading
from typing import Dict, List, Optional

import torch

from vllm.distributed.kv_transfer.kv_lookup_buffer.base import (
    KVLookupBufferBase)
from vllm.distributed.kv_transfer.kv_pipe.base import KVPipeBase
from vllm.logger import init_logger

logger = init_logger(__name__)


def _string_to_byte_tensor(s: str) -> torch.Tensor:
    byte_data = s.encode('utf-8')
    byte_list = list(byte_data)
    byte_tensor = torch.tensor(byte_list, dtype=torch.uint8, device='cpu')
    return byte_tensor


def _byte_tensor_to_string(t: torch.Tensor) -> str:
    byte_list = t.tolist()
    byte_data = bytes(byte_list)
    return byte_data.decode('utf-8')


class SimpleDictBuffer(KVLookupBufferBase):

    def __init__(self, signal_pipe: KVPipeBase, data_pipe: KVPipeBase,
                 buffer_size_thresh: float):
        """
        signal_pipe: on CPU 
        
        NOTE: on-device recv will block all threads in the process, making the 
        KV cache producer unable to listen to new request while transmitting 
        KV cache. Luckily CPU recv only blocks the current thread so we use 
        CPU recv to listen to new request.
        
        data_pipe: on device (e.g. GPU)
        """

        self.buffer: Dict[str, List[torch.Tensor]] = {}

        self.buffer_size = 0
        self.buffer_size_threshold = buffer_size_thresh
        self.buffer_lock = threading.Lock()
        self.signal_pipe = signal_pipe
        self.data_pipe = data_pipe
        self.request_handling_thread: Optional[threading.Thread] = None

        self.normal_signal = torch.tensor([0], device="cpu")
        self.end_signal = None

    def _is_end_signal(self, signal):
        return signal is None

    def drop_select_handler(self):

        try:

            while True:
                signal = self.signal_pipe.recv_tensor()
                if self._is_end_signal(signal):
                    logger.info("Received end signal!")
                    break

                key_bytes = self.data_pipe.recv_tensor()
                key_hash = _byte_tensor_to_string(key_bytes)

                with self.buffer_lock:
                    if key_hash in self.buffer:
                        self.data_pipe.send_tensor(self.buffer[key_hash])
                        del self.buffer[key_hash]
                    else:
                        self.data_pipe.send_tensor(None)

        except RuntimeError as e:
            if 'Connection closed by peer' not in str(e):
                raise e

        logger.debug("Closing drop_select_handler")

    def drop_select(self, key: str) -> List[Optional[torch.Tensor]]:

        assert self.request_handling_thread is None, \
            "drop_select should be called by the KV cache consumer "\
            "(e.g. the decode vLLM instance)"

        self.signal_pipe.send_tensor(self.normal_signal)
        self.data_pipe.send_tensor(_string_to_byte_tensor(key))

        value = self.data_pipe.recv_tensor()

        return value

    def insert(self, key: str, value: List[torch.Tensor]) -> None:

        if key in self.buffer:
            return

        tensor_size = sum(t.element_size() * t.numel() for t in value)

        if self.buffer_size + tensor_size > self.buffer_size_threshold:
            raise RuntimeError("KV transfer buffer is full.")

        with self.buffer_lock:
            self.buffer_size += tensor_size
            self.buffer[key] = value

        # when calling the insert, the current process is a sender
        # need to launch the request handler and start listening to request.
        if self.request_handling_thread is None:
            self.request_handling_thread = threading.Thread(
                target=self.drop_select_handler)
            self.request_handling_thread.start()

    def close(self):

        if hasattr(self, "request_handling_thread"
                   ) and self.request_handling_thread is not None:
            self.request_handling_thread.join()

        else:
            # TODO: have a explicit close signal and have a explicit way to
            # check if it's requester
            self.signal_pipe.send_tensor(self.end_signal)
