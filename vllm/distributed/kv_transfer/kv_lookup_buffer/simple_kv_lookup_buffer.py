
from vllm.distributed.kv_transfer.kv_lookup_buffer.base import \
    KVLookupBufferBase
from vllm.distributed.kv_transfer.kv_pipe.base import KVPipeBase
from typing import Dict, Tuple, List, Optional
import threading
import torch
from collections import deque
import time

from vllm.logger import init_logger

logger = init_logger(__name__)

class SimpleKVLookupBuffer(KVLookupBufferBase):
    
    def __init__(self, signal_pipe, data_pipe, buffer_size_thresh):
        """
        signal_pipe: on CPU -- avoid recv() stops the python intepreter
        data_pipe: on GPU
        """
        
        self.buffer = deque()
        
        self.buffer_size = 0
        self.buffer_size_threshold = buffer_size_thresh
        self.buffer_lock = threading.Lock()
        self.signal_pipe = signal_pipe
        self.data_pipe = data_pipe
        self.request_handling_thread = None

        self.normal_signal = torch.tensor([0])
        self.end_signal = None

        
    def _matches(self, tokens_roi_sender, tokens_roi_recver):

        # tokens_roi_sender: tokens and roi of the producer (in the buffer)
        # tokens_roi_recver: tokens and roi of the consumer (query)
        
        tokens_sender = tokens_roi_sender[0]
        tokens_recver = tokens_roi_recver[0]
        roi_sender = tokens_roi_sender[1]
        roi_recver = tokens_roi_recver[1]
        
        if tokens_recver is None:
            # consumer sends an empty request
            # semantics: DROP SELECT * LIMIT 1
            # so any of the data in the buffer can be drop-selected
            return True

            
        # Assuming that roi is a mask on tokens
        tokens_sender = tokens_sender[roi_sender]
        tokens_recver = tokens_recver[roi_recver]
        
        
        # simple common prefix matching
        min_length = min(len(tokens_sender), len(tokens_recver))
        if torch.allclose(tokens_sender[:min_length], tokens_recver[:min_length]):
            return min_length
        
        return 0

            
    def _send_tensor_and_dec_size(self, tensor: Optional[torch.Tensor]) -> None:

        assert tensor is not None, "Use self.data_pipe.send(None) instead"
        self.buffer_size -= tensor.element_size() * tensor.numel()
        self.data_pipe.send_tensor(tensor)

    def _get_element_size(self, data):
        
        if data == [] or data is None:
            return 0
        if isinstance(data, torch.Tensor):
            return data.element_size() * data.numel()

        assert False, "Unknown data type %s" % type(data)
        
    def _add_to_buffer(self, input_tokens, roi, key, value, hidden):

        if isinstance(input_tokens, torch.Tensor):
            input_tokens = input_tokens.clone()
        if isinstance(roi, torch.Tensor):
            roi = roi.clone()
        if isinstance(key, torch.Tensor):
            key = key.clone()
        if isinstance(value, torch.Tensor):
            value = value.clone()
        if isinstance(hidden, torch.Tensor):
            hidden = hidden.clone()

        
        buffer_item = [input_tokens, roi, key, value, hidden]
        
        with self.buffer_lock:
            for data in buffer_item:
                self.buffer_size += self._get_element_size(data)
            self.buffer.append(buffer_item)
        
    def _is_end_signal(self, signal):
        return signal is None
        
    def drop_select_handler(self):

        try:
        
            while True:
                signal = self.signal_pipe.recv_tensor()
                if self._is_end_signal(signal):
                    logger.info("Received end signal!")
                    break

                input_tokens = self.data_pipe.recv_tensor()

                roi = self.data_pipe.recv_tensor()
                tokens_roi_recver = [input_tokens, roi]
                
                matched_length = 0
                
                # perform input tokens and roi matching
                with self.buffer_lock:

                    for _ in range(len(self.buffer)):
                        
                        temp_length = self._matches(self.buffer[0], tokens_roi_recver)
                        if temp_length > 0:
                            matched_length = temp_length
                            break
                        # rotate the element we just accessed to the end
                        self.buffer.rotate(-1)
                        
                    if matched_length > 0:
                        # need to clone the tensor
                        # in case the tensor is freed before sending finishes
                        matched_item = self.buffer.popleft()
                        for tensor in matched_item:
                            self._send_tensor_and_dec_size(tensor)
                        
                    else:
                        # no match, just send None
                        for _ in range(5):
                            self.data_pipe.send_tensor(None)

        except RuntimeError as e:
            if 'Connection closed by peer' not in str(e):
                raise e

        logger.debug("Closing drop_select_handler")
                        
        
    def drop_select(self, input_tokens, roi):
        
        assert self.request_handling_thread is None, \
            "drop_select should be called by the receiver"

            
        if isinstance(input_tokens, torch.Tensor):
            input_tokens = input_tokens.clone()
        if isinstance(roi, torch.Tensor):
            roi = roi.clone()
        
        self.signal_pipe.send_tensor(self.normal_signal)
        self.data_pipe.send_tensor(input_tokens)
        self.data_pipe.send_tensor(roi)
        
        input_tokens = self.data_pipe.recv_tensor()
        roi = self.data_pipe.recv_tensor()
        key = self.data_pipe.recv_tensor()
        value = self.data_pipe.recv_tensor()
        hidden = self.data_pipe.recv_tensor()
        
        return [input_tokens, roi, key, value, hidden]

        
    def full_handler(self):
        time.sleep(0.001)
        
    
    def insert(self, input_tokens, roi, key, value, hidden) -> None:

        while self.buffer_size > self.buffer_size_threshold:
            # logger.debug("KV transfer buffer is full. Handling...")
            self.full_handler()
        
        self._add_to_buffer(input_tokens, roi, key, value, hidden)
        
        # when calling the insert, the current process is a sender
        # need to launch the request handler and start listening to request.
        if self.request_handling_thread is None:
            self.request_handling_thread = threading.Thread(
                target=self.drop_select_handler)
            self.request_handling_thread.start()
            
            
    def close(self):

        if hasattr(self, "request_handling_thread") and self.request_handling_thread is not None:
            self.request_handling_thread.join()

        else:
            # TODO: have a explicit close signal and have a explicit way to check if it's requester 
            self.signal_pipe.send_tensor(self.end_signal)
