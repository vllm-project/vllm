
from vllm.distributed.kv_transfer.kv_lookup_buffer.base import \
    KVLookupBufferBase
from typing import Dict, Tuple, List, Optional
import threading
import torch
from collections import deque

class SimpleKVLookupBuffer(KVLookupBufferBase):
    
    def __init__(self, pipe):
        
        self.buffer = deque()
        
        self.buffer_size = 0
        self.buffer_lock = threading.Lock()
        self.pipe = pipe
        self.request_handling_thread = None

        
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
        
        
        min_length = min(len(tokens_sender), len(tokens_recver))
        if torch.allclose(tokens_sender[:min_length], tokens_recver[:min_length]):
            # drastically simplified
            # common prefix matching
            print("min length is ", min_length)
            return min_length
        
        return 0

            
    def _send_tensor_and_dec_size(self, tensor: Optional[torch.Tensor]) -> None:

        assert tensor is not None, "Use self.pipe.send(None) instead"
        self.buffer_size -= tensor.element_size() * tensor.numel()
        self.pipe.send_tensor(tensor)

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
        
        
    def drop_select_handler(self):
        
        while True:
            input_tokens = self.pipe.recv_tensor()
            roi = self.pipe.recv_tensor()
            tokens_roi_recver = [input_tokens, roi]
            
            matched_idx = None
            
            # perform input tokens and roi matching
            with self.buffer_lock:
                
                for idx, tokens_roi_sender in enumerate(self.buffer):
                    if self._matches(tokens_roi_sender, tokens_roi_recver) > 0:
                        matched_idx = idx
                        break

                        
                print("Got a match ", matched_idx)
                    
                if matched_idx is not None:
                    # need to clone the tensor
                    # in case the tensor is freed before sending finishes
                    matched_item = self.buffer[matched_idx]
                    print(matched_item)
                    for tensor in matched_item:
                        self._send_tensor_and_dec_size(tensor)
                    del self.buffer[matched_idx]
                    
                else:
                    # no match, just send None
                    for _ in range(5):
                        self.pipe.send_tensor(None)
                        
        
    def drop_select(self, input_tokens, roi):
        
        assert self.request_handling_thread is None, \
            "drop_select should be called by the receiver"

            
        if isinstance(input_tokens, torch.Tensor):
            input_tokens = input_tokens.clone()
        if isinstance(roi, torch.Tensor):
            roi = roi.clone()
        
        self.pipe.send_tensor(input_tokens)
        self.pipe.send_tensor(roi)
        
        input_tokens = self.pipe.recv_tensor()
        roi = self.pipe.recv_tensor()
        key = self.pipe.recv_tensor()
        value = self.pipe.recv_tensor()
        hidden = self.pipe.recv_tensor()
        
        return [input_tokens, roi, key, value, hidden]
        
    
    def insert(self, input_tokens, roi, key, value, hidden) -> None:
        
        with self.buffer_lock:
            self._add_to_buffer(input_tokens, roi, key, value, hidden)
        
        # when calling the insert, the current process is a sender
        # need to launch the request handler and start listening to request.
        if self.request_handling_thread is None:
            self.request_handling_thread = threading.Thread(
                target=self.drop_select_handler)
            self.request_handling_thread.start()
            
            