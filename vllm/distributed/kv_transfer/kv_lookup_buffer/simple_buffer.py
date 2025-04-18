# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Deque, List, Optional, Union, Dict

import torch

from vllm.distributed.kv_transfer.kv_lookup_buffer.base import (
    KVLookupBufferBase)
from vllm.distributed.kv_transfer.kv_pipe.base import KVPipeBase
from vllm.logger import init_logger

logger = init_logger(__name__)


class SimpleBuffer(KVLookupBufferBase):

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

        self.buffer: Deque[List[torch.Tensor]] = deque()

        self.buffer_size = 0
        self.buffer_size_threshold = buffer_size_thresh
        self.buffer_cv = threading.Condition()
        self.signal_pipe = signal_pipe
        self.data_pipe = data_pipe
        self.request_handling_thread: Optional[ThreadPoolExecutor] = None

        self.normal_signal = torch.tensor([0], device="cpu")
        self.end_signal = None

    def _matches(self, tokens_roi_sender: List[torch.Tensor],
                 tokens_roi_recver: List[torch.Tensor]):

        # tokens_roi_sender: tokens and roi of the producer (in the buffer)
        # tokens_roi_recver: tokens and roi of the consumer (query)

        target_rank_sender = tokens_roi_sender[0]
        target_rank_recver = tokens_roi_recver[0]

        if target_rank_sender.item() != target_rank_recver.item():
            return 0
        
        tokens_sender = tokens_roi_sender[1]
        tokens_recver = tokens_roi_recver[1]
        roi_sender = tokens_roi_sender[2]
        roi_recver = tokens_roi_recver[2]

        if tokens_recver is None:
            # consumer sends an empty request
            # semantics: DROP SELECT * LIMIT 1
            # so any of the data in the buffer can be drop-selected
            return True

        # Assuming that roi is a binary mask on tokens
        tokens_sender = tokens_sender[roi_sender]
        tokens_recver = tokens_recver[roi_recver]

        # simple common prefix matching
        min_length = min(len(tokens_sender), len(tokens_recver))
        if torch.allclose(tokens_sender[:min_length],
                          tokens_recver[:min_length]):
            return min_length

        return 0

    def _send_tensor_and_dec_size(self, tensor: Optional[torch.Tensor],
                                  target_rank: int) -> None:

        assert tensor is not None, "Use self.data_pipe.send(None) instead"
        self.buffer_size -= tensor.element_size() * tensor.numel()
        if tensor.dtype == torch.bool:
            tensor = tensor.float()
        self.data_pipe.send_tensor(tensor, target_rank)

    def _get_element_size(self, data: Optional[Union[List, torch.Tensor]]):

        if isinstance(data, torch.Tensor):
            return data.element_size() * data.numel()
        if not data:
            # cannot perform `not data` on a tensor
            # so this check needs to go after the check above
            return 0

        raise AssertionError(f"Unknown data type {type(data)}")

    def _add_to_buffer(self, target_rank: int, input_tokens: torch.Tensor, roi: torch.Tensor,
                       key: torch.Tensor, value: torch.Tensor,
                       hidden: torch.Tensor):

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
        data_size = sum([self._get_element_size(data) for data in buffer_item])

        with self.buffer_cv:
            if self.buffer_size + data_size > self.buffer_size_threshold:
                # log outside the while loop to avoid this message being logged
                # repeatedly.
                logger.debug("KV transfer buffer is full. Handling...")
                while self.buffer_size + data_size > self.buffer_size_threshold:
                    self.buffer_cv.wait()

            self.buffer_size += data_size
            self.buffer.append(buffer_item)
            self.buffer_cv.notify()

    def _is_end_signal(self, signal):
        return signal is None

    def drop_select_handler(self, rank: int):

        try:

            signal = self.signal_pipe.recv_tensor(rank)
            if self._is_end_signal(signal):
                logger.info("Received end signal!")
                return
            target_kv_rank = self.data_pipe.recv_tensor(rank)
            # assert target_rank.item() == rank, "Target rank does not match"\
            #     "the rank of the drop-select handler"
            input_tokens = self.data_pipe.recv_tensor(rank)
            roi = self.data_pipe.recv_tensor(rank)
            assert roi is not None, "Please provide the roi when sending "\
                "drop-select request"
            roi = (roi > 0.5)
            tokens_roi_recver = [target_kv_rank, input_tokens, roi]

            matched_length = 0

            # perform input tokens and roi matching
            # FIXME: this matching is O(n), ideally it should be O(1)
            # but this buffer size won't (and shouldn't) be too large so
            # the fix is not urgent.
            with self.buffer_lock:

                for _ in range(len(self.buffer)):

                    temp_length = self._matches(self.buffer[0],
                                                tokens_roi_recver)
                    if temp_length > 0:
                        matched_length = temp_length
                        break
                    # rotate the element we just accessed to the end
                    self.buffer.rotate(-1)

                if matched_length > 0:
                    # need to clone the tensor
                    # in case the tensor is freed before sending finishes
                    matched_item = self.buffer.popleft()
                    target_rank = matched_item[0].item()
                    for tensor in matched_item[1:]:
                        self._send_tensor_and_dec_size(tensor, rank)

                else:
                    # no match, just send None
                    for _ in range(5):
                        self.data_pipe.send_tensor(None, rank)

                def is_buffer_available(
                    tokens_roi_recver: List[torch.Tensor], ) -> bool:
                    # perform input tokens and roi matching
                    # FIXME: this matching is O(n), ideally it should be O(1)
                    # but this buffer size won't (and shouldn't) be too large so
                    # the fix is not urgent.
                    for _ in range(len(self.buffer)):
                        if self._matches(self.buffer[0],
                                         tokens_roi_recver) > 0:
                            return True
                        # rotate the element we just accessed to the end
                        self.buffer.rotate(-1)
                    return False

                with self.buffer_cv:
                    while not is_buffer_available(tokens_roi_recver):
                        logger.debug(
                            "KV transfer buffer is not available. Waiting...")
                        self.buffer_cv.wait()
                    # need to clone the tensor
                    # in case the tensor is freed before sending finishes
                    matched_item = self.buffer.popleft()
                    for tensor in matched_item:
                        self._send_tensor_and_dec_size(tensor)
                    self.buffer_cv.notify()

        except RuntimeError as e:
            if 'Connection closed by peer' not in str(e):
                raise e

        logger.debug("Closing drop_select_handler")


    def drop_select(
            self, rank: int, kv_rank: int, input_tokens: Optional[torch.Tensor],
            roi: Optional[torch.Tensor]) -> List[Optional[torch.Tensor]]:

        assert not self.request_handling_thread, \
            "drop_select should be called by the KV cache consumer "\
            "(e.g. the decode vLLM instance)"

        if isinstance(input_tokens, torch.Tensor):
            input_tokens = input_tokens.clone()
        if isinstance(roi, torch.Tensor):
            roi = roi.clone().float()

        self.signal_pipe.send_tensor(self.normal_signal, rank)

        self.data_pipe.send_tensor(torch.tensor(kv_rank), rank)
        self.data_pipe.send_tensor(input_tokens, rank)
        self.data_pipe.send_tensor(roi, rank)

        input_tokens = self.data_pipe.recv_tensor(rank)
        roi = self.data_pipe.recv_tensor(rank)
        if roi is not None:
            # convert from float tensor to bool tensor
            # as PyNccl does not support sending bool tensor
            roi = (roi > 0.5)
        key = self.data_pipe.recv_tensor(rank)
        value = self.data_pipe.recv_tensor(rank)
        hidden = self.data_pipe.recv_tensor(rank)

        return [input_tokens, roi, key, value, hidden]

    def insert(self, input_tokens: torch.Tensor, roi: torch.Tensor,
               key: torch.Tensor, value: torch.Tensor,
               hidden: torch.Tensor) -> None:

        self._add_to_buffer(input_tokens, roi, key, value, hidden)

        # when calling the insert, the current process is a sender
        # need to launch the request handler and start listening to request.
        if self.request_handling_thread is None:
            self.request_handling_thread = ThreadPoolExecutor(max_workers=1)
        self.request_handling_thread.submit(self.drop_select_handler)

    def close(self):

        if hasattr(self, "request_handling_thread") and self.request_handling_thread:
            self.request_handling_thread.shutdown()

        else:
            # TODO: have a explicit close signal and have a explicit way to
            # check if it's requester
            self.signal_pipe.send_tensor(self.end_signal)
