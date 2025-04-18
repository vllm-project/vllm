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

import logging
import threading
import typing
import zmq
import socket
import time
import torch

from vllm.distributed.kv_transfer.kv_pipe.pynccl_pipe import PyNcclPipe


logger = logging.getLogger(__name__)


class DynamoNcclDataPlane:
    def __init__(
        self,
        data_pipe: PyNcclPipe,
        hostname: str = "",
        port: int = 0,
    ) -> None:
        
        self.data_pipe = data_pipe
        if not hostname:
            hostname = socket.gethostname()
        if port == 0:
            raise ValueError("Port cannot be 0")
        self._hostname = hostname
        self._port = port
        self.store = {}
        self.context = zmq.Context()
        self.rep_socket = self.context.socket(zmq.REP)
        logger.info(f"Rank {self.rank} binding to {self._hostname}:{self._port}")
        self.rep_socket.bind(f"tcp://{self._hostname}:{self._port}")
        self._listener_thread = threading.Thread(target=self.listen_for_requests, daemon=True)
        self._listener_thread.start()
        self.req_sockets = {}
        logger.info(f"Rank {self.rank} connected to the server")

    @property
    def rank(self):
        return self.data_pipe.kv_group_rank
    
    def send_tensor(
        self,
        tensor: torch.Tensor,
        tensor_id: str,
        remote_address: typing.Optional[str] = None,
    ):
        logger.debug(f"Rank {self.rank} sending tensor {tensor_id} to {remote_address}")
        return self._send_tensor(tensor, tensor_id, remote_address)

    def recv_tensor(
        self,
        tensor_id: str,
        remote_address: typing.Optional[str] = None,
    ) -> torch.Tensor:
        ret = self._recv_tensor(tensor_id, remote_address)
        return ret

    def _send_tensor(
        self,
        tensor: torch.Tensor,
        tensor_id: str,
        remote_address: typing.Optional[str] = None,
    ):
        logger.debug(f"Rank {self.rank} storing tensor with id {tensor_id} of shape {tensor.shape} and dtype {tensor.dtype}")
        if remote_address is None:
            self.store[tensor_id] = tensor
        else:
            # tensor_shape = "_".join(str(dim) for dim in tensor.shape)
            # tensor_dtype = str(tensor.dtype)
            if remote_address not in self.req_sockets:
                self.req_sockets[remote_address] = self.context.socket(zmq.REQ)
                self.req_sockets[remote_address].connect(f"tcp://{remote_address}")

            req_socket = self.req_sockets[remote_address]
            # req_socket.connect(f"tcp://{remote_address}")
            req_socket.send_string(f"PUT {self.rank} {tensor_id}")
            dst_rank = req_socket.recv_string()
            logger.debug(f"Rank {self.rank} sending tensor {tensor_id} to rank {dst_rank}")
            self.data_pipe.send_tensor(tensor, int(dst_rank))

    def _recv_tensor(
        self,
        tensor_id: str,
        remote_address: typing.Optional[str] = None,
    ) -> torch.Tensor:
        logger.debug(f"Rank {self.rank} receiving tensor")
        if remote_address is not None:
            raise NotImplementedError("Getting tensor from remote rank not implemented")
        if tensor_id in self.store:
            logger.debug(f"Popping tensor {tensor_id} from store")
            future = self.store.pop(tensor_id)
            tensor = future.result() # TODO ptarasiewicz we should run other request instead of wait
            logger.debug(f"Rank {self.rank} received tensor")
            return tensor
            
        logger.debug(f"Rank {self.rank} waiting for tensor {tensor_id}")
        time.sleep(0.001)
        return self._recv_tensor(tensor_id, remote_address)
        # raise NotImplementedError("Tensor not found in store")

    def _receive_tensor(
        self,
        tensor_id: str,
        rank: int,
    ):
        future = self.data_pipe.recv_tensor(rank)
        logger.debug(f"Rank {self.rank} storing tensor {tensor_id} in store")
        self.store[tensor_id] = future

    def listen_for_requests(self):
        while True:
            cmd, rank, tensor_id = self.rep_socket.recv_string().split()
            logger.debug(f"Rank {self.rank} received request for tensor {tensor_id}")
            self.rep_socket.send_string(f"{self.rank}")
            if cmd == "GET":
                raise NotImplementedError("Getting tensor from remote rank not implemented")
            elif cmd == "PUT":
                rank = int(rank)
                # shape = [int(dim) for dim in shape.split("_")]
                # dtype = getattr(torch, dtype)
                self._receive_tensor(tensor_id, rank)
