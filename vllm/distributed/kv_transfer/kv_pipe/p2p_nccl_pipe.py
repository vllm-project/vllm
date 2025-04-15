# SPDX-License-Identifier: Apache-2.0

import logging
import threading
import time
import typing
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import msgpack
import torch
import zmq

from vllm.config import KVTransferConfig
from vllm.distributed.device_communicators.pynccl_wrapper import (
    NCCLLibrary, buffer_type, cudaStream_t, ncclComm_t, ncclDataTypeEnum)
from vllm.utils import current_stream, get_ip

logger = logging.getLogger(__name__)


class P2pNcclPipe:

    def __init__(self,
                 local_rank: int,
                 config: KVTransferConfig,
                 hostname: str = "",
                 port_offset: int = 0,
                 library_path: Optional[str] = None) -> None:
        self.config = config
        self.rank = port_offset
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.nccl = NCCLLibrary(library_path)

        if not hostname:
            hostname = get_ip()
        port = self.config.kv_port + port_offset
        if port == 0:
            raise ValueError("Port cannot be 0")
        self._hostname = hostname
        self._port = port

        # Each card corresponds to a ZMQ address.
        self.zmq_address = f"{self._hostname}:{self._port}"

        # The `http_port` must be consistent with the port of OpenAI.
        self.http_address = (
            f"{self._hostname}:"
            f"{self.config.kv_connector_extra_config['http_port']}")

        # If `proxy_ip` or `proxy_port` is `""`,
        # then the ping thread will not be enabled.
        proxy_ip = self.config.get_from_extra_config("proxy_ip", "")
        proxy_port = self.config.get_from_extra_config("proxy_port", "")
        if proxy_ip == "" or proxy_port == "":
            self.proxy_address = ""
        else:
            self.proxy_address = proxy_ip + ":" + proxy_port

        self.context = zmq.Context()
        self.router_socket = self.context.socket(zmq.ROUTER)
        self.router_socket.bind(f"tcp://{self.zmq_address}")

        self.poller = zmq.Poller()
        self.poller.register(self.router_socket, zmq.POLLIN)

        self.send_store_cv = threading.Condition()
        self.send_queue_cv = threading.Condition()
        self.recv_store_cv = threading.Condition()
        self.comm_cv = threading.Condition()

        # The sending type includes tree mutually exclusive options:
        # PUT, GET, PUT_ASYNC.
        self.send_type = self.config.get_from_extra_config("send_type", "PUT")
        if self.send_type == "GET":
            self.send_store: Dict[str,
                                  torch.Tensor] = {}  # tensor_id: torch.Tensor
        else:
            # PUT or PUT_ASYNC
            self.send_queue: Deque[
                List[Any]] = deque()  # tensor_id: torch.Tensor
            if self.send_type == "PUT_ASYNC":
                self._send_thread = threading.Thread(target=self._send_async,
                                                     daemon=True)
                self._send_thread.start()

        self.recv_store: Dict[str,
                              torch.Tensor] = {}  # tensor_id: torch.Tensor
        self.socks: Dict[str, Any] = {}  # remote_address: client socket
        self.comms: Dict[str, Any] = {}  # remote_address: (ncclComm_t, rank)

        self.buffer_size = 0
        self.buffer_size_threshold = self.config.kv_buffer_size

        self._listener_thread = threading.Thread(
            target=self._listen_for_requests, daemon=True)
        self._listener_thread.start()

        self._ping_thread = None
        if port_offset == 0 and self.proxy_address != "":
            self._ping_thread = threading.Thread(target=self._ping,
                                                 daemon=True)
            self._ping_thread.start()

    def _create_connect(self, remote_address: typing.Optional[str] = None):
        assert remote_address is not None
        if remote_address not in self.socks:
            sock = self.context.socket(zmq.DEALER)
            sock.setsockopt_string(zmq.IDENTITY, self.zmq_address)
            sock.connect(f"tcp://{remote_address}")
            self.socks[remote_address] = sock
            if remote_address in self.comms:
                logger.info("👋comm exists, remote_address:%s, comms:%s",
                            remote_address, self.comms)
                return sock, self.comms[remote_address]

            unique_id = self.nccl.ncclGetUniqueId()
            data = {"cmd": "NEW", "unique_id": bytes(unique_id.internal)}
            sock.send(msgpack.dumps(data))

            with torch.cuda.device(self.device):
                rank = 0
                comm: ncclComm_t = self.nccl.ncclCommInitRank(
                    2, unique_id, rank)
                self.comms[remote_address] = (comm, rank)
                logger.info("🤝ncclCommInitRank Success, %s👉%s, MyRank: %s",
                            self.zmq_address, remote_address, rank)

        return self.socks[remote_address], self.comms[remote_address]

    def send_tensor(
        self,
        tensor_id: str,
        tensor: torch.Tensor,
        remote_address: typing.Optional[str] = None,
    ) -> bool:
        if remote_address is None:
            with self.recv_store_cv:
                self.recv_store[tensor_id] = tensor
                self.recv_store_cv.notify()
            return True
        else:
            if self.send_type == "PUT":
                return self._send_sync(tensor_id, tensor, remote_address)
            elif self.send_type == "PUT_ASYNC":
                with self.send_queue_cv:
                    self.send_queue.append([tensor_id, remote_address, tensor])
                    self.send_queue_cv.notify()
            else:  # GET
                with self.send_store_cv:
                    tensor_size = tensor.element_size() * tensor.numel()
                    while (self.buffer_size + tensor_size
                           > self.buffer_size_threshold):
                        oldest_tenser_id = next(iter(self.send_store))
                        oldest_tenser = self.send_store.pop(oldest_tenser_id)
                        oldest_tenser_size = oldest_tenser.element_size(
                        ) * oldest_tenser.numel()
                        self.buffer_size -= oldest_tenser_size
                        logger.info(
                            "⛔[GET]Send to %s, tensor_id:%s, tensor_size:%d,"
                            " buffer_size:%d, oldest_tenser_size:%d, rank:%d",
                            remote_address, tensor_id, tensor_size,
                            self.buffer_size, oldest_tenser_size, self.rank)

                    self.send_store[tensor_id] = tensor
                    self.buffer_size += tensor_size
                    logger.info(
                        "🔵[GET]Send to %s, tensor_id:%s, tensor_size:%d, "
                        "shape:%s, rank:%d, buffer_size:%d(%.2f%%)",
                        remote_address, tensor_id, tensor_size,
                        self.buffer_size, tensor.shape, self.rank,
                        self.buffer_size / self.buffer_size_threshold * 100)

        return True

    def recv_tensor(
        self,
        tensor_id: str,
        remote_address: typing.Optional[str] = None,
    ) -> torch.Tensor:
        if self.send_type == "PUT" or self.send_type == "PUT_ASYNC":
            start_time = time.time()
            with self.recv_store_cv:
                if tensor_id not in self.recv_store:
                    self.recv_store_cv.wait(timeout=0.001)
                tensor = self.recv_store.pop(tensor_id, None)
            duration = time.time() - start_time
            if tensor is not None:
                self.buffer_size -= (tensor.element_size() * tensor.numel())
                logger.info(
                    "🔵[PUT]Recv From %s, tensor_id:%s, shape:%s, "
                    "duration:%.3fms, size:%.3fGB, rank:%d", remote_address,
                    tensor_id, tensor.shape, duration * 1000,
                    tensor.element_size() * tensor.numel() / 1024**3,
                    self.rank)
            else:
                logger.warning(
                    "🔴[PUT]Recv From %s, tensor_id:%s, duration:%.3fms, "
                    "rank:%d", remote_address, tensor_id, duration * 1000,
                    self.rank)
            return tensor

        # GET
        if remote_address is None:
            return None

        if remote_address not in self.socks:
            self._create_connect(remote_address)

        sock = self.socks[remote_address]
        comm, rank = self.comms[remote_address]

        data = {"cmd": "GET", "tensor_id": tensor_id}
        sock.send(msgpack.dumps(data))

        message = sock.recv()
        data = msgpack.loads(message)
        if data["ret"] != 0:
            logger.warning("🔴[GET]Recv From %s, tensor_id: %s, ret: %d",
                           remote_address, tensor_id, data["ret"])
            return None

        tensor = torch.empty(data["shape"],
                             dtype=getattr(torch, data["dtype"]),
                             device=self.device)

        start_time = time.time()
        self._recv(comm, tensor, rank ^ 1)
        duration = time.time() - start_time
        logger.info(
            "🔵[GET]Recv From %s, tensor_id:%s, shape:%s, duration:%.3fms, "
            "size:%.3fGB, rank:%d", remote_address, tensor_id, tensor.shape,
            duration * 1000,
            tensor.element_size() * tensor.numel() / 1024**3, self.rank)

        return tensor

    def _listen_for_requests(self):
        while True:
            socks = dict(self.poller.poll())
            if self.router_socket in socks:
                remote_address, message = self.router_socket.recv_multipart()
                data = msgpack.loads(message)
                logger.debug("Received message from %s, data:%s",
                             remote_address.decode(), data)
                if data["cmd"] == "NEW":
                    unique_id = self.nccl.unique_id_from_bytes(
                        bytes(data["unique_id"]))
                    with torch.cuda.device(self.device):
                        rank = 1
                        comm: ncclComm_t = self.nccl.ncclCommInitRank(
                            2, unique_id, rank)
                        self.comms[remote_address.decode()] = (comm, rank)
                        logger.info(
                            "🤝ncclCommInitRank Success, %s👈%s, MyRank:%s",
                            self.zmq_address, remote_address.decode(), rank)
                elif data["cmd"] == "PUT":
                    try:
                        tensor = torch.empty(data["shape"],
                                             dtype=getattr(
                                                 torch, data["dtype"]),
                                             device=self.device)

                        tensor_size = tensor.element_size() * tensor.numel()
                        if (self.buffer_size + tensor_size
                                > self.buffer_size_threshold):
                            self.router_socket.send_multipart(
                                [remote_address, b"2"])
                            logger.warning(
                                "🔴[PUT]Recv Tensor, Out Of Threshold, "
                                "%s👈%s, data:%s", self.zmq_address,
                                remote_address.decode(), data)
                            continue

                        self.buffer_size += tensor_size
                        self.router_socket.send_multipart(
                            [remote_address, b"0"])
                        comm, rank = self.comms[remote_address.decode()]
                        self._recv(comm, tensor, rank ^ 1)

                        tensor_id = data["tensor_id"]
                        with self.recv_store_cv:
                            self.recv_store[tensor_id] = tensor
                            self.recv_store_cv.notify()
                        logger.info(
                            "🔵[PUT]Recv Tensor, %s👈%s, MyRank:%s, data:%s, "
                            "shape:%s", self.zmq_address,
                            remote_address.decode(), rank, data, tensor.shape)

                    except torch.cuda.OutOfMemoryError:
                        self.router_socket.send_multipart(
                            [remote_address, b"1"])
                        logger.warning(
                            "🔴[PUT]Recv Tensor, Out Of Memory, %s👈%s, "
                            "data:%s", self.zmq_address,
                            remote_address.decode(), data)

                elif data["cmd"] == "GET":
                    tensor_id = data["tensor_id"]
                    with self.send_store_cv:
                        tensor = self.send_store.pop(tensor_id, None)
                        if tensor is not None:
                            data = {
                                "ret": 0,
                                "shape": tensor.shape,
                                "dtype":
                                str(tensor.dtype).replace("torch.", "")
                            }
                            # LRU
                            self.send_store[tensor_id] = tensor
                        else:
                            data = {"ret": 1}

                    self.router_socket.send_multipart(
                        [remote_address, msgpack.dumps(data)])

                    if data["ret"] == 0:
                        self._send(comm, tensor.to(self.device), rank ^ 1)

                    logger.info(
                        "🔵[GET]Send Tensor, %s👉%s, "
                        "MyRank:%s, data:%s", self.zmq_address,
                        remote_address.decode(), rank, data)
                else:
                    logger.warning(
                        "🚧Unexpected, Received message from %s, data:%s",
                        remote_address, data)

    # Asynchronous sending may cause conflicts between P2P NCCL and
    # NCCL used in TP/PP, which can lead to deadlock issues.
    def _send_async(self):
        while True:
            with self.send_queue_cv:
                while not self.send_queue:
                    self.send_queue_cv.wait()
                tensor_id, remote_address, tensor = self.send_queue.popleft()
            self._send_sync(tensor_id, tensor, remote_address)

    def _send_sync(
        self,
        tensor_id: str,
        tensor: torch.Tensor,
        remote_address: typing.Optional[str] = None,
    ) -> bool:
        if remote_address is None:
            return False
        if remote_address not in self.socks:
            self._create_connect(remote_address)

        sock = self.socks[remote_address]
        comm, rank = self.comms[remote_address]
        data = {
            "cmd": "PUT",
            "tensor_id": tensor_id,
            "shape": tensor.shape,
            "dtype": str(tensor.dtype).replace("torch.", "")
        }
        sock.send(msgpack.dumps(data))

        response = sock.recv()
        if response != b"0":
            # with self.send_queue_cv:
            #     self.send_queue.append([tensor_id, remote_address, tensor])
            #     self.send_queue_cv.notify()
            logger.warning(
                "🔴Send Tensor, Peer Out Of Memory/Threshold, %s 👉 %s, "
                "MyRank:%s, data:%s, tensor:%s, size:%fGB, response:%s",
                self.zmq_address, remote_address, rank, data, tensor.shape,
                tensor.element_size() * tensor.numel() / 1024**3,
                response.decode())
            return False

        self._send(comm, tensor.to(self.device), rank ^ 1)
        logger.info("🔵Send Tensor, %s👉%s, MyRank:%s, data:%s, tensor:%s",
                    self.zmq_address, remote_address, rank, data, tensor.shape)
        return True

    def _ping(self):
        sock = self.context.socket(zmq.DEALER)
        sock.setsockopt_string(zmq.IDENTITY, self.zmq_address)
        logger.debug("ping start, zmq_address:%s", self.zmq_address)
        sock.connect(f"tcp://{self.proxy_address}")
        data = {
            "type": "P" if self.config.is_kv_producer else "D",
            "http_address": self.http_address,
            "zmq_address": self.zmq_address
        }
        while True:
            sock.send(msgpack.dumps(data))
            time.sleep(3)

    def _send(self, comm, tensor: torch.Tensor, dst: int, stream=None):
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}")
        if stream is None:
            stream = current_stream()

        with self.comm_cv:
            self.nccl.ncclSend(buffer_type(tensor.data_ptr()), tensor.numel(),
                               ncclDataTypeEnum.from_torch(tensor.dtype), dst,
                               comm, cudaStream_t(stream.cuda_stream))

    def _recv(self, comm, tensor: torch.Tensor, src: int, stream=None):
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}")
        if stream is None:
            stream = current_stream()

        with self.comm_cv:
            self.nccl.ncclRecv(buffer_type(tensor.data_ptr()), tensor.numel(),
                               ncclDataTypeEnum.from_torch(tensor.dtype), src,
                               comm, cudaStream_t(stream.cuda_stream))

    def close(self) -> None:
        self._listener_thread.join()
        if self.send_type == "PUT_ASYNC":
            self._send_thread.join()
        if self._ping_thread is not None:
            self._ping_thread.join()
