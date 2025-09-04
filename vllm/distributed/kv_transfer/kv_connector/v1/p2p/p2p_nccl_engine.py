# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import os
import threading
import time
import typing
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional

import msgpack
import torch
import zmq

from vllm.config import KVTransferConfig
from vllm.distributed.device_communicators.pynccl_wrapper import (
    NCCLLibrary, buffer_type, cudaStream_t, ncclComm_t, ncclDataTypeEnum)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.tensor_memory_pool import (  # noqa: E501
    TensorMemoryPool)
from vllm.utils import current_stream, get_ip

logger = logging.getLogger(__name__)

DEFAULT_MEM_POOL_SIZE_GB = 32


@contextmanager
def set_p2p_nccl_context(num_channels: str):
    original_values: dict[str, Any] = {}
    env_vars = [
        'NCCL_MAX_NCHANNELS',
        'NCCL_MIN_NCHANNELS',
        'NCCL_CUMEM_ENABLE',
        'NCCL_BUFFSIZE',
        'NCCL_PROTO',  # LL,LL128,SIMPLE
        'NCCL_ALGO',  # RING,TREE
    ]

    for var in env_vars:
        original_values[var] = os.environ.get(var)

    logger.info("set_p2p_nccl_context, original_values: %s", original_values)

    try:
        os.environ['NCCL_MAX_NCHANNELS'] = num_channels
        os.environ['NCCL_MIN_NCHANNELS'] = num_channels
        os.environ['NCCL_CUMEM_ENABLE'] = '1'
        yield
    finally:
        for var in env_vars:
            if original_values[var] is not None:
                os.environ[var] = original_values[var]
            else:
                os.environ.pop(var, None)


@dataclass
class SendQueueItem:
    tensor_id: str
    remote_address: str
    tensor: torch.Tensor


class P2pNcclEngine:

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
        port = int(self.config.kv_port) + port_offset
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

        self.send_stream = torch.cuda.Stream()
        self.recv_stream = torch.cuda.Stream()

        mem_pool_size_gb = float(
            self.config.get_from_extra_config("mem_pool_size_gb",
                                              DEFAULT_MEM_POOL_SIZE_GB))
        self.pool = TensorMemoryPool(max_block_size=int(mem_pool_size_gb *
                                                        1024**3))  # GB

        # The sending type includes tree mutually exclusive options:
        # PUT, GET, PUT_ASYNC.
        self.send_type = self.config.get_from_extra_config(
            "send_type", "PUT_ASYNC")
        if self.send_type == "GET":
            # tensor_id: torch.Tensor
            self.send_store: dict[str, torch.Tensor] = {}
        else:
            # PUT or PUT_ASYNC
            # tensor_id: torch.Tensor
            self.send_queue: deque[SendQueueItem] = deque()
            self.send_request_id_to_tensor_ids: dict[str, set[str]] = {}
            if self.send_type == "PUT_ASYNC":
                self._send_thread = threading.Thread(target=self.send_async,
                                                     daemon=True)
                self._send_thread.start()

        # tensor_id: torch.Tensor/(addr, dtype, shape)
        self.recv_store: dict[str, Any] = {}
        self.recv_request_id_to_tensor_ids: dict[str, set[str]] = {}
        self.socks: dict[str, Any] = {}  # remote_address: client socket
        self.comms: dict[str, Any] = {}  # remote_address: (ncclComm_t, rank)

        self.buffer_size = 0
        self.buffer_size_threshold = float(self.config.kv_buffer_size)

        self.nccl_num_channels = self.config.get_from_extra_config(
            "nccl_num_channels", "8")

        self._listener_thread = threading.Thread(
            target=self.listen_for_requests, daemon=True)
        self._listener_thread.start()

        self._ping_thread = None
        if port_offset == 0 and self.proxy_address != "":
            self._ping_thread = threading.Thread(target=self.ping, daemon=True)
            self._ping_thread.start()

        logger.info(
            "💯P2pNcclEngine init, rank:%d, local_rank:%d, http_address:%s, "
            "zmq_address:%s, proxy_address:%s, send_type:%s, buffer_size_"
            "threshold:%.2f, nccl_num_channels:%s", self.rank, self.local_rank,
            self.http_address, self.zmq_address, self.proxy_address,
            self.send_type, self.buffer_size_threshold, self.nccl_num_channels)


    def _cleanup_connection(self, remote_address: str):
        logger.warning(
            "Cleaning up stale connection for remote_address: %s",
            remote_address
        )
        if remote_address in self.comms:
            # Note: ncclCommDestroy is ideally called, but the peer process is likely gone.
            # Simply removing the handle is the primary step.
            del self.comms[remote_address]
        
        if remote_address in self.socks:
            sock = self.socks.pop(remote_address)
            try:
                # Set LINGER to 0 to close immediately without waiting
                sock.setsockopt(zmq.LINGER, 0)
                sock.close()
            except Exception as e:
                logger.debug("Error closing stale ZMQ socket: %s", e)


    def create_connect(self, remote_address: typing.Optional[str] = None):
        assert remote_address is not None
        if remote_address in self.socks or remote_address in self.comms:
            self._cleanup_connection(remote_address)

        sock = self.context.socket(zmq.DEALER)
        sock.setsockopt_string(zmq.IDENTITY, self.zmq_address)
        sock.setsockopt(zmq.SNDTIMEO, 5000) # 5 seconds send timeout
        sock.setsockopt(zmq.RCVTIMEO, 5000) # 5 seconds receive timeout
        sock.connect(f"tcp://{remote_address}")
        self.socks[remote_address] = sock

        try:
            unique_id = self.nccl.ncclGetUniqueId()
            data = {"cmd": "NEW", "unique_id": bytes(unique_id.internal)}
            sock.send(msgpack.dumps(data))

            with torch.cuda.device(self.device):
                rank = 0
                with set_p2p_nccl_context(self.nccl_num_channels):
                    comm: ncclComm_t = self.nccl.ncclCommInitRank(
                        2, unique_id, rank)
                self.comms[remote_address] = (comm, rank)
                logger.info("🤝ncclCommInitRank Success, %s👉%s, MyRank:%s",
                            self.zmq_address, remote_address, rank)

        except Exception as e:
            self._cleanup_connection(remote_address)
            # Re-raise to prevent using a partially-formed connection
            raise

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

        item = SendQueueItem(tensor_id=tensor_id,
                             remote_address=remote_address,
                             tensor=tensor)

        if self.send_type == "PUT":
            return self.send_sync(item)

        if self.send_type == "PUT_ASYNC":
            with self.send_queue_cv:
                self.send_queue.append(item)
                self.send_queue_cv.notify()
            return True

        # GET
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
                    remote_address, tensor_id, tensor_size, self.buffer_size,
                    oldest_tenser_size, self.rank)

            self.send_store[tensor_id] = tensor
            self.buffer_size += tensor_size
            logger.debug(
                "🔵[GET]Send to %s, tensor_id:%s, tensor_size:%d, "
                "shape:%s, rank:%d, buffer_size:%d(%.2f%%)", remote_address,
                tensor_id, tensor_size, tensor.shape, self.rank,
                self.buffer_size,
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
                while tensor_id not in self.recv_store:
                    self.recv_store_cv.wait()
                tensor = self.recv_store[tensor_id]

            if tensor is not None:
                if isinstance(tensor, tuple):
                    addr, dtype, shape = tensor
                    tensor = self.pool.load_tensor(addr, dtype, shape,
                                                   self.device)
                else:
                    self.buffer_size -= (tensor.element_size() *
                                         tensor.numel())
            else:
                duration = time.time() - start_time
                logger.warning(
                    "🔴[PUT]Recv From %s, tensor_id:%s, duration:%.3fms, "
                    "rank:%d", remote_address, tensor_id, duration * 1000,
                    self.rank)
            return tensor

        # GET
        if remote_address is None:
            return None

        # Check and reconnect if necessary
        try:
            # Check and reconnect if necessary
            if (remote_address not in self.socks or 
                remote_address not in self.comms):
                self.create_connect(remote_address)

            sock = self.socks[remote_address]
            comm, rank = self.comms[remote_address]
            
            data = {"cmd": "GET", "tensor_id": tensor_id}
            sock.send(msgpack.dumps(data))

            message = sock.recv()
            data = msgpack.loads(message)
            if data["ret"] != 0:
                logger.warning("🔴[GET]Recv From %s, tensor_id: %s,"
                               " peer returned error: %d",
                               remote_address, tensor_id, data["ret"])
                return None

            with torch.cuda.stream(self.recv_stream):
                tensor = torch.empty(data["shape"],
                                     dtype=getattr(torch, data["dtype"]),
                                     device=self.device)

            self.recv(comm, tensor, rank ^ 1, self.recv_stream)
            return tensor

        except (Exception, zmq.ZMQError) as e:
            self._cleanup_connection(remote_address)
            return None



    def listen_for_requests(self):
        while True:
            socks = dict(self.poller.poll())
            if self.router_socket not in socks:
                continue

            remote_address_bytes, message = self.router_socket.recv_multipart()
            remote_address = remote_address_bytes.decode()
            data = msgpack.loads(message)
            if data["cmd"] == "NEW":
                if remote_address in self.comms:
                     self._cleanup_connection(remote_address)

                unique_id = self.nccl.unique_id_from_bytes(
                    bytes(data["unique_id"]))
                with torch.cuda.device(self.device):
                    rank = 1
                    with set_p2p_nccl_context(self.nccl_num_channels):
                        comm: ncclComm_t = self.nccl.ncclCommInitRank(
                            2, unique_id, rank)
                    self.comms[remote_address] = (comm, rank)
                    logger.info("🤝ncclCommInitRank Success, %s👈%s, MyRank:%s",
                                self.zmq_address, remote_address,
                                rank)
            elif data["cmd"] == "PUT":
                tensor_id = data["tensor_id"]
                if remote_address not in self.comms:
                    logger.warning(
                        "🔴[PUT] Received PUT from %s but no NCCL comm exists."
                        "Requesting re-initialization.",
                        remote_address
                    )
                    # Respond with code '2' to signal the sender to re-initialize.
                    self.router_socket.send_multipart([
                        remote_address_bytes, b"2"])
                    continue
                
                try:
                    with torch.cuda.stream(self.recv_stream):
                        tensor = torch.empty(data["shape"],
                                             dtype=getattr(
                                                 torch, data["dtype"]),
                                             device=self.device)
                    # Send '0' to signal OK to receive NCCL data
                    self.router_socket.send_multipart([
                        remote_address_bytes, b"0"])
                    
                    # We've already confirmed comm exists
                    comm, rank = self.comms[remote_address]
                    self.recv(comm, tensor, rank ^ 1, self.recv_stream)
                    tensor_size = tensor.element_size() * tensor.numel()
                    if (self.buffer_size + tensor_size
                            > self.buffer_size_threshold):
                        # Store Tensor in memory pool
                        addr = self.pool.store_tensor(tensor)
                        tensor = (addr, tensor.dtype, tensor.shape)
                        logger.warning(
                            "🔴[PUT] Recv Tensor, Out Of Threshold, %s👈%s, "
                            "addr:%d",
                            self.zmq_address, remote_address, addr)
                    else:
                        self.buffer_size += tensor_size
                except Exception as e:
                    # In case of error during recv (e.g. OOM), 
                    # we might not be able to reply.
                    # The sender's timeout will handle this.
                    tensor = None
                    logger.warning(
                        "🔴[PUT] Recv Tensor failed after sending OK, %s👈%s, "
                        "error:%s. Cleaning up connection.", 
                        self.zmq_address, remote_address,
                        str(e))
                    # The connection might be broken, clean it up.
                    self._cleanup_connection(remote_address)

                with self.recv_store_cv:
                    self.recv_store[tensor_id] = tensor
                    self.have_received_tensor_id(tensor_id)
                    self.recv_store_cv.notify()

            elif data["cmd"] == "GET":
                tensor_id = data["tensor_id"]
                tensor = None
                with self.send_store_cv:
                    tensor = self.send_store.get(tensor_id)
                    
                if tensor is not None:
                    data = {
                        "ret": 0,
                        "shape": tensor.shape,
                        "dtype": str(tensor.dtype).replace("torch.", "")
                    }
                    self.router_socket.send_multipart(
                        [remote_address_bytes, msgpack.dumps(data)])
                    
                    try:
                        if remote_address not in self.comms:
                           raise ConnectionError(
                               f"NCCL comm not found for {remote_address}")
                        comm, rank = self.comms[remote_address]
                        self.send(comm, tensor.to(self.device), rank ^ 1,
                                  self.send_stream)
                        with self.send_store_cv:
                            self.send_store.pop(tensor_id, None)
                            self.send_store[tensor_id] = tensor
                            self.have_sent_tensor_id(tensor_id)
                    except Exception as e:
                        logger.error("Failed to send tensor for GET to %s: %s."
                                     " Cleaning up connection.",
                                     remote_address, e)
                        self._cleanup_connection(remote_address)
                else:
                    data = {"ret": 1}
                    self.router_socket.send_multipart(
                        [remote_address_bytes, msgpack.dumps(data)])
            else:
                logger.warning(
                    "🚧Unexpected, Received message from %s, data:%s",
                    remote_address, data)

    def have_sent_tensor_id(self, tensor_id: str):
        request_id = tensor_id.split('#')[0]
        if request_id not in self.send_request_id_to_tensor_ids:
            self.send_request_id_to_tensor_ids[request_id] = set()
        self.send_request_id_to_tensor_ids[request_id].add(tensor_id)

    def have_received_tensor_id(self, tensor_id: str):
        request_id = tensor_id.split('#')[0]
        if request_id not in self.recv_request_id_to_tensor_ids:
            self.recv_request_id_to_tensor_ids[request_id] = set()
        self.recv_request_id_to_tensor_ids[request_id].add(tensor_id)

    def send_async(self):
        while True:
            with self.send_queue_cv:
                while not self.send_queue:
                    self.send_queue_cv.wait()
                item = self.send_queue.popleft()
                if not self.send_queue:
                    self.send_queue_cv.notify()
            self.send_sync(item)

    def wait_for_sent(self):
        if self.send_type == "PUT_ASYNC":
            start_time = time.time()
            with self.send_queue_cv:
                while self.send_queue:
                    self.send_queue_cv.wait()
            duration = time.time() - start_time
            logger.debug(
                "🚧[PUT_ASYNC]It took %.3fms to wait for the send_queue"
                " to be empty, rank:%d", duration * 1000, self.rank)

    def send_sync(self, item: SendQueueItem) -> bool:
        if item.remote_address is None:
            return False
        
        # Check and reconnect if necessary
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Step 1: Ensure connection exists
                if (item.remote_address not in self.socks or 
                    item.remote_address not in self.comms):
                    logger.info("No connection to %s, creating one (attempt %d/%d).",
                                item.remote_address, attempt + 1, max_retries)
                    self.create_connect(item.remote_address)
                
                sock = self.socks[item.remote_address]
                comm, rank = self.comms[item.remote_address]

                # Step 2: Prepare and send PUT command
                with self.send_stream:
                    tensor = item.tensor
                data = {
                    "cmd": "PUT",
                    "tensor_id": item.tensor_id,
                    "shape": tensor.shape,
                    "dtype": str(tensor.dtype).replace("torch.", "")
                }
                
                sock.send(msgpack.dumps(data))
                response = sock.recv()

                # Step 3: Handle response
                if response == b"0":  # OK
                    logger.debug("Peer %s is ready to receive.",
                                 item.remote_address)
                    self.send(comm, tensor.to(self.device), 
                              rank ^ 1, self.send_stream)
                    if self.send_type == "PUT_ASYNC":
                        self.have_sent_tensor_id(item.tensor_id)
                    return True  # Success

                elif response == b"1":  # Peer OOM/Error
                    logger.error(
                        "🔴 Peer %s reported Out Of Memory/Threshold."
                        " Aborting send.",
                        item.remote_address)
                    return False # Unrecoverable error, do not retry

                elif response == b"2":  # Re-initialize requested
                    logger.warning(
                        "🟡 Peer %s requested re-initialization. "
                        "Cleaning up and retrying (%d/%d)...",
                        item.remote_address, attempt + 1, max_retries)
                    self._cleanup_connection(item.remote_address)
                    time.sleep(0.1)  # Small delay before retry
                    continue # Go to next attempt in the loop

                else:
                    logger.error("Received unknown response '%s' from peer %s.",
                                 response, item.remote_address)
                    self._cleanup_connection(item.remote_address)
                    continue

            except (Exception, zmq.ZMQError) as e:
                logger.error(
                    "💥 Connection error during send_sync to %s "
                    "(attempt %d/%d): %s. Cleaning up and retrying.",
                    item.remote_address, attempt + 1, max_retries, e
                )
                self._cleanup_connection(item.remote_address)
                time.sleep(0.1) # Small delay before retry

        logger.error("Failed to send tensor to %s after %d retries.",
                     item.remote_address, max_retries)
        return False


    def get_finished(
            self, finished_req_ids: set[str], no_compile_layers
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer,
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """

        # Clear the buffer upon request completion.
        for request_id in finished_req_ids:
            for layer_name in no_compile_layers:
                tensor_id = request_id + "#" + layer_name
                if tensor_id in self.recv_store:
                    with self.recv_store_cv:
                        tensor = self.recv_store.pop(tensor_id, None)
                        self.send_request_id_to_tensor_ids.pop(
                            request_id, None)
                        self.recv_request_id_to_tensor_ids.pop(
                            request_id, None)
                    if isinstance(tensor, tuple):
                        addr, _, _ = tensor
                        self.pool.free(addr)

        # TODO:Retrieve requests that have already sent the KV cache.
        finished_sending: set[str] = set()

        # TODO:Retrieve requests that have already received the KV cache.
        finished_recving: set[str] = set()

        return finished_sending or None, finished_recving or None

    def ping(self):
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

    def send(self, comm, tensor: torch.Tensor, dst: int, stream=None):
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}")
        if stream is None:
            stream = current_stream()

        with torch.cuda.stream(stream):
            self.nccl.ncclSend(buffer_type(tensor.data_ptr()), tensor.numel(),
                               ncclDataTypeEnum.from_torch(tensor.dtype), dst,
                               comm, cudaStream_t(stream.cuda_stream))
        stream.synchronize()

    def recv(self, comm, tensor: torch.Tensor, src: int, stream=None):
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}")
        if stream is None:
            stream = current_stream()

        with torch.cuda.stream(stream):
            self.nccl.ncclRecv(buffer_type(tensor.data_ptr()), tensor.numel(),
                               ncclDataTypeEnum.from_torch(tensor.dtype), src,
                               comm, cudaStream_t(stream.cuda_stream))
        stream.synchronize()

    def close(self) -> None:
        self._listener_thread.join()
        if self.send_type == "PUT_ASYNC":
            self._send_thread.join()
        if self._ping_thread is not None:
            self._ping_thread.join()
