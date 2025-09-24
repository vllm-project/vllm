# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Optional

import msgspec
import torch
import zmq

from vllm.kvserver.blocking_interface import (BlockingKVInterface,
                                              CreateKVInterface)
from vllm.kvserver.protocol import (KVServerCmd, KVServerHandshakeSchedulerMsg,
                                    KVServerHandshakeWorkerMsg,
                                    KVServerLookupRequest,
                                    KVServerOffloadFinished,
                                    KVServerOffloadRequest, decode_cmd,
                                    decode_payload, encode_cmd,
                                    send_lookup_response,
                                    send_offload_response)
from vllm.kvserver.wrapper import CudaIPCWrapper
from vllm.logger import init_logger
from vllm.utils import make_zmq_path, make_zmq_socket

logger = init_logger(__name__)


@dataclass
class KVServerConfig:
    # The host to bind the server to
    host: str
    # The port to bind the protocol socket to
    port: int


ClientId = bytes
RequestId = str
"""
The server module will have a zmq router socket doing the following thing:
    - Listening for init message and heartbeats from vLLMs
    - Receive offload/onload requests from the alive vLLMs
    - Send back the offload/onload status to the alive vLLMs

The main loop will be:
    - Process the incoming requests from clients
      - Immediately process the init message
      - Update the alive status
      - Put the offload/onload requests into a queue
    - Initiate offload/onload jobs in the queue
    - Check the offload/onload job status
    - Send back the offload/onload status to the clients
"""


class KVServer:

    def __init__(self, config: KVServerConfig):
        self.config = config
        self.context = zmq.Context()

        # Protocol socket
        self.zmq_path = make_zmq_path("tcp", config.host, config.port)
        self.main_socket = make_zmq_socket(self.context,
                                           self.zmq_path,
                                           zmq.ROUTER,
                                           bind=True)

        self.poller = zmq.Poller()
        self.poller.register(self.main_socket, zmq.POLLIN)

        self.debug_offload_queue: list[tuple[ClientId, RequestId]] = []

        self.pending_kv_caches: dict[int, list[torch.Tensor]] = {}
        self.kv_interface: Optional[BlockingKVInterface] = None

    def debug_process_offload_requests(self):
        # TODO: send the offload response back to the clients
        for client_id, req_id in self.debug_offload_queue:
            print(f"Processing offload request for client "
                  f"{client_id}, request {req_id}")
            # Simulate sending back an offload finished message
            response_msg = KVServerOffloadFinished(engine_id="",
                                                   request_id=req_id,
                                                   success=True)
            response_payload = msgspec.msgpack.encode(response_msg)
            self.main_socket.send_multipart([
                client_id,
                encode_cmd(KVServerCmd.OFFLOAD_FINISHED), response_payload
            ])
        self.debug_offload_queue.clear()

    def process_tasks(self):
        pass

    def handle_handshake_scheduler(self, client_id, cmd, payload):
        # Deserialize the handshake message
        msg = decode_payload(cmd, payload)
        assert isinstance(msg, KVServerHandshakeSchedulerMsg)
        logger.info("Got handshake from scheduler for engine %s",
                    msg.engine_id)

        # Create the KV interface
        if self.kv_interface is not None:
            logger.error("Right now only one scheduler is supported.")
            return

        self.kv_interface = CreateKVInterface(msg.model_config,
                                              msg.cache_config,
                                              msg.parallel_config,
                                              msg.scheduler_config)

        for rank, gpu_blocks in self.pending_kv_caches.items():
            self.kv_interface.register_kv_caches(rank, gpu_blocks)

    def handle_handshake_worker(self, client_id, cmd, payload):
        # Deserialize the worker handshake message
        msg = decode_payload(cmd, payload)
        assert isinstance(msg, KVServerHandshakeWorkerMsg)
        gpu_blocks = [
            CudaIPCWrapper.deserialize(b).to_tensor() for b in msg.s_gpu_blocks
        ]

        logger.info("Got handshake from worker for rank %d, gpu kv length %d",
                    msg.rank, len(gpu_blocks))

        # Add gpu blocks to pending caches if the interface is not ready
        if self.kv_interface is None:
            self.pending_kv_caches[msg.rank] = gpu_blocks
        else:
            self.kv_interface.register_kv_caches(msg.rank, gpu_blocks)

    def handle_heartbeat(self, client_id, cmd, payload):
        logger.info("Received heartbeat from client %s", client_id)

    def handle_offload_request(self, client_id, cmd, payload):
        msg = decode_payload(cmd, payload)
        assert isinstance(msg, KVServerOffloadRequest)
        logger.info(
            "Received offload request from client %s for engine %s, "
            "request_id %s, blocks %s", client_id, msg.engine_id,
            msg.request_id, msg.block_ids)
        assert self.kv_interface is not None
        logger.info("Block ids: %s", msg.block_ids)
        self.kv_interface.offload(msg.token_ids, msg.block_ids,
                                  msg.skip_leading_tokens)

        # Send back offload finished message since we are blocking
        send_offload_response(self.main_socket, client_id, msg.request_id,
                              True)

    def handle_onload_request(self, client_id, cmd, payload):
        print("Received onload request from:", client_id)

        # TODO: Do something here?

    def handle_lookup_request(self, client_id, cmd, payload):
        msg = decode_payload(cmd, payload)
        assert isinstance(msg, KVServerLookupRequest)
        logger.info(
            "Received lookup request from client %s for engine %s, "
            "model_id %s, request_id %s, token_ids %s", client_id,
            msg.engine_id, msg.model_id, msg.request_id, msg.token_ids)

        number_of_tokens = self.kv_interface.lookup(msg.token_ids)

        # Send back lookup response
        send_lookup_response(self.main_socket, client_id, msg.engine_id,
                             msg.request_id, number_of_tokens)

    def step(self):
        # Poll the main socket for incoming messages
        socks = dict(self.poller.poll(timeout=100))

        if self.main_socket in socks and socks[self.main_socket] == zmq.POLLIN:
            # Receive a message
            msg = self.main_socket.recv_multipart()
            client_id = msg[0]
            cmd = decode_cmd(msg[1])
            payload = msg[2]

            if cmd == KVServerCmd.HANDSHAKE_SCHEDULER:
                self.handle_handshake_scheduler(client_id, cmd, payload)
            elif cmd == KVServerCmd.HANDSHAKE_WORKER:
                self.handle_handshake_worker(client_id, cmd, payload)
            elif cmd == KVServerCmd.HEARTBEAT:
                self.handle_heartbeat(client_id, cmd, payload)
            elif cmd == KVServerCmd.OFFLOAD_REQUEST:
                self.handle_offload_request(client_id, cmd, payload)
            elif cmd == KVServerCmd.ONLOAD_REQUEST:
                self.handle_onload_request(client_id, cmd, payload)
            elif cmd == KVServerCmd.LOOKUP_REQUEST:
                self.handle_lookup_request(client_id, cmd, payload)
            else:
                logger.warning("Unknown command from client %s: %s", client_id, cmd)

        self.process_tasks()

    def shutdown(self):
        """Shutdown the server and clean up resources"""
        logger.info("Shutting down KV Server...")

        if self.kv_interface is not None:
            self.kv_interface.close()

        # Unregister socket from poller
        if self.main_socket and self.poller:
            self.poller.unregister(self.main_socket)

        # Close the main socket
        if self.main_socket:
            self.main_socket.close()
            self.main_socket = None

        # Terminate the ZMQ context
        if self.context:
            self.context.term()
            self.context = None

        print("KV Server shutdown complete")


if __name__ == "__main__":
    config = KVServerConfig(host="localhost", port=54332)
    server = KVServer(config)
    print("Starting the server at", config.host, ":", config.port)
    try:
        while True:
            server.step()
    except KeyboardInterrupt:
        print("Received shutdown signal...")
    except Exception:
        logger.exception("Server error")
    finally:
        server.shutdown()
