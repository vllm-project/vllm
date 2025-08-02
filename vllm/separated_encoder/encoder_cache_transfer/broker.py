# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Broker class main goals are:
    i.  Data transfer between instances
    ii. Load balance the request between different 
        prefill(or prefill+decode) instances.

The current implementation is a placeholder to make the encoder separation
code runnable, simple, and free of additional dependencies. Since the encoder
cache is much smaller than the KV-cache, the broker's data transfers do not
create a high workload. Therefore, a Redis-based broker should be sufficient
for this purpose.
"""

import logging
import queue
import socket
import struct
import threading

from vllm.logger import init_logger

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s')

logger = init_logger(__name__)


class Broker:

    BUFFER_SIZE = 65536

    def __init__(self, port):
        self.transfer_queues = [queue.Queue() for _ in range(4)]
        self.host = '0.0.0.0'
        self.port = port

    def handle_client(self, conn, addr):
        try:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            command = conn.recv(1)
            if command == b'S':
                tag_bytes = conn.recv(4)
                assert (tag_bytes is not None)
                tag = struct.unpack('>I', tag_bytes)[0]
                data_wait_log = (f"[BROKER] Received"
                                 f"send command with tag {tag}")
                logger.info(data_wait_log)
                data_size_bytes = conn.recv(4)
                assert (data_size_bytes is not None)
                data_size = struct.unpack('>I', data_size_bytes)[0]
                pickled_data = bytearray(data_size)
                view = memoryview(pickled_data)
                bytes_received = 0
                while bytes_received < data_size:
                    chunk_size = min(Broker.BUFFER_SIZE,
                                     data_size - bytes_received)
                    bytes_read = \
                        conn.recv_into(
                            view[bytes_received:bytes_received+chunk_size]
                        )
                    if bytes_read == 0:
                        break
                    bytes_received += bytes_read

                self.transfer_queues[tag].put(pickled_data)
                recv_log = ("[BROKER] Queued data with tag"
                            f"{tag}. Current queue[{tag}] size:"
                            f"{self.transfer_queues[tag].qsize()}")
                logger.info(recv_log)
            elif command == b'R':
                tag_bytes = conn.recv(4)
                assert (tag_bytes is not None)
                tag = struct.unpack('>I', tag_bytes)[0]
                data_wait_log = ("[BROKER] Received recv command with "
                                 f"tag {tag}. Waiting for data...")
                logger.info(data_wait_log)
                pickled_data = self.transfer_queues[tag].get()  # blocking call
                conn.sendall(struct.pack('>I', len(pickled_data)))
                offset = 0
                data_view = memoryview(pickled_data)
                while offset < len(pickled_data):
                    bytes_sent = conn.send(data_view[offset:])
                    offset += bytes_sent
                send_log = (f"[BROKER] Sent data to {addr} with"
                            f"tag {tag}. Current queue[{tag}] size:"
                            f"{self.transfer_queues[tag].qsize()}")
                logger.info(send_log)

        except Exception as e:
            error_log = f"[BROKER]: Error during handling {addr}: {e}."
            logger.info(error_log)
        finally:
            conn.close()


def start_server():
    broker = Broker(65432)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        s.bind((broker.host, broker.port))
        s.listen(128)
        strt_log = f"[BROKER] Server listening on {broker.host}:{broker.port}"
        logger.info(strt_log)
        while True:
            conn, addr = s.accept()
            thread = threading.Thread(target=broker.handle_client,
                                      args=(conn, addr))
            thread.daemon = True
            thread.start()


if __name__ == "__main__":
    start_server()