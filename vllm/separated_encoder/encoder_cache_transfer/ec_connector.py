# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pickle
import socket
import struct
from concurrent.futures import ThreadPoolExecutor

from vllm.separated_encoder.utils import dict_to_pos_info, mm_pos_info_to_dict


class ECConnector:

    def __init__(self,
                 transfer_workers_num: int,
                 broker_host: str = '127.0.0.1',
                 broker_port: int = 65432):
        self.broker_address = (broker_host, broker_port)
        self.executor = ThreadPoolExecutor(max_workers=transfer_workers_num)

    def _send_pickled_data(self, pickled_data, tag):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(self.broker_address)
                s.sendall(b'S')
                s.sendall(struct.pack('>I', tag))
                s.sendall(struct.pack('>I', len(pickled_data)))
                s.sendall(pickled_data)
        except Exception as e:
            raise ConnectionError(
                f"Encoder Cache Connector send data failure: {e}") from e

    def _send_alloc_notif(self, request_id, input_id):
        try:
            pickled_data = pickle.dumps((request_id, input_id), protocol=5)
            self._send_pickled_data(pickled_data, tag=0)
        except Exception as e:
            assert 0, f"Encoder Cache Connector send data failure : {e}"

    def _send_inject_notif(self, request_id, input_id):
        try:
            pickled_data = pickle.dumps((request_id, input_id), protocol=5)
            self._send_pickled_data(pickled_data, tag=1)
        except Exception as e:
            assert 0, f"Encoder Cache Connector send data failure : {e}"

    def _send_encoder_cache_metas(self, request_id, input_id,
                                  encoder_cache_size):
        try:
            pickled_data = pickle.dumps(
                (request_id, input_id, encoder_cache_size), protocol=5)
            self._send_pickled_data(pickled_data, tag=2)
        except Exception as e:
            assert 0, f"Encoder Cache Connector send data failure : {e}"

    def _send_encoder_cache(self, request_id, input_id, pos_info,
                            encoder_cache):
        try:
            pickled_data = pickle.dumps(
                (request_id, input_id, mm_pos_info_to_dict(pos_info),
                 encoder_cache.cpu().float().numpy()),
                protocol=5)
            self._send_pickled_data(pickled_data, tag=3)
        except Exception as e:
            assert 0, f"Encoder Cache Connector send data failure : {e}"

    def create_send_alloc_notif_req(self, request_id, input_id):
        self.executor.submit(self._send_alloc_notif, request_id, input_id)

    def create_send_inject_notif_req(self, request_id, input_id):
        self.executor.submit(self._send_inject_notif, request_id, input_id)

    def create_send_encoder_cache_metas_req(self, request_id, input_id,
                                            encoder_cache_size):
        self.executor.submit(self._send_encoder_cache_metas, request_id,
                             input_id, encoder_cache_size)

    def create_send_encoder_cache_req(self, request_id, input_id, pos_info,
                                      encoder_cache):
        self.executor.submit(self._send_encoder_cache, request_id, input_id,
                             pos_info, encoder_cache)

    def recv_pickled_data(self, tag):
        try:
            pickled_data = b''
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(self.broker_address)
                s.sendall(b'R')
                s.sendall(struct.pack('>I', tag))
                data_size_bytes = s.recv(4)
                assert (data_size_bytes is not None)
                data_size = struct.unpack('>I', data_size_bytes)[0]
                while len(pickled_data) < data_size:
                    packet = s.recv(data_size - len(pickled_data))
                    if not packet:
                        break
                    pickled_data += packet
            return pickled_data
        except Exception as e:
            assert 0, f"Encoder Cache Connector receive data failure : {e}"

    def _recv_alloc_notif(self, callback):
        try:
            pickled_data = self.recv_pickled_data(tag=0)
            req_id, input_id = pickle.loads(pickled_data)
            callback(req_id, input_id)
        except Exception as e:
            assert 0, f"Encoder Cache Connector receive data failure : {e}"

    def _recv_inject_notif(self, callback):
        try:
            pickled_data = self.recv_pickled_data(tag=1)
            req_id, input_id = pickle.loads(pickled_data)
            callback(req_id, input_id)
        except Exception as e:
            assert 0, f"Encoder Cache Connector receive data failure : {e}"

    def _recv_encoder_cache_metas(self, callback):
        try:
            pickled_data = self.recv_pickled_data(tag=2)
            req_id, input_id, encoder_cache_size = pickle.loads(pickled_data)
            callback(req_id, input_id, encoder_cache_size)
        except Exception as e:
            assert 0, f"Encoder Cache Connector receive data failure : {e}"

    def _recv_encoder_cache(self, callback):
        try:
            pickled_data = self.recv_pickled_data(tag=3)
            req_id, input_id, pos_info_dict, encoder_cache_numpy = \
                pickle.loads(pickled_data)
            pos_info = dict_to_pos_info(pos_info_dict)
            callback(req_id, input_id, pos_info, encoder_cache_numpy)
        except Exception as e:
            assert 0, f"Encoder Cache Connector receive data failure : {e}"

    def create_recv_alloc_notif_req(self, callback):
        self.executor.submit(self._recv_alloc_notif, callback)

    def create_recv_inject_notif_req(self, callback):
        self.executor.submit(self._recv_inject_notif, callback)

    def create_recv_encoder_cache_metas_req(self, callback):
        self.executor.submit(self._recv_encoder_cache_metas, callback)

    def create_recv_encoder_cache_req(self, callback):
        self.executor.submit(self._recv_encoder_cache, callback)
