# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import enum
import pickle
from typing import Union

import msgspec
import torch

from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, VllmConfig)
from vllm.kvserver.wrapper import CudaIPCWrapper


class KVServerCmd(enum.Enum):
    HANDSHAKE_SCHEDULER = enum.auto()
    HANDSHAKE_WORKER = enum.auto()
    HEARTBEAT = enum.auto()
    OFFLOAD_REQUEST = enum.auto()
    OFFLOAD_FINISHED = enum.auto()
    ONLOAD_REQUEST = enum.auto()
    ONLOAD_FINISHED = enum.auto()
    LOOKUP_REQUEST = enum.auto()
    LOOKUP_RESPONSE = enum.auto()


class KVServerMsgBase(msgspec.Struct, tag=True):
    pass


class KVServerHandshakeSchedulerMsg(KVServerMsgBase):
    engine_id: str
    s_model_config: bytes
    s_cache_config: bytes
    s_parallel_config: bytes
    s_scheduler_config: bytes

    @property
    def model_config(self) -> ModelConfig:
        return pickle.loads(self.s_model_config)

    @property
    def cache_config(self) -> CacheConfig:
        return pickle.loads(self.s_cache_config)

    @property
    def parallel_config(self) -> ParallelConfig:
        return pickle.loads(self.s_parallel_config)

    @property
    def scheduler_config(self) -> SchedulerConfig:
        return pickle.loads(self.s_scheduler_config)

    @staticmethod
    def from_payload(payload: bytes) -> "KVServerHandshakeSchedulerMsg":
        return msgspec.msgpack.decode(payload,
                                      type=KVServerHandshakeSchedulerMsg)


class KVServerHandshakeWorkerMsg(KVServerMsgBase):
    engine_id: str
    model_name: str
    rank: int
    world_size: int
    s_gpu_blocks: list[bytes]

    @staticmethod
    def from_payload(payload: bytes) -> "KVServerHandshakeWorkerMsg":
        return msgspec.msgpack.decode(payload, type=KVServerHandshakeWorkerMsg)


class KVServerOffloadRequest(KVServerMsgBase):
    engine_id: str
    request_id: str
    token_ids: list[int]
    block_ids: tuple[list[int], ...]
    skip_leading_tokens: int

    @staticmethod
    def from_payload(payload: bytes) -> "KVServerOffloadRequest":
        return msgspec.msgpack.decode(payload, type=KVServerOffloadRequest)


class KVServerOffloadFinished(KVServerMsgBase):
    engine_id: str
    request_id: str
    success: bool

    @staticmethod
    def from_payload(payload: bytes) -> "KVServerOffloadFinished":
        return msgspec.msgpack.decode(payload, type=KVServerOffloadFinished)


class KVServerLookupRequest(KVServerMsgBase):
    engine_id: str
    model_id: str
    request_id: str
    token_ids: list[int]

    @staticmethod
    def from_payload(payload: bytes) -> "KVServerLookupRequest":
        return msgspec.msgpack.decode(payload, type=KVServerLookupRequest)


class KVServerLookupResponse(KVServerMsgBase):
    engine_id: str
    request_id: str
    number_of_tokens: int

    @staticmethod
    def from_payload(payload: bytes) -> "KVServerLookupResponse":
        return msgspec.msgpack.decode(payload, type=KVServerLookupResponse)


KVServerMsg = Union[
    KVServerHandshakeSchedulerMsg,
    KVServerHandshakeWorkerMsg,
    KVServerOffloadRequest,
    KVServerOffloadFinished,
    KVServerLookupRequest,
    KVServerLookupResponse,
]

## HELPER FUNCTIONS


def decode_payload(cmd: KVServerCmd, payload: bytes) -> KVServerMsgBase:
    match cmd:
        case KVServerCmd.HANDSHAKE_SCHEDULER:
            return KVServerHandshakeSchedulerMsg.from_payload(payload)
        case KVServerCmd.HANDSHAKE_WORKER:
            return KVServerHandshakeWorkerMsg.from_payload(payload)
        case KVServerCmd.OFFLOAD_REQUEST:
            return KVServerOffloadRequest.from_payload(payload)
        case KVServerCmd.OFFLOAD_FINISHED:
            return KVServerOffloadFinished.from_payload(payload)
        case KVServerCmd.LOOKUP_REQUEST:
            return KVServerLookupRequest.from_payload(payload)
        case KVServerCmd.LOOKUP_RESPONSE:
            return KVServerLookupResponse.from_payload(payload)
        case _:
            raise ValueError(f"Unknown command for decoding: {cmd}")


def encode_cmd(cmd: KVServerCmd) -> bytes:
    return cmd.value.to_bytes(1, byteorder='big')


def decode_cmd(b: bytes) -> KVServerCmd:
    return KVServerCmd(int.from_bytes(b, byteorder='big'))


def send_scheduler_handshake(socket, vllm_config: VllmConfig):
    msg = KVServerHandshakeSchedulerMsg(
        engine_id="",
        s_model_config=pickle.dumps(vllm_config.model_config),
        s_cache_config=pickle.dumps(vllm_config.cache_config),
        s_parallel_config=pickle.dumps(vllm_config.parallel_config),
        s_scheduler_config=pickle.dumps(vllm_config.scheduler_config))
    payload = msgspec.msgpack.encode(msg)
    socket.send_multipart(
        [encode_cmd(KVServerCmd.HANDSHAKE_SCHEDULER), payload])


def send_worker_handshake(socket, rank: int, world_size: int,
                          gpu_kv_caches: list[torch.Tensor]):
    # Serialize the GPU blocks as bytes
    s_gpu_blocks = [
        CudaIPCWrapper(tensor).serialize() for tensor in gpu_kv_caches
    ]

    msg = KVServerHandshakeWorkerMsg(
        engine_id="",
        model_name="",
        rank=rank,
        world_size=world_size,
        s_gpu_blocks=s_gpu_blocks,
    )
    payload = msgspec.msgpack.encode(msg)
    socket.send_multipart([encode_cmd(KVServerCmd.HANDSHAKE_WORKER), payload])


def send_offload_request(socket,
                         request_id: str,
                         token_ids: list[int],
                         block_ids: tuple[list[int], ...],
                         skip_leading_tokens: int = 0):
    msg = KVServerOffloadRequest(
        engine_id="",
        request_id=request_id,
        token_ids=token_ids,
        block_ids=block_ids,
        skip_leading_tokens=skip_leading_tokens,
    )
    payload = msgspec.msgpack.encode(msg)
    socket.send_multipart([encode_cmd(KVServerCmd.OFFLOAD_REQUEST), payload])


def send_offload_response(socket, client_id, request_id: str, success: bool):
    msg = KVServerOffloadFinished(
        engine_id="",
        request_id=request_id,
        success=success,
    )
    payload = msgspec.msgpack.encode(msg)
    socket.send_multipart(
        [client_id,
         encode_cmd(KVServerCmd.OFFLOAD_FINISHED), payload])


def send_lookup_request(socket, engine_id: str, model_id: str, request_id: str,
                        token_ids: list[int]):
    msg = KVServerLookupRequest(
        engine_id=engine_id,
        model_id=model_id,
        request_id=request_id,
        token_ids=token_ids,
    )
    payload = msgspec.msgpack.encode(msg)
    socket.send_multipart([encode_cmd(KVServerCmd.LOOKUP_REQUEST), payload])


def send_lookup_response(socket, client_id, engine_id: str, request_id: str,
                         number_of_tokens: int):
    msg = KVServerLookupResponse(
        engine_id=engine_id,
        request_id=request_id,
        number_of_tokens=number_of_tokens,
    )
    payload = msgspec.msgpack.encode(msg)
    socket.send_multipart(
        [client_id,
         encode_cmd(KVServerCmd.LOOKUP_RESPONSE), payload])
