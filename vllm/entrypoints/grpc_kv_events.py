# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for streaming KV cache events over gRPC."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import grpc
import msgspec.msgpack
import zmq
import zmq.asyncio

from vllm.config.kv_events import KVEventsConfig
from vllm.distributed.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
    ZmqEventPublisher,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

_INT64_MAX = (1 << 63) - 1
_INT64_MIN = -(1 << 63)
_UINT64_MOD = 1 << 64
_UINT64_MASK = _UINT64_MOD - 1
_SEQ_BYTES = 8
_END_SEQ = -1
_DEFAULT_POLL_TIMEOUT_MS = 200
_REPLAY_IDLE_TIMEOUT_S = 5.0
_EVENT_ID_SEQ_MASK = (1 << 32) - 1
_CACHE_LEVEL_BY_MEDIUM = {
    "GPU": 0,
    "CPU": 1,
}


def _normalize_connect_endpoint(endpoint: str) -> str:
    if endpoint.startswith("tcp://*"):
        return endpoint.replace("tcp://*", "tcp://127.0.0.1", 1)
    if endpoint.startswith("tcp://0.0.0.0"):
        return endpoint.replace("tcp://0.0.0.0", "tcp://127.0.0.1", 1)
    return endpoint


def _hash_to_int64(block_hash: int | bytes) -> int:
    if isinstance(block_hash, bytes):
        value = int.from_bytes(block_hash, byteorder="big", signed=False)
    else:
        value = int(block_hash)

    if value < 0:
        if value < _INT64_MIN:
            raise OverflowError(f"block hash {value} is smaller than int64 min")
        return value

    value &= _UINT64_MASK
    if value > _INT64_MAX:
        value -= _UINT64_MOD
    return value


class GrpcKvEventStreamer:
    """Bridge from vLLM KV events (msgpack/zmq) to gRPC protobuf batches."""

    def __init__(
        self,
        kv_events_config: KVEventsConfig | None,
        data_parallel_size: int,
        pb2_module: Any,
        poll_timeout_ms: int = _DEFAULT_POLL_TIMEOUT_MS,
    ) -> None:
        self._config = kv_events_config
        self._data_parallel_size = max(1, data_parallel_size)
        self._pb2 = pb2_module
        self._decoder = msgspec.msgpack.Decoder(type=KVEventBatch)
        self._poll_timeout_ms = poll_timeout_ms

    @property
    def enabled(self) -> bool:
        return (
            self._config is not None
            and self._config.enable_kv_cache_events
            and self._config.publisher == "zmq"
        )

    async def subscribe(
        self,
        request: Any,
        context: grpc.aio.ServicerContext,
    ) -> AsyncGenerator[Any, None]:
        if not self.enabled:
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "KV cache events are not enabled. Set --kv-events-config with "
                "enable_kv_cache_events=true and publisher=zmq.",
            )
            return

        assert self._config is not None
        start_sequence = max(0, int(getattr(request, "start_sequence_number", 0)))

        ctx = zmq.asyncio.Context.instance()
        sub_socket = ctx.socket(zmq.SUB)
        replay_sockets: list[zmq.asyncio.Socket] = []
        last_seq_by_dp: dict[int, int] = {}
        try:
            sub_socket.setsockopt(zmq.SUBSCRIBE, self._config.topic.encode("utf-8"))
            for endpoint in self._offset_endpoints(self._config.endpoint):
                sub_socket.connect(_normalize_connect_endpoint(endpoint))

            if self._config.replay_endpoint:
                start_seq_bytes = start_sequence.to_bytes(8, "big", signed=False)
                for endpoint in self._offset_endpoints(self._config.replay_endpoint):
                    replay = ctx.socket(zmq.REQ)
                    replay.connect(_normalize_connect_endpoint(endpoint))
                    replay_sockets.append(replay)
                    await replay.send(start_seq_bytes)

                for replay in replay_sockets:
                    async for message in self._drain_replay(
                        replay, context, start_sequence, last_seq_by_dp
                    ):
                        yield message

            topic_bytes = self._config.topic.encode("utf-8")
            while not self._is_cancelled(context):
                if not await sub_socket.poll(self._poll_timeout_ms):
                    continue

                frames = await sub_socket.recv_multipart()
                if len(frames) != 3:
                    logger.warning("Invalid KV event frame count=%d", len(frames))
                    continue

                recv_topic, seq_bytes, payload = frames
                if recv_topic != topic_bytes:
                    continue
                if len(seq_bytes) != _SEQ_BYTES:
                    logger.warning(
                        "Invalid KV event sequence length=%d (expected %d)",
                        len(seq_bytes),
                        _SEQ_BYTES,
                    )
                    continue

                sequence_number = int.from_bytes(seq_bytes, "big", signed=False)
                if sequence_number < start_sequence:
                    continue

                try:
                    decoded_batch = self._decoder.decode(payload)
                except Exception as exc:
                    logger.warning(
                        "Failed to decode KV event batch seq=%d: %s",
                        sequence_number,
                        exc,
                    )
                    continue
                if not self._should_emit(
                    decoded_batch, sequence_number, last_seq_by_dp
                ):
                    continue
                yield self._to_proto_batch(sequence_number, decoded_batch)

        except asyncio.CancelledError:
            return
        finally:
            sub_socket.close(linger=0)
            for replay in replay_sockets:
                replay.close(linger=0)

    async def _drain_replay(
        self,
        replay_socket: zmq.asyncio.Socket,
        context: grpc.aio.ServicerContext,
        start_sequence: int,
        last_seq_by_dp: dict[int, int],
    ) -> AsyncGenerator[Any, None]:
        loop = asyncio.get_running_loop()
        idle_deadline = loop.time() + _REPLAY_IDLE_TIMEOUT_S

        while not self._is_cancelled(context):
            if not await replay_socket.poll(self._poll_timeout_ms):
                if loop.time() >= idle_deadline:
                    logger.warning("Timed out waiting for KV replay response.")
                    break
                continue

            frames = await replay_socket.recv_multipart()
            idle_deadline = loop.time() + _REPLAY_IDLE_TIMEOUT_S
            if len(frames) != 2:
                logger.warning("Invalid KV replay frame count=%d", len(frames))
                continue

            seq_bytes, payload = frames
            if len(seq_bytes) != _SEQ_BYTES:
                logger.warning(
                    "Invalid KV replay sequence length=%d (expected %d)",
                    len(seq_bytes),
                    _SEQ_BYTES,
                )
                continue
            sequence_number = int.from_bytes(seq_bytes, "big", signed=True)
            if sequence_number == _END_SEQ:
                break

            if sequence_number < start_sequence:
                continue

            try:
                decoded_batch = self._decoder.decode(payload)
            except Exception as exc:
                logger.warning(
                    "Failed to decode KV replay batch seq=%d: %s",
                    sequence_number,
                    exc,
                )
                continue
            if not self._should_emit(decoded_batch, sequence_number, last_seq_by_dp):
                continue
            yield self._to_proto_batch(sequence_number, decoded_batch)

    def _offset_endpoints(self, endpoint: str) -> list[str]:
        endpoints: list[str] = []
        for dp_rank in range(self._data_parallel_size):
            maybe_endpoint = ZmqEventPublisher.offset_endpoint_port(endpoint, dp_rank)
            if maybe_endpoint is None:
                raise ValueError("KV event endpoint must not be None")
            endpoints.append(maybe_endpoint)
        return endpoints

    def _should_emit(
        self,
        decoded_batch: KVEventBatch,
        sequence_number: int,
        last_seq_by_dp: dict[int, int],
    ) -> bool:
        dp_rank = (
            int(decoded_batch.data_parallel_rank)
            if decoded_batch.data_parallel_rank is not None
            else -1
        )
        prev_seq = last_seq_by_dp.get(dp_rank, -1)
        if sequence_number <= prev_seq:
            return False
        last_seq_by_dp[dp_rank] = sequence_number
        return True

    def _to_proto_batch(self, sequence_number: int, batch: KVEventBatch) -> Any:
        proto_events = [
            self._to_proto_event(self._make_event_id(sequence_number, event_idx), event)
            for event_idx, event in enumerate(batch.events)
        ]
        message = self._pb2.KvEventBatch(
            sequence_number=sequence_number,
            timestamp=batch.ts,
            events=proto_events,
        )
        if batch.data_parallel_rank is not None:
            message.dp_rank = int(batch.data_parallel_rank)
        return message

    def _to_proto_event(self, event_id: int, event: Any) -> Any:
        if isinstance(event, BlockStored):
            return self._to_proto_block_stored(event_id, event)
        if isinstance(event, BlockRemoved):
            return self._to_proto_block_removed(event_id, event)
        if isinstance(event, AllBlocksCleared):
            return self._pb2.KvCacheEvent(
                event_id=event_id,
                cleared=self._pb2.KvCacheCleared(),
            )
        raise TypeError(f"Unsupported KV event type: {type(event)}")

    def _to_proto_block_stored(self, event_id: int, event: BlockStored) -> Any:
        cache_level = self._cache_level(event.medium)
        blocks = []
        token_stride = max(int(event.block_size), 0)
        for block_idx, block_hash in enumerate(event.block_hashes):
            start = block_idx * token_stride
            end = start + token_stride
            block_token_ids = (
                event.token_ids[start:end] if token_stride > 0 else event.token_ids
            )
            block = self._pb2.KvBlock(
                block_hash=_hash_to_int64(block_hash),
                token_ids=block_token_ids,
                block_size=event.block_size,
            )
            if event.lora_id is not None:
                block.lora_id = event.lora_id
            if cache_level is not None:
                block.cache_level = cache_level
            blocks.append(block)

        stored = self._pb2.KvBlocksStored(blocks=blocks)
        if event.parent_block_hash is not None:
            stored.parent_block_hash = _hash_to_int64(event.parent_block_hash)

        return self._pb2.KvCacheEvent(event_id=event_id, stored=stored)

    def _to_proto_block_removed(self, event_id: int, event: BlockRemoved) -> Any:
        removed = self._pb2.KvBlocksRemoved(
            block_hashes=[_hash_to_int64(bh) for bh in event.block_hashes]
        )
        cache_level = self._cache_level(event.medium)
        if cache_level is not None:
            removed.cache_level = cache_level
        return self._pb2.KvCacheEvent(event_id=event_id, removed=removed)

    @staticmethod
    def _cache_level(medium: str | None) -> int | None:
        if medium is None:
            return None
        return _CACHE_LEVEL_BY_MEDIUM.get(medium.upper())

    @staticmethod
    def _make_event_id(sequence_number: int, event_idx: int) -> int:
        # Pack 32 bits of sequence and 32 bits of per-batch event index.
        seq_hi = sequence_number & _EVENT_ID_SEQ_MASK
        idx_lo = event_idx & _EVENT_ID_SEQ_MASK
        return (seq_hi << 32) | idx_lo

    @staticmethod
    def _is_cancelled(context: grpc.aio.ServicerContext) -> bool:
        for method_name in ("done", "cancelled"):
            method = getattr(context, method_name, None)
            if callable(method):
                try:
                    if bool(method()):
                        return True
                except Exception:
                    continue
        return False
