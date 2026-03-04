# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import msgspec
import pytest
import zmq

from vllm.config.kv_events import KVEventsConfig
from vllm.distributed.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
)
from vllm.distributed.kv_events import (
    KVEventBatch as InternalKVEventBatch,
)
from vllm.entrypoints.grpc_kv_events import GrpcKVEventBridge


class _FakeServicerContext:
    def __init__(self) -> None:
        self._done = False

    def done(self) -> bool:
        return self._done

    def cancelled(self) -> bool:
        return False


class _FakeSocket:
    def __init__(self, frames):
        self.frames = list(frames)
        self.closed = False
        self.connected = None
        self.subscribed_topic = None
        self.sent = []

    def connect(self, endpoint):
        self.connected = endpoint

    def setsockopt_string(self, opt, value):
        assert opt == zmq.SUBSCRIBE
        self.subscribed_topic = value

    def send(self, payload):
        self.sent.append(payload)

    def poll(self, _timeout):
        return 1 if self.frames else 0

    def recv_multipart(self):
        return self.frames.pop(0)

    def close(self, linger=0):
        self.closed = True


class _FakeZmqContext:
    def __init__(self, sub_socket: _FakeSocket, replay_socket: _FakeSocket):
        self.sub_socket = sub_socket
        self.replay_socket = replay_socket

    def socket(self, kind):
        if kind == zmq.SUB:
            return self.sub_socket
        if kind == zmq.REQ:
            return self.replay_socket
        raise AssertionError(f"Unexpected socket kind: {kind}")


def _encode_batch(batch: InternalKVEventBatch) -> bytes:
    return msgspec.msgpack.Encoder().encode(batch)


def test_bridge_enablement_backwards_compatibility():
    assert not GrpcKVEventBridge(None).enabled

    # Explicitly "null" keeps legacy no-streaming behavior even when
    # kv-cache events are enabled internally.
    disabled = KVEventsConfig(enable_kv_cache_events=True, publisher="null")
    assert not GrpcKVEventBridge(disabled).enabled

    enabled = KVEventsConfig(
        enable_kv_cache_events=True,
        publisher="zmq",
        endpoint="tcp://*:5557",
        replay_endpoint="tcp://*:5558",
        topic="kv-events",
    )
    assert GrpcKVEventBridge(enabled).enabled


def test_bridge_offsets_and_normalizes_tcp_wildcard_endpoints():
    cfg = KVEventsConfig(
        enable_kv_cache_events=True,
        publisher="zmq",
        endpoint="tcp://*:5557",
        replay_endpoint="tcp://*:5558",
        topic="kv-events",
    )
    bridge = GrpcKVEventBridge(cfg, data_parallel_rank=2)

    assert bridge.enabled
    assert bridge._config is not None
    assert bridge._config.endpoint == "tcp://127.0.0.1:5559"
    assert bridge._config.replay_endpoint == "tcp://127.0.0.1:5560"


def test_convert_batch_handles_int_and_bytes_hashes_backwards_compatibility():
    batch = InternalKVEventBatch(
        ts=123.0,
        data_parallel_rank=None,
        events=[
            BlockStored(
                block_hashes=[(1 << 63) + 5, 42],
                parent_block_hash=(1 << 63) + 3,
                token_ids=[10, 11, 12, 13],
                block_size=2,
                lora_id=7,
                medium="GPU",
                lora_name="adapter-a",
            ),
            BlockStored(
                block_hashes=[(b"\xff" * 32)],
                parent_block_hash=None,
                token_ids=[99],
                block_size=1,
                lora_id=None,
                medium="GPU",
                lora_name=None,
            ),
            BlockRemoved(block_hashes=[(b"\xff" * 32), 9], medium="GPU"),
            AllBlocksCleared(),
        ],
    )

    out = GrpcKVEventBridge._convert_batch(77, batch)

    assert out.sequence_number == 77
    assert len(out.events) == 4

    stored0 = out.events[0]
    assert stored0.HasField("stored")
    assert stored0.stored.parent_block_hash == -(1 << 63) + 3
    assert [b.block_hash for b in stored0.stored.blocks] == [-(1 << 63) + 5, 42]
    assert [list(b.token_ids) for b in stored0.stored.blocks] == [[10, 11], [12, 13]]
    assert [b.lora_id for b in stored0.stored.blocks] == [7, 7]

    stored1 = out.events[1]
    assert stored1.HasField("stored")
    assert stored1.stored.blocks[0].block_hash == -1

    removed = out.events[2]
    assert removed.HasField("removed")
    assert list(removed.removed.block_hashes) == [-1, 9]

    cleared = out.events[3]
    assert cleared.HasField("cleared")

    # dp_rank is optional and should stay unset when absent in source.
    assert not out.HasField("dp_rank")

    # Event IDs stay in uint64 range even for large sequence numbers.
    big_seq_out = GrpcKVEventBridge._convert_batch((1 << 40) + 9, batch)
    assert big_seq_out.events[0].event_id == (9 << 32)


@pytest.mark.asyncio
async def test_stream_replay_then_live_with_sequence_filtering(monkeypatch):
    replay_seq_old = (2).to_bytes(8, byteorder="big", signed=False)
    replay_seq_keep = (3).to_bytes(8, byteorder="big", signed=False)
    replay_end = (-1).to_bytes(8, byteorder="big", signed=True)

    replay_payload_old = _encode_batch(
        InternalKVEventBatch(
            ts=1.0,
            data_parallel_rank=None,
            events=[AllBlocksCleared()],
        )
    )
    replay_payload_keep = _encode_batch(
        InternalKVEventBatch(
            ts=2.0,
            data_parallel_rank=1,
            events=[AllBlocksCleared()],
        )
    )
    live_payload_dup = _encode_batch(
        InternalKVEventBatch(
            ts=3.0,
            data_parallel_rank=2,
            events=[AllBlocksCleared()],
        )
    )
    live_payload_keep = _encode_batch(
        InternalKVEventBatch(
            ts=4.0,
            data_parallel_rank=3,
            events=[AllBlocksCleared()],
        )
    )

    replay_socket = _FakeSocket(
        frames=[
            (replay_seq_old, replay_payload_old),
            (replay_seq_keep, replay_payload_keep),
            (replay_end, b""),
        ]
    )
    sub_socket = _FakeSocket(
        frames=[
            (b"kv-events", replay_seq_keep, live_payload_dup),
            (
                b"kv-events",
                (4).to_bytes(8, byteorder="big", signed=False),
                live_payload_keep,
            ),
        ]
    )
    fake_ctx = _FakeZmqContext(sub_socket=sub_socket, replay_socket=replay_socket)

    async def _inline_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(
        "vllm.entrypoints.grpc_kv_events.asyncio.to_thread", _inline_to_thread
    )
    monkeypatch.setattr(
        "vllm.entrypoints.grpc_kv_events.zmq.Context.instance", lambda: fake_ctx
    )

    cfg = KVEventsConfig(
        enable_kv_cache_events=True,
        publisher="zmq",
        endpoint="tcp://localhost:5557",
        replay_endpoint="tcp://localhost:5558",
        topic="kv-events",
    )
    bridge = GrpcKVEventBridge(cfg)

    context = _FakeServicerContext()
    got = []
    async for item in bridge.stream(start_sequence_number=3, context=context):
        got.append(item)
        if len(got) == 2:
            break

    assert replay_socket.sent == [(3).to_bytes(8, byteorder="big", signed=False)]
    assert [b.sequence_number for b in got] == [3, 4]
    assert [b.dp_rank for b in got] == [1, 3]
    assert sub_socket.subscribed_topic == "kv-events"
    assert replay_socket.closed
    assert sub_socket.closed


@pytest.mark.asyncio
async def test_stream_replay_handles_high_bit_sequence_numbers(monkeypatch):
    start_seq = (1 << 63) + 9
    replay_seq_keep = start_seq.to_bytes(8, byteorder="big", signed=False)
    replay_end = (-1).to_bytes(8, byteorder="big", signed=True)

    replay_payload = _encode_batch(
        InternalKVEventBatch(
            ts=10.0,
            data_parallel_rank=0,
            events=[AllBlocksCleared()],
        )
    )
    live_payload = _encode_batch(
        InternalKVEventBatch(
            ts=11.0,
            data_parallel_rank=0,
            events=[AllBlocksCleared()],
        )
    )

    replay_socket = _FakeSocket(
        frames=[
            (replay_seq_keep, replay_payload),
            (replay_end, b""),
        ]
    )
    sub_socket = _FakeSocket(
        frames=[
            (
                b"kv-events",
                (start_seq + 1).to_bytes(8, byteorder="big", signed=False),
                live_payload,
            ),
        ]
    )
    fake_ctx = _FakeZmqContext(sub_socket=sub_socket, replay_socket=replay_socket)

    async def _inline_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(
        "vllm.entrypoints.grpc_kv_events.asyncio.to_thread", _inline_to_thread
    )
    monkeypatch.setattr(
        "vllm.entrypoints.grpc_kv_events.zmq.Context.instance", lambda: fake_ctx
    )

    cfg = KVEventsConfig(
        enable_kv_cache_events=True,
        publisher="zmq",
        endpoint="tcp://localhost:5557",
        replay_endpoint="tcp://localhost:5558",
        topic="kv-events",
    )
    bridge = GrpcKVEventBridge(cfg)

    context = _FakeServicerContext()
    got = []
    async for item in bridge.stream(start_sequence_number=start_seq, context=context):
        got.append(item)
        if len(got) == 2:
            break

    assert [b.sequence_number for b in got] == [start_seq, start_seq + 1]
