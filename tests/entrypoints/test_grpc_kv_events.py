# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

pytest.importorskip("grpc")

import grpc

from vllm.config.kv_events import KVEventsConfig
from vllm.distributed.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
)
from vllm.entrypoints.grpc_kv_events import GrpcKvEventStreamer, _hash_to_int64
from vllm.grpc import vllm_engine_pb2


class _FakeContext:
    def __init__(self, cancelled: bool):
        self._cancelled = cancelled

    def cancelled(self) -> bool:
        return self._cancelled

    def done(self):
        # Regression guard: _is_cancelled must not rely on done(),
        # which may return a truthy future object.
        return object()


class _FakePeerContext(_FakeContext):
    def __init__(self, cancelled: bool, peer: str):
        super().__init__(cancelled=cancelled)
        self._peer = peer

    def peer(self) -> str:
        return self._peer


class _FakeAbortContext(_FakePeerContext):
    def __init__(self, cancelled: bool, peer: str):
        super().__init__(cancelled=cancelled, peer=peer)
        self.aborted: tuple[grpc.StatusCode, str] | None = None

    async def abort(self, code: grpc.StatusCode, details: str):
        self.aborted = (code, details)


class _FakeStreamContext(_FakeAbortContext):
    def __init__(self, cancelled: bool, peer: str):
        super().__init__(cancelled=cancelled, peer=peer)
        self.initial_metadata_sent = False

    async def send_initial_metadata(self, _metadata):
        self.initial_metadata_sent = True


def _make_streamer(
    kv_events_config: KVEventsConfig | None = None,
) -> GrpcKvEventStreamer:
    return GrpcKvEventStreamer(
        kv_events_config=kv_events_config,
        data_parallel_size=1,
        pb2_module=vllm_engine_pb2,
    )


def _make_enabled_kv_events_config(
    topic: str = "",
    allow_remote_subscribe: bool = False,
) -> KVEventsConfig:
    return KVEventsConfig(
        enable_kv_cache_events=True,
        publisher="zmq",
        topic=topic,
        allow_remote_subscribe=allow_remote_subscribe,
    )


def _make_request(
    start_sequence_number: int = 0,
) -> vllm_engine_pb2.SubscribeKvEventsRequest:
    return vllm_engine_pb2.SubscribeKvEventsRequest(
        start_sequence_number=start_sequence_number
    )


def test_hash_to_int64_normalizes_unsigned_values():
    assert _hash_to_int64(7) == 7
    assert _hash_to_int64((1 << 63) + 5) == -9223372036854775803
    assert _hash_to_int64(b"\xff" * 32) == -1


def test_to_proto_batch_maps_all_event_variants():
    streamer = _make_streamer()

    stored = BlockStored(
        block_hashes=[1, 2],
        parent_block_hash=3,
        token_ids=[10, 11, 12],
        block_size=16,
        lora_id=5,
        medium="CPU",
        lora_name=None,
    )
    removed = BlockRemoved(block_hashes=[4, 5], medium="GPU")
    cleared = AllBlocksCleared()
    batch = KVEventBatch(
        ts=123.45,
        events=[stored, removed, cleared],
        data_parallel_rank=2,
    )

    proto_batch = streamer._to_proto_batch(sequence_number=7, batch=batch)

    assert proto_batch.sequence_number == 7
    assert proto_batch.timestamp == 123.45
    assert proto_batch.dp_rank == 2
    assert len(proto_batch.events) == 3

    proto_stored = proto_batch.events[0]
    assert proto_stored.event_id == (7 << 32)
    assert proto_stored.HasField("stored")
    assert proto_stored.stored.parent_block_hash == 3
    assert len(proto_stored.stored.blocks) == 2
    assert proto_stored.stored.blocks[0].block_hash == 1
    assert proto_stored.stored.blocks[0].token_ids == [10, 11, 12]
    assert proto_stored.stored.blocks[0].block_size == 16
    assert proto_stored.stored.blocks[0].lora_id == 5
    assert proto_stored.stored.blocks[0].cache_level == 1

    proto_removed = proto_batch.events[1]
    assert proto_removed.event_id == (7 << 32) + 1
    assert proto_removed.HasField("removed")
    assert proto_removed.removed.block_hashes == [4, 5]
    assert proto_removed.removed.cache_level == 0

    proto_cleared = proto_batch.events[2]
    assert proto_cleared.event_id == (7 << 32) + 2
    assert proto_cleared.HasField("cleared")


def test_unknown_medium_does_not_set_cache_level():
    streamer = _make_streamer()

    removed = BlockRemoved(block_hashes=[1], medium="DISK")
    proto_event = streamer._to_proto_event(event_id=0, event=removed)
    assert proto_event.removed.block_hashes == [1]
    assert not proto_event.removed.HasField("cache_level")


def test_block_stored_token_ids_are_split_per_block():
    streamer = _make_streamer()

    stored = BlockStored(
        block_hashes=[101, 202],
        parent_block_hash=None,
        token_ids=[1, 2, 3, 4],
        block_size=2,
        lora_id=None,
        medium="GPU",
        lora_name=None,
    )
    proto_event = streamer._to_proto_event(event_id=0, event=stored)

    assert proto_event.stored.blocks[0].token_ids == [1, 2]
    assert proto_event.stored.blocks[1].token_ids == [3, 4]


@pytest.mark.parametrize("cancelled", [False, True])
def test_is_cancelled_uses_cancelled_method(cancelled: bool):
    context = _FakeContext(cancelled=cancelled)
    assert GrpcKvEventStreamer._is_cancelled(context) is cancelled


@pytest.mark.parametrize(
    ("peer", "is_local"),
    [
        ("unix:/tmp/vllm.sock", True),
        ("ipv4:127.0.0.1:50051", True),
        ("ipv4:10.1.2.3:50051", False),
        ("ipv6:[::1]:50051", True),
        ("ipv6:[2001:db8::1]:50051", False),
        ("unknown:anything", False),
        ("", False),
    ],
)
def test_is_local_peer(peer: str, is_local: bool):
    assert GrpcKvEventStreamer._is_local_peer(peer) is is_local


def test_extract_ipv6_host_only_strips_bracketed_port():
    assert GrpcKvEventStreamer._extract_ipv6_host("[::1]:50051") == "::1"
    assert GrpcKvEventStreamer._extract_ipv6_host("::1") == "::1"
    assert GrpcKvEventStreamer._extract_ipv6_host("2001:db8::1") == "2001:db8::1"
    # Unbracketed inputs are treated as full IPv6 host text.
    assert GrpcKvEventStreamer._extract_ipv6_host("::1:50051") == "::1:50051"


def test_is_subscriber_allowed_respects_allow_remote_subscribe():
    streamer_default = _make_streamer(kv_events_config=_make_enabled_kv_events_config())
    remote_context = _FakePeerContext(cancelled=False, peer="ipv4:10.1.2.3:50051")
    assert not streamer_default._is_subscriber_allowed(remote_context)

    streamer_remote_allowed = _make_streamer(
        kv_events_config=_make_enabled_kv_events_config(
            allow_remote_subscribe=True,
        )
    )
    assert streamer_remote_allowed._is_subscriber_allowed(remote_context)


@pytest.mark.asyncio
async def test_subscribe_denies_remote_peer_by_default():
    streamer = _make_streamer(
        kv_events_config=_make_enabled_kv_events_config(
            topic="kv-events",
        )
    )
    context = _FakeAbortContext(cancelled=False, peer="ipv4:10.1.2.3:50051")
    request = _make_request()

    responses = [batch async for batch in streamer.subscribe(request, context)]

    assert responses == []
    assert context.aborted is not None
    code, details = context.aborted
    assert code == grpc.StatusCode.PERMISSION_DENIED
    assert "allow_remote_subscribe=true" in details


@pytest.mark.asyncio
async def test_subscribe_returns_unimplemented_when_disabled():
    streamer = _make_streamer()
    context = _FakeAbortContext(cancelled=False, peer="")
    request = _make_request()

    responses = [batch async for batch in streamer.subscribe(request, context)]

    assert responses == []
    assert context.aborted is not None
    code, details = context.aborted
    assert code == grpc.StatusCode.UNIMPLEMENTED
    assert "KV cache events are not enabled" in details


@pytest.mark.asyncio
async def test_subscribe_sends_initial_metadata_before_stream():
    streamer = _make_streamer(
        kv_events_config=_make_enabled_kv_events_config(
            topic="kv-events",
        )
    )
    context = _FakeStreamContext(cancelled=True, peer="ipv4:127.0.0.1:50051")
    request = _make_request()

    responses = [batch async for batch in streamer.subscribe(request, context)]

    assert responses == []
    assert context.aborted is None
    assert context.initial_metadata_sent
