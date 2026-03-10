# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

pytest.importorskip("grpc")

from vllm.distributed.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
)
from vllm.entrypoints.grpc_kv_events import GrpcKvEventStreamer, _hash_to_int64
from vllm.grpc import vllm_engine_pb2


def test_hash_to_int64_normalizes_unsigned_values():
    assert _hash_to_int64(7) == 7
    assert _hash_to_int64((1 << 63) + 5) == -9223372036854775803
    assert _hash_to_int64(b"\xff" * 32) == -1


def test_to_proto_batch_maps_all_event_variants():
    streamer = GrpcKvEventStreamer(
        kv_events_config=None,
        data_parallel_size=1,
        pb2_module=vllm_engine_pb2,
    )

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
    streamer = GrpcKvEventStreamer(
        kv_events_config=None,
        data_parallel_size=1,
        pb2_module=vllm_engine_pb2,
    )

    removed = BlockRemoved(block_hashes=[1], medium="DISK")
    proto_event = streamer._to_proto_event(event_id=0, event=removed)
    assert proto_event.removed.block_hashes == [1]
    assert not proto_event.removed.HasField("cache_level")


def test_block_stored_token_ids_are_split_per_block():
    streamer = GrpcKvEventStreamer(
        kv_events_config=None,
        data_parallel_size=1,
        pb2_module=vllm_engine_pb2,
    )

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
