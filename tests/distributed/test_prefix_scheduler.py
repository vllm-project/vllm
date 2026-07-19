# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from types import SimpleNamespace

import pytest

from vllm.config.kv_events import KVEventsConfig
from vllm.distributed.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    EventPublisherFactory,
    HttpPrefixCacheEventUploader,
    KVEventBatch,
    NullEventPublisher,
    PrefixCacheEventUploaderFactory,
)
from vllm.distributed.prefix_scheduler import (
    GlobalPrefixScheduler,
    NodePrefixCacheState,
    PrefixCacheSnapshot,
)
from vllm.entrypoints.openai.prefix_routing import (
    PrefixRoutingProxy,
    _parse_prefix_routing_config,
)
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    ExternalBlockHash,
    maybe_convert_block_hash,
)


def _hash(value: int) -> BlockHash:
    return BlockHash(bytes([value]) * 32)


def _external_hash(value: BlockHash) -> ExternalBlockHash:
    return maybe_convert_block_hash(value)


def test_node_prefix_cache_state_applies_events_and_matches_longest_prefix():
    state = NodePrefixCacheState(node_id="node-a", hash_block_size=16)
    block_hashes = [_hash(1), _hash(2), _hash(3)]

    state.apply_events(
        [
            BlockStored(
                block_hashes=[
                    _external_hash(block_hashes[0]),
                    _external_hash(block_hashes[1]),
                ],
                parent_block_hash=None,
                token_ids=[],
                block_size=16,
                lora_id=None,
                medium="GPU",
                lora_name=None,
                group_idx=0,
            )
        ]
    )

    assert state.longest_prefix_match(block_hashes, prompt_num_tokens=64) == 32

    state.apply_events(
        [
            BlockRemoved(
                block_hashes=[_external_hash(block_hashes[1])],
                medium="GPU",
                group_idx=0,
            )
        ]
    )
    assert state.longest_prefix_match(block_hashes, prompt_num_tokens=64) == 16

    state.apply_events([AllBlocksCleared()])
    assert state.longest_prefix_match(block_hashes, prompt_num_tokens=64) == 0


def test_global_prefix_scheduler_routes_to_longest_match():
    scheduler = GlobalPrefixScheduler()
    block_hashes = [_hash(1), _hash(2), _hash(3)]

    scheduler.update_snapshot(
        PrefixCacheSnapshot(
            node_id="node-a",
            data_parallel_rank=0,
            hash_block_size=16,
            group_block_sizes={0: 16},
            group_hashes={0: {_external_hash(block_hashes[0])}},
        )
    )
    scheduler.update_snapshot(
        PrefixCacheSnapshot(
            node_id="node-b",
            data_parallel_rank=1,
            hash_block_size=16,
            group_block_sizes={0: 16},
            group_hashes={
                0: {_external_hash(block_hashes[0]), _external_hash(block_hashes[1])}
            },
        )
    )

    decision = scheduler.choose_node(block_hashes, prompt_num_tokens=64)

    assert decision is not None
    assert decision.node_id == "node-b"
    assert decision.data_parallel_rank == 1
    assert decision.matched_tokens == 32


def test_global_prefix_scheduler_round_robins_ties():
    scheduler = GlobalPrefixScheduler()
    block_hashes = [_hash(1)]
    for node_id in ("node-a", "node-b"):
        scheduler.update_snapshot(
            PrefixCacheSnapshot(
                node_id=node_id,
                data_parallel_rank=None,
                hash_block_size=16,
                group_block_sizes={0: 16},
                group_hashes={0: {_external_hash(block_hashes[0])}},
            )
        )

    decisions = [
        scheduler.choose_node(block_hashes, prompt_num_tokens=32).node_id
        for _ in range(4)
    ]

    assert decisions == ["node-a", "node-b", "node-a", "node-b"]


def test_global_prefix_scheduler_applies_event_batch_rank():
    scheduler = GlobalPrefixScheduler()
    block_hashes = [_hash(1)]
    scheduler.register_node("node-a", hash_block_size=16)

    scheduler.apply_event_batch(
        "node-a",
        KVEventBatch(
            ts=1.0,
            data_parallel_rank=3,
            events=[
                BlockStored(
                    block_hashes=[_external_hash(block_hashes[0])],
                    parent_block_hash=None,
                    token_ids=[],
                    block_size=16,
                    lora_id=None,
                    medium="GPU",
                    lora_name=None,
                    group_idx=0,
                )
            ],
        ),
    )

    decision = scheduler.choose_node(block_hashes, prompt_num_tokens=32)

    assert decision is not None
    assert decision.data_parallel_rank == 3
    assert decision.matched_tokens == 16


def test_node_prefix_cache_state_matches_larger_group_block_size():
    block_hashes = [_hash(1), _hash(2), _hash(3), _hash(4)]
    state = NodePrefixCacheState(
        node_id="node-a",
        hash_block_size=16,
        group_block_sizes={0: 32},
        group_hashes={0: {_external_hash(block_hashes[1])}},
    )

    assert state.longest_prefix_match(block_hashes, prompt_num_tokens=80) == 32


def test_node_prefix_cache_state_requires_a_hit_in_every_cache_group():
    block_hashes = [_hash(1), _hash(2)]
    state = NodePrefixCacheState(
        node_id="node-a",
        hash_block_size=16,
        group_block_sizes={0: 16, 1: 16},
        group_hashes={0: {_external_hash(block_hashes[0])}},
    )

    assert state.longest_prefix_match(block_hashes, prompt_num_tokens=32) == 0


def test_prefix_cache_upload_is_independent_from_event_publisher():
    config = KVEventsConfig(
        enable_kv_cache_events=False,
        publisher="null",
        prefix_cache_upload_endpoint="http://127.0.0.1:9/prefix_routing",
    )

    publisher = EventPublisherFactory.create(config)
    uploader = PrefixCacheEventUploaderFactory.create(config)

    try:
        assert isinstance(publisher, NullEventPublisher)
        assert isinstance(uploader, HttpPrefixCacheEventUploader)
    finally:
        uploader.shutdown()


def test_prefix_routing_renders_completion_engine_inputs():
    class Renderer:
        async def render_completion(self, request):
            return [
                {"prompt_token_ids": [1, 2], "cache_salt": "salt"},
                {"prompt_token_ids": [3]},
            ]

    proxy = object.__new__(PrefixRoutingProxy)
    proxy.app_state = SimpleNamespace(online_renderer=Renderer())

    rendered = asyncio.run(
        proxy._render_request(
            "/v1/completions",
            {"model": "test-model", "prompt": ["first", "second"]},
        )
    )

    assert rendered == [([1, 2], "salt"), ([3], None)]


def test_prefix_routing_renders_chat_engine_inputs():
    class Renderer:
        async def render_chat(self, request):
            return [], [{"prompt_token_ids": [4, 5], "cache_salt": "chat-salt"}]

    proxy = object.__new__(PrefixRoutingProxy)
    proxy.app_state = SimpleNamespace(online_renderer=Renderer())

    rendered = asyncio.run(
        proxy._render_request(
            "/v1/chat/completions",
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
    )

    assert rendered == [([4, 5], "chat-salt")]


@pytest.mark.parametrize(
    "config, error",
    [
        (
            {
                "nodes": [
                    {"id": "node-a", "url": "local"},
                    {"id": "node-a", "url": "local"},
                ]
            },
            "duplicate prefix routing node id",
        ),
        (
            {"nodes": [{"id": "node-a", "url": "ftp://node-a"}]},
            r"requires an HTTP\(S\) URL",
        ),
        (
            {"nodes": [{"id": "node-a", "url": "local"}], "hash_block_size": 0},
            "hash_block_size must be a positive integer",
        ),
        (
            {"nodes": [{"id": "node-a", "url": "local"}], "request_timeout": 0},
            "request_timeout must be positive",
        ),
    ],
)
def test_prefix_routing_config_rejects_invalid_values(config, error):
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(hash_block_size=16, block_size=16)
    )

    with pytest.raises(ValueError, match=error):
        _parse_prefix_routing_config(config, vllm_config)
