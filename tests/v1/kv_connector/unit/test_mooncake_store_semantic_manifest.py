# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest
from vllm.v1.core.kv_cache_utils import BlockHash

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    ChunkedTokenDatabase,
    KeyMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.semantic_manifest import (  # noqa: E501
    SemanticCommitProtocol,
    SemanticRegionManifest,
)


class FakeStore:
    def __init__(self) -> None:
        self.objects: set[str] = set()
        self.put_result = 0
        self.short_exists_result = False

    def batch_is_exist(self, keys: list[str]) -> list[int]:
        results = [1 if key in self.objects else 0 for key in keys]
        return results[:-1] if self.short_exists_result else results

    def batch_put_from_multi_buffers(
        self,
        keys: list[str],
        addrs: list[list[int]],
        sizes: list[list[int]],
        replicate_config: Any,
    ) -> list[int]:
        assert len(keys) == len(addrs) == len(sizes) == 1
        if self.put_result >= 0:
            self.objects.update(keys)
        return [self.put_result]


def _manifest(
    *,
    pp_rank: int = 0,
    group_id: int = 0,
    stride_scale: int = 1,
    schema: str = "target-state-v1",
) -> SemanticRegionManifest:
    database = ChunkedTokenDatabase(
        KeyMetadata(
            "test-model",
            tp_rank=2,
            pcp_rank=0,
            dcp_rank=2,
            pp_rank=pp_rank,
            group_id=group_id,
            store_namespace="@semantic:v1",
        ),
        block_size=16,
    )
    regions = tuple(
        ChunkedTokenDatabase.from_semantic_region(
            database,
            region_id=region_id,
            base_addr=base_addr,
            block_stride=content_len * stride_scale,
            content_len=content_len,
        )
        for region_id, base_addr, content_len in (
            ("layer.4:target_conv", 0x1000, 256),
            ("layer.4:base_recurrent", 0x2000, 512),
        )
    )
    return SemanticRegionManifest(schema, regions)


def test_manifest_identity_ignores_endpoint_layout():
    producer = _manifest(pp_rank=0, group_id=1, stride_scale=1)
    consumer = _manifest(pp_rank=1, group_id=5, stride_scale=2)
    block_hash = BlockHash(b"hash")

    assert producer.wire_fingerprint == consumer.wire_fingerprint
    assert producer.data_keys(block_hash) == consumer.data_keys(block_hash)
    assert producer.commit_key(block_hash) == consumer.commit_key(block_hash)


def test_publish_requires_every_region_then_verifies_commit():
    store = FakeStore()
    manifest = _manifest()
    block_hash = BlockHash(b"hash")
    protocol = SemanticCommitProtocol(store, replicate_config=object())

    data_keys = manifest.data_keys(block_hash)
    store.objects.add(data_keys[0])
    assert not protocol.publish(manifest, block_hash, marker_addr=0x3000)
    assert manifest.commit_key(block_hash) not in store.objects

    store.objects.add(data_keys[1])
    assert protocol.publish(manifest, block_hash, marker_addr=0x3000)
    assert protocol.is_loadable(manifest, block_hash)


def test_load_rejects_region_evicted_after_commit():
    store = FakeStore()
    manifest = _manifest()
    block_hash = BlockHash(b"hash")
    protocol = SemanticCommitProtocol(store, replicate_config=object())
    store.objects.update(manifest.data_keys(block_hash))
    assert protocol.publish(manifest, block_hash, marker_addr=0x3000)

    store.objects.remove(manifest.data_keys(block_hash)[0])
    assert manifest.commit_key(block_hash) in store.objects
    assert not protocol.is_loadable(manifest, block_hash)


def test_protocol_fails_closed_on_malformed_store_results():
    store = FakeStore()
    manifest = _manifest()
    block_hash = BlockHash(b"hash")
    protocol = SemanticCommitProtocol(store, replicate_config=object())
    store.objects.update(manifest.data_keys(block_hash))
    store.short_exists_result = True

    assert not protocol.publish(manifest, block_hash, marker_addr=0x3000)
    assert not protocol.is_loadable(manifest, block_hash)


def test_protocol_fails_closed_when_commit_put_fails():
    store = FakeStore()
    manifest = _manifest()
    block_hash = BlockHash(b"hash")
    protocol = SemanticCommitProtocol(store, replicate_config=object())
    store.objects.update(manifest.data_keys(block_hash))
    store.put_result = -1

    assert not protocol.publish(manifest, block_hash, marker_addr=0x3000)


def test_schema_change_uses_a_different_commit_key():
    block_hash = BlockHash(b"hash")
    assert _manifest(schema="v1").commit_key(block_hash) != _manifest(
        schema="v2"
    ).commit_key(block_hash)


def test_manifest_rejects_duplicate_region_ids():
    manifest = _manifest()
    duplicate = (manifest.region_databases[0],) * 2
    with pytest.raises(ValueError, match="unique"):
        SemanticRegionManifest(manifest.schema, duplicate)
