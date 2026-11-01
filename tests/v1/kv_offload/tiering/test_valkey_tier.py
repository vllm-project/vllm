# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ValkeyTierManager against an in-memory client.

The integration test at the bottom runs against a real Valkey when
VLLM_TEST_VALKEY_NODES (``host:port``) is set.
"""

import os

import pytest
import torch

from tests.v1.kv_offload.tiering.pooled_tier_utils import (
    CTX,
    assert_store_then_load_roundtrip,
    drain,
    key,
    lookup_and_wait,
    make_job,
    make_offloading_spec,
    page_aligned_zero_tensor,
)
from tests.v1.kv_offload.tiering.valkey_fakes import FakeValkeyClient
from vllm.v1.kv_offload.base import LookupResult
from vllm.v1.kv_offload.tiering.valkey.client import ValkeyClient
from vllm.v1.kv_offload.tiering.valkey.manager import ValkeyTierManager

_NUM_CHUNKS = 4
_CHUNK_ELEMENTS = 1024


def make_tier(
    client: ValkeyClient, tensor: torch.Tensor, **tier_kwargs
) -> ValkeyTierManager:
    class FakeClientTier(ValkeyTierManager):
        def _create_client(self) -> ValkeyClient:
            return client

    return FakeClientTier(
        offloading_spec=make_offloading_spec(),
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="valkey",
        startup_nodes="fake:6379",
        n_read_threads=2,
        n_write_threads=2,
        **tier_kwargs,
    )


@pytest.fixture
def tensor() -> torch.Tensor:
    return page_aligned_zero_tensor(_NUM_CHUNKS, _CHUNK_ELEMENTS)


@pytest.fixture
def client() -> FakeValkeyClient:
    return FakeValkeyClient()


@pytest.fixture
def tier(client, tensor):
    tier = make_tier(client, tensor)
    yield tier
    tier.shutdown()


def test_store_then_load_roundtrip(tier, tensor):
    tensor[:] = torch.rand_like(tensor)
    expected = tensor.clone()
    keys = [key(i) for i in range(_NUM_CHUNKS)]
    tier.submit_store(make_job(1, keys))
    assert [r.success for r in drain(tier)] == [True]

    tensor.zero_()
    # Load into reversed slots to show each key lands in its job's chunk id.
    slots = list(reversed(range(_NUM_CHUNKS)))
    tier.submit_load(make_job(2, keys, chunk_ids=slots, is_promotion=True))
    assert [r.success for r in drain(tier)] == [True]
    for index, slot in enumerate(slots):
        assert torch.equal(tensor[slot], expected[index])


def test_lookup_hits_only_stored_keys(tier):
    tier.submit_store(make_job(1, [key(0)]))
    drain(tier)
    assert lookup_and_wait(tier, [key(0), key(1)]) == [
        LookupResult.HIT,
        LookupResult.MISS,
    ]


def test_evicted_key_mid_load_keeps_prefix_and_marks_rest_miss(tier, client):
    keys = [key(i) for i in range(_NUM_CHUNKS)]
    tier.submit_store(make_job(1, keys))
    drain(tier)
    assert lookup_and_wait(tier, keys) == [LookupResult.HIT] * _NUM_CHUNKS

    # Valkey evicts chunk 2 between lookup and load.
    del client.values[tier.key_mapper.get_file_name(keys[2])]
    tier.submit_load(make_job(2, keys, is_promotion=True))
    (result,) = drain(tier)

    assert not result.success
    assert tuple(result.successful_keys) == (keys[0], keys[1])
    assert [tier.lookup(k, CTX) for k in keys] == [
        LookupResult.HIT,
        LookupResult.HIT,
        LookupResult.MISS,
        LookupResult.MISS,
    ]


def test_short_value_fails_chunk(tier, client):
    client.values[tier.key_mapper.get_file_name(key(0))] = b"short"
    tier.submit_load(make_job(1, [key(0)], is_promotion=True))
    (result,) = drain(tier)
    assert not result.success
    assert result.successful_keys is None


def test_load_error_fails_chunk(tier, client):
    tier.submit_store(make_job(1, [key(0)]))
    drain(tier)
    client.failures["get"].append(ConnectionError("reset"))
    tier.submit_load(make_job(2, [key(0)], is_promotion=True))
    (result,) = drain(tier)
    assert not result.success


def test_store_error_fails_job(tier, client):
    client.failures["set"].append(ConnectionError("reset"))
    tier.submit_store(make_job(1, [key(0)]))
    (result,) = drain(tier)
    assert not result.success


def test_ttl_applies_to_chunks_not_run_config(client, tensor):
    tier = make_tier(client, tensor, ttl_seconds=60)
    try:
        tier.submit_store(make_job(1, [key(0)]))
        drain(tier)
        chunk_name = tier.key_mapper.get_file_name(key(0))
        config_name = tier.key_mapper.get_config_file_path()
        assert client.ttls == {config_name: None, chunk_name: 60}
    finally:
        tier.shutdown()


def test_unreachable_server_raises_at_construction(tensor):
    client = FakeValkeyClient()
    client.failures["set"].append(ConnectionError("refused"))
    with pytest.raises(RuntimeError, match="cannot reach"):
        make_tier(client, tensor)


def test_shutdown_closes_client(client, tensor):
    tier = make_tier(client, tensor)
    tier.shutdown()
    assert client.closed


@pytest.mark.skipif(
    "VLLM_TEST_VALKEY_NODES" not in os.environ,
    reason="needs a Valkey server at VLLM_TEST_VALKEY_NODES",
)
def test_roundtrip_against_valkey(tensor):
    pytest.importorskip("glide_sync")
    tier = ValkeyTierManager(
        offloading_spec=make_offloading_spec(),
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="valkey",
        startup_nodes=os.environ["VLLM_TEST_VALKEY_NODES"],
        key_prefix=f"vllm-test-{os.getpid()}",
        ttl_seconds=60,
        n_read_threads=2,
        n_write_threads=2,
    )
    try:
        assert_store_then_load_roundtrip(tier, tensor)
    finally:
        tier.shutdown()
