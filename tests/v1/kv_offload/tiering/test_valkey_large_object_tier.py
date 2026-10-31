# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ValkeyLargeObjectTierManager against in-memory clients:
the RDMA data path, transport recovery, and the watchdog."""

import os
import time

import pytest
import torch

from tests.v1.kv_offload.tiering.pooled_tier_utils import (
    assert_store_then_load_roundtrip,
    drain,
    key,
    make_job,
    make_offloading_spec,
    page_aligned_zero_tensor,
)
from tests.v1.kv_offload.tiering.valkey_fakes import FakeLargeObjectClient
from vllm.v1.kv_offload.tiering.valkey_large_object.client import (
    LargeObjectClient,
    ValkeyTransportError,
)
from vllm.v1.kv_offload.tiering.valkey_large_object.manager import (
    ValkeyLargeObjectTierManager,
)

_NUM_CHUNKS = 4
_CHUNK_ELEMENTS = 1024
# Watchdog ticks every 10 ms.
_TIMEOUT_MS = 40


class FakeConnections:
    """Hands out a new client per connect, all sharing one server's data."""

    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.clients: list[FakeLargeObjectClient] = []
        self.refusals = 0

    def connect(self) -> FakeLargeObjectClient:
        if self.refusals:
            raise ConnectionError("refused")
        client = FakeLargeObjectClient(self.values)
        self.clients.append(client)
        return client

    def wait_for_clients(self, count: int, timeout: float = 1.0) -> None:
        deadline = time.monotonic() + timeout
        while len(self.clients) < count:
            assert time.monotonic() < deadline, f"never reached {count} clients"
            time.sleep(0.005)


def make_tier(
    connections: FakeConnections, tensor: torch.Tensor, **tier_kwargs
) -> ValkeyLargeObjectTierManager:
    class FakeClientTier(ValkeyLargeObjectTierManager):
        def _create_client(self) -> LargeObjectClient:
            client = connections.connect()
            self._register_regions(client)
            return client

    return FakeClientTier(
        offloading_spec=make_offloading_spec(),
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="valkey-large-object",
        startup_nodes="fake:6379",
        timeout_ms=_TIMEOUT_MS,
        n_read_threads=2,
        n_write_threads=2,
        **tier_kwargs,
    )


@pytest.fixture
def tensor() -> torch.Tensor:
    return page_aligned_zero_tensor(_NUM_CHUNKS, _CHUNK_ELEMENTS)


@pytest.fixture
def connections() -> FakeConnections:
    return FakeConnections()


@pytest.fixture
def tier(connections, tensor):
    tier = make_tier(connections, tensor)
    yield tier
    tier.shutdown()


def store_all(tier: ValkeyLargeObjectTierManager, job_id: int = 1) -> list:
    tier.submit_store(make_job(job_id, [key(i) for i in range(_NUM_CHUNKS)]))
    return drain(tier)


def load_all(tier: ValkeyLargeObjectTierManager, job_id: int) -> list:
    keys = [key(i) for i in range(_NUM_CHUNKS)]
    tier.submit_load(make_job(job_id, keys, is_promotion=True))
    return drain(tier)


def test_store_then_load_roundtrip_over_rdma(tier, tensor):
    assert_store_then_load_roundtrip(tier, tensor)


def test_ttl_expires_stored_chunks(connections, tensor):
    tier = make_tier(connections, tensor, ttl_seconds=60)
    try:
        store_all(tier)
        chunk_ttls = {
            name: ttl
            for name, ttl in connections.clients[0].ttls.items()
            if name.endswith(".bin")
        }
        assert list(chunk_ttls.values()) == [60] * _NUM_CHUNKS
    finally:
        tier.shutdown()


def test_transport_failure_replaces_client(tier, connections):
    store_all(tier)
    first = connections.clients[0]
    first.failures["rdma_get"].append(ValkeyTransportError("connection reset"))

    (result,) = load_all(tier, 2)
    assert not result.success
    connections.wait_for_clients(2)

    assert first.events == ["register_region", "close", "release_regions"]
    assert [r.success for r in load_all(tier, 3)] == [True]


def test_concurrent_transport_failures_reconnect_once(tier, connections):
    first = connections.clients[0]
    for _ in range(2):
        first.failures["rdma_set"].append(ValkeyTransportError("connection reset"))
    tier.submit_store(make_job(1, [key(0)]))
    tier.submit_store(make_job(2, [key(1)]))
    drain(tier)
    connections.wait_for_clients(2)

    # Several watchdog ticks: a second reconnect would have happened by now.
    time.sleep(5 * _TIMEOUT_MS / 4000)
    assert len(connections.clients) == 2


def test_watchdog_cancels_hung_transfer(tier, connections):
    store_all(tier)
    first = connections.clients[0]
    first.hang_transfers = True

    (result,) = load_all(tier, 2)
    assert not result.success
    # Teardown finishes before the replacement client is created.
    connections.wait_for_clients(2)
    # The region is released only after the cancelled transfer returned.
    assert first.events == [
        "register_region",
        "close",
        "transfer_cancelled",
        "release_regions",
    ]
    assert [r.success for r in load_all(tier, 3)] == [True]


def test_reconnect_retries_until_server_returns(tier, connections):
    store_all(tier)
    connections.refusals = 1
    connections.clients[0].failures["rdma_get"].append(
        ValkeyTransportError("connection reset")
    )
    load_all(tier, 2)

    # Fails fast without a transport rather than queueing behind the outage.
    (result,) = load_all(tier, 3)
    assert not result.success

    connections.refusals = 0
    connections.wait_for_clients(2)
    assert [r.success for r in load_all(tier, 4)] == [True]


@pytest.mark.skipif(
    "VLLM_TEST_VALKEY_LARGE_OBJECT_NODES" not in os.environ,
    reason="needs Valkey with valkey-large-object at "
    "VLLM_TEST_VALKEY_LARGE_OBJECT_NODES",
)
def test_roundtrip_against_valkey_large_object(tensor):
    glide_sync = pytest.importorskip("glide_sync")
    if not glide_sync.GlideClient.rdma_usable():
        pytest.skip("valkey-glide-sync has no RDMA support or no libfabric")
    tier = ValkeyLargeObjectTierManager(
        offloading_spec=make_offloading_spec(),
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="valkey-large-object",
        startup_nodes=os.environ["VLLM_TEST_VALKEY_LARGE_OBJECT_NODES"],
        rdma_provider="tcp",
        key_prefix=f"vllm-test-{os.getpid()}",
        ttl_seconds=60,
        n_read_threads=2,
        n_write_threads=2,
    )
    try:
        assert_store_then_load_roundtrip(tier, tensor)
    finally:
        tier.shutdown()


def test_shutdown_tears_down_transport(connections, tensor):
    tier = make_tier(connections, tensor)
    tier.shutdown()
    assert connections.clients[0].events == [
        "register_region",
        "close",
        "release_regions",
    ]


def test_regions_hold_whole_chunks(connections, tensor):
    chunk_bytes = tensor[0].numel() * tensor.element_size()
    # One byte short of three chunks rounds down to two per region.
    tier = make_tier(connections, tensor, rdma_region_bytes=3 * chunk_bytes - 1)
    try:
        regions = connections.clients[0].regions
        assert [len(region) for region in regions] == [2 * chunk_bytes] * 2
        assert_store_then_load_roundtrip(tier, tensor)
    finally:
        tier.shutdown()


def test_transfers_never_overlap_on_a_region(connections, tensor):
    chunk_bytes = tensor[0].numel() * tensor.element_size()
    tier = make_tier(connections, tensor, rdma_region_bytes=2 * chunk_bytes)
    try:
        client = connections.clients[0]
        client.transfer_seconds = 0.005
        # One job per chunk so the pool threads transfer concurrently.
        for chunk in range(_NUM_CHUNKS):
            tier.submit_store(make_job(chunk, [key(chunk)], chunk_ids=[chunk]))
        assert all(r.success for r in drain(tier))
        assert client.overlapping_transfers == 0
    finally:
        tier.shutdown()
