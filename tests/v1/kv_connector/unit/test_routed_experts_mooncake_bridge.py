# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the routed-experts Mooncake bridge (direct GPU<->Mooncake).

These exercise the bridge's addressing / dedup / fail-fast logic against a fake
``MooncakeDistributedStore`` and a real ``/dev/shm`` shared slot buffer, with no
Mooncake or GPU dependency. The end-to-end RDMA path is covered separately by
the real-Mooncake E2E.
"""

import uuid

import numpy as np
import pytest

from vllm.model_executor.layers.fused_moe.routed_experts_capture.mooncake_bridge import (  # noqa: E501
    RoutedExpertsMooncakeBridge,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capture.shared_region import (  # noqa: E501
    RoutedExpertsWorkerWriter,
    SharedRoutingRegion,
    shared_routing_mmap_path,
)

NUM_BLOCKS = 8
BLOCK_SIZE = 4
LAYERS = 2
TOP_K = 2
DTYPE = "uint8"


def _slot_shape() -> tuple[int, int, int]:
    return (NUM_BLOCKS * BLOCK_SIZE, LAYERS, TOP_K)


class _FakeStore:
    """In-memory stand-in for MooncakeDistributedStore.

    ``register_buffer`` records the base; put/get copy bytes between the
    registered process memory (addressed by absolute pointer) and a dict keyed
    by routing key, exactly like the RDMA pool would.
    """

    def __init__(self) -> None:
        self.pool: dict[str, bytes] = {}
        self.registered: list[tuple[int, int]] = []
        self.last_replicate_config = "UNSET"

    def register_buffer(self, base: int, nbytes: int) -> int:
        self.registered.append((base, nbytes))
        return 0

    def batch_is_exist(self, keys):
        return [1 if k in self.pool else 0 for k in keys]

    def batch_put_from_multi_buffers(self, keys, addrs, sizes, replicate_config=None):
        self.last_replicate_config = replicate_config
        for key, addr_list, size_list in zip(keys, addrs, sizes):
            buf = bytearray()
            for addr, size in zip(addr_list, size_list):
                buf += (
                    np.ctypeslib.as_array(
                        (np.ctypeslib.ctypes.c_uint8 * size).from_address(addr)
                    )
                ).tobytes()
            self.pool[key] = bytes(buf)
        return [0] * len(keys)

    def batch_get_into_multi_buffers(self, keys, addrs, sizes):
        res = []
        for key, addr_list, size_list in zip(keys, addrs, sizes):
            payload = self.pool.get(key)
            if payload is None:
                res.append(-1)
                continue
            off = 0
            for addr, size in zip(addr_list, size_list):
                dst = np.ctypeslib.as_array(
                    (np.ctypeslib.ctypes.c_uint8 * size).from_address(addr)
                )
                dst[:] = np.frombuffer(payload[off : off + size], dtype=np.uint8)
                off += size
            res.append(0)
        return res


@pytest.fixture
def region_and_writer():
    """Create the scheduler-side region, then a worker writer that attaches it."""
    instance_id = f"retest_{uuid.uuid4().hex}"
    dp_rank = 0
    region = SharedRoutingRegion(
        path=shared_routing_mmap_path(instance_id, dp_rank),
        shape=_slot_shape(),
        dtype=DTYPE,
    )
    writer = RoutedExpertsWorkerWriter(
        instance_id=instance_id,
        dp_rank=dp_rank,
        slot_shape=_slot_shape(),
        dtype=DTYPE,
        block_size=BLOCK_SIZE,
    )
    writer.attach()
    yield region, writer
    writer.close()
    region.close()


def _fill_region_block(region: SharedRoutingRegion, block_id: int, value: int):
    """Set every slot of one block's routing row to ``value``."""
    arr = region.array.reshape(NUM_BLOCKS, BLOCK_SIZE, LAYERS, TOP_K)
    arr[block_id] = value


def test_bridge_put_get_roundtrip(region_and_writer):
    region, writer = region_and_writer
    store = _FakeStore()
    bridge = RoutedExpertsMooncakeBridge(store, writer)
    bridge.register()
    # Whole slot buffer registered once.
    assert store.registered == [(writer.region_base_address(), writer.region_nbytes())]

    # Writer wrote routing for blocks 1 and 3; PUT them.
    _fill_region_block(region, 1, 11)
    _fill_region_block(region, 3, 33)
    bridge.put_routing(["k1", "k3"], [1, 3])
    assert set(store.pool) == {"re:k1", "re:k3"}

    # Clobber the buffer (simulate GPU block reuse), then GET back.
    region.array[:] = 0
    bridge.get_routing(["k1", "k3"], [1, 3])
    arr = region.array.reshape(NUM_BLOCKS, BLOCK_SIZE, LAYERS, TOP_K)
    assert (arr[1] == 11).all()
    assert (arr[3] == 33).all()
    # Untouched blocks stay zero.
    assert (arr[0] == 0).all()
    assert (arr[2] == 0).all()


def test_bridge_put_dedup(region_and_writer):
    region, writer = region_and_writer
    store = _FakeStore()
    bridge = RoutedExpertsMooncakeBridge(store, writer)
    _fill_region_block(region, 2, 7)
    bridge.put_routing(["k2"], [2])
    # Pre-seed pool with a sentinel for k2; a second put must skip it (dedup).
    store.pool["re:k2"] = b"SENTINEL"
    _fill_region_block(region, 2, 9)
    bridge.put_routing(["k2"], [2])
    assert store.pool["re:k2"] == b"SENTINEL"


def test_bridge_get_missing_raises(region_and_writer):
    region, writer = region_and_writer
    store = _FakeStore()
    bridge = RoutedExpertsMooncakeBridge(store, writer)
    bridge.register()
    # No PUT happened: GET must raise (KV-present-but-routing-absent invariant).
    with pytest.raises(RuntimeError, match="routing rows are absent"):
        bridge.get_routing(["missing"], [0])


def test_bridge_block_row_nbytes(region_and_writer):
    _, writer = region_and_writer
    # block_size * layers * top_k * itemsize(uint8=1)
    assert writer.block_row_nbytes() == BLOCK_SIZE * LAYERS * TOP_K


def test_bridge_put_passes_replicate_config(region_and_writer):
    """PUT forwards the connector's ReplicateConfig (same policy as KV PUT)."""
    region, writer = region_and_writer
    store = _FakeStore()
    sentinel = object()
    bridge = RoutedExpertsMooncakeBridge(store, writer, replicate_config=sentinel)
    _fill_region_block(region, 1, 5)
    bridge.put_routing(["k1"], [1])
    assert store.last_replicate_config is sentinel


def test_bridge_put_omits_replicate_config_when_none(region_and_writer):
    """No replicate_config -> 3-arg PUT (store default), matching a connector
    configured without one."""
    region, writer = region_and_writer
    store = _FakeStore()
    bridge = RoutedExpertsMooncakeBridge(store, writer)  # replicate_config=None
    _fill_region_block(region, 1, 5)
    bridge.put_routing(["k1"], [1])
    # 3-arg call path -> fake store's replicate_config param keeps its default
    # (None), never the explicit sentinel a 4-arg call would pass.
    assert store.last_replicate_config is None
