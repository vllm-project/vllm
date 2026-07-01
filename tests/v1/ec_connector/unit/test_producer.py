# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
import uuid

import torch

from vllm.distributed.ec_transfer.ec_connector.cpu.common import ECRegionContext
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.producer import (
    ECCPUProducer,
)
from vllm.distributed.ec_transfer.ec_connector.ec_shared_region import (
    ECSharedRegion,
)

_N = 8
_BS = 64


def _region() -> ECSharedRegion:
    return ECSharedRegion(
        instance_id=str(uuid.uuid4()), num_blocks=_N, block_size_bytes=_BS
    )


def _producer(region: ECSharedRegion) -> ECCPUProducer:
    ctx = ECRegionContext(
        region=region,
        dtype=torch.float16,
        hidden_dim=32,
        element_size=2,
        block_size_bytes=_BS,
        num_blocks=_N,
    )
    return ECCPUProducer(
        memory_context=ctx,
        local_encodings={},
        blocks={},
        lock=threading.Lock(),
    )


class _Pos:
    def __init__(self, offset, length):
        self.offset = offset
        self.length = length


class _Feature:
    def __init__(self, mm_hash, offset, length):
        self.mm_hash = mm_hash
        self.identifier = mm_hash
        self.mm_position = _Pos(offset, length)


class _Request:
    def __init__(self, features):
        self.mm_features = features


def test_fifo_alloc_evicts_oldest_unpinned():
    r = _region()
    p = _producer(r)
    old = r.alloc(4)
    new = r.alloc(4)
    p._local_encodings["old"] = None
    p._blocks["old"] = old
    p._local_encodings["new"] = None
    p._blocks["new"] = new
    result = p._fifo_alloc(3)
    assert len(result) == 3
    assert "old" not in p._local_encodings
    assert "new" in p._local_encodings
    r.cleanup()


def test_fifo_alloc_skips_pinned():
    r = _region()
    p = _producer(r)
    pinned = r.alloc(4)
    unpinned = r.alloc(4)
    r.pin(pinned)
    p._local_encodings["pinned"] = None
    p._blocks["pinned"] = pinned
    p._local_encodings["unpinned"] = None
    p._blocks["unpinned"] = unpinned
    result = p._fifo_alloc(3)
    assert len(result) == 3
    assert "pinned" in p._local_encodings
    assert "unpinned" not in p._local_encodings
    r.unpin(pinned)
    r.cleanup()


def test_build_saves_allocates_and_promotes():
    r = _region()
    p = _producer(r)
    # 1 token * hidden_dim(32) * element_size(2) = 64 bytes = 1 block.
    req = _Request([_Feature("h1", offset=0, length=1)])
    p.update_state_after_alloc(req, 0)
    saves = p.build_saves()
    assert "h1" in saves and len(saves["h1"]) == 1
    assert p._blocks["h1"] == saves["h1"]
    # Not promoted to local cache until the NEXT build_saves.
    assert "h1" not in p._local_encodings
    saves2 = p.build_saves()
    assert saves2 == {}
    assert "h1" in p._local_encodings
    r.cleanup()


def test_ensure_cache_available_defers_when_region_full():
    r = _region()
    p = _producer(r)
    # Fill the region and pin so eviction cannot reclaim.
    full = r.alloc(_N)
    r.pin(full)
    p._local_encodings["x"] = None
    p._blocks["x"] = full
    req = _Request([_Feature("new", offset=0, length=1)])
    assert p.ensure_cache_available(req, num_computed_tokens=0) is False
    r.unpin(full)
    r.cleanup()


def test_ensure_cache_available_ok_when_room():
    r = _region()
    p = _producer(r)
    req = _Request([_Feature("new", offset=0, length=1)])
    assert p.ensure_cache_available(req, num_computed_tokens=0) is True
    r.cleanup()


def test_ensure_cache_available_skips_already_computed_feature():
    r = _region()
    p = _producer(r)
    # offset+length <= num_computed_tokens -> skipped, always True.
    req = _Request([_Feature("done", offset=0, length=2)])
    assert p.ensure_cache_available(req, num_computed_tokens=2) is True
    assert p._pending_new_encodings == {}
    r.cleanup()
