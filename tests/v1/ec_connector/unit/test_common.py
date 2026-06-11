# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import uuid
from unittest.mock import MagicMock

import pytest

from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.consumer import (
    ECCPUConsumer,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.producer import (
    ECCPUProducer,
)
from vllm.distributed.ec_transfer.ec_connector.ec_shared_region import (
    AllocationError,
    ECSharedRegion,
)

_N = 8
_BS = 64


def _region() -> ECSharedRegion:
    return ECSharedRegion(
        instance_id=str(uuid.uuid4()), num_blocks=_N, block_size_bytes=_BS
    )


def _make_producer(region: ECSharedRegion) -> ECCPUProducer:
    return ECCPUProducer(
        region=region,
        engine=MagicMock(),
        agent_metadata=b"meta",
        mem_descriptor_bytes=b"desc",
        compat_hash="hash",
        hidden_dim=32,
        element_size=2,
        block_size_bytes=_BS,
        peer_host="localhost",
        peer_port=12345,
    )


def _make_consumer(region: ECSharedRegion) -> ECCPUConsumer:
    return ECCPUConsumer(
        region=region,
        transport=MagicMock(),
        engine=MagicMock(),
        local_xfer_handle=0,
        compat_hash="hash",
        hidden_dim=32,
        element_size=2,
        block_size_bytes=_BS,
    )


# ── ECCPUProducer._fifo_alloc ─────────────────────────────────────────────────


def test_producer_fifo_alloc_evicts_oldest_unpinned():
    r = _region()
    p = _make_producer(r)
    old = r.alloc(4)
    new = r.alloc(4)
    p._local_encodings["old"] = old
    p._local_encodings["new"] = new
    result = p._fifo_alloc(3)
    assert len(result) == 3
    assert "old" not in p._local_encodings
    assert "new" in p._local_encodings
    r.cleanup()


def test_producer_fifo_alloc_skips_pinned():
    r = _region()
    p = _make_producer(r)
    pinned = r.alloc(4)
    unpinned = r.alloc(4)
    r.pin(pinned)
    p._local_encodings["pinned"] = pinned
    p._local_encodings["unpinned"] = unpinned
    result = p._fifo_alloc(3)
    assert len(result) == 3
    assert "pinned" in p._local_encodings
    assert "unpinned" not in p._local_encodings
    r.unpin(pinned)
    r.cleanup()


def test_producer_fifo_alloc_raises_when_all_pinned():
    r = _region()
    p = _make_producer(r)
    all_idx = r.alloc(_N)
    r.pin(all_idx)
    p._local_encodings["only"] = all_idx
    with pytest.raises(AllocationError):
        p._fifo_alloc(1)
    r.unpin(all_idx)
    r.cleanup()


# ── ECCPUConsumer._fifo_alloc ─────────────────────────────────────────────────


def test_consumer_fifo_alloc_protects_pending_reload():
    r = _region()
    c = _make_consumer(r)
    full = r.alloc(_N)
    c._loaded["A"] = full
    c._pending_reload = {"A"}
    with pytest.raises(AllocationError):
        c._fifo_alloc(1)
    assert "A" in c._loaded
    r.cleanup()


def test_consumer_fifo_alloc_evicts_unprotected():
    r = _region()
    c = _make_consumer(r)
    cold = r.alloc(_N // 2)
    hot = r.alloc(_N // 2)
    c._loaded["cold"] = cold
    c._loaded["hot"] = hot
    c._pending_reload = {"hot"}
    result = c._fifo_alloc(1)
    assert len(result) == 1
    assert "hot" in c._loaded
    assert "cold" not in c._loaded
    r.cleanup()
