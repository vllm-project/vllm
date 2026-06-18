# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
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
    import torch

    from vllm.distributed.ec_transfer.ec_connector.cpu.common import ECRegionContext

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
        compat_hash="hash",
        addr=("localhost", 12345),
        local_encodings={},
        blocks={},
        lock=threading.Lock(),
    )


def _make_consumer(region: ECSharedRegion) -> ECCPUConsumer:
    import torch

    from vllm.distributed.ec_transfer.ec_connector.cpu.common import ECRegionContext

    ctx = ECRegionContext(
        region=region,
        dtype=torch.float16,
        hidden_dim=32,
        element_size=2,
        block_size_bytes=_BS,
        num_blocks=_N,
    )
    return ECCPUConsumer(
        memory_context=ctx,
        transport=MagicMock(),
        data=MagicMock(),
        compat_hash="hash",
        local_encodings={},
        blocks={},
        lock=threading.Lock(),
    )


# ── ECCPUProducer._fifo_alloc ─────────────────────────────────────────────────


def test_producer_fifo_alloc_evicts_oldest_unpinned():
    r = _region()
    p = _make_producer(r)
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


def test_producer_fifo_alloc_skips_pinned():
    r = _region()
    p = _make_producer(r)
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


def test_producer_fifo_alloc_raises_when_all_pinned():
    r = _region()
    p = _make_producer(r)
    all_idx = r.alloc(_N)
    r.pin(all_idx)
    p._local_encodings["only"] = None
    p._blocks["only"] = all_idx
    with pytest.raises(AllocationError):
        p._fifo_alloc(1)
    r.unpin(all_idx)
    r.cleanup()


# ── ECCPUConsumer._fifo_alloc ─────────────────────────────────────────────────


def test_consumer_fifo_alloc_protects_pending_reload():
    r = _region()
    c = _make_consumer(r)
    full = r.alloc(_N)
    c._local_encodings["A"] = None
    c._blocks["A"] = full
    r.pin(full)
    c._pending_reload = {"A"}
    with pytest.raises(AllocationError):
        c._fifo_alloc(1)
    assert "A" in c._local_encodings
    r.unpin(full)
    r.cleanup()


def test_consumer_fifo_alloc_evicts_unprotected():
    r = _region()
    c = _make_consumer(r)
    cold = r.alloc(_N // 2)
    hot = r.alloc(_N // 2)
    c._local_encodings["cold"] = None
    c._local_encodings["hot"] = None
    c._blocks["cold"] = cold
    c._blocks["hot"] = hot
    r.pin(hot)
    c._pending_reload = {"hot"}
    result = c._fifo_alloc(1)
    assert len(result) == 1
    assert "hot" in c._local_encodings
    assert "cold" not in c._local_encodings
    r.unpin(hot)
    r.cleanup()
