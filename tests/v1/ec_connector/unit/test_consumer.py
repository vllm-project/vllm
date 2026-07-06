# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
import uuid

import torch

from vllm.distributed.ec_transfer.ec_connector.cpu.common import ECRegionContext
from vllm.distributed.ec_transfer.ec_connector.cpu.ec_shared_region import (
    ECSharedRegion,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.consumer import (
    ECCPUConsumer,
)

_N = 8
_BS = 64


def _region() -> ECSharedRegion:
    return ECSharedRegion(
        instance_id=str(uuid.uuid4()), num_blocks=_N, block_size_bytes=_BS
    )


def _consumer(region, local_encodings, blocks) -> ECCPUConsumer:
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
        local_encodings=local_encodings,
        blocks=blocks,
        lock=threading.Lock(),
    )


class _Feature:
    def __init__(self, mm_hash):
        self.mm_hash = mm_hash
        self.identifier = mm_hash


class _Request:
    def __init__(self, features):
        self.mm_features = features


def test_has_cache_item():
    r = _region()
    c = _consumer(r, {"a": None}, {"a": [0]})
    assert c.has_cache_item("a") is True
    assert c.has_cache_item("missing") is False
    r.cleanup()


def test_ensure_cache_available_always_true():
    r = _region()
    c = _consumer(r, {}, {})
    assert c.ensure_cache_available(_Request([_Feature("x")]), 0) is True
    r.cleanup()


def test_update_state_after_alloc_pins_and_marks_reload():
    r = _region()
    blocks = r.alloc(2)
    c = _consumer(r, {"a": None}, {"a": blocks})
    c.update_state_after_alloc(_Request([_Feature("a")]), 0)
    assert "a" in c._pending_reload
    # Pinned: try_free must refuse.
    assert r.try_free(blocks) is False
    r.cleanup()


def test_build_loads_reserves_and_unpins():
    r = _region()
    blocks = r.alloc(2)
    c = _consumer(r, {"a": None}, {"a": blocks})
    c.update_state_after_alloc(_Request([_Feature("a")]), 0)
    loads = c.build_loads()
    assert loads == {"a": blocks}
    # Unpinned after build_loads: try_free now succeeds.
    assert r.try_free(blocks) is True
    assert c._pending_reload == set()
    r.cleanup()


def test_update_state_after_alloc_ignores_uncached():
    r = _region()
    c = _consumer(r, {}, {})
    c.update_state_after_alloc(_Request([_Feature("missing")]), 0)
    assert c._pending_reload == set()
    r.cleanup()


def test_repeated_alloc_pins_once_and_single_build_releases():
    # Two update_state_after_alloc calls for the same mm_hash in one step
    # must pin exactly once, so a single build_loads unpin fully releases
    # the blocks. Guards the `if mm_hash not in self._pending_reload` check.
    r = _region()
    blocks = r.alloc(2)
    c = _consumer(r, {"a": None}, {"a": blocks})
    c.update_state_after_alloc(_Request([_Feature("a")]), 0)
    c.update_state_after_alloc(_Request([_Feature("a")]), 0)
    assert c._pending_reload == {"a"}
    assert r.try_free(blocks) is False  # still pinned
    c.build_loads()
    # If the guard were missing, ref count would be 2 and this would fail.
    assert r.try_free(blocks) is True
    r.cleanup()


def test_build_loads_empty_when_no_pending():
    r = _region()
    c = _consumer(r, {"a": None}, {"a": r.alloc(1)})
    assert c.build_loads() == {}
    r.cleanup()


def test_build_loads_serves_only_pinned_hashes():
    # Two cached items, only one alloc'd this step: build_loads must return
    # just the pinned one, not every locally cached hash.
    r = _region()
    blocks_a = r.alloc(2)
    blocks_b = r.alloc(2)
    c = _consumer(r, {"a": None, "b": None}, {"a": blocks_a, "b": blocks_b})
    c.update_state_after_alloc(_Request([_Feature("a")]), 0)
    assert c.build_loads() == {"a": blocks_a}
    r.cleanup()


def test_shutdown_unpins_pending_blocks():
    # A step that pinned blocks but never reached build_loads (e.g. crash /
    # teardown) must not leak pins: shutdown releases them.
    r = _region()
    blocks = r.alloc(2)
    c = _consumer(r, {"a": None}, {"a": blocks})
    c.update_state_after_alloc(_Request([_Feature("a")]), 0)
    assert r.try_free(blocks) is False  # pinned
    c.shutdown()
    assert c._pending_reload == set()
    assert r.try_free(blocks) is True  # released
    r.cleanup()


def test_uses_identifier_when_mm_hash_is_none():
    # mm_hash falls back to feature.identifier when None.
    r = _region()
    blocks = r.alloc(1)
    c = _consumer(r, {"ident": None}, {"ident": blocks})
    feature = _Feature("ident")
    feature.mm_hash = None
    c.update_state_after_alloc(_Request([feature]), 0)
    assert c._pending_reload == {"ident"}
    r.cleanup()
