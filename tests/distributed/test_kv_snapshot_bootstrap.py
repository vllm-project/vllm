# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import Counter

import msgspec
import pytest

from vllm.distributed.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
)
from vllm.distributed.kv_events_snapshot import KVCacheSnapshot

pytestmark = pytest.mark.skip_global_cleanup


def store(hashes, parent=None, medium="GPU", tokens=None):
    return BlockStored(
        block_hashes=hashes,
        parent_block_hash=parent,
        token_ids=tokens
        if tokens is not None
        else [h * 4 + j for h in hashes for j in range(4)],
        block_size=4 if medium == "GPU" else 0,
        lora_id=None,
        lora_name=None,
        medium=medium,
    )


def remove(hashes, medium="GPU"):
    return BlockRemoved(block_hashes=hashes, medium=medium)


def wire(events):
    """Decode exported events as a snapshot consumer receives them."""
    batch = msgspec.msgpack.encode(KVEventBatch(ts=0, events=list(events)))
    return msgspec.msgpack.decode(batch, type=KVEventBatch).events


def consume(events):
    """Check reconstruction dependencies as well as resident references."""
    known = set()
    live: Counter = Counter()
    for e in events:
        if isinstance(e, BlockStored):
            if e.token_ids:
                assert e.parent_block_hash is None or e.parent_block_hash in known
                known.update(e.block_hashes)
            else:
                assert set(e.block_hashes) <= known
            live.update((e.medium, e.group_idx, h) for h in e.block_hashes)
        elif isinstance(e, BlockRemoved):
            live.subtract((e.medium, e.group_idx, h) for h in e.block_hashes)
    return +live


def test_evicted_parent_metadata_is_retained():
    snap = KVCacheSnapshot()
    history = [store([1]), store([2], parent=1), remove([1])]
    snap.apply(history)
    assert consume(wire(snap.export())) == consume(history)


def test_cpu_only_block_retains_gpu_metadata():
    snap = KVCacheSnapshot()
    history = [store([1]), store([1], medium="CPU", tokens=[]), remove([1])]
    snap.apply(history)
    assert consume(wire(snap.export())) == consume(history)


def test_restated_block_with_unknown_parent_is_unavailable():
    # A strict consumer cannot resolve the parent either.
    snap = KVCacheSnapshot()
    update = store([1], parent=99)
    update.medium = "CPU"
    snap.apply([store([1]), remove([1])])
    snap.apply([update])
    assert snap.tainted == 1 and 99 not in snap._records


def test_duplicate_references_survive_one_remove():
    snap = KVCacheSnapshot()
    history = [store([1]), store([1]), remove([1])]
    snap.apply(history)
    assert consume(wire(snap.export())) == consume(history)


def test_sparse_store_is_unavailable():
    # Block records need one token span per hash; consumers cannot index
    # sparse spans either, so the recorder reports itself unavailable.
    snap = KVCacheSnapshot()
    snap.apply([store([1, 3], tokens=list(range(12)))])
    assert snap.tainted == 2


def test_reset_keeps_cpu_dependencies():
    snap = KVCacheSnapshot()
    snap.apply([store([1]), store([1], medium="CPU", tokens=[])])
    snap.apply([AllBlocksCleared()])
    assert consume(wire(snap.export())) == Counter({("CPU", None, 1): 1})


def test_offload_bytes_resolve_integer_gpu_hash(monkeypatch):
    monkeypatch.setenv("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", "1")
    snap = KVCacheSnapshot()
    gpu = store([1])
    cpu = store([(1).to_bytes(32, "big")], medium="CPU", tokens=[])
    snap.apply([gpu, cpu, remove([1])])
    exported = wire(snap.export())
    assert exported[0].block_hashes == [1] and exported[0].token_ids == gpu.token_ids
    assert consume(exported) == Counter({("CPU", None, 1): 1})


@pytest.mark.parametrize("field", ["ownership", "locality"])
def test_residency_scopes_fail_closed(field):
    # Snapshot consumers reject these scopes, so a snapshot cannot carry them.
    event = store([1])
    setattr(event, field, "REMOTE")
    with pytest.raises(ValueError, match="locality or ownership"):
        KVCacheSnapshot().apply([event])
