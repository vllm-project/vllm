# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import Counter

import pytest

from vllm.distributed.kv_events import AllBlocksCleared, BlockRemoved, BlockStored
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
    assert consume(list(snap.export())) == consume(history)


def test_cpu_only_block_retains_gpu_metadata():
    snap = KVCacheSnapshot()
    history = [store([1]), store([1], medium="CPU", tokens=[]), remove([1])]
    snap.apply(history)
    assert consume(list(snap.export())) == consume(history)


def test_unknown_parent_tier_update_exports_a_closed_chain():
    snap = KVCacheSnapshot()
    source = store([1])
    update = store([1], parent=99)
    update.medium = "CPU"
    snap.apply([source, remove([1]), update])
    assert consume(list(snap.export())) == Counter({("CPU", None, 1): 1})


def test_duplicate_references_survive_one_remove():
    snap = KVCacheSnapshot()
    history = [store([1]), store([1]), remove([1])]
    snap.apply(history)
    assert consume(list(snap.export())) == consume(history)


def test_sparse_source_event_is_not_split():
    snap = KVCacheSnapshot()
    original = store([1, 3], tokens=list(range(12)))
    history = [original, remove([1])]
    snap.apply(history)
    exported = list(snap.export())
    assert exported[0] == original
    assert consume(exported) == consume(history)


def test_reset_keeps_cpu_dependencies():
    snap = KVCacheSnapshot()
    snap.apply([store([1]), store([1], medium="CPU", tokens=[])])
    snap.apply([AllBlocksCleared()])
    assert consume(list(snap.export())) == Counter({("CPU", None, 1): 1})


def test_offload_bytes_resolve_integer_gpu_hash(monkeypatch):
    monkeypatch.setenv("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", "1")
    snap = KVCacheSnapshot()
    gpu = store([1])
    cpu = store([(1).to_bytes(32, "big")], medium="CPU", tokens=[])
    snap.apply([gpu, cpu, remove([1])])
    exported = list(snap.export())
    assert exported[:2] == [gpu, cpu]


@pytest.mark.parametrize(
    "field,values", [("ownership", (None, "disk")), ("locality", ("LOCAL", "REMOTE"))]
)
def test_snapshot_preserves_independent_residency_scopes(field, values):
    snap = KVCacheSnapshot()
    first, second = store([1]), store([1, 2])
    setattr(first, field, values[0])
    setattr(second, field, values[1])
    removed = remove([1])
    setattr(removed, field, values[1])
    snap.apply([first, second, removed])
    residency: Counter = Counter()
    for event in snap.export():
        for h in event.block_hashes:
            key = (event.medium, event.group_idx, event.locality, event.ownership, h)
            residency[key] += 1 if isinstance(event, BlockStored) else -1
    assert +residency == Counter(
        {
            ("GPU", None, first.locality, first.ownership, 1): 1,
            ("GPU", None, second.locality, second.ownership, 2): 1,
        }
    )
