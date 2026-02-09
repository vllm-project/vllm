"""
Test to understand GC behavior with Request objects and gc.freeze().
Measures how often gen0/gen1/gen2 collections happen and whether
cyclic garbage from Request objects is collected.
"""
import gc
import os
import sys
import weakref

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from functools import partial

from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
)
from vllm.pooling_params import PoolingParams
from vllm.v1.request import Request


def test_gc_collection_frequency():
    """Check how often GC runs and whether it collects Request cycles."""

    init_none_hash(sha256)
    block_size = 16
    block_hasher = get_request_block_hasher(block_size, sha256)
    pooling_params = PoolingParams()

    # Simulate gc.freeze() like EngineCore does
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)
    gc.freeze()

    print(f"GC thresholds: {gc.get_threshold()}")
    print(f"After freeze - GC counts: {gc.get_count()}, frozen: {gc.get_freeze_count()}")

    weak_refs = []
    collected_counts = []

    # Track GC events
    gc_events = []
    original_callbacks = gc.callbacks[:]

    def gc_callback(phase, info):
        if phase == "stop":
            gc_events.append({
                "generation": info["generation"],
                "collected": info["collected"],
                "uncollectable": info["uncollectable"],
            })

    gc.callbacks.append(gc_callback)

    print(f"\nCreating 1000 Request objects with prefix caching (simulating serving)...")

    for i in range(1000):
        prompt_tokens = list(range(i * 100, i * 100 + 200))
        req = Request(
            request_id=f"req-{i}",
            prompt_token_ids=prompt_tokens,
            sampling_params=None,
            pooling_params=pooling_params,
            eos_token_id=None,
            block_hasher=block_hasher,
        )
        weak_refs.append(weakref.ref(req))

        # Simulate freeing the request (like scheduler does)
        del req

        if (i + 1) % 100 == 0:
            alive = sum(1 for ref in weak_refs if ref() is not None)
            gen0, gen1, gen2 = gc.get_count()
            gen0_events = sum(1 for e in gc_events if e["generation"] == 0)
            gen1_events = sum(1 for e in gc_events if e["generation"] == 1)
            gen2_events = sum(1 for e in gc_events if e["generation"] == 2)
            total_collected = sum(e["collected"] for e in gc_events)
            print(f"  After {i+1} requests: alive={alive}, "
                  f"gc_counts=({gen0},{gen1},{gen2}), "
                  f"collections: gen0={gen0_events}, gen1={gen1_events}, gen2={gen2_events}, "
                  f"total_collected={total_collected}")

    alive_final = sum(1 for ref in weak_refs if ref() is not None)
    print(f"\nFinal alive count: {alive_final}")
    print(f"Total GC events: {len(gc_events)}")
    for gen in range(3):
        events = [e for e in gc_events if e["generation"] == gen]
        total = sum(e["collected"] for e in events)
        print(f"  Gen {gen}: {len(events)} collections, {total} objects collected")

    gc.callbacks.remove(gc_callback)
    gc.unfreeze()
    gc.collect()


def test_gc_with_fewer_tracked_objects():
    """Simulate v0.11.1's optimization that creates fewer GC-tracked objects."""

    init_none_hash(sha256)
    block_size = 16
    block_hasher = get_request_block_hasher(block_size, sha256)
    pooling_params = PoolingParams()

    gc.collect()
    gc.freeze()

    print(f"\n{'='*60}")
    print("Test: Fewer GC-tracked objects (v0.11.1 behavior)")
    print(f"{'='*60}")

    weak_refs = []
    gc_events = []

    def gc_callback(phase, info):
        if phase == "stop":
            gc_events.append({
                "generation": info["generation"],
                "collected": info["collected"],
            })

    gc.callbacks.append(gc_callback)

    for i in range(1000):
        prompt_tokens = list(range(i * 100, i * 100 + 200))
        req = Request(
            request_id=f"req-{i}",
            prompt_token_ids=prompt_tokens,
            sampling_params=None,
            pooling_params=pooling_params,
            eos_token_id=None,
            block_hasher=block_hasher,
        )
        weak_refs.append(weakref.ref(req))
        del req

        # Simulate v0.11.1's optimization: use tuples instead of lists
        # This creates fewer GC-tracked objects per iteration
        _ = ()  # empty tuple (not tracked by GC)

    alive = sum(1 for ref in weak_refs if ref() is not None)
    gen0_events = sum(1 for e in gc_events if e["generation"] == 0)
    gen2_events = sum(1 for e in gc_events if e["generation"] == 2)
    print(f"After 1000 requests: alive={alive}")
    print(f"GC collections: gen0={gen0_events}, gen2={gen2_events}")

    gc.callbacks.remove(gc_callback)
    gc.unfreeze()
    gc.collect()


def test_gc_with_large_data():
    """Test with larger data (simulating mm_features) to see memory impact."""
    import resource

    init_none_hash(sha256)
    block_size = 16
    block_hasher = get_request_block_hasher(block_size, sha256)
    pooling_params = PoolingParams()

    gc.collect()
    gc.freeze()

    print(f"\n{'='*60}")
    print("Test: Request objects with large auxiliary data")
    print(f"{'='*60}")

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    weak_refs = []

    for batch in range(10):
        for i in range(100):
            prompt_tokens = list(range(200))
            req = Request(
                request_id=f"req-{batch}-{i}",
                prompt_token_ids=prompt_tokens,
                sampling_params=None,
                pooling_params=pooling_params,
                eos_token_id=None,
                block_hasher=block_hasher,
            )
            # Simulate large mm_features data
            req._large_data = bytearray(1024 * 1024)  # 1MB per request

            weak_refs.append(weakref.ref(req))
            del req

        alive = sum(1 for ref in weak_refs if ref() is not None)
        rss_now = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        gen0, gen1, gen2 = gc.get_count()
        print(f"  Batch {batch+1}: alive={alive}, "
              f"RSS={rss_now:.1f}MB (delta={rss_now-rss_before:+.1f}MB), "
              f"gc_counts=({gen0},{gen1},{gen2})")

    gc.unfreeze()
    gc.collect()

    alive_after = sum(1 for ref in weak_refs if ref() is not None)
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"\nAfter gc.collect(): alive={alive_after}, "
          f"RSS={rss_after:.1f}MB (delta={rss_after-rss_before:+.1f}MB)")


if __name__ == "__main__":
    test_gc_collection_frequency()
    test_gc_with_fewer_tracked_objects()
    test_gc_with_large_data()
