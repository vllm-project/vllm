"""
Test to verify that the fix for the Request reference cycle works.
Request objects should be freed immediately by reference counting
when prefix caching is enabled (no cyclic GC needed).
"""
import gc
import os
import sys
import weakref

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
)
from vllm.pooling_params import PoolingParams
from vllm.v1.request import Request


def test_request_gc_with_prefix_caching():
    """Test that Request objects are properly freed WITHOUT needing cyclic GC."""

    init_none_hash(sha256)
    block_size = 16
    block_hasher = get_request_block_hasher(block_size, sha256)

    weak_refs = []

    # Disable automatic GC to test reference counting alone
    gc.disable()

    pooling_params = PoolingParams()
    for i in range(100):
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

    alive_before = sum(1 for ref in weak_refs if ref() is not None)
    print(f"With prefix caching:")
    print(f"  Alive Request objects before gc.collect(): {alive_before}")

    collected = gc.collect()
    print(f"  GC collected {collected} objects")

    alive_after = sum(1 for ref in weak_refs if ref() is not None)
    print(f"  Alive Request objects after gc.collect(): {alive_after}")

    gc.enable()

    if alive_before > 0:
        print(f"\n  FAIL: {alive_before} Request objects were NOT freed "
              "without explicit gc.collect()!")
        print("  Reference cycle still exists.")
        return False
    else:
        print(f"\n  PASS: All Request objects freed immediately by "
              "reference counting (no cycle).")
        return True


def test_request_gc_without_prefix_caching():
    """Test that Request objects without prefix caching don't have cycles."""

    weak_refs = []
    gc.disable()

    pooling_params = PoolingParams()
    for i in range(100):
        prompt_tokens = list(range(i * 100, i * 100 + 200))
        req = Request(
            request_id=f"req-{i}",
            prompt_token_ids=prompt_tokens,
            sampling_params=None,
            pooling_params=pooling_params,
            eos_token_id=None,
            block_hasher=None,
        )
        weak_refs.append(weakref.ref(req))
        del req

    alive_before = sum(1 for ref in weak_refs if ref() is not None)
    print(f"\nWithout prefix caching:")
    print(f"  Alive Request objects before gc.collect(): {alive_before}")

    collected = gc.collect()
    alive_after = sum(1 for ref in weak_refs if ref() is not None)
    print(f"  GC collected {collected} objects")
    print(f"  Alive Request objects after gc.collect(): {alive_after}")

    gc.enable()

    if alive_before == 0:
        print(f"\n  PASS: All Request objects freed immediately.")
        return True
    else:
        print(f"\n  FAIL: {alive_before} Request objects leaked.")
        return False


def test_memory_growth_with_freeze():
    """Test that there's no memory growth even with gc.freeze()."""
    import resource

    init_none_hash(sha256)
    block_size = 16
    block_hasher = get_request_block_hasher(block_size, sha256)

    gc.collect()
    gc.freeze()

    print(f"\n{'='*60}")
    print("Test: Memory stability with gc.freeze() + prefix caching")
    print(f"{'='*60}")

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    weak_refs = []

    pooling_params = PoolingParams()
    for round_idx in range(10):
        for i in range(500):
            prompt_tokens = list(range(500))
            req = Request(
                request_id=f"req-{round_idx}-{i}",
                prompt_token_ids=prompt_tokens,
                sampling_params=None,
                pooling_params=pooling_params,
                eos_token_id=None,
                block_hasher=block_hasher,
            )
            weak_refs.append(weakref.ref(req))
            del req

        alive = sum(1 for ref in weak_refs if ref() is not None)
        rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"  Round {round_idx + 1}: alive={alive}, "
              f"RSS={rss_after:.1f} MB (delta={rss_after - rss_before:+.1f} MB)")

    gc.unfreeze()
    gc.collect()

    alive_final = sum(1 for ref in weak_refs if ref() is not None)
    print(f"\n  Final alive: {alive_final}")
    if alive_final == 0:
        print("  PASS: All objects freed.")
        return True
    else:
        print(f"  FAIL: {alive_final} objects still alive.")
        return False


if __name__ == "__main__":
    results = []
    results.append(("prefix caching GC", test_request_gc_with_prefix_caching()))
    results.append(("no prefix caching GC", test_request_gc_without_prefix_caching()))
    results.append(("memory growth with freeze", test_memory_growth_with_freeze()))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll tests passed! The reference cycle fix is working.")
    else:
        print("\nSome tests failed!")
    sys.exit(0 if all_pass else 1)
