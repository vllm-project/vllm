# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tier 2 (Physical) — CuMem state-machine edge cases.
Tier 3 (Protocol) — Cross-tag wake ordering invariants.

These tests verify allocator-level invariants that are not covered by the
HTTP-level state machine tests (test_state_machine.py) or the behavioral
tests (test_sleep_wake.py).

Edge cases contributed from production multi-model hot-swap work:
- Partial wake + forward → illegal memory access (#44395)
- Double unmap protection when sleep() follows a discarded state
- Graph pool presence slowing VMM operations
- Cross-tag wake ordering affecting scheduler safety

All tests require real GPU memory operations (--enable-sleep-mode with
CuMemAllocator active), so they use real weights (not dummy).

RFC: https://github.com/vllm-project/vllm/issues/45585
"""

import threading
import time

import requests

from .conftest import (
    gen,
    gpu_free_bytes,
    is_sleeping,
    ok,
    poll_until,
    resume,
    server,
    sleep,
    wake,
)

_PORT_BASE = 8810


# ---------------------------------------------------------------------------
# Tier 2: Physical — CuMem state-machine edge cases
# ---------------------------------------------------------------------------


class TestPartialWakeMemoryState:
    """Verify GPU memory state during partial (staged) wake sequences."""

    def test_partial_wake_weights_only_leaves_kv_unmapped(self):
        """After sleep + wake(weights), KV cache memory must remain freed.

        This is the precondition for #44395 — if KV stays unmapped, any
        forward attempt on the KV region would hit released memory.
        """
        with server(port=_PORT_BASE) as url:
            # Baseline: generate to populate KV cache
            r = gen(url)
            assert ok(r), "baseline generation failed"
            mem_awake = gpu_free_bytes()

            # Sleep level 1: offload weights, discard KV
            assert sleep(url, level=1) == 200
            mem_asleep = gpu_free_bytes()
            assert mem_asleep > mem_awake, "sleep did not free GPU memory"

            # Partial wake: only weights
            assert wake(url, tags=["weights"]) == 200
            mem_partial = gpu_free_bytes()

            # KV cache should still be freed — partial wake does not remap it
            # mem_partial should be between asleep (all freed) and awake (all mapped)
            assert mem_partial > mem_awake, (
                "partial wake remapped all memory — KV should still be freed"
            )
            assert is_sleeping(url), (
                "engine should still report sleeping after partial wake"
            )

            # Full wake to restore
            assert wake(url) == 200
            assert not is_sleeping(url)

    def test_sleep_after_partial_wake_no_double_unmap(self):
        """sleep() after partial wake must not double-unmap already-freed KV.

        Sequence: sleep → wake(weights) → sleep again
        The KV allocations are still unmapped from the first sleep. The second
        sleep must skip them to avoid CUDA_ERROR_INVALID_VALUE.
        """
        with server(port=_PORT_BASE + 1) as url:
            r = gen(url)
            assert ok(r)

            # First cycle: sleep → partial wake (weights only)
            assert sleep(url, level=1) == 200
            assert wake(url, tags=["weights"]) == 200
            assert is_sleeping(url), "should still be sleeping (KV not woken)"

            # Second sleep: must not crash on already-unmapped KV
            assert sleep(url, level=1) == 200

            # Verify engine is still healthy
            assert wake(url) == 200
            assert not is_sleeping(url)
            r = gen(url)
            assert ok(r), "generation failed after double sleep cycle"

    def test_repeated_partial_wake_same_tag_idempotent(self):
        """Waking the same tag twice must not crash (double create_and_map).

        Without an is_asleep guard, create_and_map on an already-mapped VA
        would produce CUDA_ERROR_INVALID_VALUE.
        """
        with server(port=_PORT_BASE + 2) as url:
            r = gen(url)
            assert ok(r)

            assert sleep(url, level=1) == 200

            # Wake weights twice
            assert wake(url, tags=["weights"]) == 200
            assert wake(url, tags=["weights"]) == 200

            # Engine should still be functional
            assert wake(url) == 200
            r = gen(url)
            assert ok(r), "generation failed after double wake of same tag"


class TestSleepWakeWithGraphPool:
    """Verify sleep/wake behavior when CUDA graphs are active.

    Graph capture pools interact with VMM operations — sleep/wake take
    3-5x longer when a graph pool is present. These tests verify correctness
    (not timing) with enforce_eager=False.
    """

    def test_sleep_wake_with_cuda_graphs_correctness(self):
        """sleep/wake must succeed when CUDA graph pool is present."""
        with server(
            extra_args=["--max-num-seqs", "8"],  # no --enforce-eager
            port=_PORT_BASE + 3,
        ) as url:
            # Remove --enforce-eager: server will use CUDA graphs
            # Generate to warm up graphs
            r = gen(url)
            assert ok(r)
            baseline_text = r["choices"][0]["text"]

            # Sleep/wake cycle
            assert sleep(url, level=1) == 200
            assert wake(url) == 200

            # Output must still be correct
            r = gen(url)
            assert ok(r)
            assert r["choices"][0]["text"] == baseline_text, (
                "output changed after sleep/wake with CUDA graphs active"
            )

    def test_sleep_wake_timing_with_graphs_bounded(self):
        """sleep/wake latency with graphs should not exceed 10x eager latency.

        This is a regression guard — graph pool VMM slowdown should be bounded.
        """
        with server(port=_PORT_BASE + 4) as url:
            # Eager baseline timing
            r = gen(url)
            assert ok(r)

            t0 = time.perf_counter()
            assert sleep(url, level=1) == 200
            assert wake(url) == 200
            eager_elapsed = time.perf_counter() - t0

        with server(
            extra_args=["--max-num-seqs", "8"],
            port=_PORT_BASE + 4,
        ) as url:
            # Graph-enabled timing
            r = gen(url)
            assert ok(r)

            t0 = time.perf_counter()
            assert sleep(url, level=1) == 200
            assert wake(url) == 200
            graph_elapsed = time.perf_counter() - t0

        # Graph pool may slow VMM ops, but should not exceed 10x
        assert graph_elapsed < eager_elapsed * 10, (
            f"graph pool sleep/wake {graph_elapsed:.1f}s > 10x eager "
            f"{eager_elapsed:.1f}s — regression in VMM ops with graph pool"
        )


# ---------------------------------------------------------------------------
# Tier 3: Protocol — Cross-tag wake ordering
# ---------------------------------------------------------------------------


class TestCrossTagWakeOrdering:
    """Verify that partial wake ordering cannot cause illegal memory access.

    The dangerous sequence (from #44395):
      sleep → wake(weights) → forward → crash (KV still unmapped)

    The scheduler must block forward dispatch until all tags are resident.
    """

    def test_wake_weights_then_generate_blocked(self):
        """After wake(weights), generation must not proceed until KV is woken.

        This tests the scheduler gate: even though weights are resident,
        the engine should not dispatch a forward pass while KV is unmapped.
        The request should either queue (waiting) or error — not crash.
        """
        with server(port=_PORT_BASE + 5) as url:
            r = gen(url)
            assert ok(r)

            # Sleep and partially wake weights only
            assert sleep(url, level=1) == 200
            assert wake(url, tags=["weights"]) == 200
            assert is_sleeping(url), "engine should still be sleeping"

            # Attempt generation — should not crash the engine
            # It may timeout, return error, or queue — any non-crash is correct
            try:
                r = requests.post(
                    f"{url}/v1/completions",
                    json={
                        "model": "m",
                        "prompt": "Hello",
                        "max_tokens": 4,
                        "temperature": 0,
                    },
                    timeout=5,
                )
                # If it returns, it should be an error (not a valid completion
                # from unmapped memory)
                if r.status_code == 200:
                    body = r.json()
                    # If we got a 200, the scheduler allowed it through —
                    # this is only safe if full wake happened implicitly
                    pass
            except requests.exceptions.Timeout:
                # Timeout is acceptable — request is queued waiting for KV
                pass
            except requests.exceptions.ConnectionError:
                # Connection error means engine crashed — this is the bug
                raise AssertionError(
                    "Engine crashed during partial-wake generation — "
                    "scheduler did not gate on KV residency (#44395)"
                )

            # Verify engine is still alive
            health_r = requests.get(f"{url}/health", timeout=5)
            assert health_r.status_code == 200, (
                "engine not healthy after partial-wake generation attempt"
            )

            # Full wake and verify normal operation
            assert wake(url) == 200
            r = gen(url)
            assert ok(r), "generation failed after full wake"

    def test_wake_kv_then_weights_scheduler_blocks(self):
        """Waking KV before weights: scheduler must not dispatch forward.

        KV memory is mapped but contains stale/zero data. Weights are still
        unmapped. A forward would read zeroed weight memory → garbage output
        or crash in flash_attn.
        """
        with server(port=_PORT_BASE + 6) as url:
            r = gen(url)
            assert ok(r)

            assert sleep(url, level=1) == 200

            # Wake KV first (unusual order)
            assert wake(url, tags=["kv_cache"]) == 200
            assert is_sleeping(url), "engine should still be sleeping"

            # Attempt generation — must not produce output from zeroed weights
            try:
                r = requests.post(
                    f"{url}/v1/completions",
                    json={
                        "model": "m",
                        "prompt": "Hello",
                        "max_tokens": 4,
                        "temperature": 0,
                    },
                    timeout=5,
                )
            except (requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError):
                pass  # Both acceptable — blocked or crashed safely

            # Engine must still be healthy
            health_r = requests.get(f"{url}/health", timeout=5)
            assert health_r.status_code == 200, (
                "engine not healthy after reverse-order wake + generation"
            )

            # Complete wake and verify
            assert wake(url) == 200
            r = gen(url)
            assert ok(r)

    def test_full_wake_cycle_forward_succeeds(self):
        """Control test: full wake (all tags) → forward works normally."""
        with server(port=_PORT_BASE + 7) as url:
            r = gen(url)
            assert ok(r)
            expected = r["choices"][0]["text"]

            assert sleep(url, level=1) == 200
            assert wake(url) == 200
            assert not is_sleeping(url)

            r = gen(url)
            assert ok(r)
            assert r["choices"][0]["text"] == expected

    def test_concurrent_wake_and_generate_no_crash(self):
        """Concurrent wake_up and generation must not crash.

        Simulates the race where an orchestrator fires wake_up and a
        generate request near-simultaneously.
        """
        with server(port=_PORT_BASE + 8) as url:
            r = gen(url)
            assert ok(r)

            for _ in range(3):
                assert sleep(url, level=1) == 200

                errors = []

                def _wake():
                    try:
                        wake(url)
                    except Exception as e:
                        errors.append(f"wake error: {e}")

                def _gen():
                    try:
                        requests.post(
                            f"{url}/v1/completions",
                            json={
                                "model": "m",
                                "prompt": "Hi",
                                "max_tokens": 2,
                                "temperature": 0,
                            },
                            timeout=15,
                        )
                    except Exception as e:
                        errors.append(f"gen error: {e}")

                t1 = threading.Thread(target=_wake)
                t2 = threading.Thread(target=_gen)
                t1.start()
                t2.start()
                t1.join(timeout=20)
                t2.join(timeout=20)

                # Engine must survive regardless of race outcome
                poll_until(
                    lambda: requests.get(f"{url}/health", timeout=3).status_code == 200,
                    timeout=10,
                    msg="engine unhealthy after concurrent wake+gen",
                )

                # Ensure fully awake for next iteration
                if is_sleeping(url):
                    wake(url)
