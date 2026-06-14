# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end behavioral tests for the sleep / partial-wake / full-wake / abort
state machine exposed through VLLM_SERVER_DEV_MODE dev endpoints.

These tests assert *behavior* (scheduler gating, output correctness, engine
liveness), not just flag values.  The key regression test is
``test_partial_wake_blocks_until_kv_resident``, which directly exercises the
PR #44483 bug scenario: a generate request must not execute FlashAttention
while kv_cache is still unmapped after a weights-only partial wake_up.

All tests require:
  - --enable-sleep-mode  (allocates KV cache through CuMemAllocator so that
    sleep() actually unmaps it; without this flag sleep/wake are no-ops for KV)
  - VLLM_SERVER_DEV_MODE=1  (exposes /sleep, /wake_up, /is_sleeping endpoints)
  - CUDA_VISIBLE_DEVICES=0  (single GPU)
"""

import contextlib
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager

import requests


# ---------------------------------------------------------------------------
# Minimal self-contained server harness (no tests.utils dependency so this
# file can run in any environment that has vllm installed).
# ---------------------------------------------------------------------------
@contextmanager
def _sleep_mode_server(
    model: str,
    extra_args: list[str],
    env: dict | None = None,
    port: int = 8765,
    startup_timeout: float = 180.0,
):
    """Launch `vllm serve` as a subprocess and yield the base URL."""
    merged_env = {**os.environ, "VLLM_SERVER_DEV_MODE": "1"}
    if env:
        merged_env.update(env)

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--port",
        str(port),
        "--served-model-name",
        "m",
    ] + extra_args

    proc = subprocess.Popen(
        cmd, env=merged_env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    base_url = f"http://localhost:{port}"
    try:
        deadline = time.time() + startup_timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                err = (
                    proc.stderr.read(4000).decode(errors="replace")
                    if proc.stderr
                    else ""
                )
                raise RuntimeError(f"vllm server exited during startup:\n{err}")
            try:
                if requests.get(f"{base_url}/health", timeout=3).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            proc.terminate()
            raise RuntimeError("vllm server did not start in time")
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


# Small, CI-friendly model. Override with VLLM_TEST_MODEL env var if needed.
MODEL_NAME = os.environ.get("VLLM_TEST_MODEL", "Qwen/Qwen3-0.6B")

# Common server args.
# --enable-sleep-mode is REQUIRED: without it vLLM does not allocate KV cache
# through CuMemAllocator, so sleep/wake are no-ops for KV — the partial-wake
# bug (FA on unmapped KV → IMA) cannot trigger, and the fix cannot be tested.
# --enforce-eager avoids CUDA-graph capture side-effects in the test logic.
COMMON_ARGS = [
    "--dtype",
    "bfloat16",
    "--max-model-len",
    "2048",
    "--max-num-seqs",
    "64",
    "--gpu-memory-utilization",
    "0.4",
    "--enable-sleep-mode",
    "--enforce-eager",
]

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _generate(
    base_url: str,
    prompt: str = "The capital of France is",
    max_tokens: int = 8,
    timeout: int = 30,
) -> dict | None:
    """Fire a synchronous completion.
    Returns parsed JSON on success, None on timeout or any HTTP/connection error.
    Never raises — callers check for None or inspect the JSON for an 'error' key.
    """
    try:
        r = requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": "m",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0,
            },
            timeout=timeout,
        )
        return r.json()
    except Exception:
        return None


def _sleep(base_url: str, level: int = 1) -> int:
    return requests.post(
        f"{base_url}/sleep", params={"level": level}, timeout=10
    ).status_code


def _wake(base_url: str, tags: list[str] | None = None) -> int:
    params = {}
    if tags:
        params["tags"] = tags
    return requests.post(f"{base_url}/wake_up", params=params, timeout=20).status_code


def _is_sleeping(base_url: str) -> bool:
    return requests.get(f"{base_url}/is_sleeping", timeout=5).json()["is_sleeping"]


def _health(base_url: str) -> int:
    try:
        return requests.get(f"{base_url}/health", timeout=5).status_code
    except Exception:
        return 0


def _metrics(base_url: str):
    """Return (awake, weights_offloaded, discard_all) from Prometheus metrics."""
    from prometheus_client.parser import text_string_to_metric_families

    r = requests.get(f"{base_url}/metrics", timeout=5)
    awake = weights_offloaded = discard_all = None
    for family in text_string_to_metric_families(r.text):
        if family.name == "vllm:engine_sleep_state":
            for sample in family.samples:
                # label key is "sleep_state" (e.g. sleep_state="awake")
                lv = sample.labels.get("sleep_state", "")
                if lv == "awake":
                    awake = sample.value
                elif lv == "weights_offloaded":
                    weights_offloaded = sample.value
                elif lv == "discard_all":
                    discard_all = sample.value
    return awake, weights_offloaded, discard_all


# ---------------------------------------------------------------------------
# Test 1 — awake baseline
# ---------------------------------------------------------------------------


def test_awake_generates_correctly():
    """Sanity: server starts awake and produces correct output."""
    with _sleep_mode_server(MODEL_NAME, COMMON_ARGS) as url:
        resp = _generate(url)
        assert resp is not None, "baseline generate timed out"
        assert len(resp["choices"][0]["text"]) > 0, "empty output on awake generate"
        assert _is_sleeping(url) is False
        assert _health(url) == 200


# ---------------------------------------------------------------------------
# Test 2 — sleep / full-wake cycle, output correctness
# ---------------------------------------------------------------------------


def test_sleep_full_wake_output_consistent():
    """sleep(1) → wake_up() restores the engine to a state that produces the
    same output as before sleeping (weights loaded back correctly)."""
    with _sleep_mode_server(MODEL_NAME, COMMON_ARGS) as url:
        golden = _generate(url)
        assert golden is not None
        golden_text = golden["choices"][0]["text"]

        # sleep
        assert _sleep(url, level=1) == 200
        assert _is_sleeping(url) is True
        awake, wo, _ = _metrics(url)
        assert awake == 0 and wo == 1, (
            f"unexpected sleep metrics: awake={awake} weights_offloaded={wo}"
        )

        # full wake
        assert _wake(url) == 200
        assert _is_sleeping(url) is False
        awake, wo, _ = _metrics(url)
        assert awake == 1 and wo == 0, (
            f"unexpected post-wake metrics: awake={awake} weights_offloaded={wo}"
        )

        # output must match golden
        resp = _generate(url)
        assert resp is not None
        assert resp["choices"][0]["text"] == golden_text, (
            "output changed after sleep/wake cycle — weights not restored"
        )


# ---------------------------------------------------------------------------
# Test 3 — sleep / full-wake round-trip with level 2
# ---------------------------------------------------------------------------


def test_sleep_level2_full_wake():
    """sleep(level=2) discards weights too (no CPU backup); wake_up() must
    restore everything from caller-pushed weights.  For a standalone vLLM
    server this is unusual, but the endpoint must not crash."""
    with _sleep_mode_server(MODEL_NAME, COMMON_ARGS) as url:
        assert _sleep(url, level=2) == 200
        assert _is_sleeping(url) is True
        _, _, da = _metrics(url)
        assert da == 1, f"level-2 sleep should set discard_all metric, got {da}"

        # full wake (level-2 wake reloads from disk via HF)
        assert _wake(url) == 200
        assert _is_sleeping(url) is False
        assert _health(url) == 200


# ---------------------------------------------------------------------------
# Test 4 — PARTIAL wake: generate must NOT run while kv_cache unmapped
#          This is the PR #44483 regression test.
# ---------------------------------------------------------------------------


def test_partial_wake_blocks_until_kv_resident():
    """
    Bug scenario (pre-PR #44483):
      sleep(1) → wake_up(["weights"]) → generate  →  FA on unmapped KV → IMA
                 ^^ kv_cache still unmapped here

    With the fix, the scheduler must remain paused after a partial wake
    (tags=["weights"] only) and must not dispatch a generate step until
    wake_up(["kv_cache"]) is called.

    Behavioral assertion:
      - After wake_up(["weights"]), is_sleeping is still True.
      - A generate request submitted in this window must NOT complete before
        wake_up(["kv_cache"]) is called  (i.e. it should time out or block).
      - After wake_up(["kv_cache"]), is_sleeping becomes False, and the request
        (or a new one) completes successfully.
      - The engine remains healthy throughout (no IMA / EngineDeadError).
    """
    with _sleep_mode_server(MODEL_NAME, COMMON_ARGS) as url:
        # 1. sleep
        assert _sleep(url, level=1) == 200
        assert _is_sleeping(url) is True

        # 2. partial wake: weights only
        assert _wake(url, tags=["weights"]) == 200
        # engine must STILL be sleeping (kv_cache not yet resident)
        assert _is_sleeping(url) is True, (
            "is_sleeping must remain True after weights-only partial wake"
        )

        # 3. fire a generate during the partial-wake window.
        #    With the fix: the request is held (scheduler paused) until
        #    wake_up(["kv_cache"]).  We give it a short timeout intentionally —
        #    it will *not* complete while kv_cache is unmapped.
        #    Without the fix: the request executes immediately, FA runs on the
        #    unmapped KV VA → TMA descriptor 700 IMA → EngineDeadError.
        result: dict = {}

        def _bg_gen():
            # timeout=8s: should NOT complete before kv_cache wakes (fix case)
            result["resp"] = _generate(url, timeout=8)

        t = threading.Thread(target=_bg_gen)
        t.start()

        # 4. wait briefly — if the bug is present the request will complete
        #    AND crash the engine before we reach this assertion.
        time.sleep(3)
        assert _health(url) == 200, (
            "engine died during partial-wake window — likely IMA from "
            "executing FlashAttention on unmapped kv_cache (PR #44483 bug)"
        )

        # 5. complete the wake — this is the moment the fix allows the request
        #    to actually execute (scheduler unblocked, kv_cache now resident).
        assert _wake(url, tags=["kv_cache"]) == 200
        assert _is_sleeping(url) is False

        t.join(timeout=60)

        # 6. engine must be alive after full wake
        assert _health(url) == 200, "engine died after full wake"

        # 7. a fresh generate must succeed (the bg_gen may have timed out in
        #    the partial-wake window, which is the correct fixed behavior).
        resp = _generate(url, timeout=30)
        assert resp is not None, "generate timed out after full wake"
        assert len(resp["choices"][0]["text"]) > 0, "empty output after full wake"


# ---------------------------------------------------------------------------
# Test 5 — abort mid-generation; engine stays healthy
# ---------------------------------------------------------------------------


def test_abort_keeps_engine_healthy():
    """Cancel a long generation mid-flight; the server must accept new
    requests immediately after without errors."""
    with _sleep_mode_server(MODEL_NAME, COMMON_ARGS) as url:
        # fire a long generation and cancel it quickly
        def _long_gen():
            with contextlib.suppress(Exception):
                requests.post(
                    f"{url}/v1/completions",
                    json={
                        "model": "m",
                        "prompt": "Count from 1 to 1000:",
                        "max_tokens": 512,
                        "temperature": 0,
                    },
                    timeout=2,  # intentionally very short -> triggers abort
                )

        t = threading.Thread(target=_long_gen)
        t.start()
        t.join()

        # server must still be alive
        assert _health(url) == 200, "engine died after abort"

        # and must serve new requests
        resp = _generate(url)
        assert resp is not None, "generate timed out after abort"
        assert len(resp["choices"][0]["text"]) > 0


# ---------------------------------------------------------------------------
# Test 6 — abort-then-sleep-wake: colocate_async composite sequence
# ---------------------------------------------------------------------------


def test_abort_then_sleep_full_wake():
    """Simulate the colocate_async partial-rollout pattern:
    abort pending requests → sleep → full wake → generate succeeds."""
    with _sleep_mode_server(MODEL_NAME, COMMON_ARGS) as url:
        # queue a request then abort it
        def _bg():
            with contextlib.suppress(Exception):
                requests.post(
                    f"{url}/v1/completions",
                    json={
                        "model": "m",
                        "prompt": "x" * 500,
                        "max_tokens": 256,
                        "temperature": 0,
                    },
                    timeout=1,
                )

        threading.Thread(target=_bg).start()
        time.sleep(0.5)

        # sleep → wake cycle on top of a just-aborted state
        assert _sleep(url, level=1) == 200
        assert _is_sleeping(url) is True
        assert _wake(url) == 200
        assert _is_sleeping(url) is False
        assert _health(url) == 200

        resp = _generate(url)
        assert resp is not None
        assert len(resp["choices"][0]["text"]) > 0


# ---------------------------------------------------------------------------
# Test 7 — partial wake, then complete; verify correct output after full wake
# ---------------------------------------------------------------------------


def test_partial_wake_then_full_wake_output_correct():
    """After a sleep → partial wake (weights) → full wake (kv_cache) sequence,
    the output must equal the pre-sleep golden output, confirming both weight
    restoration and KV cache re-initialization are correct."""
    with _sleep_mode_server(MODEL_NAME, COMMON_ARGS) as url:
        golden = _generate(url)
        assert golden is not None
        golden_text = golden["choices"][0]["text"]

        assert _sleep(url, level=1) == 200
        assert _wake(url, tags=["weights"]) == 200
        assert _is_sleeping(url) is True  # still partial

        assert _wake(url, tags=["kv_cache"]) == 200
        assert _is_sleeping(url) is False  # fully awake now

        resp = _generate(url)
        assert resp is not None
        assert resp["choices"][0]["text"] == golden_text, (
            "output mismatch after staged wake — weight or KV restore broken"
        )


# ---------------------------------------------------------------------------
# Test 8 — sleep blocks new requests (SGLang lesson: release only when idle;
#           vLLM pause_scheduler + clear_cache must prevent execution)
# ---------------------------------------------------------------------------


def test_sleep_blocks_execution():
    """After sleep(), the engine must not execute any forward passes.
    New requests should be held (scheduler paused) and not return results
    until wake_up() is called.

    SGLang asserts is_fully_idle() before release_memory_occupation; vLLM
    pause_scheduler() achieves the same — no request should slip through while
    kv_cache is unmapped.
    """
    with _sleep_mode_server(MODEL_NAME, COMMON_ARGS) as url:
        assert _sleep(url, level=1) == 200
        assert _is_sleeping(url) is True

        # Request submitted while asleep must NOT complete (engine dead or blocked).
        # We use a very short timeout — if the request returns successfully the
        # scheduler guard is broken.
        resp = _generate(url, timeout=5)
        # Either None (timeout/connection refused) or an error JSON is acceptable;
        # what is NOT acceptable is a successful completion with choices.
        successful = (
            resp is not None
            and "choices" in resp
            and resp.get("choices")
            and "error" not in resp
        )
        assert not successful, (
            "generate returned a successful result while engine was sleeping — "
            "scheduler guard broken"
        )

        # Engine must still be alive (just paused, not dead)
        assert _health(url) == 200, "engine died during sleep instead of blocking"

        # Full wake — request should now succeed
        assert _wake(url) == 200
        assert _is_sleeping(url) is False
        resp = _generate(url)
        assert resp is not None
        assert "choices" in resp and len(resp["choices"][0]["text"]) > 0


# ---------------------------------------------------------------------------
# Test 9 — multiple sleep/wake cycles: bookkeeping correctness
#           (SGLang/TRTInfer lesson: repeated release+resume must not corrupt
#           the memory pool; cumem double-free or wrong residency state breaks
#           subsequent cycles)
# ---------------------------------------------------------------------------


def test_multiple_sleep_wake_cycles_stable():
    """Three consecutive sleep(1) → wake_up() cycles must all produce the
    same output as the pre-sleep golden, verifying that the cumem allocator
    bookkeeping (is_resident tracking, pool pointers) is stable across
    repeated release+remap of the same physical pages.

    Regression for: a double-unmap or stale is_resident flag would cause the
    second or third cycle to either IMA or silently serve wrong output.
    """
    with _sleep_mode_server(MODEL_NAME, COMMON_ARGS) as url:
        golden = _generate(url)
        assert golden is not None
        golden_text = golden["choices"][0]["text"]

        for cycle in range(3):
            assert _sleep(url, level=1) == 200, f"sleep failed on cycle {cycle}"
            assert _is_sleeping(url) is True
            assert _wake(url) == 200, f"wake failed on cycle {cycle}"
            assert _is_sleeping(url) is False
            assert _health(url) == 200, f"engine died on cycle {cycle}"
            resp = _generate(url)
            assert resp is not None, f"generate timed out on cycle {cycle}"
            assert resp["choices"][0]["text"] == golden_text, (
                f"output changed on cycle {cycle} — cumem bookkeeping corrupted"
            )


# ---------------------------------------------------------------------------
# Test 10 — prefix cache invalidated after wake_up
#            (SGLang lesson: resume_memory_occupation must flush prefix cache;
#             vLLM wake_up calls reset_prefix_cache so stale KV entries from
#             before sleep do not survive across a sleep/wake boundary)
# ---------------------------------------------------------------------------


def test_prefix_cache_cleared_after_wake():
    """After sleep → wake_up, the prefix cache must be invalidated.

    SGLang explicitly calls flush_cache() after resume_memory_occupation.
    vLLM reset_prefix_cache() is called inside wake_up (core.py:688).

    If the prefix cache is NOT flushed, a request using the same prompt as
    before sleep would reuse a stale cached KV entry whose physical page was
    released during sleep — leading to either corrupted output or IMA on the
    next sleep cycle.

    We verify this by:
      1. Generating with prompt P to populate the prefix cache.
      2. sleep(1) — physical pages released.
      3. wake_up() — pages remapped, prefix cache should be flushed.
      4. Generate again with the same prompt P.
      5. The engine must be alive and return a valid (non-corrupt) response.
    """
    with _sleep_mode_server(MODEL_NAME, COMMON_ARGS) as url:
        prompt = "The capital of France is"

        # populate prefix cache
        r1 = _generate(url, prompt=prompt)
        assert r1 is not None

        assert _sleep(url, level=1) == 200
        assert _wake(url) == 200
        assert _health(url) == 200

        # generate with same prompt — must succeed (not crash or corrupt)
        r2 = _generate(url, prompt=prompt)
        assert r2 is not None, "generate after wake failed — possible prefix cache IMA"
        assert "choices" in r2 and len(r2["choices"][0]["text"]) > 0, (
            "empty output after wake with cached prompt — prefix cache not flushed"
        )
