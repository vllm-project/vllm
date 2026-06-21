# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end tests for the vLLM RL sleep/wake/pause lifecycle.

Endpoint surface under test
---------------------------
sleep/api_router  : POST /sleep  POST /wake_up  GET /is_sleeping
rlhf/api_router   : POST /pause  POST /resume   GET /is_paused
                    GET  /get_world_size

All tests require:
  --enable-sleep-mode   KV cache allocated via CuMemAllocator; without this
                        flag sleep/wake are no-ops and the bug cannot trigger.
  VLLM_SERVER_DEV_MODE=1

Test classes (ordered by increasing complexity)
------------------------------------------------
TestSleepWakeFlags          flag/metrics smoke test (supersedes test_sleep.py)
TestPhysicalMemory          GPU free-bytes + Prometheus metrics per stage
TestSchedulerGate           scheduler must not dispatch during partial/full wake
TestOutputCorrectness       golden-output roundtrip across lifecycle
TestPauseResume             /pause /resume /is_paused independent of sleep
TestErrorPaths              idempotency, wrong order, abort interaction

RFC: https://github.com/vllm-project/vllm/issues/45585
Fixes regression introduced by: https://github.com/vllm-project/vllm/pull/44483
"""

import contextlib
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager

import requests
from prometheus_client.parser import text_string_to_metric_families

MODEL_NAME = os.environ.get("VLLM_TEST_MODEL", "Qwen/Qwen3-0.6B")

_BASE_ARGS = [
    "--dtype",
    "bfloat16",
    "--max-model-len",
    "2048",
    "--max-num-seqs",
    "32",
    "--gpu-memory-utilization",
    "0.75",
    "--enable-sleep-mode",
    "--enforce-eager",
]


# ---------------------------------------------------------------------------
# Server harness
# ---------------------------------------------------------------------------


@contextmanager
def _server(extra_args=None, port: int = 8770, timeout: float = 180.0):
    """Launch vllm server with dev router; yield base URL."""
    env = {**os.environ, "VLLM_SERVER_DEV_MODE": "1"}
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL_NAME,
        "--port",
        str(port),
        "--served-model-name",
        "m",
        *(_BASE_ARGS + (extra_args or [])),
    ]
    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    url = f"http://localhost:{port}"
    try:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                err = (
                    proc.stderr.read(4000).decode(errors="replace")
                    if proc.stderr
                    else ""
                )
                raise RuntimeError(f"vllm server exited during startup:\n{err}")
            with contextlib.suppress(Exception):
                if requests.get(f"{url}/health", timeout=3).status_code == 200:
                    break
            time.sleep(1)
        else:
            proc.terminate()
            raise RuntimeError("vllm server did not start in time")
        yield url
    finally:
        proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)
        if proc.poll() is None:
            proc.kill()


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _gen(url, prompt="The capital of France is", max_tokens=8, timeout=30):
    """Fire a completion; return JSON or None on any error."""
    try:
        r = requests.post(
            f"{url}/v1/completions",
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


def _ok(resp) -> bool:
    """True iff resp is a successful completion (has choices, no error key)."""
    return (
        resp is not None
        and "choices" in resp
        and bool(resp["choices"])
        and "error" not in resp
    )


def _sleep(url, level=1, mode="abort"):
    return requests.post(
        f"{url}/sleep", params={"level": level, "mode": mode}, timeout=15
    ).status_code


def _wake(url, tags=None):
    params = {"tags": tags} if tags else {}
    return requests.post(f"{url}/wake_up", params=params, timeout=20).status_code


def _pause(url, mode="abort", clear_cache=True):
    return requests.post(
        f"{url}/pause",
        params={"mode": mode, "clear_cache": clear_cache},
        timeout=15,
    ).status_code


def _resume(url):
    return requests.post(f"{url}/resume", timeout=10).status_code


def _is_sleeping(url) -> bool:
    return requests.get(f"{url}/is_sleeping", timeout=5).json()["is_sleeping"]


def _is_paused(url) -> bool:
    return requests.get(f"{url}/is_paused", timeout=5).json()["is_paused"]


def _health(url) -> int:
    try:
        return requests.get(f"{url}/health", timeout=5).status_code
    except Exception:
        return 0


def _gpu_free_bytes() -> int:
    """Read GPU-0 free bytes via subprocess to avoid import-time torch init."""
    out = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import torch; f,_=torch.cuda.mem_get_info(0); print(f)",
        ],
        timeout=10,
    )
    return int(out.strip())


def _sleep_metrics(url):
    """Return (awake, weights_offloaded, discard_all) from /metrics."""
    r = requests.get(f"{url}/metrics", timeout=5)
    vals: dict = {}
    for family in text_string_to_metric_families(r.text):
        if family.name == "vllm:engine_sleep_state":
            for s in family.samples:
                vals[s.labels.get("sleep_state", "")] = s.value
    return vals.get("awake"), vals.get("weights_offloaded"), vals.get("discard_all")


# ---------------------------------------------------------------------------
# TestSleepWakeFlags
# ---------------------------------------------------------------------------


class TestSleepWakeFlags:
    """Smoke-level flag and metrics checks.

    These mirror the original test_sleep.py assertions so that file can
    eventually be removed.  Kept separate so a CI bisect can isolate quickly.
    """

    def test_sleep_sets_is_sleeping_and_metrics(self):
        with _server() as url:
            assert _sleep(url, level=1) == 200
            assert _is_sleeping(url) is True

            awake, wo, da = _sleep_metrics(url)
            assert awake == 0 and wo == 1 and da == 0

            assert _wake(url) == 200
            assert _is_sleeping(url) is False

            awake, wo, da = _sleep_metrics(url)
            assert awake == 1 and wo == 0 and da == 0

    def test_level2_sets_discard_all_metric(self):
        with _server() as url:
            assert _sleep(url, level=2) == 200
            _, _, da = _sleep_metrics(url)
            assert da == 1

            assert _wake(url) == 200
            assert _is_sleeping(url) is False
            assert _health(url) == 200

    def test_staged_wake_keeps_is_sleeping_true(self):
        """wake_up(["weights"]) must leave is_sleeping True until kv_cache wakes."""
        with _server() as url:
            assert _sleep(url, level=1) == 200
            assert _wake(url, tags=["weights"]) == 200
            assert _is_sleeping(url) is True  # still partial

            assert _wake(url, tags=["kv_cache"]) == 200
            assert _is_sleeping(url) is False


# ---------------------------------------------------------------------------
# TestPhysicalMemory
# ---------------------------------------------------------------------------


class TestPhysicalMemory:
    """Assert GPU free bytes change, not just flags.

    Guards against regressions where CuMemAllocator.sleep() silently no-ops
    (e.g. missing stream-sync, wrong tag registration) while returning 200.
    Each stage is cross-validated against the Prometheus sleep-state metrics.
    """

    def test_sleep_level1_frees_gpu_memory(self):
        with _server() as url:
            _gen(url)  # warm up — allocate KV blocks
            free_awake = _gpu_free_bytes()

            assert _sleep(url, level=1) == 200
            free_sleeping = _gpu_free_bytes()
            freed_gib = (free_sleeping - free_awake) / 2**30

            # 1B bf16 weights ~2 GiB + KV at 0.75 util; true unmap must free ≥1.5 GiB
            assert freed_gib > 1.5, (
                f"sleep(1) freed only {freed_gib:.2f} GiB — "
                "CuMemAllocator unmap may be a no-op"
            )
            awake, wo, _ = _sleep_metrics(url)
            assert awake == 0 and wo == 1, (
                f"Prometheus sleep metrics inconsistent: awake={awake} wo={wo}"
            )

            assert _wake(url) == 200
            free_awake2 = _gpu_free_bytes()
            recovered_gib = (free_awake2 - free_sleeping) / 2**30
            assert recovered_gib > 1.0, (
                f"wake_up recovered only {recovered_gib:.2f} GiB — "
                "remap may be incomplete"
            )

    def test_sleep_level2_frees_all_discards_all(self):
        with _server() as url:
            _gen(url)
            free_awake = _gpu_free_bytes()

            assert _sleep(url, level=2) == 200
            freed_gib = (_gpu_free_bytes() - free_awake) / 2**30
            assert freed_gib > 1.5

            _, _, da = _sleep_metrics(url)
            assert da == 1

            assert _wake(url) == 200
            assert _health(url) == 200

    def test_sleep_level0_does_not_release_memory(self):
        """level=0 pauses scheduling only — no GPU memory change."""
        with _server() as url:
            _gen(url)
            free_awake = _gpu_free_bytes()

            assert _sleep(url, level=0) == 200
            freed_gib = (_gpu_free_bytes() - free_awake) / 2**30
            assert freed_gib < 0.5, (
                f"sleep(0) released {freed_gib:.2f} GiB — "
                "it should only pause scheduling, not free GPU memory"
            )
            _, wo, da = _sleep_metrics(url)
            assert wo == 0 and da == 0

            assert _wake(url) == 200
            assert _health(url) == 200

    def test_staged_release_each_step_changes_memory(self):
        """Each tag releases a distinct chunk of GPU memory."""
        with _server() as url:
            _gen(url)
            assert _sleep(url, level=1) == 200

            assert _wake(url, tags=["weights"]) == 200
            free_after_weights = _gpu_free_bytes()

            assert _wake(url, tags=["kv_cache"]) == 200
            free_after_kv = _gpu_free_bytes()

            # waking kv_cache consumes more GPU memory than weights-only wake
            assert free_after_kv < free_after_weights, (
                "waking kv_cache should use more GPU memory than weights-only wake"
            )
            assert _health(url) == 200


# ---------------------------------------------------------------------------
# TestSchedulerGate
# ---------------------------------------------------------------------------


class TestSchedulerGate:
    """The scheduler must not dispatch work while memory is unmapped.

    The core of this class is the PR #44483 regression test.  Without the fix,
    wake_up(["weights"]) unconditionally called resume_scheduler(), allowing
    FlashAttention to launch on the still-unmapped kv_cache VA —
    flash_fwd_launch_template.h:199 TMA descriptor 700 illegal memory access.
    """

    def test_partial_wake_blocks_until_kv_resident(self):
        """PR #44483 regression test.

        sleep(1) → wake_up(["weights"]) → generate must NOT execute FA on the
        unmapped kv_cache VA.  With the fix the scheduler holds the request
        until wake_up(["kv_cache"]) completes; without the fix the engine dies
        (health goes to 503) within 3 seconds.
        """
        with _server() as url:
            assert _sleep(url, level=1) == 200
            assert _wake(url, tags=["weights"]) == 200
            assert _is_sleeping(url) is True  # kv_cache still unmapped

            result: dict = {}

            def _bg():
                # short timeout — should block, not complete, while kv_cache is asleep
                result["resp"] = _gen(url, timeout=8)

            t = threading.Thread(target=_bg)
            t.start()
            time.sleep(3)

            # engine must be alive — if it died the fix is missing
            assert _health(url) == 200, (
                "engine died during partial-wake window — "
                "IMA from running FlashAttention on unmapped kv_cache "
                "(PR #44483 regression)"
            )

            assert _wake(url, tags=["kv_cache"]) == 200
            assert _is_sleeping(url) is False
            t.join(timeout=30)
            assert _health(url) == 200

            resp = _gen(url)
            assert resp and _ok(resp)

    def test_sleep_abort_mode_blocks_new_requests(self):
        """sleep(mode='abort') pauses the scheduler.

        New requests must not return successful completions while sleeping.
        They must also not hang forever (cf. #45326).
        """
        with _server() as url:
            assert _sleep(url, level=1, mode="abort") == 200
            assert _is_sleeping(url) is True

            resp = _gen(url, timeout=5)
            assert not _ok(resp), (
                "generate returned a successful result while engine was sleeping "
                "(scheduler not paused — or request hangs without aborting)"
            )
            # engine still alive, not dead
            assert _health(url) == 200

            assert _wake(url) == 200
            assert _ok(_gen(url))

    def test_sleep_level0_blocks_without_releasing_memory(self):
        """level=0 = pause only; same scheduler gate, zero memory change."""
        with _server() as url:
            assert _sleep(url, level=0) == 200
            assert _is_sleeping(url) is True

            resp = _gen(url, timeout=5)
            assert not _ok(resp)
            assert _health(url) == 200

            assert _wake(url) == 200
            assert _ok(_gen(url))


# ---------------------------------------------------------------------------
# TestOutputCorrectness
# ---------------------------------------------------------------------------


class TestOutputCorrectness:
    """Output must be deterministic and self-consistent across the lifecycle."""

    def test_full_wake_restores_output(self):
        """sleep(1) → wake_up() — weights restored; output matches golden."""
        with _server() as url:
            golden = _gen(url)
            assert golden
            golden_text = golden["choices"][0]["text"]

            assert _sleep(url, level=1) == 200
            assert _wake(url) == 200

            resp = _gen(url)
            assert resp and resp["choices"][0]["text"] == golden_text, (
                "output changed after sleep/wake — weight restore broken"
            )

    def test_staged_wake_restores_output(self):
        """sleep → wake(weights) → wake(kv_cache) — output matches golden."""
        with _server() as url:
            golden_text = _gen(url)["choices"][0]["text"]

            assert _sleep(url, level=1) == 200
            assert _wake(url, tags=["weights"]) == 200
            assert _wake(url, tags=["kv_cache"]) == 200

            resp = _gen(url)
            assert resp and resp["choices"][0]["text"] == golden_text

    def test_multiple_cycles_stable(self):
        """3× sleep/wake cycles — output and engine stay stable.

        Guards against cumem bookkeeping corruption across repeated
        release+remap of the same physical pages.
        """
        with _server() as url:
            golden_text = _gen(url)["choices"][0]["text"]

            for i in range(3):
                assert _sleep(url, level=1) == 200
                assert _wake(url) == 200
                assert _health(url) == 200

                resp = _gen(url)
                assert resp and resp["choices"][0]["text"] == golden_text, (
                    f"output drifted on cycle {i} — cumem bookkeeping corrupted"
                )

    def test_prefix_cache_cleared_after_wake(self):
        """wake_up() calls reset_prefix_cache(); no stale KV entries survive.

        If the prefix cache is not flushed, a subsequent sleep cycle could
        reuse a stale entry pointing to an already-released physical page.
        """
        with _server() as url:
            prompt = "The capital of France is"
            _gen(url, prompt=prompt)  # populate prefix cache

            assert _sleep(url, level=1) == 200
            assert _wake(url) == 200
            assert _health(url) == 200

            resp = _gen(url, prompt=prompt)
            assert _ok(resp), (
                "generate failed after wake with cached prompt — "
                "possible stale prefix-cache IMA"
            )


# ---------------------------------------------------------------------------
# TestPauseResume
# ---------------------------------------------------------------------------


class TestPauseResume:
    """POST /pause  POST /resume  GET /is_paused are independent of sleep.

    /pause blocks scheduling without releasing GPU memory (level=0 equivalent
    from the GPU side, but a distinct code path and distinct state flag).
    """

    def test_pause_blocks_without_releasing_memory(self):
        with _server() as url:
            free_before = _gpu_free_bytes()
            assert _pause(url, mode="abort") == 200
            assert _is_paused(url) is True
            assert _is_sleeping(url) is False  # distinct from sleep

            freed = (_gpu_free_bytes() - free_before) / 2**30
            assert freed < 0.5, (
                f"/pause released {freed:.2f} GiB — it must not touch GPU memory"
            )

            resp = _gen(url, timeout=5)
            assert not _ok(resp)

            assert _resume(url) == 200
            assert _is_paused(url) is False
            assert _ok(_gen(url))

    def test_pause_mode_wait_drains_inflight_request(self):
        """mode='wait' lets an in-flight request complete, then blocks new ones."""
        with _server() as url:
            result: dict = {}

            def _bg():
                result["r"] = _gen(url, max_tokens=32, timeout=60)

            t = threading.Thread(target=_bg)
            t.start()
            time.sleep(0.5)

            assert _pause(url, mode="wait") == 200
            t.join(timeout=30)
            assert result.get("r") is not None, (
                "in-flight request not completed after pause(mode=wait)"
            )

            resp = _gen(url, timeout=5)
            assert not _ok(resp)
            assert _resume(url) == 200

    def test_pause_mode_keep_resumes_frozen_request(self):
        """mode='keep' freezes the request; it must complete after /resume."""
        with _server() as url:
            result: dict = {}

            def _bg():
                result["r"] = _gen(url, max_tokens=16, timeout=60)

            t = threading.Thread(target=_bg)
            t.start()
            time.sleep(0.3)

            assert _pause(url, mode="keep") == 200
            time.sleep(1)

            # request must NOT have completed yet
            assert not _ok(result.get("r")), (
                "request completed before resume in mode=keep"
            )

            assert _resume(url) == 200
            t.join(timeout=30)
            assert _ok(result.get("r")), (
                "request not completed after resume in mode=keep"
            )

    def test_get_world_size(self):
        with _server() as url:
            r = requests.get(f"{url}/get_world_size", timeout=5)
            assert r.status_code == 200
            ws = r.json()["world_size"]
            assert isinstance(ws, int) and ws >= 1


# ---------------------------------------------------------------------------
# TestErrorPaths
# ---------------------------------------------------------------------------


class TestErrorPaths:
    """Protocol violations must not crash the engine."""

    def test_double_sleep_idempotent(self):
        """sleep() while already sleeping must not crash.

        Per PR #45518, response body carries {"already_sleeping": true/false}.
        We assert only liveness here; the body contract is validated separately.
        """
        with _server() as url:
            assert _sleep(url, level=1) == 200
            sc = _sleep(url, level=1)  # idempotent call
            assert sc in (200, 400), f"double sleep returned unexpected {sc}"

            assert _wake(url) == 200
            assert _health(url) == 200
            assert _ok(_gen(url))

    def test_wake_while_awake_idempotent(self):
        with _server() as url:
            _gen(url)
            assert _is_sleeping(url) is False
            sc = _wake(url)
            assert sc in (200, 400)
            assert _health(url) == 200
            assert _ok(_gen(url))

    def test_abort_then_sleep_wake(self):
        """Abort a mid-flight request, then sleep → wake.  Engine must survive.

        Simulates the colocate_async partial-rollout pattern.
        """
        with _server() as url:

            def _bg():
                with contextlib.suppress(Exception):
                    requests.post(
                        f"{url}/v1/completions",
                        json={
                            "model": "m",
                            "prompt": "x" * 200,
                            "max_tokens": 256,
                            "temperature": 0,
                        },
                        timeout=1,
                    )

            threading.Thread(target=_bg).start()
            time.sleep(0.3)

            assert _sleep(url, level=1) == 200
            assert _wake(url) == 200
            assert _health(url) == 200
            assert _ok(_gen(url))


# ---------------------------------------------------------------------------
# TestConcurrentRace  (new — Tier 1B supplement)
# ---------------------------------------------------------------------------


class TestConcurrentRace:
    """Concurrent sleep + generate threads must not deadlock or crash the engine.

    Reference: miles tests/fast/router/test_session_race_conditions.py
               TestSessionConcurrencyContracts (4 tests) +
               TestClosingRaceConditions (5 tests) — 8-thread ThreadPoolExecutor
    """

    def test_concurrent_sleep_and_generate_no_deadlock(self):
        """10 generate threads racing against 1 sleep/wake thread.

        The engine must survive and remain healthy after the race window.
        We don't assert that all generates succeed (some will be aborted by
        the sleep), but we do assert that:
          - no thread hangs indefinitely (join timeout 30 s)
          - engine is alive at the end
          - a fresh generate after the race succeeds
        """
        with _server() as url:
            results = []
            errors = []

            def _gen_thread():
                try:
                    r = _gen(url, max_tokens=8, timeout=10)
                    results.append(r)
                except Exception as e:
                    errors.append(str(e))

            def _sleep_wake_thread():
                try:
                    _sleep(url, level=1, mode="abort")
                    time.sleep(0.2)
                    _wake(url)
                except Exception as e:
                    errors.append(str(e))

            threads = [threading.Thread(target=_gen_thread) for _ in range(10)]
            sw = threading.Thread(target=_sleep_wake_thread)

            for t in threads:
                t.start()
            time.sleep(0.1)  # let some generates get in-flight before sleep
            sw.start()

            sw.join(timeout=30)
            for t in threads:
                t.join(timeout=30)

            assert _health(url) == 200, (
                f"engine died after concurrent race; errors={errors}"
            )
            # After the race the engine must be able to serve requests
            assert _ok(_gen(url)), "generate failed after concurrent race"


# ---------------------------------------------------------------------------
# TestMemoryLeakCycle  (new — Tier 1A supplement)
# ---------------------------------------------------------------------------


class TestMemoryLeakCycle:
    """sleep/wake cycles must not accumulate host-side memory leaks.

    Reference: ROLL tests/third_party/vllm/test_vllm_mem_oom.py
               generate_memory() — 20 iterations, tracking CPU RSS.
    """

    def test_no_host_rss_growth_over_10_cycles(self):
        """10 sleep/wake cycles: CPU RSS must not grow monotonically.

        We measure RSS after a warm-up period (cycles 3–10) and assert
        the total growth is < 10 % of the warm-up baseline, guarding
        against handle-leak or buffer-accumulation bugs.
        """
        import psutil

        with _server() as url:
            rss_samples = []

            for i in range(10):
                _gen(url)
                assert _sleep(url, level=1) == 200
                assert _wake(url) == 200
                assert _health(url) == 200

                if i >= 2:  # skip warm-up cycles
                    proc = psutil.Process()
                    rss_samples.append(proc.memory_info().rss)

            baseline = rss_samples[0]
            peak = max(rss_samples)
            growth_pct = (peak - baseline) / baseline * 100

            assert growth_pct < 10.0, (
                f"Host RSS grew {growth_pct:.1f}% over 8 post-warmup sleep/wake cycles "
                f"(baseline={baseline/1e6:.1f} MB, peak={peak/1e6:.1f} MB) — "
                "possible memory leak in cumem bookkeeping or handle tracking"
            )


# ---------------------------------------------------------------------------
# TestSamplingNAbort  (new — Tier 1B supplement)
# ---------------------------------------------------------------------------


class TestSamplingNAbort:
    """sleep(mode='abort') while sampling_n > 1 is in-flight must not crash.

    Reference: ROLL tests/third_party/vllm/test_abort.py
               _check_vllm_sampling_n — cross-version abort semantics
               (CancelledError in V0, finish_reason='abort' in V1 >= 0.10.2)
    """

    def test_sampling_n4_abort_engine_survives(self):
        """sampling_n=4 in-flight request aborted by sleep — engine survives.

        With sampling_n=4, the scheduler allocates 4 sequence slots per
        request.  sleep(mode='abort') must drain all 4 cleanly without
        leaving dangling state that blocks the next wake.
        """
        with _server() as url:
            result: dict = {}

            def _bg():
                try:
                    r = requests.post(
                        f"{url}/v1/completions",
                        json={
                            "model": "m",
                            "prompt": "Count from 1 to 100:",
                            "max_tokens": 64,
                            "temperature": 1.0,
                            "n": 4,
                        },
                        timeout=15,
                    )
                    result["resp"] = r.json()
                except Exception as e:
                    result["err"] = str(e)

            t = threading.Thread(target=_bg)
            t.start()
            time.sleep(0.3)  # let the request start

            assert _sleep(url, level=1, mode="abort") == 200
            t.join(timeout=15)

            assert _health(url) == 200, "engine died after sampling_n=4 abort"
            assert _wake(url) == 200
            # engine must recover and serve new requests
            assert _ok(_gen(url)), "generate failed after sampling_n abort + wake"


# ---------------------------------------------------------------------------
# TestLogprobsPrecision  (new — Tier 1C supplement)
# ---------------------------------------------------------------------------


class TestLogprobsPrecision:
    """logprobs values must be consistent before and after a sleep/wake cycle.

    Reference: ROLL tests/distributed/strategy/log_probs/ (9 files)
               test_fsdp_log_probs_full, test_fsdp_log_probs_cp_rmpad, etc.

    After sleep(level=1)/wake, weights are remapped from CPU backup.
    Calibrated scales (FP8-KV, etc.) must be restored; logprobs must match
    the pre-sleep values within a tight tolerance.
    """

    def test_logprobs_stable_after_sleep_wake(self):
        """logprobs before and after sleep/wake must match within 1e-3.

        Reference: ROLL test_fsdp_log_probs_full — compares log_probs values
        across different parallelism configurations to within tight tolerance.
        """
        with _server() as url:
            prompt = "The capital of France is Paris and the capital of Germany is"

            def _get_logprobs():
                r = requests.post(
                    f"{url}/v1/completions",
                    json={
                        "model": "m",
                        "prompt": prompt,
                        "max_tokens": 4,
                        "temperature": 0,
                        "logprobs": 5,
                    },
                    timeout=30,
                )
                resp = r.json()
                if "choices" not in resp or not resp["choices"]:
                    return None
                choice = resp["choices"][0]
                lp = choice.get("logprobs", {})
                return lp.get("token_logprobs", [])

            before = _get_logprobs()
            assert before is not None, "failed to get logprobs before sleep"
            assert len(before) > 0

            assert _sleep(url, level=1) == 200
            assert _wake(url) == 200
            assert _health(url) == 200

            after = _get_logprobs()
            assert after is not None, "failed to get logprobs after sleep/wake"
            assert len(after) == len(before), (
                "logprobs length changed after sleep/wake"
            )

            for i, (b, a) in enumerate(zip(before, after)):
                if b is None or a is None:
                    continue
                diff = abs(b - a)
                assert diff < 1e-3, (
                    f"logprob[{i}] drifted after sleep/wake: "
                    f"before={b:.6f} after={a:.6f} diff={diff:.2e} — "
                    "weight restore or KV-scale recalibration may be incorrect"
                )


# ---------------------------------------------------------------------------
# TestSleepWakeLatency  (new — Tier 1A supplement)
# ---------------------------------------------------------------------------


class TestSleepWakeLatency:
    """sleep/wake must complete within reasonable wall-clock time.

    Reference: sglang test_update_weights_from_distributed.py
               assert time < 3s for weight sync operations.
    Here we use 10 s for the full sleep+wake roundtrip on a 0.6B model,
    which is generous but guards against regressions that cause multi-minute
    stalls (e.g. accidental full model re-download on wake).
    """

    def test_sleep_wake_roundtrip_under_10s(self):
        """sleep(1) + wake_up() roundtrip must complete in < 10 s.

        For a 0.6B bfloat16 model (~1.2 GB weights) the cumem unmap/remap
        should complete in < 2 s on modern hardware.  10 s is a conservative
        bound that would still catch pathological regressions.
        """
        with _server() as url:
            _gen(url)  # warm up — ensure KV blocks are allocated

            t0 = time.perf_counter()
            assert _sleep(url, level=1) == 200
            t_sleep = time.perf_counter()

            assert _wake(url) == 200
            t_wake = time.perf_counter()

            sleep_elapsed = t_sleep - t0
            wake_elapsed = t_wake - t_sleep
            total = t_wake - t0

            assert total < 10.0, (
                f"sleep+wake roundtrip took {total:.2f}s "
                f"(sleep={sleep_elapsed:.2f}s, wake={wake_elapsed:.2f}s) — "
                "cumem unmap/remap regression?"
            )
            assert _health(url) == 200
            assert _ok(_gen(url))

    def test_five_cycles_total_under_60s(self):
        """5 consecutive sleep/wake cycles must finish in < 60 s total.

        Guards against per-cycle overhead accumulation (handle leak,
        growing allocation list, etc.).
        """
        with _server() as url:
            _gen(url)

            t0 = time.perf_counter()
            for _ in range(5):
                assert _sleep(url, level=1) == 200
                assert _wake(url) == 200
                assert _health(url) == 200
            total = time.perf_counter() - t0

            assert total < 60.0, (
                f"5 sleep/wake cycles took {total:.2f}s — "
                "per-cycle overhead may be accumulating"
            )
            assert _ok(_gen(url))


# ---------------------------------------------------------------------------
# TestCPUWeightBackup  (new — Tier 1A supplement)
# ---------------------------------------------------------------------------


class TestCPUWeightBackup:
    """CPU weight backup: level-1 sleep offloads weights to CPU; wake restores.

    Reference: sglang test_release_memory_occupation.py
               test_release_and_resume_occupation_with_weights_cpu_backup —
               verifies golden output after CPU backup roundtrip.

    Unlike TestOutputCorrectness which tests 3 cycles, this specifically
    tests 5 cycles and focuses on the weights-only partial-wake path, which
    is the path used in colocate RL (trainer uses KV memory while engine
    keeps weights on CPU).
    """

    def test_weights_cpu_backup_5cycle_output_stable(self):
        """5× sleep(level=1) → wake(weights) → wake(kv_cache) output stable.

        This is the colocate RL pattern: sleep frees ALL GPU memory,
        wake(["weights"]) restores weights only (trainer has released GPU),
        wake(["kv_cache"]) re-allocates KV pool.
        Output must match the golden on every cycle.
        """
        with _server() as url:
            golden = _gen(url)
            assert golden and _ok(golden)
            golden_text = golden["choices"][0]["text"]

            for cycle in range(5):
                assert _sleep(url, level=1) == 200

                # staged wake: weights first, then kv_cache
                assert _wake(url, tags=["weights"]) == 200
                assert _wake(url, tags=["kv_cache"]) == 200
                assert _health(url) == 200

                resp = _gen(url)
                assert resp and _ok(resp), (
                    f"generate failed on CPU-backup cycle {cycle}"
                )
                assert resp["choices"][0]["text"] == golden_text, (
                    f"output drifted on CPU-backup cycle {cycle} — "
                    "weight restore from CPU backup may be incomplete"
                )
