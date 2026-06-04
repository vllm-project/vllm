#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that /collective_rpc reload_weights actually swaps in new weights.

The server is booted from a *corrupted* mirror of the checkpoint
(``run_collective_rpc_reload`` uses
``tests/cohere/scripts/zero_safetensor_param.py`` to zero
``model.language_model.embed_tokens.weight``, which in this checkpoint
is tied to the LM head — zeroing it kills both input encoding and
output projection). The test then:

1. Runs a bee-eval task on the broken model — expects a *low* score.
2. Submits a long-running generation in a background thread so the
   engine has an in-flight request, then POSTs ``/pause?mode=wait``.
   The in-flight request prevents the engine from draining within
   ``_PAUSE_WAIT_TIMEOUT``, so the ``_pause`` helper must fall back
   to ``mode=abort`` — proving the fallback path against a real
   server. Follows up with ``/collective_rpc reload_weights`` (good
   checkpoint), ``/collective_rpc recapture_cudagraphs``, ``/resume``.
3. Re-runs the same task — expects a *passing* score.

This validates that ``reload_weights`` actually loads new weights into
the live engine (rather than just leaving the existing tensors intact)
AND that the pause-fallback path works end-to-end. The CUDA graph
recapture step is required because vLLM's ``reload_weights`` does not
invalidate captured graphs; see
``docs/cohere/code_notes/reload-weights.md``.

All HTTP control-plane calls use only ``requests`` — no torch / vLLM
on the client side.

Env vars (set by run_tests.sh):
  RELOAD_MODEL          (required) served model name
  RELOAD_MODEL_PATH     (required) filesystem path to the GOOD checkpoint
                                   (the path passed to reload_weights)
  RELOAD_BASE_URL       server URL (default: http://localhost:8000)
  RELOAD_DATA_DIR       bee eval data (default: tests/cohere/bee_eval_data)
  RELOAD_TASK           task to evaluate (default: ocrbench)
  RELOAD_MIN_SCORE      Phase 3 lower bound (default: 0.40)
  RELOAD_BROKEN_MAX_SCORE  Phase 1 upper bound (default: 0.20)
  RELOAD_MAX_SAMPLES    samples per task (default: 16)
  RELOAD_SERVER_LOG     optional path to server log; printed on failure
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from pathlib import Path
from typing import Literal

import pytest
import requests
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_bee_samples import TaskSummary, run_task


def _env(name: str, default: str | None = None, required: bool = False) -> str:
    val = os.environ.get(name, default)
    if required and not val:
        pytest.skip(f"{name} not set; run via run_tests.sh")
    assert val is not None
    return val


@pytest.fixture(scope="module")
def cfg():
    return {
        "model": _env("RELOAD_MODEL", required=True),
        "model_path": _env("RELOAD_MODEL_PATH", required=True),
        "base_url": _env("RELOAD_BASE_URL", "http://localhost:8000"),
        "data_dir": _env("RELOAD_DATA_DIR", "tests/cohere/bee_eval_data"),
        "task": _env("RELOAD_TASK", "ocrbench"),
        "min_score": float(_env("RELOAD_MIN_SCORE", "0.40")),
        "broken_max_score": float(_env("RELOAD_BROKEN_MAX_SCORE", "0.20")),
        "max_samples": int(_env("RELOAD_MAX_SAMPLES", "16")),
        "server_log": os.environ.get("RELOAD_SERVER_LOG", ""),
    }


def _dump_server_log(server_log: str, n: int = 80) -> None:
    """Print last n lines of the server log if available."""
    if not server_log:
        return
    try:
        with open(server_log) as f:
            lines = f.readlines()
    except OSError as exc:
        print(f"  (could not read server log {server_log}: {exc})")
        return
    print(f"\n=== Last {min(n, len(lines))} lines of server log ({server_log}) ===")
    print("".join(lines[-n:]))


def _run_task(cfg: dict) -> TaskSummary:
    client = AsyncOpenAI(base_url=f"{cfg['base_url']}/v1", api_key="not-needed")
    return asyncio.run(
        run_task(
            cfg["task"],
            cfg["data_dir"],
            client,
            cfg["model"],
            cfg["max_samples"],
            enable_thinking_budget=False,
        )
    )


# Wait timeout is intentionally short so the integration test can
# exercise the abort fallback by submitting a long-running generation
# before /pause; the engine won't drain within the window, mode=wait
# times out, and the helper falls back to mode=abort.
_PAUSE_WAIT_TIMEOUT = 2
_PAUSE_ABORT_TIMEOUT = 60
# reload_weights re-reads safetensors from disk and recapture_cudagraphs
# re-runs the full capture pipeline, both of which take real time.
_RPC_TIMEOUT = 600
# Sized so the injected request is still decoding when /pause?mode=wait
# times out, even on the fastest GPU profile we run on. On B200/GB200
# the c5-3a30t-class model decodes at ~1k tok/s single-batch, so 32k
# tokens ≈ 32 s of decode — comfortably above the 2 s wait window
# (plus the 1 s warm-up sleep before /pause), with ~10× margin to
# absorb future kernel/scheduler speedups.
_LOAD_INJECTION_MAX_TOKENS = 32768


def _post(
    base_url: str, path: str, *, timeout: float = _RPC_TIMEOUT, **kwargs
) -> requests.Response:
    resp = requests.post(f"{base_url}{path}", timeout=timeout, **kwargs)
    resp.raise_for_status()
    return resp


def _pause(base_url: str) -> Literal["wait", "abort"]:
    """Pause generation with a ``wait`` → ``abort`` fallback.

    ``mode=wait`` lets any in-flight requests drain cleanly so they
    don't show up as spurious aborts in the server log. If the engine
    doesn't drain within ``_PAUSE_WAIT_TIMEOUT`` (typically because a
    request is hung), fall back to ``mode=abort`` to force-finish the
    remaining requests and unblock the reload sequence — better than
    leaving the engine in a half-paused state for the full HTTP read
    window. There is no engine-side pause timeout; the server keeps
    awaiting the idle callback until either the engine drains or the
    client closes the connection.

    Returns ``"wait"`` if the engine drained in time, ``"abort"`` if
    the helper had to fall back. Callers can assert on the return
    value when they need to verify which path was taken (e.g. the
    integration test injects an in-flight load to force the fallback).
    """
    try:
        requests.post(
            f"{base_url}/pause",
            params={"mode": "wait"},
            timeout=_PAUSE_WAIT_TIMEOUT,
        ).raise_for_status()
        return "wait"
    except requests.exceptions.Timeout:
        print(
            f"  /pause?mode=wait did not drain within "
            f"{_PAUSE_WAIT_TIMEOUT}s — falling back to mode=abort"
        )

    _post(
        base_url,
        "/pause",
        params={"mode": "abort"},
        timeout=_PAUSE_ABORT_TIMEOUT,
    )
    return "abort"


def _start_long_generation(cfg: dict) -> threading.Thread:
    """Submit a long generation in a background thread.

    The request sits in the engine so that the next ``/pause?mode=wait``
    call must wait for it to drain — long enough to exceed
    ``_PAUSE_WAIT_TIMEOUT`` and exercise the abort fallback in
    ``_pause``. The fallback will issue ``/pause?mode=abort`` which
    marks the request ``FINISHED_ABORTED``; the background thread
    then exits with whatever response shape the server emits for
    aborted requests, which we don't depend on.
    """

    def worker() -> None:
        try:
            requests.post(
                f"{cfg['base_url']}/v1/chat/completions",
                json={
                    "model": cfg["model"],
                    "messages": [
                        {
                            "role": "user",
                            "content": "Write a very long detailed story.",
                        }
                    ],
                    "max_tokens": _LOAD_INJECTION_MAX_TOKENS,
                    "stream": False,
                },
                timeout=_RPC_TIMEOUT,
            )
        except Exception as exc:  # pragma: no cover - thread is fire-and-forget
            print(f"  load-injection request ended with: {exc!r}")

    thread = threading.Thread(target=worker, name="pause-fallback-load", daemon=True)
    thread.start()
    # Give the request time to reach the engine and start generating
    # so that /pause?mode=wait actually has work to wait on.
    time.sleep(1.0)
    return thread


def _reload_weights(base_url: str, model_path: str) -> Literal["wait", "abort"]:
    """Pause (wait for in-flight) → reload_weights → recapture graphs → resume.

    Quant ``process_weights_after_loading`` rebuilds backend-specific
    Python helper objects (e.g. ``moe_kernel`` / ``moe_quant_config`` on
    FP8 MoE methods) on every reload. CUDA graphs captured at startup
    bind to method addresses on the original objects, so the rebuild
    leaves the graphs pointing at freed memory and the next forward pass
    faults with an illegal memory access.

    Calling ``recapture_cudagraphs`` (a Cohere fork addition on the GPU
    worker) drops every wrapper's ``concrete_cudagraph_entries`` and
    re-runs ``capture_model`` against the freshly reloaded weights.
    Warmup inside ``_warmup_and_capture`` dispatches with
    ``CUDAGraphMode.NONE``, so the recapture path itself does not replay
    the stale graphs.

    See docs/cohere/code_notes/reload-weights.md for the full investigation.
    """
    pause_path = _pause(base_url)

    _post(
        base_url,
        "/collective_rpc",
        json={
            "method": "reload_weights",
            "kwargs": {"weights_path": model_path},
        },
    )

    # recapture_cudagraphs returns the total number of CUDAGraphEntry
    # instances that exist on CUDAGraphWrapper instances after recapture
    # (summed per worker). A zero result means the worker silently
    # bypassed the recapture path (enforce_eager=True or
    # cudagraph_mode=NONE) — in that case the test would still pass
    # (eager mode doesn't crash on stale graphs), but we wouldn't
    # actually be exercising the path under test. Assert non-zero on
    # every worker to catch misconfigurations that would render the
    # test meaningless.
    #
    # NOTE: we intentionally do NOT use ``capture_model()``'s return
    # value (a byte delta from ``torch.cuda.mem_get_info``) as the
    # signal here. CUDA graphs share a global memory pool that is not
    # released between captures, so on recapture the delta is often
    # ``0`` (or negative) even though graphs were successfully
    # recaptured — which would cause spurious failures with a
    # ``> 0`` assertion.
    resp = _post(
        base_url,
        "/collective_rpc",
        json={"method": "recapture_cudagraphs"},
    )
    results = resp.json().get("results", [])
    assert results and all(int(n) > 0 for n in results), (
        f"recapture_cudagraphs returned no-op (results={results}) — "
        f"server likely booted with --enforce-eager or "
        f"cudagraph_mode=NONE, so the CUDA-graph recapture path was "
        f"not actually exercised by this test"
    )

    _post(base_url, "/resume")
    return pause_path


def _print_phase(phase: int, label: str, cfg: dict) -> None:
    print(f"\n=== Phase {phase}: {label} (task={cfg['task']}) ===")


def _print_score(label: str, summary: TaskSummary, elapsed: float) -> None:
    print(
        f"  {label} score: {summary.avg_score:.3f} "
        f"({summary.passed}/{summary.total} passed, "
        f"{summary.errors} errors, {elapsed:.1f}s)"
    )


def test_reload_weights_fixes_broken_model(cfg):
    """Broken model boots low, reload_weights brings it back to passing."""
    # --- Phase 1: server booted from corrupted mirror → expect bad score ---
    _print_phase(1, "evaluating broken model", cfg)
    t0 = time.monotonic()
    before = _run_task(cfg)
    _print_score("broken", before, time.monotonic() - t0)

    assert not before.skipped, f"Task {cfg['task']} was skipped: {before.reason}"
    assert before.avg_score <= cfg["broken_max_score"], (
        f"Phase 1: avg_score={before.avg_score:.3f} > "
        f"broken_max={cfg['broken_max_score']:.3f} — the corruption did "
        f"not degrade the model enough; either the corruption script did "
        f"not run, or the model is too robust to the chosen perturbation"
    )

    # --- Phase 2: pause → reload from GOOD weights → recapture → resume ---
    # Inject an in-flight long generation so /pause?mode=wait must wait
    # on it, exceed _PAUSE_WAIT_TIMEOUT, and force _pause to fall back
    # to mode=abort. This validates the fallback path against the real
    # server every test run; without injection the engine would drain
    # immediately and the fallback path would never execute.
    _print_phase(2, f"reloading weights from {cfg['model_path']}", cfg)
    print("  injecting in-flight load to force /pause?mode=wait timeout")
    inflight = _start_long_generation(cfg)
    t0 = time.monotonic()
    try:
        pause_path = _reload_weights(cfg["base_url"], cfg["model_path"])
    except requests.RequestException as exc:
        print(f"  reload request failed: {exc}")
        _dump_server_log(cfg["server_log"])
        raise
    print(f"  reload completed in {time.monotonic() - t0:.1f}s")

    # The in-flight request was force-aborted by mode=abort; let the
    # background thread close out its HTTP connection.
    inflight.join(timeout=30)

    assert pause_path == "abort", (
        f"expected /pause?mode=wait to time out (forcing mode=abort) "
        f"given the injected in-flight generation, but pause completed "
        f"as {pause_path!r} — either the load-injection request didn't "
        f"reach the engine, or _PAUSE_WAIT_TIMEOUT="
        f"{_PAUSE_WAIT_TIMEOUT}s is too long for this hardware. The "
        f"fallback path was not actually exercised."
    )

    # --- Phase 3: after reload → score should be good ---
    _print_phase(3, "evaluating after reload", cfg)
    t0 = time.monotonic()
    after = _run_task(cfg)
    _print_score("fixed", after, time.monotonic() - t0)

    # If every request errored out, the engine likely crashed during reload —
    # dump the server log so the root cause is visible without re-running.
    if after.errors == after.total and after.total > 0:
        print("\n  All requests errored — engine likely crashed during reload.")
        _dump_server_log(cfg["server_log"])

    assert after.avg_score >= cfg["min_score"], (
        f"Phase 3: avg_score={after.avg_score:.3f} < "
        f"min={cfg['min_score']:.3f} "
        f"({after.passed}/{after.total} passed, {after.errors} errors) — "
        f"reload_weights did not restore the model"
    )

    delta = after.avg_score - before.avg_score
    print(
        f"\n=== PASSED: score delta={delta:+.3f} "
        f"(broken={before.avg_score:.3f}, fixed={after.avg_score:.3f}) ==="
    )
