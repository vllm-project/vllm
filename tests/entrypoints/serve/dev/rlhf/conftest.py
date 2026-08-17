# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Shared fixtures and helpers for the RL lifecycle test suite.

All test modules under this directory import from here to avoid duplication.

RFC: https://github.com/vllm-project/vllm/issues/45585
PR:  https://github.com/vllm-project/vllm/pull/45586
"""

import contextlib
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Model / server defaults
# ---------------------------------------------------------------------------


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


# Lightweight args for state-machine / protocol tests that don't need real
# weights (avoids spending time downloading a 1B checkpoint in T0 tests).
_DUMMY_ARGS = [
    "--dtype",
    "bfloat16",
    "--max-model-len",
    "128",
    "--max-num-seqs",
    "8",
    "--gpu-memory-utilization",
    "0.5",
    "--enable-sleep-mode",
    "--enforce-eager",
    "--load-format",
    "dummy",
]


# ---------------------------------------------------------------------------
# Server harness
# ---------------------------------------------------------------------------


@contextmanager
def server(
    extra_args=None,
    port: int = 8770,
    timeout: float = 180.0,
    dummy_weights: bool = False,
):
    """Launch a vLLM server with the dev router; yield its base URL.

    Args:
        extra_args:      Additional CLI flags appended after the base args.
        port:            HTTP port to bind (caller is responsible for uniqueness).
        timeout:         Seconds to wait for /health before giving up.
        dummy_weights:   If True, use --load-format dummy (fast, no real weights).
    """
    env = {**os.environ, "VLLM_SERVER_DEV_MODE": "1"}
    base = _DUMMY_ARGS if dummy_weights else _BASE_ARGS
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
        *(base + (extra_args or [])),
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
# Polling helper (200-lie workaround)
# ---------------------------------------------------------------------------


def poll_until(
    predicate: Callable[[], bool],
    timeout: float = 10.0,
    interval: float = 0.5,
) -> bool:
    """Poll predicate() until it returns True or timeout expires.

    Workaround for the vLLM sleep/wake "200-lie" — the HTTP endpoints may
    return 200 before the underlying operation is complete, so callers that
    need to verify state *after* an operation can use this helper instead of
    assuming the 200 means completion.

    Returns True if predicate became true within timeout, False otherwise.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if predicate():
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


# ---------------------------------------------------------------------------
# HTTP helpers — generation
# ---------------------------------------------------------------------------


def gen(url, prompt="The capital of France is", max_tokens=8, timeout=30):
    """Fire a /v1/completions request; return JSON or None on any error."""
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


def gen_with_logprobs(
    url, prompt="The capital of France is", max_tokens=8, logprobs=5, timeout=30
):
    """Fire a /v1/completions request with logprobs; return JSON or None."""
    try:
        r = requests.post(
            f"{url}/v1/completions",
            json={
                "model": "m",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0,
                "logprobs": logprobs,
            },
            timeout=timeout,
        )
        return r.json()
    except Exception:
        return None


def ok(resp) -> bool:
    """True iff resp is a successful completion (has choices, no error key)."""
    return (
        resp is not None
        and "choices" in resp
        and bool(resp["choices"])
        and "error" not in resp
    )


# ---------------------------------------------------------------------------
# HTTP helpers — stream generation
# ---------------------------------------------------------------------------


@dataclass
class StreamResult:
    started: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None
    error: Exception | None = None


def stream_completion(url: str, result: StreamResult, max_tokens: int) -> None:
    try:
        with requests.post(
            f"{url}/v1/completions",
            json={
                "model": "m",
                "prompt": "Count upward slowly: one, two, three,",
                "max_tokens": max_tokens,
                "temperature": 0,
                "ignore_eos": True,
                "stream": True,
            },
            stream=True,
            timeout=(5, 60),
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line or line == "data: [DONE]":
                    continue
                assert line.startswith("data: ")
                chunk = json.loads(line.removeprefix("data: "))
                result.chunks.append(chunk)
                choice = chunk["choices"][0]
                if choice.get("text"):
                    result.started.set()
                if choice.get("finish_reason") is not None:
                    result.finish_reason = choice["finish_reason"]
    except Exception as error:
        result.error = error
    finally:
        result.done.set()


def start_stream(url: str, max_tokens: int) -> tuple[StreamResult, threading.Thread]:
    result = StreamResult()
    thread = threading.Thread(
        target=stream_completion,
        args=(url, result, max_tokens),
    )
    thread.start()
    started = result.started.wait(timeout=10)
    if not started or result.done.is_set():
        pause(url, mode="abort")
        resume(url)
        thread.join(timeout=10)
    assert started, "request did not start generating"
    assert not result.done.is_set(), "request completed before it could be paused"
    return result, thread


# ---------------------------------------------------------------------------
# HTTP helpers — pause / resume
# ---------------------------------------------------------------------------


def pause(url, mode="abort", clear_cache=True):
    return requests.post(
        f"{url}/pause",
        params={"mode": mode, "clear_cache": clear_cache},
        timeout=15,
    ).status_code


def resume(url):
    return requests.post(f"{url}/resume", timeout=10).status_code


def completion_with_cache_details(url: str, prompt: str) -> dict[str, Any]:
    response = requests.post(
        f"{url}/v1/completions",
        json={
            "model": "m",
            "prompt": prompt,
            "max_tokens": 8,
            "temperature": 0,
            "logprobs": 1,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def golden_output(response: dict[str, Any]) -> dict[str, Any]:
    choice = response["choices"][0]
    usage = response["usage"]
    return {
        "text": choice["text"],
        "finish_reason": choice["finish_reason"],
        "tokens": choice["logprobs"]["tokens"],
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
    }


def cached_tokens(response: dict[str, Any]) -> int:
    return response["usage"]["prompt_tokens_details"]["cached_tokens"]


# ---------------------------------------------------------------------------
# HTTP helpers — sleep / wake
# ---------------------------------------------------------------------------


def sleep(url, level=1, mode="abort"):
    return requests.post(
        f"{url}/sleep", params={"level": level, "mode": mode}, timeout=15
    ).status_code


def wake(url, tags=None):
    params = {"tags": tags} if tags else {}
    return requests.post(f"{url}/wake_up", params=params, timeout=20).status_code


def is_sleeping(url) -> bool:
    return requests.get(f"{url}/is_sleeping", timeout=5).json()["is_sleeping"]


def is_paused(url) -> bool:
    return requests.get(f"{url}/is_paused", timeout=5).json()["is_paused"]


def health(url) -> int:
    try:
        return requests.get(f"{url}/health", timeout=5).status_code
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# HTTP helpers — weight transfer
# ---------------------------------------------------------------------------


def start_weight_update(url, is_checkpoint_format=True):
    return requests.post(
        f"{url}/start_weight_update",
        json={"is_checkpoint_format": is_checkpoint_format},
        timeout=10,
    )


def finish_weight_update(url):
    return requests.post(f"{url}/finish_weight_update", timeout=10)


def get_world_size(url, include_dp=True):
    return requests.get(
        f"{url}/get_world_size",
        params={"include_dp": include_dp},
        timeout=5,
    )


# ---------------------------------------------------------------------------
# GPU / metrics helpers
# ---------------------------------------------------------------------------


def gpu_free_bytes(device: int = 0) -> int:
    """Read GPU free bytes via subprocess to avoid import-time torch init."""
    out = subprocess.check_output(
        [
            sys.executable,
            "-c",
            f"import torch; f,_=torch.accelerator.get_memory_info({device}); print(f)",
        ],
        timeout=10,
    )
    return int(out.strip())


def sleep_metrics(url):
    """Return (awake, weights_offloaded, discard_all) from /metrics."""
    try:
        from prometheus_client.parser import text_string_to_metric_families
    except ImportError:
        return None, None, None

    r = requests.get(f"{url}/metrics", timeout=5)
    vals: dict = {}
    for family in text_string_to_metric_families(r.text):
        if family.name == "vllm:engine_sleep_state":
            for s in family.samples:
                vals[s.labels.get("sleep_state", "")] = s.value
    return vals.get("awake"), vals.get("weights_offloaded"), vals.get("discard_all")
