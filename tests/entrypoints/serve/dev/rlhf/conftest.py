# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared HTTP helpers; coverage ownership is documented in __init__.py."""

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
    port: int | None = None,
    timeout: float = 180.0,
    dummy_weights: bool = False,
    *,
    model: str | None = None,
    env_dict: dict[str, str] | None = None,
    enable_sleep_mode: bool = True,
    enforce_eager: bool = True,
    weight_transfer_config: dict | None = None,
):
    """Yield a dev-server URL using the repository's process/port lifecycle.

    Args:
        extra_args: Additional vLLM CLI options.
        port: Optional explicit port; otherwise allocate an available port.
        timeout: Server startup timeout in seconds.
        dummy_weights: Skip checkpoint loading for protocol-only tests.
        model: Model to load; defaults to VLLM_TEST_MODEL.
        env_dict: Environment overrides passed only to the server.
        enable_sleep_mode: Allocate weights/cache through the sleep backend.
        enforce_eager: Disable graphs unless the scenario explicitly tests them.
        weight_transfer_config: Optional serialized transfer-backend config.
    """
    from tests.utils import RemoteOpenAIServer

    args = [
        arg
        for arg in (_DUMMY_ARGS if dummy_weights else _BASE_ARGS)
        if arg not in ("--enable-sleep-mode", "--enforce-eager")
    ]
    if enable_sleep_mode:
        args.append("--enable-sleep-mode")
    if enforce_eager:
        args.append("--enforce-eager")
    args += ["--served-model-name", "m", *(extra_args or [])]
    if port is not None:
        args += ["--port", str(port)]
    if weight_transfer_config is not None:
        args += ["--weight-transfer-config", json.dumps(weight_transfer_config)]
    with RemoteOpenAIServer(
        model or MODEL_NAME,
        args,
        auto_port=port is None,
        seed=None if "--seed" in args else 0,
        env_dict={"VLLM_SERVER_DEV_MODE": "1", **(env_dict or {})},
        max_wait_seconds=timeout,
    ) as remote:
        if not dummy_weights:
            gen(remote.url_root, max_tokens=4, timeout=120)
        yield remote.url_root


@contextmanager
def reusable_server(*args, **kwargs):
    """Reuse one server and clean the parent only after server shutdown.

    Callers use pytest.mark.skip_global_cleanup so function-scoped cleanup
    does not run while this longer-lived server still owns GPU resources.
    """
    try:
        with server(*args, **kwargs) as url:
            yield url
    finally:
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

        cleanup_dist_env_and_memory()


# ---------------------------------------------------------------------------
# Polling helper (200-lie workaround)
# ---------------------------------------------------------------------------


def poll_until(
    predicate: Callable[[], bool],
    timeout: float = 10.0,
    interval: float = 0.5,
) -> bool:
    """Wait for an asynchronous observation; propagate unexpected failures."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# ---------------------------------------------------------------------------
# HTTP helpers — generation
# ---------------------------------------------------------------------------


def gen(url, prompt="The capital of France is", max_tokens=8, timeout=30):
    """Generate a completion, propagating transport and server failures."""
    response = requests.post(
        f"{url}/v1/completions",
        json={
            "model": "m",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def gen_with_logprobs(
    url, prompt="The capital of France is", max_tokens=8, logprobs=5, timeout=30
):
    """Generate a completion, propagating transport and server failures."""
    response = requests.post(
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
    response.raise_for_status()
    return response.json()


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


# First-token wait for a streaming request; loaded machines need the slack.
STREAM_START_TIMEOUT = 20.0


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
    started = result.started.wait(timeout=STREAM_START_TIMEOUT)
    if not started or result.done.is_set():
        # Best-effort: on a stalled server these time out too, and would then
        # mask the assertions below.
        with contextlib.suppress(requests.RequestException):
            pause(url, mode="abort")
            resume(url)
        thread.join(timeout=10)
    assert started, (
        f"request did not start generating within {STREAM_START_TIMEOUT}s "
        f"(stream error: {result.error})"
    )
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


def sleep_response(url, level=1, mode="abort"):
    return requests.post(
        f"{url}/sleep", params={"level": level, "mode": mode}, timeout=15
    )


def sleep(url, level=1, mode="abort"):
    return sleep_response(url, level=level, mode=mode).status_code


def wake_response(url, tags=None):
    params = {"tags": tags} if tags else {}
    return requests.post(f"{url}/wake_up", params=params, timeout=20)


def wake(url, tags=None):
    return wake_response(url, tags=tags).status_code


def is_sleeping(url) -> bool:
    return requests.get(f"{url}/is_sleeping", timeout=5).json()["is_sleeping"]


def ensure_awake(url, timeout=30.0) -> None:
    response = wake_response(url)
    response.raise_for_status()
    assert poll_until(lambda: not is_sleeping(url), timeout=timeout), (
        f"engine did not return to the awake state: {response.text}"
    )


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


def weight_info_response(url):
    return requests.get(f"{url}/weight_info", timeout=5)


def update_weight_version_response(url, new_version):
    return requests.post(
        f"{url}/update_weight_version",
        json={"new_version": new_version},
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
