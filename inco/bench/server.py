# SPDX-License-Identifier: Apache-2.0
"""Talk to a running vLLM server: wait for readiness, capture and audit config.

The audit is the part that matters for a trustworthy baseline. A Pareto curve
measured against a server that silently fell back to eager mode is not a
baseline, so before any measurement we read the effective ``VllmConfig`` from
``/server_info`` (requires ``VLLM_SERVER_DEV_MODE=1``) and assert that the
standard perf features are actually on.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class ServerNotReady(RuntimeError):
    pass


def _get_json(url: str, timeout: float = 10.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode())


def _get_status(url: str, timeout: float = 10.0) -> int:
    """HTTP status only. `/health` answers 200 with an empty body, so parsing
    it as JSON would raise and be misread as "server not up yet" forever."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
        return resp.status


def _post(url: str, timeout: float = 60.0) -> int:
    req = urllib.request.Request(url, data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.status


def wait_for_server(
    base_url: str,
    timeout_s: float = 1800.0,
    interval_s: float = 5.0,
    sleep=time.sleep,
    clock=time.monotonic,
    getter=_get_status,
    on_wait=None,
) -> float:
    """Block until ``/health`` answers. Returns seconds waited.

    Args:
        on_wait: Called with seconds elapsed on every retry, so a caller can
            emit a heartbeat instead of waiting in silence.
    """
    started = clock()
    deadline = started + timeout_s
    while True:
        try:
            getter(f"{base_url.rstrip('/')}/health", timeout=5.0)
            return clock() - started
        except Exception:  # noqa: BLE001 - any failure means "not up yet"
            if clock() >= deadline:
                raise ServerNotReady(
                    f"{base_url} did not become healthy within {timeout_s:.0f}s"
                ) from None
            if on_wait is not None:
                on_wait(clock() - started)
            sleep(interval_s)


def fetch_server_info(base_url: str, getter=_get_json) -> dict[str, Any] | None:
    """Effective server config, or None if dev mode is off."""
    try:
        return getter(f"{base_url.rstrip('/')}/server_info?config_format=json")
    except Exception:  # noqa: BLE001
        return None


def reset_prefix_cache(base_url: str, poster=_post) -> bool:
    """Flush the prefix cache so each concurrency point starts cold."""
    try:
        poster(f"{base_url.rstrip('/')}/reset_prefix_cache")
        return True
    except Exception:  # noqa: BLE001
        return False


@dataclass(frozen=True)
class PerfFeatureAudit:
    """What the server is actually doing, extracted from /server_info."""

    cudagraph_mode: str | None
    enforce_eager: bool | None
    async_scheduling: bool | None
    prefix_caching: bool | None
    max_num_seqs: int | None
    max_num_batched_tokens: int | None
    tensor_parallel_size: int | None
    expert_parallel: bool | None
    dtype: str | None
    kv_cache_dtype: str | None
    attention_backend: str | None
    kv_cache_size_tokens: int | None
    kv_cache_max_concurrency: float | None
    gpu_model: str | None
    kv_cache_memory_bytes: int | None

    def problems(self, max_concurrency: int | None = None) -> list[str]:
        """Reasons this server is not a credible high-perf baseline.

        Args:
            max_concurrency: Highest client concurrency the sweep will drive.
                The engine must be able to batch that wide, otherwise the top
                of the Pareto curve measures the scheduler queue.
        """
        found = []
        if self.enforce_eager:
            found.append("enforce_eager=True: CUDA graphs are disabled")
        if _cudagraphs_disabled(self.cudagraph_mode):
            found.append(f"cudagraph_mode={self.cudagraph_mode}: no CUDA graphs")
        if self.async_scheduling is False:
            found.append(
                "async_scheduling=False: no CPU/GPU overlap (overlap scheduler off)"
            )
        if (
            max_concurrency is not None
            and self.max_num_seqs is not None
            and self.max_num_seqs < max_concurrency
        ):
            found.append(
                f"max_num_seqs={self.max_num_seqs} < sweep max concurrency "
                f"{max_concurrency}: the top of the curve will be queue-bound"
            )
        return found

    def servable_concurrency(self, seq_len: int) -> float | None:
        """Requests of ``seq_len`` tokens the KV cache can hold concurrently.

        Reported by the engine after it profiles free memory, so this is the
        real capacity rather than an estimate. None when the server did not
        report it (older vLLM, or dev mode off).
        """
        if not self.kv_cache_size_tokens or seq_len < 1:
            return None
        return self.kv_cache_size_tokens / seq_len


def _cudagraphs_disabled(mode: Any) -> bool:
    """Whether ``cudagraph_mode`` means "no graphs at all".

    The JSON form of vLLM's CUDAGraphMode is its enum *value*: an int for
    NONE/PIECEWISE/FULL (0/1/2), or a pair for the composite modes --
    FULL_AND_PIECEWISE serialises as ``[2, 1]``. A string comparison would
    therefore never match a disabled server and the gate would pass it.
    """
    if mode is None:
        return False
    if isinstance(mode, str):
        return mode.strip().upper() in ("NONE", "0")
    if isinstance(mode, (list, tuple)):
        return bool(mode) and all(_cudagraphs_disabled(part) for part in mode)
    return mode == 0


def _first_gpu_model(system_env: Any) -> str | None:
    """GPU name from ``/server_info``'s collect_env block.

    Worth recording with every run: a provider that advertises one accelerator
    can hand out variants with different memory (H100 80GB HBM3 vs H100 NVL),
    which silently changes KV capacity and makes two curves incomparable.
    """
    if not isinstance(system_env, dict):
        return None
    models = system_env.get("nvidia_gpu_models")
    if not isinstance(models, str) or not models.strip():
        return None
    first = models.strip().splitlines()[0]
    return first.split(": ", 1)[-1].strip() or None


def _dig(obj: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def audit_perf_features(server_info: dict[str, Any] | None) -> PerfFeatureAudit:
    """Pull the perf-relevant knobs out of a ``/server_info`` payload."""
    cfg = _dig(server_info or {}, "vllm_config")
    env = _dig(server_info or {}, "vllm_env")
    # /server_info returns vllm_config as a repr string unless config_format=json.
    cfg = cfg if isinstance(cfg, dict) else {}
    env = env if isinstance(env, dict) else {}
    compilation = cfg.get("compilation_config") or {}
    scheduler = cfg.get("scheduler_config") or {}
    cache = cfg.get("cache_config") or {}
    model = cfg.get("model_config") or {}
    parallel = cfg.get("parallel_config") or {}

    return PerfFeatureAudit(
        cudagraph_mode=compilation.get("cudagraph_mode"),
        enforce_eager=model.get("enforce_eager"),
        async_scheduling=scheduler.get("async_scheduling"),
        prefix_caching=cache.get("enable_prefix_caching"),
        max_num_seqs=scheduler.get("max_num_seqs"),
        max_num_batched_tokens=scheduler.get("max_num_batched_tokens"),
        tensor_parallel_size=parallel.get("tensor_parallel_size"),
        expert_parallel=parallel.get("enable_expert_parallel"),
        dtype=str(model.get("dtype")) if model.get("dtype") is not None else None,
        kv_cache_dtype=cache.get("cache_dtype"),
        attention_backend=env.get("VLLM_ATTENTION_BACKEND"),
        kv_cache_size_tokens=cache.get("kv_cache_size_tokens"),
        kv_cache_max_concurrency=cache.get("kv_cache_max_concurrency"),
        gpu_model=_first_gpu_model(_dig(server_info or {}, "system_env")),
        kv_cache_memory_bytes=cache.get("kv_cache_memory_bytes"),
    )
