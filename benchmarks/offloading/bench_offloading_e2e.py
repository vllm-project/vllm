#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end recompute vs. OffloadingConnector benchmark client.

The benchmark deliberately uses Prometheus counter deltas to prove where every
prompt token came from. Run it against an otherwise idle vLLM server.
"""

from __future__ import annotations

import argparse
import http.client
import json
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
BENCHMARK_NAME = "vllm-offloading-connector-e2e"
CUDAGRAPH_RUNTIME_MODES = ("NONE", "PIECEWISE", "FULL")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {}
    mean = statistics.mean(values)
    return {
        "count": len(values),
        "mean": mean,
        "median": statistics.median(values),
        "p95": (
            statistics.quantiles(values, n=20, method="inclusive")[18]
            if len(values) > 1
            else values[0]
        ),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.pstdev(values),
        "cv": statistics.pstdev(values) / mean if mean else 0.0,
    }


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def metric_value(
    metrics: str,
    name: str,
    engine: int,
    labels: tuple[str, ...] = (),
) -> float:
    engine_label = f'engine="{engine}"'
    for line in metrics.splitlines():
        if not line.startswith(name + "{") or engine_label not in line:
            continue
        if all(label in line for label in labels):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


def metric_delta(
    before: str,
    after: str,
    name: str,
    engine: int,
    *labels: str,
) -> float:
    return metric_value(after, name, engine, labels) - metric_value(
        before, name, engine, labels
    )


def cudagraph_dispatch_metrics(
    before: str,
    after: str,
    engine: int,
) -> dict[str, Any]:
    metric_prefix = "vllm:cudagraph_dispatch_total{"
    available = any(
        line.startswith(metric_prefix)
        for metrics in (before, after)
        for line in metrics.splitlines()
    )
    counts = {
        mode: int(
            metric_delta(
                before,
                after,
                "vllm:cudagraph_dispatch_total",
                engine,
                f'runtime_mode="{mode}"',
            )
        )
        for mode in CUDAGRAPH_RUNTIME_MODES
    }
    active_modes = [mode for mode, count in counts.items() if count]
    runtime_mode = (
        active_modes[0] if len(active_modes) == 1 else "MIXED" if active_modes else None
    )
    return {
        "cudagraph_metrics_available": available,
        "cudagraph_runtime_mode_counts": counts,
        "cudagraph_runtime_mode": runtime_mode,
        "cudagraph_used": (
            any(mode != "NONE" for mode in active_modes) if active_modes else None
        ),
    }


class VllmClient:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        model: str,
        engine: int,
        settle_seconds: float,
        request_timeout: float,
        metrics_timeout: float,
        require_cudagraph_metrics: bool = False,
    ) -> None:
        self.host = host
        self.port = port
        self.model = model
        self.engine = engine
        self.settle_seconds = settle_seconds
        self.request_timeout = request_timeout
        self.metrics_timeout = metrics_timeout
        self.require_cudagraph_metrics = require_cudagraph_metrics

    def get(self, path: str, timeout: float) -> str:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        conn.request("GET", path)
        response = conn.getresponse()
        body = response.read().decode()
        conn.close()
        if response.status != 200:
            raise RuntimeError(f"GET {path} returned HTTP {response.status}: {body}")
        return body

    def check_server(self) -> dict[str, Any]:
        self.get("/health", self.metrics_timeout)
        metrics = self.get("/metrics", self.metrics_timeout)
        required = (
            "vllm:prompt_tokens_by_source_total",
            "vllm:time_to_first_token_seconds",
            "vllm:request_prefill_time_seconds",
        )
        if self.require_cudagraph_metrics:
            required += ("vllm:cudagraph_dispatch_total",)
        missing = [name for name in required if name not in metrics]
        if missing:
            raise RuntimeError(f"server metrics are missing: {', '.join(missing)}")

        version = json.loads(self.get("/version", self.metrics_timeout)).get("version")
        models = json.loads(self.get("/v1/models", self.metrics_timeout)).get(
            "data", []
        )
        model = next((item for item in models if item.get("id") == self.model), None)
        if model is None:
            available = [item.get("id") for item in models]
            raise RuntimeError(
                f"model {self.model!r} is not served; available models: {available}"
            )
        return {
            "version": version,
            "model": {
                "id": model.get("id"),
                "root": model.get("root"),
                "owned_by": model.get("owned_by"),
                "max_model_len": model.get("max_model_len"),
            },
        }

    def measure_completion(
        self,
        *,
        prompt_token: int,
        num_tokens: int,
        cache_salt: str,
    ) -> dict[str, Any]:
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": [prompt_token] * num_tokens,
                "max_tokens": 1,
                "temperature": 0,
                "add_special_tokens": False,
                "return_token_ids": True,
                "cache_salt": cache_salt,
            },
            separators=(",", ":"),
        )
        headers = {
            "Content-Type": "application/json",
            "X-data-parallel-rank": str(self.engine),
        }

        before = self.get("/metrics", self.metrics_timeout)
        conn = http.client.HTTPConnection(
            self.host, self.port, timeout=self.request_timeout
        )
        start = time.perf_counter()
        conn.request("POST", "/v1/completions", body=payload, headers=headers)
        response = conn.getresponse()
        body = response.read().decode()
        client_seconds = time.perf_counter() - start
        conn.close()
        if response.status != 200:
            raise RuntimeError(f"completion returned HTTP {response.status}: {body}")

        if self.settle_seconds:
            time.sleep(self.settle_seconds)
        after = self.get("/metrics", self.metrics_timeout)
        result = json.loads(body)
        actual_tokens = int(result["usage"]["prompt_tokens"])
        if actual_tokens != num_tokens:
            raise RuntimeError(
                f"expected {num_tokens} prompt tokens, got {actual_tokens}"
            )

        choice = result["choices"][0]
        cudagraph = cudagraph_dispatch_metrics(before, after, self.engine)
        if self.require_cudagraph_metrics and not any(
            cudagraph["cudagraph_runtime_mode_counts"].values()
        ):
            raise RuntimeError(
                "request produced no vllm:cudagraph_dispatch_total delta; "
                "start the server with --cudagraph-metrics"
            )
        return {
            "cache_salt": cache_salt,
            "prompt_token": prompt_token,
            "prompt_tokens": actual_tokens,
            "client_e2e_seconds": client_seconds,
            "server_ttft_seconds": metric_delta(
                before,
                after,
                "vllm:time_to_first_token_seconds_sum",
                self.engine,
            ),
            "server_prefill_seconds": metric_delta(
                before,
                after,
                "vllm:request_prefill_time_seconds_sum",
                self.engine,
            ),
            "local_compute_tokens": metric_delta(
                before,
                after,
                "vllm:prompt_tokens_by_source_total",
                self.engine,
                'source="local_compute"',
            ),
            "local_cache_hit_tokens": metric_delta(
                before,
                after,
                "vllm:prompt_tokens_by_source_total",
                self.engine,
                'source="local_cache_hit"',
            ),
            "external_kv_tokens": metric_delta(
                before,
                after,
                "vllm:prompt_tokens_by_source_total",
                self.engine,
                'source="external_kv_transfer"',
            ),
            "cpu_to_gpu_bytes": metric_delta(
                before,
                after,
                "vllm:kv_offload_total_bytes_total",
                self.engine,
                'transfer_type="CPU_to_GPU"',
            ),
            "gpu_to_cpu_bytes": metric_delta(
                before,
                after,
                "vllm:kv_offload_total_bytes_total",
                self.engine,
                'transfer_type="GPU_to_CPU"',
            ),
            "output_token_ids": list(choice.get("token_ids") or []),
            "output_text": choice.get("text", ""),
            "request_id": result.get("id"),
            **cudagraph,
        }

    def measure_decode_store(
        self,
        *,
        prompt_token: int,
        decode_tokens: int,
        cache_salt: str,
    ) -> dict[str, Any]:
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": [prompt_token],
                "max_tokens": decode_tokens,
                "temperature": 0,
                "ignore_eos": True,
                "add_special_tokens": False,
                "return_token_ids": True,
                "cache_salt": cache_salt,
            },
            separators=(",", ":"),
        )
        headers = {
            "Content-Type": "application/json",
            "X-data-parallel-rank": str(self.engine),
        }

        before = self.get("/metrics", self.metrics_timeout)
        conn = http.client.HTTPConnection(
            self.host, self.port, timeout=self.request_timeout
        )
        started = time.perf_counter()
        conn.request("POST", "/v1/completions", body=payload, headers=headers)
        response = conn.getresponse()
        body = response.read().decode()
        client_seconds = time.perf_counter() - started
        conn.close()
        if response.status != 200:
            raise RuntimeError(f"completion returned HTTP {response.status}: {body}")

        if self.settle_seconds:
            time.sleep(self.settle_seconds)
        after = self.get("/metrics", self.metrics_timeout)
        result = json.loads(body)
        usage = result["usage"]
        cudagraph = cudagraph_dispatch_metrics(before, after, self.engine)
        if self.require_cudagraph_metrics and not any(
            cudagraph["cudagraph_runtime_mode_counts"].values()
        ):
            raise RuntimeError(
                "request produced no vllm:cudagraph_dispatch_total delta; "
                "start the server with --cudagraph-metrics"
            )
        return {
            "cache_salt": cache_salt,
            "prompt_tokens": int(usage["prompt_tokens"]),
            "completion_tokens": int(usage["completion_tokens"]),
            "client_e2e_seconds": client_seconds,
            "gpu_to_cpu_operations": metric_delta(
                before,
                after,
                "vllm:kv_offload_size_count",
                self.engine,
                'transfer_type="GPU_to_CPU"',
            ),
            "gpu_to_cpu_bytes": metric_delta(
                before,
                after,
                "vllm:kv_offload_total_bytes_total",
                self.engine,
                'transfer_type="GPU_to_CPU"',
            ),
            "request_id": result.get("id"),
            **cudagraph,
        }

    def measure_decode_store_lifecycle(
        self,
        *,
        prompt_token: int,
        decode_tokens: int,
        cache_salt: str,
        drain_prompt_token: int,
    ) -> dict[str, Any]:
        """Measure a decode store through its deferred transfer completion."""
        before = self.get("/metrics", self.metrics_timeout)
        measurement = self.measure_decode_store(
            prompt_token=prompt_token,
            decode_tokens=decode_tokens,
            cache_salt=cache_salt,
        )
        drain = self.measure_decode_store(
            prompt_token=drain_prompt_token,
            decode_tokens=1,
            cache_salt=f"{cache_salt}-drain-{uuid.uuid4().hex}",
        )
        # The drain request supplies the next engine step needed to report a
        # store submitted at the end of the measured request.
        time.sleep(max(1.0, self.settle_seconds))
        after = self.get("/metrics", self.metrics_timeout)
        measurement["gpu_to_cpu_operations"] = metric_delta(
            before,
            after,
            "vllm:kv_offload_size_count",
            self.engine,
            'transfer_type="GPU_to_CPU"',
        )
        measurement["gpu_to_cpu_bytes"] = metric_delta(
            before,
            after,
            "vllm:kv_offload_total_bytes_total",
            self.engine,
            'transfer_type="GPU_to_CPU"',
        )
        measurement["metrics_drain_request_id"] = drain["request_id"]
        return measurement


def output_signature(measurement: dict[str, Any]) -> tuple[tuple[int, ...], str]:
    return (
        tuple(int(token) for token in measurement["output_token_ids"]),
        str(measurement["output_text"]),
    )


def validate_cold(measurement: dict[str, Any], tokens: int) -> list[str]:
    errors = []
    expected = {
        "prompt_tokens": tokens,
        "local_compute_tokens": float(tokens),
        "local_cache_hit_tokens": 0.0,
        "external_kv_tokens": 0.0,
    }
    for field, value in expected.items():
        if measurement[field] != value:
            errors.append(f"{field}: expected {value}, got {measurement[field]}")
    return errors


def expected_replay_sources(tokens: int, block_size: int) -> tuple[int, int]:
    external = tokens - tokens % block_size
    local_compute = tokens - external
    if local_compute == 0:
        # vLLM recomputes the final token of a fully cached request for logits.
        local_compute = 1
    return external, local_compute


def validate_replay(
    measurement: dict[str, Any],
    *,
    tokens: int,
    block_size: int,
    require_transfer_bytes: bool,
) -> list[str]:
    expected_external, expected_compute = expected_replay_sources(tokens, block_size)
    errors = []
    expected = {
        "prompt_tokens": tokens,
        "local_compute_tokens": float(expected_compute),
        "local_cache_hit_tokens": 0.0,
        "external_kv_tokens": float(expected_external),
    }
    for field, value in expected.items():
        if measurement[field] != value:
            errors.append(f"{field}: expected {value}, got {measurement[field]}")
    if require_transfer_bytes and measurement["cpu_to_gpu_bytes"] <= 0:
        errors.append(
            "cpu_to_gpu_bytes: expected a positive transfer counter delta, "
            f"got {measurement['cpu_to_gpu_bytes']}"
        )
    return errors


def new_salt(run_id: str, tokens: int, attempt: int, role: str) -> str:
    return f"{run_id}-{tokens}-{attempt}-{role}-{uuid.uuid4().hex}"


def run_recompute_attempt(
    client: VllmClient,
    *,
    run_id: str,
    tokens: int,
    attempt: int,
    prompt_token: int,
) -> dict[str, Any]:
    measurement = client.measure_completion(
        prompt_token=prompt_token,
        num_tokens=tokens,
        cache_salt=new_salt(run_id, tokens, attempt, "recompute"),
    )
    errors = validate_cold(measurement, tokens)
    return {
        "attempt": attempt,
        "timestamp": utc_now(),
        "valid": not errors,
        "validation_errors": errors,
        "request": measurement,
    }


def run_offload_attempt(
    client: VllmClient,
    *,
    run_id: str,
    tokens: int,
    attempt: int,
    prompt_token: int,
    eviction_prompt_token: int,
    eviction_plan: list[int],
    block_size: int,
    require_transfer_bytes: bool,
) -> dict[str, Any]:
    target_salt = new_salt(run_id, tokens, attempt, "target")
    cold = client.measure_completion(
        prompt_token=prompt_token,
        num_tokens=tokens,
        cache_salt=target_salt,
    )

    evictions = []
    for index, eviction_tokens in enumerate(eviction_plan):
        evictions.append(
            client.measure_completion(
                prompt_token=eviction_prompt_token + index,
                num_tokens=eviction_tokens,
                cache_salt=new_salt(run_id, tokens, attempt, f"eviction-{index}"),
            )
        )

    replay = client.measure_completion(
        prompt_token=prompt_token,
        num_tokens=tokens,
        cache_salt=target_salt,
    )

    errors = [f"cold: {error}" for error in validate_cold(cold, tokens)]
    for index, (measurement, eviction_tokens) in enumerate(
        zip(evictions, eviction_plan, strict=True)
    ):
        errors.extend(
            f"eviction[{index}]: {error}"
            for error in validate_cold(measurement, eviction_tokens)
        )
    errors.extend(
        f"replay: {error}"
        for error in validate_replay(
            replay,
            tokens=tokens,
            block_size=block_size,
            require_transfer_bytes=require_transfer_bytes,
        )
    )
    output_matches = output_signature(cold) == output_signature(replay)
    if not output_matches:
        errors.append("cold and replay outputs differ")

    return {
        "attempt": attempt,
        "timestamp": utc_now(),
        "valid": not errors,
        "validation_errors": errors,
        "output_matches": output_matches,
        "cold": cold,
        "evictions": evictions,
        "replay": replay,
    }


def validate_decode_block_stores(
    calibration: dict[str, Any], multi_block: dict[str, Any], blocks: int
) -> list[str]:
    errors = []
    for label, measurement in (
        ("calibration", calibration),
        ("multi_block", multi_block),
    ):
        if measurement["prompt_tokens"] != 1:
            errors.append(
                f"{label}.prompt_tokens: expected 1, got {measurement['prompt_tokens']}"
            )
        if measurement["gpu_to_cpu_operations"] <= 0:
            errors.append(
                f"{label}.gpu_to_cpu_operations: expected positive delta, "
                f"got {measurement['gpu_to_cpu_operations']}"
            )
        if measurement["gpu_to_cpu_bytes"] <= 0:
            errors.append(
                f"{label}.gpu_to_cpu_bytes: expected positive delta, "
                f"got {measurement['gpu_to_cpu_bytes']}"
            )

    for field in ("gpu_to_cpu_operations", "gpu_to_cpu_bytes"):
        expected = calibration[field] * blocks
        if multi_block[field] != expected:
            errors.append(
                f"multi_block.{field}: expected {expected} "
                f"({blocks}x calibration), got {multi_block[field]}"
            )
    return errors


def run_decode_block_verification(
    client: VllmClient,
    *,
    run_id: str,
    prompt_token: int,
    block_size: int,
    blocks: int,
    tail_tokens: int = 1,
) -> dict[str, Any]:
    calibration = client.measure_decode_store_lifecycle(
        prompt_token=prompt_token,
        decode_tokens=block_size + tail_tokens,
        cache_salt=new_salt(run_id, block_size, 0, "decode-calibration"),
        drain_prompt_token=prompt_token + 2,
    )
    multi_block = client.measure_decode_store_lifecycle(
        prompt_token=prompt_token + 1,
        decode_tokens=block_size * blocks + tail_tokens,
        cache_salt=new_salt(run_id, block_size * blocks, 0, "decode-multi-block"),
        drain_prompt_token=prompt_token + 3,
    )
    errors = []
    if calibration["completion_tokens"] != block_size + tail_tokens:
        errors.append(
            "calibration.completion_tokens: "
            f"expected {block_size + tail_tokens}, "
            f"got {calibration['completion_tokens']}"
        )
    if multi_block["completion_tokens"] != block_size * blocks + tail_tokens:
        errors.append(
            "multi_block.completion_tokens: "
            f"expected {block_size * blocks + tail_tokens}, "
            f"got {multi_block['completion_tokens']}"
        )
    errors.extend(validate_decode_block_stores(calibration, multi_block, blocks))
    return {
        "timestamp": utc_now(),
        "valid": not errors,
        "validation_errors": errors,
        "block_size": block_size,
        "blocks": blocks,
        "tail_tokens": tail_tokens,
        "calibration": calibration,
        "multi_block": multi_block,
    }


def primary_measurement(mode: str, attempt: dict[str, Any]) -> dict[str, Any]:
    return attempt["request"] if mode == "recompute" else attempt["replay"]


def refresh_result(mode: str, result: dict[str, Any]) -> None:
    valid = [attempt for attempt in result["attempts"] if attempt["valid"]]
    result["valid_samples"] = len(valid)
    result["total_attempts"] = len(result["attempts"])
    if not valid:
        result["summary"] = {}
        return

    measurements = [primary_measurement(mode, attempt) for attempt in valid]
    result["summary"] = {
        field: summarize([float(measurement[field]) for measurement in measurements])
        for field in (
            "client_e2e_seconds",
            "server_ttft_seconds",
            "server_prefill_seconds",
            "cpu_to_gpu_bytes",
        )
    }
    if all("cudagraph_runtime_mode_counts" in item for item in measurements):
        observed = [
            item
            for item in measurements
            if item.get("cudagraph_metrics_available")
            and item.get("cudagraph_runtime_mode") is not None
        ]
        runtime_mode_counts = {
            mode: sum(
                int(item["cudagraph_runtime_mode_counts"][mode]) for item in observed
            )
            for mode in CUDAGRAPH_RUNTIME_MODES
        }
        result["cudagraph_summary"] = {
            "runtime_mode_counts": runtime_mode_counts,
            "used_samples": sum(
                1 for item in observed if item["cudagraph_used"] is True
            ),
            "not_used_samples": sum(
                1 for item in observed if item["cudagraph_used"] is False
            ),
            "unavailable_samples": len(measurements) - len(observed),
        }
    else:
        result.pop("cudagraph_summary", None)
    if mode == "offload":
        result["summary"]["cold_client_e2e_seconds"] = summarize(
            [float(attempt["cold"]["client_e2e_seconds"]) for attempt in valid]
        )


def build_config(args: argparse.Namespace, eviction_plan: list[int]) -> dict[str, Any]:
    return {
        "host": args.host,
        "port": args.port,
        "model": args.model,
        "engine": args.engine,
        "sizes": args.sizes,
        "repeats": args.repeats,
        "max_attempts": args.max_attempts,
        "prompt_token": args.prompt_token,
        "eviction_prompt_token": args.eviction_prompt_token,
        "eviction_plan": eviction_plan,
        "block_size": args.block_size,
        "settle_seconds": args.settle_seconds,
        "request_timeout": args.request_timeout,
        "metrics_timeout": args.metrics_timeout,
        "require_transfer_bytes": args.require_transfer_bytes,
        "require_cudagraph_metrics": args.require_cudagraph_metrics,
        "verify_decode_blocks": args.verify_decode_blocks,
        "verify_decode_tail_tokens": args.verify_decode_tail_tokens,
        "max_tokens": 1,
        "notes": args.notes,
    }


def resume_config_errors(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    mutable = {
        "repeats",
        "max_attempts",
        "notes",
        "verify_decode_blocks",
        "verify_decode_tail_tokens",
    }
    keys = (set(old) | set(new)) - mutable
    return [
        f"{key}: existing={old.get(key)!r}, requested={new.get(key)!r}"
        for key in sorted(keys)
        if old.get(key) != new.get(key)
    ]


def load_or_create_run(
    args: argparse.Namespace,
    output_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    if args.resume:
        if not output_path.exists():
            raise RuntimeError(f"cannot resume missing output file: {output_path}")
        data = json.loads(output_path.read_text())
        if data.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError("cannot resume an output with a different schema")
        if data.get("mode") != args.mode:
            raise RuntimeError(
                f"cannot resume mode {data.get('mode')!r} as {args.mode!r}"
            )
        errors = resume_config_errors(data["config"], config)
        if errors:
            raise RuntimeError("resume configuration mismatch:\n" + "\n".join(errors))
        data["config"].update(
            repeats=args.repeats,
            max_attempts=args.max_attempts,
            notes=args.notes,
            verify_decode_blocks=args.verify_decode_blocks,
            verify_decode_tail_tokens=args.verify_decode_tail_tokens,
        )
        data["status"] = "running"
        data.pop("error", None)
        return data

    if output_path.exists() and not args.overwrite:
        raise RuntimeError(
            f"output already exists: {output_path}; use --overwrite or --resume"
        )
    run_id = args.run_id or uuid.uuid4().hex
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "mode": args.mode,
        "status": "running",
        "run_id": run_id,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "config": config,
        "results": [
            {
                "tokens": tokens,
                "valid_samples": 0,
                "total_attempts": 0,
                "attempts": [],
                "summary": {},
            }
            for tokens in args.sizes
        ],
    }


def checkpoint(path: Path, data: dict[str, Any]) -> None:
    data["updated_at"] = utc_now()
    atomic_write_json(path, data)


def validate_server_metadata(
    existing: dict[str, Any] | None, current: dict[str, Any]
) -> None:
    if existing is not None and existing != current:
        raise RuntimeError(
            "server metadata changed since the original run: "
            f"existing={existing!r}, current={current!r}"
        )


def decode_verification_needs_refresh(
    existing: dict[str, Any] | None,
    *,
    blocks: int,
    block_size: int,
    tail_tokens: int,
) -> bool:
    return (
        not existing
        or not existing.get("valid")
        or existing.get("blocks") != blocks
        or existing.get("block_size") != block_size
        or existing.get("tail_tokens") != tail_tokens
    )


def concise_attempt(mode: str, tokens: int, attempt: dict[str, Any]) -> dict[str, Any]:
    measurement = primary_measurement(mode, attempt)
    return {
        "mode": mode,
        "tokens": tokens,
        "attempt": attempt["attempt"],
        "valid": attempt["valid"],
        "validation_errors": attempt["validation_errors"],
        "client_e2e_seconds": measurement.get("client_e2e_seconds"),
        "server_ttft_seconds": measurement.get("server_ttft_seconds"),
        "local_compute_tokens": measurement.get("local_compute_tokens"),
        "local_cache_hit_tokens": measurement.get("local_cache_hit_tokens"),
        "external_kv_tokens": measurement.get("external_kv_tokens"),
        "cpu_to_gpu_bytes": measurement.get("cpu_to_gpu_bytes"),
        "cudagraph_metrics_available": measurement.get("cudagraph_metrics_available"),
        "cudagraph_runtime_mode": measurement.get("cudagraph_runtime_mode"),
        "cudagraph_used": measurement.get("cudagraph_used"),
    }


def make_parser(
    *,
    default_mode: str,
    default_sizes: list[int],
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark cold recompute or CPU OffloadingConnector replay with "
            "strict Prometheus metric validation."
        )
    )
    parser.add_argument(
        "--mode", choices=("recompute", "offload"), default=default_mode
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--model", default="M1-0710")
    parser.add_argument("--engine", type=int, default=0)
    parser.add_argument("--sizes", type=int, nargs="+", default=default_sizes)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=30)
    parser.add_argument("--prompt-token", type=int, default=31318)
    parser.add_argument("--eviction-prompt-token", type=int, default=31319)
    parser.add_argument("--eviction-tokens", type=int, default=12000)
    parser.add_argument("--eviction-requests", type=int, default=2)
    parser.add_argument(
        "--eviction-token-counts",
        type=int,
        nargs="+",
        help="Explicit per-request eviction plan; overrides the two legacy options.",
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--verify-decode-blocks",
        type=int,
        default=4,
        help=(
            "In offload mode, verify incremental decode stores against a "
            "one-block calibration; use 0 to disable."
        ),
    )
    parser.add_argument(
        "--verify-decode-tail-tokens",
        type=int,
        default=1,
        help=(
            "Tokens generated past each full-block boundary during decode-store "
            "verification. Use num_speculative_tokens + 1 for MTP/EAGLE."
        ),
    )
    parser.add_argument("--settle-seconds", type=float, default=0.2)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument("--metrics-timeout", type=float, default=30)
    parser.add_argument(
        "--require-transfer-bytes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-cudagraph-metrics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Require the optional per-request CUDA graph runtime-mode Prometheus "
            "counter. Standard vLLM builds do not export this counter."
        ),
    )
    parser.add_argument("--run-id")
    parser.add_argument("--notes", default="")
    parser.add_argument("--output", required=True)
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--overwrite", action="store_true")
    output_group.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.sizes or any(tokens <= 0 for tokens in args.sizes):
        raise RuntimeError("all --sizes values must be positive")
    if len(set(args.sizes)) != len(args.sizes):
        raise RuntimeError("--sizes must not contain duplicates")
    if args.repeats <= 0 or args.max_attempts < args.repeats:
        raise RuntimeError("require 0 < repeats <= max-attempts")
    if args.block_size <= 0:
        raise RuntimeError("--block-size must be positive")
    if args.verify_decode_blocks < 0 or args.verify_decode_blocks == 1:
        raise RuntimeError("--verify-decode-blocks must be 0 or at least 2")
    if not 0 < args.verify_decode_tail_tokens < args.block_size:
        raise RuntimeError(
            "--verify-decode-tail-tokens must be positive and less than block size"
        )
    if args.settle_seconds < 0:
        raise RuntimeError("--settle-seconds must not be negative")
    if args.mode == "offload":
        plan = (
            args.eviction_token_counts
            or [args.eviction_tokens] * args.eviction_requests
        )
        if not plan or any(tokens <= 0 for tokens in plan):
            raise RuntimeError("offload mode requires a positive eviction plan")


def main(
    argv: list[str] | None = None,
    *,
    default_mode: str = "offload",
    default_sizes: list[int] | None = None,
) -> int:
    parser = make_parser(
        default_mode=default_mode,
        default_sizes=default_sizes or [256, 512, 1024, 2048, 4096, 8192],
    )
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        eviction_plan = (
            args.eviction_token_counts
            if args.eviction_token_counts is not None
            else [args.eviction_tokens] * args.eviction_requests
        )
        config = build_config(args, eviction_plan)
        output_path = Path(args.output)
        data = load_or_create_run(args, output_path, config)
        checkpoint(output_path, data)

        client = VllmClient(
            host=args.host,
            port=args.port,
            model=args.model,
            engine=args.engine,
            settle_seconds=args.settle_seconds,
            request_timeout=args.request_timeout,
            metrics_timeout=args.metrics_timeout,
            require_cudagraph_metrics=args.require_cudagraph_metrics,
        )
        server_metadata = client.check_server()
        validate_server_metadata(data.get("server"), server_metadata)
        data["server"] = server_metadata
        checkpoint(output_path, data)

        if args.mode == "offload":
            if not args.verify_decode_blocks:
                data.pop("decode_block_verification", None)
                checkpoint(output_path, data)
            else:
                existing_verification = data.get("decode_block_verification")
                if decode_verification_needs_refresh(
                    existing_verification,
                    blocks=args.verify_decode_blocks,
                    block_size=args.block_size,
                    tail_tokens=args.verify_decode_tail_tokens,
                ):
                    verification = run_decode_block_verification(
                        client,
                        run_id=data["run_id"],
                        prompt_token=(
                            args.eviction_prompt_token + len(eviction_plan) + 1
                        ),
                        block_size=args.block_size,
                        blocks=args.verify_decode_blocks,
                        tail_tokens=args.verify_decode_tail_tokens,
                    )
                    data["decode_block_verification"] = verification
                    checkpoint(output_path, data)
                    if not args.quiet:
                        print(
                            json.dumps(
                                {"decode_block_verification": verification},
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                    if not verification["valid"]:
                        raise RuntimeError(
                            "decode block verification failed: "
                            + "; ".join(verification["validation_errors"])
                        )

        for result in data["results"]:
            tokens = int(result["tokens"])
            refresh_result(args.mode, result)
            while (
                result["valid_samples"] < args.repeats
                and result["total_attempts"] < args.max_attempts
            ):
                attempt_number = result["total_attempts"] + 1
                try:
                    if args.mode == "recompute":
                        attempt = run_recompute_attempt(
                            client,
                            run_id=data["run_id"],
                            tokens=tokens,
                            attempt=attempt_number,
                            prompt_token=args.prompt_token,
                        )
                    else:
                        attempt = run_offload_attempt(
                            client,
                            run_id=data["run_id"],
                            tokens=tokens,
                            attempt=attempt_number,
                            prompt_token=args.prompt_token,
                            eviction_prompt_token=args.eviction_prompt_token,
                            eviction_plan=eviction_plan,
                            block_size=args.block_size,
                            require_transfer_bytes=args.require_transfer_bytes,
                        )
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    attempt = {
                        "attempt": attempt_number,
                        "timestamp": utc_now(),
                        "valid": False,
                        "validation_errors": [f"{type(exc).__name__}: {exc}"],
                    }

                result["attempts"].append(attempt)
                refresh_result(args.mode, result)
                checkpoint(output_path, data)
                if not args.quiet:
                    if "request" in attempt or "replay" in attempt:
                        message = concise_attempt(args.mode, tokens, attempt)
                    else:
                        message = {
                            "mode": args.mode,
                            "tokens": tokens,
                            "attempt": attempt_number,
                            "valid": False,
                            "validation_errors": attempt["validation_errors"],
                        }
                    print(json.dumps(message, sort_keys=True), flush=True)

            if result["valid_samples"] < args.repeats:
                raise RuntimeError(
                    f"collected {result['valid_samples']} valid samples for "
                    f"{tokens} tokens after {result['total_attempts']} attempts"
                )

        data["status"] = "completed"
        data.pop("error", None)
        checkpoint(output_path, data)
        print(
            json.dumps(
                {
                    "status": data["status"],
                    "mode": data["mode"],
                    "output": str(output_path),
                    "results": [
                        {
                            "tokens": result["tokens"],
                            "valid_samples": result["valid_samples"],
                            "total_attempts": result["total_attempts"],
                            "summary": result["summary"],
                        }
                        for result in data["results"]
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except KeyboardInterrupt:
        if "data" in locals() and "output_path" in locals():
            data["status"] = "interrupted"
            checkpoint(output_path, data)
        print("benchmark interrupted; checkpoint saved", file=sys.stderr)
        return 130
    except Exception as exc:
        if "data" in locals() and "output_path" in locals():
            data["status"] = "failed"
            data["error"] = f"{type(exc).__name__}: {exc}"
            checkpoint(output_path, data)
        print(f"benchmark failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
