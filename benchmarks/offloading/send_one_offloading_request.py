#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import http.client
import json
import time

CUDAGRAPH_RUNTIME_MODES = ("NONE", "PIECEWISE", "FULL")


def get_metrics(host: str, port: int, timeout: float) -> str:
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    conn.request("GET", "/metrics")
    response = conn.getresponse()
    body = response.read().decode()
    conn.close()
    if response.status != 200:
        raise RuntimeError(f"metrics returned HTTP {response.status}: {body}")
    return body


def metric_value(
    metrics: str, name: str, engine: int, labels: tuple[str, ...] = ()
) -> float:
    engine_label = f'engine="{engine}"'
    for line in metrics.splitlines():
        if not line.startswith(name + "{") or engine_label not in line:
            continue
        if all(label in line for label in labels):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


def metric_delta(
    before: str, after: str, name: str, engine: int, *labels: str
) -> float:
    return metric_value(after, name, engine, labels) - metric_value(
        before, name, engine, labels
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send exactly one completion and print its metric deltas."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--model", default="M1-0710")
    parser.add_argument("--engine", type=int, default=0)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--prompt-token", type=int, required=True)
    parser.add_argument("--cache-salt", required=True)
    parser.add_argument("--settle-seconds", type=float, default=0.2)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument("--metrics-timeout", type=float, default=30)
    args = parser.parse_args()
    if args.tokens <= 0:
        parser.error("--tokens must be positive")
    if args.settle_seconds < 0:
        parser.error("--settle-seconds must not be negative")

    payload = json.dumps(
        {
            "model": args.model,
            "prompt": [args.prompt_token] * args.tokens,
            "max_tokens": 1,
            "temperature": 0,
            "add_special_tokens": False,
            "return_token_ids": True,
            "cache_salt": args.cache_salt,
        },
        separators=(",", ":"),
    )
    headers = {
        "Content-Type": "application/json",
        "X-data-parallel-rank": str(args.engine),
    }

    before = get_metrics(args.host, args.port, args.metrics_timeout)
    conn = http.client.HTTPConnection(
        args.host, args.port, timeout=args.request_timeout
    )
    start = time.perf_counter()
    conn.request("POST", "/v1/completions", body=payload, headers=headers)
    response = conn.getresponse()
    body = response.read().decode()
    client_seconds = time.perf_counter() - start
    conn.close()
    if response.status != 200:
        raise RuntimeError(f"completion returned HTTP {response.status}: {body}")

    time.sleep(args.settle_seconds)
    after = get_metrics(args.host, args.port, args.metrics_timeout)
    result = json.loads(body)
    cudagraph_metrics_available = any(
        line.startswith("vllm:cudagraph_dispatch_total{")
        for metrics in (before, after)
        for line in metrics.splitlines()
    )
    cudagraph_counts = {
        mode: int(
            metric_delta(
                before,
                after,
                "vllm:cudagraph_dispatch_total",
                args.engine,
                f'runtime_mode="{mode}"',
            )
        )
        for mode in CUDAGRAPH_RUNTIME_MODES
    }
    active_modes = [mode for mode, count in cudagraph_counts.items() if count]
    output = {
        "cache_salt": args.cache_salt,
        "prompt_token": args.prompt_token,
        "prompt_tokens": result["usage"]["prompt_tokens"],
        "output_token_ids": result["choices"][0].get("token_ids") or [],
        "client_e2e_ms": client_seconds * 1000,
        "server_ttft_ms": metric_delta(
            before, after, "vllm:time_to_first_token_seconds_sum", args.engine
        )
        * 1000,
        "server_prefill_ms": metric_delta(
            before, after, "vllm:request_prefill_time_seconds_sum", args.engine
        )
        * 1000,
        "local_compute_tokens": metric_delta(
            before,
            after,
            "vllm:prompt_tokens_by_source_total",
            args.engine,
            'source="local_compute"',
        ),
        "local_cache_hit_tokens": metric_delta(
            before,
            after,
            "vllm:prompt_tokens_by_source_total",
            args.engine,
            'source="local_cache_hit"',
        ),
        "external_kv_tokens": metric_delta(
            before,
            after,
            "vllm:prompt_tokens_by_source_total",
            args.engine,
            'source="external_kv_transfer"',
        ),
        "cpu_to_gpu_bytes": metric_delta(
            before,
            after,
            "vllm:kv_offload_total_bytes_total",
            args.engine,
            'transfer_type="CPU_to_GPU"',
        ),
        "gpu_to_cpu_bytes": metric_delta(
            before,
            after,
            "vllm:kv_offload_total_bytes_total",
            args.engine,
            'transfer_type="GPU_to_CPU"',
        ),
        "cudagraph_metrics_available": cudagraph_metrics_available,
        "cudagraph_runtime_mode_counts": cudagraph_counts,
        "cudagraph_runtime_mode": (
            active_modes[0]
            if len(active_modes) == 1
            else "MIXED"
            if active_modes
            else None
        ),
        "cudagraph_used": (
            any(mode != "NONE" for mode in active_modes) if active_modes else None
        ),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
