#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Create a Markdown comparison from two schema-v2 benchmark outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .bench_offloading_e2e import BENCHMARK_NAME, SCHEMA_VERSION
except ImportError:
    from bench_offloading_e2e import BENCHMARK_NAME, SCHEMA_VERSION


def load_result(path: Path, expected_mode: str) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if data.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"{path}: unsupported schema version")
    if data.get("benchmark") != BENCHMARK_NAME:
        raise RuntimeError(f"{path}: not an OffloadingConnector benchmark")
    if data.get("mode") != expected_mode:
        raise RuntimeError(
            f"{path}: expected mode {expected_mode}, got {data.get('mode')}"
        )
    if data.get("status") != "completed":
        raise RuntimeError(f"{path}: benchmark status is {data.get('status')}")
    if not data.get("server"):
        raise RuntimeError(
            f"{path}: missing server metadata; resume the benchmark once with "
            "the current runner to upgrade the result"
        )
    return data


def validate_compatible(recompute: dict[str, Any], offload: dict[str, Any]) -> None:
    keys = ("model", "engine", "prompt_token", "block_size", "max_tokens")
    mismatches = [
        f"{key}: recompute={recompute['config'].get(key)!r}, "
        f"offload={offload['config'].get(key)!r}"
        for key in keys
        if recompute["config"].get(key) != offload["config"].get(key)
    ]
    if mismatches:
        raise RuntimeError("incompatible benchmark files:\n" + "\n".join(mismatches))
    if recompute.get("server") != offload.get("server"):
        raise RuntimeError(
            "incompatible server metadata: "
            f"recompute={recompute.get('server')!r}, "
            f"offload={offload.get('server')!r}"
        )


def stat(result: dict[str, Any], metric: str, field: str) -> float:
    return float(result["summary"][metric][field])


def result_map(data: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(result["tokens"]): result for result in data["results"]}


def cudagraph_modes(result: dict[str, Any], measurement_key: str) -> str:
    modes = {
        attempt[measurement_key].get("cudagraph_runtime_mode") or "unavailable"
        for attempt in result["attempts"]
        if attempt["valid"]
    }
    return ",".join(sorted(modes))


def render_markdown(
    recompute: dict[str, Any],
    offload: dict[str, Any],
    recompute_path: Path,
    offload_path: Path,
) -> str:
    recompute_results = result_map(recompute)
    offload_results = result_map(offload)
    sizes = sorted(set(recompute_results) & set(offload_results))
    offload_only_sizes = sorted(set(offload_results) - set(recompute_results))
    if not sizes:
        raise RuntimeError("benchmark files have no common context sizes")

    lines = [
        "# OffloadingConnector Benchmark Comparison",
        "",
        f"- Recompute: `{recompute_path}`",
        f"- Offload: `{offload_path}`",
        f"- Model: `{recompute['config']['model']}`",
        f"- vLLM: `{recompute['server']['version']}`",
        f"- Model root: `{recompute['server']['model']['root']}`",
        f"- Block size: {recompute['config']['block_size']}",
        "",
        "## Server TTFT",
        "",
        "| Context | Recompute median / P95 (ms) | CPU load median / P95 "
        "(ms) | Saved (ms) | Speedup | Recompute CV | CPU load CV |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tokens in sizes:
        rec = recompute_results[tokens]
        load = offload_results[tokens]
        rec_median = stat(rec, "server_ttft_seconds", "median")
        load_median = stat(load, "server_ttft_seconds", "median")
        lines.append(
            f"| {tokens:,} | {rec_median * 1000:.3f} / "
            f"{stat(rec, 'server_ttft_seconds', 'p95') * 1000:.3f} | "
            f"{load_median * 1000:.3f} / "
            f"{stat(load, 'server_ttft_seconds', 'p95') * 1000:.3f} | "
            f"{(rec_median - load_median) * 1000:.3f} | "
            f"{rec_median / load_median:.2f}x | "
            f"{stat(rec, 'server_ttft_seconds', 'cv') * 100:.2f}% | "
            f"{stat(load, 'server_ttft_seconds', 'cv') * 100:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Client E2E",
            "",
            "| Context | Recompute median / P95 (ms) | CPU load median / P95 "
            "(ms) | Speedup |",
            "|---:|---:|---:|---:|",
        ]
    )
    for tokens in sizes:
        rec = recompute_results[tokens]
        load = offload_results[tokens]
        rec_median = stat(rec, "client_e2e_seconds", "median")
        load_median = stat(load, "client_e2e_seconds", "median")
        lines.append(
            f"| {tokens:,} | {rec_median * 1000:.3f} / "
            f"{stat(rec, 'client_e2e_seconds', 'p95') * 1000:.3f} | "
            f"{load_median * 1000:.3f} / "
            f"{stat(load, 'client_e2e_seconds', 'p95') * 1000:.3f} | "
            f"{rec_median / load_median:.2f}x |"
        )

    lines.extend(
        [
            "",
            "## Validation",
            "",
            "| Context | Recompute valid / attempts | Offload valid / attempts "
            "| Recompute / replay CUDA Graph | Replay HBM hit | Replay external "
            "hit | CPU-to-GPU bytes |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for tokens in sizes:
        rec = recompute_results[tokens]
        load = offload_results[tokens]
        valid_load = [attempt for attempt in load["attempts"] if attempt["valid"]]
        local_hits = {
            int(attempt["replay"]["local_cache_hit_tokens"]) for attempt in valid_load
        }
        external_hits = {
            int(attempt["replay"]["external_kv_tokens"]) for attempt in valid_load
        }
        lines.append(
            f"| {tokens:,} | {rec['valid_samples']} / {rec['total_attempts']} | "
            f"{load['valid_samples']} / {load['total_attempts']} | "
            f"{cudagraph_modes(rec, 'request')} / "
            f"{cudagraph_modes(load, 'replay')} | "
            f"{','.join(map(str, sorted(local_hits)))} | "
            f"{','.join(map(str, sorted(external_hits)))} | "
            f"{stat(load, 'cpu_to_gpu_bytes', 'median'):,.0f} |"
        )

    if offload_only_sizes:
        lines.extend(
            [
                "",
                "## Offload-Only Mixed Replay Validation",
                "",
                "These non-block-aligned contexts validate mixed external KV "
                "load and local compute. They are excluded from speedup tables "
                "because no matching recompute samples were provided.",
                "",
                "| Context | Valid / attempts | Replay HBM hit | External KV "
                "tokens | Local compute tokens | CUDA Graph mode | CPU-to-GPU "
                "bytes |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for tokens in offload_only_sizes:
            load = offload_results[tokens]
            valid_load = [attempt for attempt in load["attempts"] if attempt["valid"]]
            local_hits = {
                int(attempt["replay"]["local_cache_hit_tokens"])
                for attempt in valid_load
            }
            external_hits = {
                int(attempt["replay"]["external_kv_tokens"]) for attempt in valid_load
            }
            local_compute = {
                int(attempt["replay"]["local_compute_tokens"]) for attempt in valid_load
            }
            lines.append(
                f"| {tokens:,} | {load['valid_samples']} / "
                f"{load['total_attempts']} | "
                f"{','.join(map(str, sorted(local_hits)))} | "
                f"{','.join(map(str, sorted(external_hits)))} | "
                f"{','.join(map(str, sorted(local_compute)))} | "
                f"{cudagraph_modes(load, 'replay')} | "
                f"{stat(load, 'cpu_to_gpu_bytes', 'median'):,.0f} |"
            )

    lines.extend(
        [
            "",
            "The comparison excludes eviction-request setup time. Inspect the "
            "raw attempt records for all validation errors and request metrics.",
            "",
        ]
    )
    verification = offload.get("decode_block_verification")
    if verification:
        calibration = verification["calibration"]
        multi_block = verification["multi_block"]
        lines.extend(
            [
                "## Decode Block Store",
                "",
                f"- Status: `{'PASS' if verification['valid'] else 'FAIL'}`",
                f"- Logical blocks: 1 -> {verification['blocks']}",
                "- GPU-to-CPU operations: "
                f"{calibration['gpu_to_cpu_operations']:,.0f} -> "
                f"{multi_block['gpu_to_cpu_operations']:,.0f}",
                "- GPU-to-CPU bytes: "
                f"{calibration['gpu_to_cpu_bytes']:,.0f} -> "
                f"{multi_block['gpu_to_cpu_bytes']:,.0f}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", type=Path, required=True)
    parser.add_argument("--offload", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        recompute = load_result(args.recompute, "recompute")
        offload = load_result(args.offload, "offload")
        validate_compatible(recompute, offload)
        markdown = render_markdown(recompute, offload, args.recompute, args.offload)
        if args.output:
            if args.output.exists() and not args.overwrite:
                raise RuntimeError(
                    f"output already exists: {args.output}; use --overwrite"
                )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(markdown)
        print(markdown)
        return 0
    except Exception as exc:
        print(f"summary failed: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
