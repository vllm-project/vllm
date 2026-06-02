# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CI regression test for thinking-token-budget sampling overhead.

Starts a vLLM server (same style as test_request_cancellation.py), then for each
thinking budget and concurrency level runs ``vllm bench serve`` with and without
``thinking_token_budget`` in ``--extra-body``. Fails if median TPOT overhead exceeds 4%.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import regex as re

from tests.utils import RemoteOpenAIServer

THINKING_TOKEN_BUDGETS = [500, 1000]
CONCURRENCY_LEVELS = [1, 8, 32, 64]
MAX_OVERHEAD_PCT = 4.0
DEFAULT_PROMPTS_PER_CONCURRENCY = 10

DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parent / "bee_eval_data" / "aime_2025_16samples.jsonl"
)

_MEDIAN_TPOT_RE = re.compile(r"Median TPOT \(ms\):\s+([\d.]+)")


def _resolve_model_path(model_arg: str | None) -> str:
    if model_arg:
        return model_arg
    engines_dir = os.environ.get("ENGINES_DIR", "/root/engines")
    default_name = os.environ.get("THINKING_BUDGET_OVERHEAD_MODEL", "c5-3a30t_fp8")
    candidate = os.path.join(engines_dir, default_name)
    if os.path.isdir(candidate):
        return candidate
    raise ValueError(
        "Model path required: pass --model or set ENGINES_DIR with "
        f"{default_name} present."
    )


def _resolve_dataset_path(dataset_arg: str | None) -> str:
    if dataset_arg:
        path = Path(dataset_arg)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")
        return str(path)

    candidates: list[str] = []
    if env_path := os.environ.get("THINKING_BUDGET_OVERHEAD_DATASET"):
        candidates.append(env_path)
    candidates.append(str(DEFAULT_DATASET_PATH))

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        "Dataset not found. Pass --dataset-path or set "
        "THINKING_BUDGET_OVERHEAD_DATASET to the path of the dataset."
    )


def _parse_median_tpot_ms(stdout: str) -> float:
    match = _MEDIAN_TPOT_RE.search(stdout)
    if not match:
        raise ValueError("Median TPOT (ms) not found in vllm bench serve output")
    return float(match.group(1))


def _run_bench_serve(
    *,
    host: str,
    port: int,
    model: str,
    dataset_path: str,
    concurrency: int,
    output_len: int,
    num_prompts: int,
    thinking_token_budget: int | None,
) -> float:
    cmd = [
        "vllm",
        "bench",
        "serve",
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        model,
        "--backend",
        "openai-chat",
        "--endpoint",
        "/v1/chat/completions",
        "--dataset-name",
        "custom",
        "--dataset-path",
        dataset_path,
        "--output-len",
        str(output_len),
        "--num-prompts",
        str(num_prompts),
        "--max-concurrency",
        str(concurrency),
        "--request-rate",
        "inf",
        "--temperature",
        "0.6",
        "--top-p",
        "0.95",
        "--percentile-metrics",
        "tpot",
        "--ready-check-timeout-sec",
        "600",
    ]
    if thinking_token_budget is not None:
        cmd.extend(
            [
                "--extra-body",
                json.dumps({"thinking_token_budget": thinking_token_budget}),
            ]
        )

    print(f"Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"vllm bench serve failed (exit {proc.returncode}): {proc.stderr}"
        )

    combined = f"{proc.stdout}\n{proc.stderr}"
    failed_match = re.search(r"Failed requests:\s+(\d+)", combined)
    if failed_match and int(failed_match.group(1)) > 0:
        raise RuntimeError(f"Benchmark had {failed_match.group(1)} failed requests")

    return _parse_median_tpot_ms(proc.stdout)


def _overhead_pct(tpot_with_budget_ms: float, tpot_baseline_ms: float) -> float:
    if tpot_baseline_ms <= 0:
        raise ValueError(f"Invalid baseline TPOT: {tpot_baseline_ms}")
    return 100.0 * (tpot_with_budget_ms - tpot_baseline_ms) / tpot_baseline_ms


def run_overhead_sweep(
    *,
    model: str,
    dataset_path: str,
    tp_size: int,
    prompts_per_concurrency: int,
    max_overhead_pct: float,
) -> list[str]:
    server_args = [
        "--tensor-parallel-size",
        str(tp_size),
        "--trust-remote-code",
        "--no-enable-prefix-caching",
        "--reasoning-parser",
        "cohere_command4",
    ]

    failures: list[str] = []

    # Profile defaults are applied inside the server process via
    # VLLM_ENABLE_COHERE_AUTO_CONFIG (passed through env_dict below).
    with RemoteOpenAIServer(
        model,
        server_args,
        env_dict={"VLLM_ENABLE_COHERE_AUTO_CONFIG": "1"},
    ) as server:
        print(f"Server ready at http://{server.host}:{server.port}")

        for budget in THINKING_TOKEN_BUDGETS:
            output_len = budget + 3
            for concurrency in CONCURRENCY_LEVELS:
                num_prompts = concurrency * prompts_per_concurrency
                tag = f"budget={budget}_concurrency={concurrency}"

                print(f"\n{'=' * 60}\n  {tag}\n{'=' * 60}")

                try:
                    baseline_tpot = _run_bench_serve(
                        host=server.host,
                        port=server.port,
                        model=model,
                        dataset_path=dataset_path,
                        concurrency=concurrency,
                        output_len=output_len,
                        num_prompts=num_prompts,
                        thinking_token_budget=None,
                    )
                    budget_tpot = _run_bench_serve(
                        host=server.host,
                        port=server.port,
                        model=model,
                        dataset_path=dataset_path,
                        concurrency=concurrency,
                        output_len=output_len,
                        num_prompts=num_prompts,
                        thinking_token_budget=budget,
                    )
                except Exception as exc:
                    msg = f"{tag}: {exc}"
                    print(f"ERROR: {msg}")
                    failures.append(msg)
                    continue

                overhead = _overhead_pct(budget_tpot, baseline_tpot)
                print(
                    f"{tag}: baseline median TPOT={baseline_tpot:.3f} ms, "
                    f"with budget median TPOT={budget_tpot:.3f} ms, "
                    f"overhead={overhead:.2f}%"
                )

                if overhead > max_overhead_pct:
                    msg = (
                        f"{tag}: overhead {overhead:.2f}% exceeds "
                        f"{max_overhead_pct}% limit"
                    )
                    print(f"FAIL: {msg}")
                    failures.append(msg)
                else:
                    print(f"PASS: {tag}")

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Thinking-token-budget TPOT overhead regression (CI)."
    )
    parser.add_argument("--model", type=str, default=None, help="Model directory.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to custom JSONL dataset (e.g. aime_2025_16samples.jsonl).",
    )
    parser.add_argument(
        "--tp-size",
        "--tensor-parallel-size",
        type=int,
        default=int(os.environ.get("TP", "1")),
        help="Tensor parallel size for the server.",
    )
    parser.add_argument(
        "--prompts-per-concurrency",
        type=int,
        default=int(
            os.environ.get(
                "THINKING_BUDGET_OVERHEAD_PROMPTS_PER_CONCURRENCY",
                str(DEFAULT_PROMPTS_PER_CONCURRENCY),
            )
        ),
        help="num-prompts = concurrency * this value for each bench run.",
    )
    parser.add_argument(
        "--max-overhead-pct",
        type=float,
        default=float(
            os.environ.get("THINKING_BUDGET_MAX_OVERHEAD_PCT", str(MAX_OVERHEAD_PCT))
        ),
        help="Maximum allowed median TPOT overhead (percent) when budget is enabled.",
    )
    args = parser.parse_args()

    model = _resolve_model_path(args.model)
    dataset_path = _resolve_dataset_path(args.dataset_path)

    print("Thinking budget overhead regression")
    print(f"  model: {model}")
    print(f"  dataset: {dataset_path}")
    print(f"  budgets: {THINKING_TOKEN_BUDGETS}")
    print(f"  concurrencies: {CONCURRENCY_LEVELS}")
    print(f"  max overhead: {args.max_overhead_pct}%")

    failures = run_overhead_sweep(
        model=model,
        dataset_path=dataset_path,
        tp_size=args.tp_size,
        prompts_per_concurrency=args.prompts_per_concurrency,
        max_overhead_pct=args.max_overhead_pct,
    )

    print("\n" + "=" * 80)
    if failures:
        print("THINKING BUDGET OVERHEAD TEST FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)

    print("All thinking budget overhead checks passed.")
    print("=" * 80)


if __name__ == "__main__":
    main()
