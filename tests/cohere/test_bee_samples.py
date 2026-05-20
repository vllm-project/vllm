#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bee sample checker: send a few samples per task to a live vLLM server and
validate responses with bee_eval_checker.

All samples within each task are fired concurrently (following the pattern
in vllm/benchmarks/throughput.py and vllm/benchmarks/serve.py).

Assumes the vLLM server is already running (started by the caller script).

Usage:
    python3 tests/cohere/test_bee_samples.py \\
        --base-url http://localhost:8000/v1 \\
        --model command-r7b_fp8 \\
        --tasks mmlupro,mgsm,mbpp_plus,ocrbench,infovqa,mathvista,aime,niah \\
        --data-dir tests/cohere/bee_eval_data
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bee_eval_checker import BeeEvalTask, EvalSample, get_task

DEFAULT_MAX_TOKENS = 4096
# since most test pass with a accuracy above 0.25 on the nightly CI tests
# we use 0.24 as the default minimum score to pass the test
DEFAULT_MIN_SCORE = 0.24
DEFAULT_THINKING_BUDGET = 2048


@dataclass
class TaskConfig:
    data_file: str
    max_tokens: int = DEFAULT_MAX_TOKENS
    min_score: float = DEFAULT_MIN_SCORE
    thinking_budget: int | None = DEFAULT_THINKING_BUDGET


TASK_CONFIG: dict[str, TaskConfig] = {
    "mmlupro": TaskConfig("mmlupro_16samples.jsonl", min_score=0.4),
    "ocrbench": TaskConfig("ocrbench_16samples.jsonl"),
    "infovqa": TaskConfig("infovqa_16samples.jsonl"),
    "mathvista": TaskConfig("mathvista_16samples.json"),
    "aime": TaskConfig(
        "aime_2025_16samples.csv",
        thinking_budget=16384,
        max_tokens=32768,
        min_score=0.24,
    ),
    "mgsm": TaskConfig("mgsm_ja_16samples.csv"),
    "mbpp_plus": TaskConfig(
        "mbpp_plus_16samples.jsonl", thinking_budget=4096, max_tokens=8192
    ),
    "niah": TaskConfig("niah_16samples.jsonl"),
}


@dataclass
class SampleResult:
    """Outcome of a single sample request."""

    index: int
    passed: bool
    score: float
    elapsed: float
    generation: str = ""
    ground_truth: str = ""
    reasoning: str = ""
    error: str | None = None


@dataclass
class TaskSummary:
    """Aggregated result for one eval task."""

    task: str
    skipped: bool = False
    reason: str = ""
    total: int = 0
    passed: int = 0
    errors: int = 0
    avg_score: float = 0.0
    min_score: float = 0.0
    elapsed: float = 0.0
    sample_results: list[SampleResult] = field(default_factory=list)

    @property
    def meets_threshold(self) -> bool:
        return self.skipped or self.avg_score >= self.min_score


async def send_sample(
    client: AsyncOpenAI,
    model: str,
    sample: EvalSample,
    max_tokens: int,
    index: int,
    task: BeeEvalTask,
    thinking_budget: int | None = None,
) -> SampleResult:
    """Send one sample to the server and validate the response."""
    t0 = time.monotonic()
    try:
        extra_body = (
            {"thinking_token_budget": thinking_budget}
            if thinking_budget is not None
            else None
        )
        resp = await client.chat.completions.create(
            model=model,
            messages=sample.messages,
            max_tokens=max_tokens,
            temperature=0,
            extra_body=extra_body,
        )
        msg = resp.choices[0].message
        generation = msg.content or ""
        reasoning = getattr(msg, "reasoning", None) or ""
        result = task.check_response(sample, generation)
        gt = sample.ground_truth
        return SampleResult(
            index=index,
            passed=result.passed,
            score=result.score,
            elapsed=time.monotonic() - t0,
            generation=generation,
            ground_truth=gt if isinstance(gt, str) else " | ".join(gt),
            reasoning=reasoning,
        )
    except Exception as exc:
        return SampleResult(
            index=index,
            passed=False,
            score=0.0,
            elapsed=time.monotonic() - t0,
            error=str(exc),
        )


async def run_task(
    task_name: str,
    data_dir: str,
    client: AsyncOpenAI,
    model: str,
    max_samples: int | None = None,
    enable_thinking_budget: bool = False,
    min_score_override: float | None = None,
) -> TaskSummary:
    """Load samples, fire all requests concurrently, return aggregated result."""
    cfg = TASK_CONFIG.get(task_name)
    if cfg is None:
        return TaskSummary(task=task_name, skipped=True, reason="no task config")

    min_score = min_score_override if min_score_override is not None else cfg.min_score
    thinking_budget = cfg.thinking_budget if enable_thinking_budget else None
    task = get_task(task_name)

    data_path = Path(data_dir) / cfg.data_file
    if not data_path.exists():
        return TaskSummary(
            task=task_name, skipped=True, reason=f"{data_path} not found"
        )

    samples = task.load_samples(str(data_path), n=max_samples)
    if not samples:
        return TaskSummary(task=task_name, skipped=True, reason="0 samples loaded")

    t0 = time.monotonic()
    coros = [
        send_sample(client, model, sample, cfg.max_tokens, i, task, thinking_budget)
        for i, sample in enumerate(samples)
    ]
    results: list[SampleResult] = await asyncio.gather(*coros)
    wall_time = time.monotonic() - t0

    scores = [r.score for r in results if r.error is None]
    return TaskSummary(
        task=task_name,
        total=len(results),
        passed=sum(1 for r in results if r.passed),
        errors=sum(1 for r in results if r.error is not None),
        avg_score=sum(scores) / len(scores) if scores else 0.0,
        min_score=min_score,
        elapsed=wall_time,
        sample_results=results,
    )


def print_summaries(summaries: list[TaskSummary], total_elapsed: float) -> list[str]:
    """Print per-sample details and a summary table.

    Returns below-threshold task descriptions.
    """
    for s in summaries:
        print(f"=== {s.task} ===")
        if s.skipped:
            print(f"  SKIPPED: {s.reason}\n")
            continue
        for r in s.sample_results:
            if r.error:
                print(f"  [{r.index + 1}/{s.total}] ERROR: {r.error}")
            else:
                status = "PASS" if r.passed else "FAIL"
                gen_preview = r.generation[:120].replace("\n", "\\n")
                gt_preview = r.ground_truth[:120].replace("\n", "\\n")
                print(
                    f"  [{r.index + 1}/{s.total}] {status}  "
                    f"score={r.score:.2f}  ({r.elapsed:.1f}s)"
                )
                print(f"    expected: {gt_preview}")
                print(f"    got:      {gen_preview}")
        rate = s.passed / s.total if s.total else 0
        print(
            f"  Result: {s.passed}/{s.total} passed ({rate:.0%}), "
            f"avg_score={s.avg_score:.2f}, errors={s.errors}\n"
        )

    print("=" * 60)
    print("BEE SAMPLES SUMMARY")
    print("=" * 60)
    below_threshold: list[str] = []
    for s in summaries:
        if s.skipped:
            print(f"  {s.task:15s}  SKIPPED  ({s.reason})")
            continue
        rate = s.passed / s.total if s.total else 0
        mark = "PASS" if s.meets_threshold else "FAIL"
        err_note = f"  ({s.errors} errors)" if s.errors else ""
        print(
            f"  {s.task:15s}  {s.passed:2d}/{s.total:2d} "
            f"({rate:5.0%})  avg={s.avg_score:.2f}  "
            f"min={s.min_score:.2f}  [{mark}]{err_note}"
        )
        if not s.meets_threshold:
            below_threshold.append(
                f"{s.task} (avg={s.avg_score:.2f} < min={s.min_score:.2f})"
            )

    print(f"\nTotal wall time: {total_elapsed:.1f}s")
    return below_threshold


async def run(args: argparse.Namespace) -> int:
    """Async entry point: resolve tasks, run concurrently, print results."""
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    if not tasks:
        print("ERROR: No tasks to run.")
        return 1

    enable_thinking_budget: bool = args.enable_thinking_budget
    print(f"Model:    {args.model}")
    print(f"Tasks:    {', '.join(tasks)}")
    print(f"Data dir: {args.data_dir}")
    print(f"Base URL: {args.base_url}")
    if enable_thinking_budget:
        print("Thinking: per-task budgets from TASK_CONFIG (--enable-thinking-budget)")
    else:
        print("Thinking: disabled (no thinking_token_budget sent)")
    print()

    min_score_override: float | None = args.min_score
    client = AsyncOpenAI(base_url=args.base_url, api_key="not-needed")

    t_start = time.monotonic()
    coros = [
        run_task(
            task_name,
            args.data_dir,
            client,
            args.model,
            args.max_samples,
            enable_thinking_budget,
            min_score_override,
        )
        for task_name in tasks
    ]
    summaries: list[TaskSummary] = await asyncio.gather(*coros)
    total_elapsed = time.monotonic() - t_start

    below_threshold = print_summaries(summaries, total_elapsed)

    if args.output_json:
        _write_results(args.output_json, summaries, total_elapsed, args.model)

    if below_threshold:
        print("\nRESULT: FAILED — tasks below expected score:")
        for msg in below_threshold:
            print(f"  - {msg}")
        return 1

    print("\nRESULT: PASSED")
    return 0


def _build_task_summary(s: TaskSummary) -> dict[str, Any]:
    """Build the per-task dict shared by both summary and detailed output."""
    task_obj: dict[str, Any] = {
        "task": s.task,
        "skipped": s.skipped,
    }
    if s.skipped:
        task_obj["reason"] = s.reason
    else:
        task_obj.update(
            {
                "total": s.total,
                "passed": s.passed,
                "errors": s.errors,
                "avg_score": round(s.avg_score, 4),
                "min_score": s.min_score,
                "meets_threshold": s.meets_threshold,
                "wall_time_s": round(s.elapsed, 2),
            }
        )
    return task_obj


def _write_results(
    path: str,
    summaries: list[TaskSummary],
    total_elapsed: float,
    model: str,
) -> None:
    """Write a compact summary (no per-sample data) and a detailed companion file.

    The summary at *path* contains only per-task aggregates and is small enough
    for CI artifact uploads.  A ``_detailed`` sibling (e.g.
    ``bee_samples_model_detailed.json``) keeps full sample-level output
    including generations for local debugging.
    """
    summary: dict[str, Any] = {
        "model": model,
        "total_wall_time_s": round(total_elapsed, 2),
        "tasks": [_build_task_summary(s) for s in summaries],
    }
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary written to {path}")

    detailed_path = Path(path)
    detailed_path = detailed_path.with_name(
        detailed_path.stem + "_detailed" + detailed_path.suffix
    )
    detailed: dict[str, Any] = {
        "model": model,
        "total_wall_time_s": round(total_elapsed, 2),
        "tasks": [],
    }
    for s in summaries:
        task_obj = _build_task_summary(s)
        if not s.skipped:
            task_obj["samples"] = [
                {
                    "index": r.index,
                    "passed": r.passed,
                    "score": round(r.score, 4),
                    "elapsed_s": round(r.elapsed, 2),
                    "ground_truth": r.ground_truth,
                    "generation": r.generation,
                    **({"reasoning": r.reasoning} if r.reasoning else {}),
                    **({"error": r.error} if r.error else {}),
                }
                for r in s.sample_results
            ]
        detailed["tasks"].append(task_obj)

    with open(detailed_path, "w") as f:
        json.dump(detailed, f, indent=2, ensure_ascii=False)
    print(f"Detailed results written to {detailed_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bee sample checker against a running vLLM server"
    )
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", required=True, help="Served model name")
    parser.add_argument(
        "--tasks",
        required=True,
        help="Comma-separated task names",
    )
    parser.add_argument("--data-dir", default="tests/cohere/bee_eval_data")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=16,
        help="Max samples per task (default: 16)",
    )
    parser.add_argument(
        "--enable-thinking-budget",
        action="store_true",
        default=False,
        help="Enable per-task thinking budgets from TASK_CONFIG. "
        "Without this flag, no thinking_token_budget is sent to the server.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Override the minimum expected avg_score for all tasks. "
        "If omitted, per-task defaults from TASK_CONFIG are used.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Write per-task summary to this JSON file and a *_detailed.json "
        "sibling with full per-sample generations.",
    )
    args = parser.parse_args()
    return asyncio.run(run(args))


# ── pytest integration ──────────────────────────────────────────────────────
# When invoked via ``pytest``, the fixture below runs all tasks concurrently
# (same as standalone mode) and caches results.  Each parametrized
# ``test_bee_task`` then checks one task's outcome, giving per-task JUnit XML
# reporting.  Configuration is read from environment variables set by
# ``run_tests.sh``.
#
# Env vars:
#   BEE_MODEL           (required) served model name
#   BEE_BASE_URL        server URL          (default: http://localhost:8000/v1)
#   BEE_DATA_DIR        sample data dir     (default: tests/cohere/bee_eval_data)
#   BEE_MAX_SAMPLES     samples per task    (default: 16)
#   BEE_TASKS           comma-separated     (default: all TASK_CONFIG keys)
#   ENABLE_THINKING_BUDGET  "1" to enable   (default: disabled)
#   BEE_OUTPUT_JSON     path for summary JSON output
#   BEE_MIN_SCORE_OVERRIDES  per-task min_score, e.g. "aime:0.3,mgsm:0.4"

_BEE_DEFAULT_TASKS = list(TASK_CONFIG.keys())

try:
    import os as _os

    import pytest as _pytest

    @_pytest.fixture(scope="module")
    def bee_summaries():
        """Run all bee sample tasks concurrently once and cache results."""
        model = _os.environ.get("BEE_MODEL")
        if not model:
            _pytest.skip("BEE_MODEL not set; run via run_tests.sh")
        assert model is not None

        base_url = _os.environ.get("BEE_BASE_URL", "http://localhost:8000/v1")
        data_dir = _os.environ.get("BEE_DATA_DIR", "tests/cohere/bee_eval_data")
        max_samples = int(_os.environ.get("BEE_MAX_SAMPLES", "16"))
        enable_thinking = _os.environ.get("ENABLE_THINKING_BUDGET") == "1"
        output_json = _os.environ.get("BEE_OUTPUT_JSON")
        for pair in _os.environ.get("BEE_MIN_SCORE_OVERRIDES", "").split(","):
            pair = pair.strip()
            if ":" in pair:
                k, v = pair.split(":", 1)
                k = k.strip()
                if k in TASK_CONFIG:
                    TASK_CONFIG[k].min_score = float(v.strip())
        tasks = [
            t.strip()
            for t in _os.environ.get("BEE_TASKS", ",".join(_BEE_DEFAULT_TASKS)).split(
                ","
            )
            if t.strip()
        ]

        client = AsyncOpenAI(base_url=base_url, api_key="not-needed")
        t_start = time.monotonic()

        async def _run_all() -> list[TaskSummary]:
            return list(
                await asyncio.gather(
                    *[
                        run_task(
                            t, data_dir, client, model, max_samples, enable_thinking
                        )
                        for t in tasks
                    ]
                )
            )

        summaries: list[TaskSummary] = asyncio.run(_run_all())
        total_elapsed = time.monotonic() - t_start

        print_summaries(summaries, total_elapsed)

        if output_json:
            _write_results(output_json, summaries, total_elapsed, model)

        return {s.task: s for s in summaries}

    @_pytest.mark.parametrize("task_name", _BEE_DEFAULT_TASKS)
    def test_bee_task(task_name: str, bee_summaries: dict[str, TaskSummary]):
        summary = bee_summaries.get(task_name)
        if summary is None:
            _pytest.skip(f"Task {task_name} not included in this run")
        assert summary is not None
        if summary.skipped:
            _pytest.skip(f"{task_name}: {summary.reason}")
        assert summary.meets_threshold, (
            f"{task_name}: avg_score={summary.avg_score:.2f} "
            f"< min={summary.min_score:.2f} "
            f"({summary.passed}/{summary.total} passed, "
            f"{summary.errors} errors)"
        )

except ImportError:
    pass


if __name__ == "__main__":
    sys.exit(main())
