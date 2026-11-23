#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Analyze encoder-admission steps from vLLM iteration logs (#69306).

This classifies the *existing* eager-fallback trigger, not a CUDA-graph fix.
It does not claim that graph replay can safely handle changing cross-attention
K/V; a separate correctness implementation is needed.
"""

import argparse
import json
import math
import re
import statistics
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

ITERATION_RE = re.compile(
    r"Iteration\((?P<iteration>\d+)\): .*?KV cache usage: [0-9.]+%"
    r"(?:, encoder inputs: (?P<encoder_inputs>\d+), "
    r"encoder output embeddings: \d+)?"
)
ELAPSED_RE = re.compile(r"iteration elapsed time: (?P<elapsed>[^ ,]+) ms")


@dataclass(frozen=True)
class Step:
    index: int
    encoder_inputs: int
    elapsed_ms: float | None


def parse_line(line: str) -> Step | None:
    if " (dummy)" in line or "Iteration(" not in line:
        return None
    match = ITERATION_RE.search(line)
    if match is None:
        return None
    elapsed_match = ELAPSED_RE.search(line)
    elapsed_ms = None
    if elapsed_match:
        with suppress(ValueError):
            elapsed_ms = float(elapsed_match["elapsed"])
        if elapsed_ms is not None and (not math.isfinite(elapsed_ms) or elapsed_ms < 0):
            elapsed_ms = None
    return Step(
        index=int(match["iteration"]),
        encoder_inputs=int(match["encoder_inputs"] or 0),
        elapsed_ms=elapsed_ms,
    )


def summarize(steps: list[Step]) -> dict:
    if not steps:
        raise ValueError("No non-dummy Iteration(...) records found")
    admitted = [step for step in steps if step.encoder_inputs > 0]
    other = [step for step in steps if step.encoder_inputs == 0]
    admission_times = [
        step.elapsed_ms for step in admitted if step.elapsed_ms is not None
    ]
    other_times = [step.elapsed_ms for step in other if step.elapsed_ms is not None]
    total_timed = len(admission_times) + len(other_times)
    total_ms = sum(admission_times) + sum(other_times)
    return {
        "total_steps": len(steps),
        "encoder_admission_steps": len(admitted),
        "encoder_admission_fraction_pct": 100 * len(admitted) / len(steps),
        "steps_with_timing": total_timed,
        "admission_steps_missing_timing": len(admitted) - len(admission_times),
        "non_admission_steps_missing_timing": len(other) - len(other_times),
        "timing_coverage_pct": 100 * total_timed / len(steps),
        "median_admission_step_ms": (
            statistics.median(admission_times) if admission_times else None
        ),
        "median_non_admission_step_ms": (
            statistics.median(other_times) if other_times else None
        ),
        "observed_admission_step_time_fraction_pct": (
            100 * sum(admission_times) / total_ms
            if total_timed == len(steps) and total_ms > 0
            else None
        ),
        "interpretation": (
            "Current GPUModelRunner sets skip_compiled for encoder-decoder "
            "steps that schedule encoder inputs. This is a correlated timing "
            "proxy, not a measured CUDA-graph speedup or proof that replay is safe."
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log", type=Path, help="vLLM engine log with Iteration(...) lines")
    p.add_argument("--output-json", type=Path)
    args = p.parse_args()
    with args.log.open() as source:
        steps = [step for line in source if (step := parse_line(line)) is not None]
    output = json.dumps(summarize(steps), indent=2, allow_nan=False)
    print(output)
    if args.output_json:
        args.output_json.write_text(output + "\n")


if __name__ == "__main__":
    main()
