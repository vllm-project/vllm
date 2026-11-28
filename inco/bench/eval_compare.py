# SPDX-License-Identifier: Apache-2.0
"""Compare lm-eval results between two checkpoints.

    python -m bench.eval_compare Qwen3-30B-A3B-Instruct-2507 \
        layerwise_reap-renorm_true-seed_42-0.50

Reads the JSON that ``modal_reap.py::evaluate`` writes per model
(``{task: {"acc,none": ..., "acc_stderr,none": ...}}``) and reports the
accuracy delta per task.

Two choices here are deliberate:

* **``acc_norm`` wins over ``acc``** when a task reports both. Raw ``acc``
  compares unnormalized logprobs and is length-biased toward short options;
  ``acc_norm`` divides by byte length and is what the REAP paper reports. A
  pruned model can score below the random-choice floor on ``acc`` while
  ``acc_norm`` is healthy, so reading ``acc`` alone invents a regression.
* **A delta is only called a regression if it clears 1.96 combined standard
  errors.** At 277 (RTE) to 1267 (WinoGrande) questions the per-task stderr is
  1-3pp, so sub-2pp deltas are noise. Quoting them as wins or losses is the
  single easiest way to overstate a compression result.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench import report

DEFAULT_EVAL_ROOT = "results/evals"

# Checked in order; the first one a task reports becomes its headline metric.
METRIC_PREFERENCE = ("acc_norm", "pass@1", "acc", "exact_match")

# Random-choice accuracy, used only to flag a headline number that has fallen
# to chance -- which means "model is guessing", not "model is 40% as good".
CHANCE_LEVEL = {
    "arc_challenge": 0.25,
    "arc_easy": 0.25,
    "openbookqa": 0.25,
    "hellaswag": 0.25,
    "mmlu": 0.25,
    "boolq": 0.5,
    "rte": 0.5,
    "winogrande": 0.5,
}


class MalformedEval(ValueError):
    pass


@dataclass(frozen=True)
class Score:
    """One task's headline metric for one model."""

    task: str
    metric: str
    value: float
    stderr: float | None = None

    @property
    def at_chance(self) -> bool:
        floor = CHANCE_LEVEL.get(self.task)
        if floor is None:
            return False
        margin = 1.96 * self.stderr if self.stderr else 0.0
        return self.value <= floor + margin


def _split_key(key: str) -> str:
    """``"acc_norm,none"`` -> ``"acc_norm"``; lm-eval appends the filter name."""
    return key.split(",", 1)[0]


def parse_task_metrics(metrics: dict[str, Any], task: str) -> Score | None:
    """Pick the headline metric out of one task's lm-eval metric block."""
    by_name: dict[str, float] = {}
    stderrs: dict[str, float] = {}
    for key, value in metrics.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        name = _split_key(key)
        if name.endswith("_stderr"):
            stderrs[name[: -len("_stderr")]] = float(value)
        else:
            by_name[name] = float(value)

    for candidate in METRIC_PREFERENCE:
        if candidate in by_name:
            return Score(
                task=task,
                metric=candidate,
                value=by_name[candidate],
                stderr=stderrs.get(candidate),
            )
    # Unknown task type: fall back to the sole metric if there is exactly one,
    # so a new benchmark shows up in the table instead of vanishing silently.
    remaining = sorted(by_name)
    if len(remaining) == 1:
        name = remaining[0]
        return Score(task, name, by_name[name], stderrs.get(name))
    return None


def load_eval_dir(path: str | Path) -> dict[str, Score]:
    """Merge every ``*.json`` under ``path`` into one task -> Score map.

    Later files win on collision, so re-running a single task overrides the
    value from an earlier bundled run.
    """
    path = Path(path)
    files = sorted(path.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no eval JSON under {path}")

    scores: dict[str, Score] = {}
    for file in files:
        try:
            payload = json.loads(file.read_text())
        except json.JSONDecodeError as exc:
            raise MalformedEval(f"{file}: {exc}") from exc
        if not isinstance(payload, dict):
            raise MalformedEval(f"{file}: expected a JSON object")
        for task, metrics in payload.items():
            if not isinstance(metrics, dict):
                continue
            if score := parse_task_metrics(metrics, task):
                scores[task] = score
    if not scores:
        raise MalformedEval(f"{path}: no usable metrics in {len(files)} file(s)")
    return scores


def combined_stderr(base: Score, other: Score) -> float | None:
    """Stderr of the difference of two independent means."""
    if base.stderr is None or other.stderr is None:
        return None
    return math.sqrt(base.stderr**2 + other.stderr**2)


def verdict(base: Score, other: Score) -> str:
    """Whether the delta clears 1.96 combined stderr."""
    delta = other.value - base.value
    sigma = combined_stderr(base, other)
    if sigma is None:
        return "no stderr"
    if sigma == 0:
        return "same" if delta == 0 else "significant"
    if abs(delta) <= 1.96 * sigma:
        return "noise"
    return "significant"


def compare(
    baseline: dict[str, Score],
    other: dict[str, Score],
    base_label: str = "baseline",
    other_label: str = "pruned",
) -> str:
    """Markdown table of per-task deltas, baseline-ordered."""
    tasks = list(baseline) + [t for t in other if t not in baseline]
    header = [
        "task",
        "metric",
        base_label,
        other_label,
        "delta",
        "rel",
        "1.96se",
        "verdict",
    ]
    rows: list[list[str]] = []
    for task in tasks:
        base = baseline.get(task)
        new = other.get(task)
        if base is None or new is None:
            present = base or new
            rows.append(
                [
                    task,
                    present.metric if present else "-",
                    f"{base.value:.4f}" if base else "-",
                    f"{new.value:.4f}" if new else "-",
                    "-",
                    "-",
                    "-",
                    "missing",
                ]
            )
            continue
        delta = new.value - base.value
        sigma = combined_stderr(base, new)
        rel = f"{delta / base.value * 100:+.1f}%" if base.value else "-"
        note = verdict(base, new)
        if new.at_chance and not base.at_chance:
            note += ", at chance"
        rows.append(
            [
                task,
                base.metric if base.metric == new.metric else f"{base.metric}/{new.metric}",
                f"{base.value:.4f}",
                f"{new.value:.4f}",
                f"{delta:+.4f}",
                rel,
                f"{1.96 * sigma:.4f}" if sigma is not None else "-",
                note,
            ]
        )
    return report.render_table(header, rows)


def mean_delta(baseline: dict[str, Score], other: dict[str, Score]) -> float | None:
    """Unweighted mean headline delta over tasks both models scored.

    Unweighted on purpose: this is a mean over *benchmarks*, matching how the
    REAP paper's "MC Avg" column is built. It is not a pooled accuracy.
    """
    shared = [t for t in baseline if t in other]
    if not shared:
        return None
    return sum(other[t].value - baseline[t].value for t in shared) / len(shared)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", help="eval directory or label")
    parser.add_argument("other", help="eval directory or label")
    parser.add_argument(
        "--eval-root",
        default=DEFAULT_EVAL_ROOT,
        help=f"root holding per-model eval dirs (default: {DEFAULT_EVAL_ROOT})",
    )
    args = parser.parse_args(argv)

    def resolve(name: str) -> Path:
        """Labels win over cwd-relative paths.

        Checked eval-root-first on purpose: `inco/reap/` is a real directory
        in this repo, so resolving a bare `reap` label against the cwd would
        silently point at the REAP source checkout instead of the results.
        An absolute path still resolves to itself -- `Path("a") / "/b"` is
        `/b` -- so explicit paths keep working.
        """
        under_root = Path(args.eval_root) / name
        return under_root if under_root.is_dir() else Path(name)

    base_dir, other_dir = resolve(args.baseline), resolve(args.other)
    try:
        baseline = load_eval_dir(base_dir)
        other = load_eval_dir(other_dir)
    except (FileNotFoundError, MalformedEval) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"baseline: {base_dir}\nother:    {other_dir}\n")
    print(compare(baseline, other, args.baseline, args.other))
    if (avg := mean_delta(baseline, other)) is not None:
        shared = len([t for t in baseline if t in other])
        print(f"\nmean headline delta over {shared} shared task(s): {avg:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
