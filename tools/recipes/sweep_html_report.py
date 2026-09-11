#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Generate a self-contained HTML summary from completed recipe sweeps."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

STAGES = (
    (
        "parallel-layout",
        "Parallel layout",
        "parallel-layout-recommendation.json",
    ),
    (
        "concurrency-tuning",
        "Concurrency",
        "concurrency-recommendation.json",
    ),
    ("runtime-tuning", "Scheduler", "recommendation.json"),
)

DEFAULT_TTFT_SLA_MS: float | None = None
DEFAULT_TPOT_SLA_MS: float | None = None


def _read_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object.")
    return value


def _summary_runs(stage_dir: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in sorted(stage_dir.glob("**/summary.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            runs.extend(item for item in value if isinstance(item, dict))
    return runs


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _fmt(value: Any, digits: int = 1) -> str:
    number = _number(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}"


def _fmt_int(value: Any) -> str:
    number = _number(value)
    return "-" if number is None else f"{number:.0f}"


def _fmt_percent(value: Any) -> str:
    number = _number(value)
    return "-" if number is None else f"{number:.1f}%"


def _text(value: Any) -> str:
    if value is None:
        return "-"
    return escape(str(value))


def _resolve_sla(
    reports: dict[str, dict[str, Any] | None],
    key: str,
    expected: float | None,
) -> float | None:
    values: dict[str, float | None] = {}
    for stage, label, _ in STAGES:
        report = reports[stage]
        if report is not None:
            values[label] = _number((report.get("slo") or {}).get(key))

    non_null = {value for value in values.values() if value is not None}
    if expected is not None:
        mismatched = {
            label: value for label, value in values.items() if value != expected
        }
        if mismatched:
            details = ", ".join(
                f"{label}={value}" for label, value in mismatched.items()
            )
            raise ValueError(
                f"Recommendation {key} SLA does not match the command-line "
                f"value {expected}: {details}"
            )
        return expected

    if len(non_null) > 1:
        details = ", ".join(f"{label}={value}" for label, value in values.items())
        raise ValueError(f"Recommendation {key} SLA values disagree: {details}")
    return next(iter(non_null), None)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return '<p class="muted">No completed candidates were found.</p>'
    head = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = []
    for row in rows:
        cells = "".join(f"<td>{_text(cell)}</td>" for cell in row)
        body.append("<tr>" + cells + "</tr>")
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def _status_badge(value: Any) -> str:
    status = str(value or "unknown").replace("_", " ")
    css_class = "pass" if value == "sla_feasible" else "warn"
    return f'<span class="badge {css_class}">{escape(status)}</span>'


def _sla_label(candidate: dict[str, Any]) -> str:
    if candidate.get("p99_sla_eligible") is True:
        return "PASS"
    if candidate.get("p99_sla_eligible") is False:
        return "FAIL"
    return "-"


def _scheduler_value(candidate: dict[str, Any], key: str) -> str:
    value = candidate.get(key)
    return "vLLM default" if value is None else str(value)


def _selected_settings(
    parallel: dict[str, Any] | None,
    concurrency: dict[str, Any] | None,
    scheduler: dict[str, Any] | None,
) -> list[list[Any]]:
    parallel_rec = (parallel or {}).get("recommended") or {}
    concurrency_rec = (concurrency or {}).get("recommended") or {}
    scheduler_rec = (scheduler or {}).get("recommended") or {}
    max_num_seqs = (
        _scheduler_value(scheduler_rec, "max_num_seqs")
        if scheduler is not None
        else "-"
    )
    max_num_batched_tokens = (
        _scheduler_value(scheduler_rec, "max_num_batched_tokens")
        if scheduler is not None
        else "-"
    )
    return [
        ["tensor-parallel-size", parallel_rec.get("tensor_parallel_size")],
        ["data-parallel-size", parallel_rec.get("data_parallel_size")],
        ["max concurrency", concurrency_rec.get("max_concurrency")],
        ["max-num-seqs", max_num_seqs],
        ["max-num-batched-tokens", max_num_batched_tokens],
    ]


def _stage_summary(
    reports: dict[str, dict[str, Any] | None],
) -> list[list[Any]]:
    rows = []
    for stage, label, _ in STAGES:
        report = reports[stage]
        if report is None:
            rows.append([label, "not available", "-", "-", "-", "-"])
            continue
        measured = report.get("measured") or {}
        rows.append(
            [
                label,
                str(report.get("status", "unknown")).replace("_", " "),
                _fmt(measured.get("mean_output_throughput")),
                _fmt(measured.get("mean_request_goodput"), 3),
                _fmt_int(measured.get("mean_p99_ttft_ms")),
                _fmt(measured.get("mean_p99_tpot_ms")),
            ]
        )
    return rows


def _parallel_rows(report: dict[str, Any] | None) -> list[list[Any]]:
    rows = []
    for candidate in (report or {}).get("candidates", []):
        rows.append(
            [
                candidate.get("tensor_parallel_size"),
                candidate.get("data_parallel_size"),
                candidate.get("max_num_seqs"),
                candidate.get("max_num_batched_tokens"),
                _fmt(candidate.get("mean_output_throughput")),
                _fmt_int(candidate.get("mean_p99_ttft_ms")),
                _fmt(candidate.get("mean_p99_tpot_ms")),
                _fmt_percent(candidate.get("combined_compliance_percent")),
                _sla_label(candidate),
            ]
        )
    return rows


def _concurrency_rows(report: dict[str, Any] | None) -> list[list[Any]]:
    candidates = (report or {}).get("candidates", [])
    candidates = sorted(candidates, key=lambda item: item.get("max_concurrency", 0))
    rows = []
    for candidate in candidates:
        rows.append(
            [
                candidate.get("max_concurrency"),
                _fmt(candidate.get("mean_output_throughput")),
                _fmt(candidate.get("mean_request_throughput"), 3),
                _fmt(candidate.get("mean_request_goodput"), 3),
                _fmt_percent(candidate.get("combined_compliance_percent")),
                _fmt_int(candidate.get("mean_p99_ttft_ms")),
                _fmt(candidate.get("mean_p99_tpot_ms")),
                _sla_label(candidate),
            ]
        )
    return rows


def _scheduler_rows(report: dict[str, Any] | None) -> list[list[Any]]:
    rows = []
    for candidate in (report or {}).get("candidates", []):
        rows.append(
            [
                _scheduler_value(candidate, "max_num_seqs"),
                _scheduler_value(candidate, "max_num_batched_tokens"),
                _fmt(candidate.get("mean_output_throughput")),
                _fmt(candidate.get("mean_request_goodput"), 3),
                _fmt_int(candidate.get("mean_p99_ttft_ms")),
                _fmt(candidate.get("mean_p99_tpot_ms")),
                _fmt_percent(candidate.get("combined_compliance_percent")),
                _sla_label(candidate),
            ]
        )
    return rows


def _metadata(runs: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    for stage, _, _ in STAGES:
        if runs[stage]:
            first = runs[stage][0]
            return {
                "model": first.get("model_id", "unknown"),
                "input_tokens": first.get("random_input_len"),
                "output_tokens": first.get("random_output_len"),
                "request_rate": first.get("request_rate"),
            }
    return {
        "model": "unknown",
        "input_tokens": None,
        "output_tokens": None,
        "request_rate": None,
    }


def _observations(
    reports: dict[str, dict[str, Any] | None],
    runs: dict[str, list[dict[str, Any]]],
) -> list[tuple[str, str]]:
    observations: list[tuple[str, str]] = []
    parallel = reports["parallel-layout"]
    concurrency = reports["concurrency-tuning"]
    scheduler = reports["runtime-tuning"]

    parallel_candidates = (parallel or {}).get("candidates", [])
    if len(parallel_candidates) == 1:
        observations.append(
            (
                "warning",
                (
                    "Only one parallel-layout candidate completed. The result "
                    "validates that layout but does not establish an optimum."
                ),
            )
        )

    if parallel and concurrency and runs["parallel-layout"]:
        representative = runs["parallel-layout"][0].get("max_concurrency")
        concurrency_values = {
            item.get("max_concurrency") for item in concurrency.get("candidates", [])
        }
        parallel_passed = parallel_candidates and parallel_candidates[0].get(
            "p99_sla_eligible"
        )
        if representative not in concurrency_values and parallel_passed:
            observations.append(
                (
                    "warning",
                    (
                        f"Concurrency {representative} passed in the "
                        "parallel-layout stage but was not included in the "
                        "concurrency grid. Treat the automated concurrency "
                        "choice as conservative."
                    ),
                )
            )

    concurrency_candidates = (concurrency or {}).get("candidates", [])
    first_failure = next(
        (
            item
            for item in sorted(
                concurrency_candidates,
                key=lambda candidate: candidate.get("max_concurrency", 0),
            )
            if item.get("p99_sla_eligible") is False
        ),
        None,
    )
    if first_failure and concurrency:
        slo = concurrency.get("slo") or {}
        ttft = _number(first_failure.get("mean_p99_ttft_ms"))
        tpot = _number(first_failure.get("mean_p99_tpot_ms"))
        ttft_slo = _number(slo.get("ttft_ms"))
        tpot_slo = _number(slo.get("tpot_ms"))
        if (
            ttft
            and ttft_slo
            and ttft > ttft_slo
            and tpot is not None
            and (tpot_slo is None or tpot <= tpot_slo)
        ):
            observations.append(
                (
                    "info",
                    (
                        "TTFT is the first limiting SLA; TPOT remains within "
                        "its objective at the first failing concurrency."
                    ),
                )
            )

    scheduler_candidates = (scheduler or {}).get("candidates", [])
    scheduler_throughputs = [
        number
        for number in (
            _number(item.get("mean_output_throughput")) for item in scheduler_candidates
        )
        if number is not None
    ]
    if len(scheduler_throughputs) > 1 and min(scheduler_throughputs) > 0:
        spread = (max(scheduler_throughputs) / min(scheduler_throughputs) - 1) * 100
        if spread <= 3:
            observations.append(
                (
                    "info",
                    (
                        "Scheduler output throughput is tightly grouped: "
                        f"{spread:.2f}% total spread across measured candidates."
                    ),
                )
            )

    if not observations:
        observations.append(
            ("info", "No automatic coverage or SLA warnings were detected.")
        )
    return observations


def _observation_html(observations: list[tuple[str, str]]) -> str:
    return "".join(
        f'<div class="notice {escape(level)}">{escape(message)}</div>'
        for level, message in observations
    )


def generate_report(
    root: Path,
    output: Path,
    title: str,
    ttft_sla_ms: float | None = None,
    tpot_sla_ms: float | None = None,
) -> None:
    reports = {stage: _read_object(root / filename) for stage, _, filename in STAGES}
    if not any(reports.values()):
        names = ", ".join(filename for _, _, filename in STAGES)
        raise ValueError(f"No recommendation files found under {root}: {names}")

    ttft_sla_ms = _resolve_sla(reports, "ttft_ms", ttft_sla_ms)
    tpot_sla_ms = _resolve_sla(reports, "tpot_ms", tpot_sla_ms)

    runs = {stage: _summary_runs(root / "results" / stage) for stage, _, _ in STAGES}
    meta = _metadata(runs)
    observations = _observations(reports, runs)
    concurrency = reports["concurrency-tuning"] or {}
    scheduler = reports["runtime-tuning"] or {}
    measured = scheduler.get("measured") or concurrency.get("measured") or {}
    recommended_concurrency = (concurrency.get("recommended") or {}).get(
        "max_concurrency"
    )
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    styles = """
:root { color-scheme: light; --ink: #17324d; --blue: #146c94;
  --teal: #2a9d8f; --green: #2e7d32; --amber: #d97706;
  --red: #c0392b; --line: #d8dee4; --soft: #f3f5f7; }
* { box-sizing: border-box; }
body { margin: 0; background: #eef2f5; color: var(--ink);
  font: 15px/1.45 Arial, sans-serif; }
main { max-width: 1180px; margin: 28px auto; padding: 0 20px 40px; }
header, section { background: white; border: 1px solid var(--line);
  border-radius: 12px; margin-bottom: 18px; padding: 24px; }
h1 { margin: 0 0 6px; font-size: 30px; }
h2 { margin: 0 0 14px; font-size: 21px; }
h3 { margin: 20px 0 10px; font-size: 16px; color: var(--blue); }
.sub, .muted { color: #607486; }
.cards { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px; margin-top: 20px; }
.card { background: var(--soft); border-radius: 9px; padding: 14px; }
.card strong { display: block; font-size: 22px; }
.card span { color: #607486; font-size: 12px; }
.badge { border-radius: 99px; display: inline-block; font-size: 12px;
  font-weight: 700; padding: 4px 9px; text-transform: uppercase; }
.badge.pass { background: #eaf5ec; color: var(--green); }
.badge.warn { background: #fff4e5; color: var(--amber); }
.notice { border-left: 5px solid var(--blue); margin: 9px 0;
  padding: 11px 14px; background: #eaf4f8; }
.notice.warning { border-color: var(--amber); background: #fff4e5; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { background: var(--ink); color: white; text-align: left; }
th, td { border: 1px solid var(--line); padding: 8px 9px; white-space: nowrap; }
tbody tr:nth-child(even) { background: var(--soft); }
footer { color: #607486; font-size: 12px; text-align: center; }
@media (max-width: 760px) { .cards { grid-template-columns: 1fr 1fr; } }
@media print { body { background: white; } main { margin: 0; max-width: none; }
  header, section { border: 0; break-inside: avoid; padding: 12px 0; }
  .table-wrap { overflow: visible; } }
"""

    body = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<style>{styles}</style>
</head>
<body><main>
<header>
  <div class="sub">vLLM recipe sweep report</div>
  <h1>{escape(title)}</h1>
  <div class="sub">{_text(meta["model"])} | {_text(meta["input_tokens"])}
  input / {_text(meta["output_tokens"])} output tokens |
  request rate {_text(meta["request_rate"])}</div>
  <div class="cards">
    <div class="card"><strong>{_text(recommended_concurrency)}</strong>
      <span>recommended concurrency</span></div>
    <div class="card"><strong>{_fmt(measured.get("mean_output_throughput"))}</strong>
      <span>output tokens/s</span></div>
    <div class="card"><strong>{_fmt_int(measured.get("mean_p99_ttft_ms"))} ms</strong>
      <span>mean p99 TTFT</span></div>
    <div class="card"><strong>{_fmt(measured.get("mean_p99_tpot_ms"))} ms</strong>
      <span>mean p99 TPOT</span></div>
    <div class="card"><strong>{_fmt_int(ttft_sla_ms)} ms</strong>
      <span>TTFT SLA</span></div>
    <div class="card"><strong>{_fmt(tpot_sla_ms)} ms</strong>
      <span>TPOT SLA</span></div>
  </div>
</header>
<section>
  <h2>Recommended configuration</h2>
  {
        _table(
            ["Setting", "Value"],
            _selected_settings(
                reports["parallel-layout"],
                reports["concurrency-tuning"],
                reports["runtime-tuning"],
            ),
        )
    }
</section>
<section>
  <h2>Stage summary</h2>
  {
        _table(
            [
                "Stage",
                "Status",
                "Output tok/s",
                "Goodput req/s",
                "p99 TTFT ms",
                "p99 TPOT ms",
            ],
            _stage_summary(reports),
        )
    }
  <h3>Automatic observations</h3>
  {_observation_html(observations)}
</section>
<section>
  <h2>Parallel-layout candidates {
        _status_badge((reports["parallel-layout"] or {}).get("status"))
    }</h2>
  {
        _table(
            [
                "TP",
                "DP",
                "Max seqs",
                "Batch tokens",
                "Output tok/s",
                "p99 TTFT ms",
                "p99 TPOT ms",
                "Compliance",
                "SLA",
            ],
            _parallel_rows(reports["parallel-layout"]),
        )
    }
</section>
<section>
  <h2>Concurrency candidates {
        _status_badge((reports["concurrency-tuning"] or {}).get("status"))
    }</h2>
  {
        _table(
            [
                "Concurrency",
                "Output tok/s",
                "Req/s",
                "Goodput req/s",
                "Compliance",
                "p99 TTFT ms",
                "p99 TPOT ms",
                "SLA",
            ],
            _concurrency_rows(reports["concurrency-tuning"]),
        )
    }
</section>
<section>
  <h2>Scheduler candidates {
        _status_badge((reports["runtime-tuning"] or {}).get("status"))
    }</h2>
  {
        _table(
            [
                "Max seqs",
                "Batch tokens",
                "Output tok/s",
                "Goodput req/s",
                "p99 TTFT ms",
                "p99 TPOT ms",
                "Compliance",
                "SLA",
            ],
            _scheduler_rows(reports["runtime-tuning"]),
        )
    }
</section>
<footer>Generated {escape(generated)} from completed recommendation and
summary JSON files. Warmup results are excluded by the recommenders.</footer>
</main></body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(body, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a self-contained HTML report from recipe sweeps."
    )
    parser.add_argument(
        "--sweep-dir",
        default=None,
        help="Sweep package directory (default: beside this script).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output HTML path (default: <sweep-dir>/sweep-report.html).",
    )
    parser.add_argument(
        "--title",
        default="vLLM Recipe Sweep",
        help="Report title.",
    )
    parser.add_argument(
        "--ttft-sla-ms",
        type=float,
        default=DEFAULT_TTFT_SLA_MS,
        help="Expected TTFT objective; must match every recommendation file.",
    )
    parser.add_argument(
        "--tpot-sla-ms",
        type=float,
        default=DEFAULT_TPOT_SLA_MS,
        help="Expected TPOT objective; must match every recommendation file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in ("ttft_sla_ms", "tpot_sla_ms"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            option = "--" + name.replace("_", "-")
            raise ValueError(f"{option} must be greater than zero.")

    script_dir = Path(__file__).absolute().parent
    root = Path(args.sweep_dir).resolve() if args.sweep_dir else script_dir
    output = Path(args.output).resolve() if args.output else root / "sweep-report.html"
    generate_report(
        root,
        output,
        args.title,
        ttft_sla_ms=args.ttft_sla_ms,
        tpot_sla_ms=args.tpot_sla_ms,
    )
    print(f"Wrote HTML sweep report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
