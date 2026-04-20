#!/usr/bin/env python3
"""Summarize skip-softmax threshold results into markdown tables.

Reads accuracy (lm-evaluation-harness) and performance (benchmark_serving)
results from ./accuracy/ and ./perf/, groups them by skip-softmax threshold,
and prints nicely formatted markdown tables to stdout.
"""

import json
import os
import re
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_accuracy_dir(dirpath: Path) -> dict | None:
    """Return {model, threshold, kv_dtype, benchmark, metric_value, metric_stderr}."""
    name = dirpath.name
    # Pattern: <model_slug>_thresh-pf<PF>-dc<DC>_kvdtype-<KV>_<benchmark>
    m = re.match(
        r"^(.+?)_thresh-pf([\w.]+)-dc([\w.]+)_kvdtype-(\w+)_(gsm8k|mmlu_pro)$",
        name,
    )
    if not m:
        return None

    model_slug, thresh_prefill, thresh_decode, kv_dtype, benchmark = m.groups()
    threshold = f"pf{thresh_prefill}/dc{thresh_decode}"
    model = model_slug.replace("_", "/", 1)  # first _ is org separator

    # Find the results JSON inside the nested directory
    json_files = list(dirpath.rglob("results_*.json"))
    if not json_files:
        return None

    with open(json_files[0]) as f:
        data = json.load(f)

    results = data.get("results", {})
    if benchmark == "gsm8k":
        entry = results.get("gsm8k", {})
        value = entry.get("exact_match,strict-match")
        stderr = entry.get("exact_match_stderr,strict-match")
        metric_name = "exact_match (strict)"
    elif benchmark == "mmlu_pro":
        entry = results.get("mmlu_pro", {})
        value = entry.get("exact_match,custom-extract")
        stderr = entry.get("exact_match_stderr,custom-extract")
        metric_name = "exact_match"
    else:
        return None

    return {
        "model": model,
        "threshold": threshold,
        "kv_dtype": kv_dtype,
        "benchmark": benchmark,
        "metric_name": metric_name,
        "value": value,
        "stderr": stderr,
    }


def parse_perf_file(filepath: Path) -> dict | None:
    """Return parsed perf result dict."""
    name = filepath.stem
    m = re.match(
        r"^(.+?)_thresh-pf([\w.]+)-dc([\w.]+)_kvdtype-(\w+)"
        r"_isl-(\d+)_osl-(\d+)_conc-(\d+)$",
        name,
    )
    if not m:
        return None

    model_slug, thresh_prefill, thresh_decode, kv_dtype, isl, osl, conc = m.groups()
    threshold = f"pf{thresh_prefill}/dc{thresh_decode}"
    model = model_slug.replace("_", "/", 1)

    with open(filepath) as f:
        data = json.load(f)

    return {
        "model": model,
        "threshold": threshold,
        "kv_dtype": kv_dtype,
        "isl": int(isl),
        "osl": int(osl),
        "concurrency": int(conc),
        "request_throughput": data.get("request_throughput"),
        "output_throughput": data.get("output_throughput"),
        "total_token_throughput": data.get("total_token_throughput"),
        "mean_ttft_ms": data.get("mean_ttft_ms"),
        "median_ttft_ms": data.get("median_ttft_ms"),
        "p99_ttft_ms": data.get("p99_ttft_ms"),
        "mean_tpot_ms": data.get("mean_tpot_ms"),
        "median_tpot_ms": data.get("median_tpot_ms"),
        "p99_tpot_ms": data.get("p99_tpot_ms"),
        "mean_itl_ms": data.get("mean_itl_ms"),
        "median_itl_ms": data.get("median_itl_ms"),
        "p99_itl_ms": data.get("p99_itl_ms"),
        "completed": data.get("completed"),
        "failed": data.get("failed"),
        "duration": data.get("duration"),
    }


def fmt_threshold(t: str) -> str:
    """Pretty-print threshold value."""
    if t in ("none", "pfnone/dcnone"):
        return "None (baseline)"
    return str(t)


def _threshold_sort_key(t: str) -> tuple:
    """Sort key for (possibly composite) threshold strings.

    Accepts either the legacy single-value form (``"none"`` / ``"15000"``) or
    the composite ``"pf<PF>/dc<DC>"`` form emitted by :func:`parse_perf_file`
    and :func:`parse_accuracy_dir`.
    """
    if t in ("none", "pfnone/dcnone"):
        return (0, float("inf"), float("inf"))

    def _num(x: str) -> float:
        if x == "none":
            return float("inf")
        try:
            return float(x)
        except ValueError:
            return float("inf")

    if t.startswith("pf") and "/dc" in t:
        pf = t[len("pf"):].split("/dc", 1)[0]
        dc = t.split("/dc", 1)[1]
        return (1, _num(pf), _num(dc))
    # Legacy single-value form.
    return (1, _num(t), _num(t))


def fmt_val(v, precision=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{precision}f}"
    return str(v)


def fmt_pct(v, precision=2):
    if v is None:
        return "—"
    return f"{v * 100:.{precision}f}%"


def fmt_int(v):
    if v is None:
        return "—"
    return f"{int(v):,}"


# ---------------------------------------------------------------------------
# Collect data
# ---------------------------------------------------------------------------

def collect_accuracy():
    acc_dir = RESULTS_DIR / "accuracy"
    results = []
    if not acc_dir.exists():
        return results
    for entry in sorted(acc_dir.iterdir()):
        if entry.is_dir():
            parsed = parse_accuracy_dir(entry)
            if parsed:
                results.append(parsed)
    return results


def collect_perf():
    perf_dir = RESULTS_DIR / "perf"
    results = []
    if not perf_dir.exists():
        return results
    for entry in sorted(perf_dir.iterdir()):
        if entry.suffix == ".json":
            parsed = parse_perf_file(entry)
            if parsed:
                results.append(parsed)
    return results


# ---------------------------------------------------------------------------
# Render markdown
# ---------------------------------------------------------------------------

def render_accuracy_table(acc_results: list[dict]) -> str:
    """Render a markdown table of accuracy results grouped by model."""
    lines = []
    lines.append("## Accuracy Results")
    lines.append("")

    # Group by model
    models = sorted(set(r["model"] for r in acc_results))
    benchmarks = sorted(set(r["benchmark"] for r in acc_results))

    for model in models:
        model_results = [r for r in acc_results if r["model"] == model]
        thresholds = sorted(
            set(r["threshold"] for r in model_results),
            key=_threshold_sort_key,
        )

        lines.append(f"### {model}")
        lines.append("")

        # Header
        header = "| Threshold |"
        separator = "|---|"
        for bm in benchmarks:
            header += f" {bm} |"
            separator += "---|"
        lines.append(header)
        lines.append(separator)

        # Rows
        for thresh in thresholds:
            row = f"| {fmt_threshold(thresh)} |"
            for bm in benchmarks:
                match = [
                    r for r in model_results
                    if r["threshold"] == thresh and r["benchmark"] == bm
                ]
                if match:
                    r = match[0]
                    stderr_str = f" ± {fmt_val(r['stderr'])}" if r["stderr"] else ""
                    row += f" {fmt_pct(r['value'])}{stderr_str} |"
                else:
                    row += " — |"
            lines.append(row)

        lines.append("")

    return "\n".join(lines)


def render_perf_table(perf_results: list[dict]) -> str:
    """Render markdown tables of perf results grouped by (model, ISL, concurrency)."""
    lines = []
    lines.append("## Performance Results")
    lines.append("")

    models = sorted(set(r["model"] for r in perf_results))

    for model in models:
        model_results = [r for r in perf_results if r["model"] == model]
        lines.append(f"### {model}")
        lines.append("")

        # Group by (ISL, OSL)
        workloads = sorted(set((r["isl"], r["osl"]) for r in model_results))

        for isl, osl in workloads:
            wl_results = [
                r for r in model_results if r["isl"] == isl and r["osl"] == osl
            ]
            concurrencies = sorted(set(r["concurrency"] for r in wl_results))
            thresholds = sorted(
                set(r["threshold"] for r in wl_results),
                key=_threshold_sort_key,
            )

            lines.append(f"#### ISL={fmt_int(isl)}, OSL={osl}")
            lines.append("")

            # --- Throughput sub-table ---
            lines.append("**Throughput (tokens/s)**")
            lines.append("")
            header = "| Threshold |"
            separator = "|---|"
            for conc in concurrencies:
                header += f" conc={conc} (req/s) | conc={conc} (out tok/s) | conc={conc} (total tok/s) |"
                separator += "---|---|---|"
            lines.append(header)
            lines.append(separator)

            for thresh in thresholds:
                row = f"| {fmt_threshold(thresh)} |"
                for conc in concurrencies:
                    match = [
                        r for r in wl_results
                        if r["threshold"] == thresh and r["concurrency"] == conc
                    ]
                    if match:
                        r = match[0]
                        row += f" {fmt_val(r['request_throughput'], 2)} |"
                        row += f" {fmt_val(r['output_throughput'], 1)} |"
                        row += f" {fmt_val(r['total_token_throughput'], 1)} |"
                    else:
                        row += " — | — | — |"
                lines.append(row)

            lines.append("")

            # --- Latency sub-table ---
            lines.append("**Latency (ms)**")
            lines.append("")
            header = "| Threshold |"
            separator = "|---|"
            for conc in concurrencies:
                header += f" conc={conc} TTFT (mean) | conc={conc} TTFT (p99) | conc={conc} TPOT (mean) | conc={conc} ITL (mean) |"
                separator += "---|---|---|---|"
            lines.append(header)
            lines.append(separator)

            for thresh in thresholds:
                row = f"| {fmt_threshold(thresh)} |"
                for conc in concurrencies:
                    match = [
                        r for r in wl_results
                        if r["threshold"] == thresh and r["concurrency"] == conc
                    ]
                    if match:
                        r = match[0]
                        row += f" {fmt_val(r['mean_ttft_ms'], 1)} |"
                        row += f" {fmt_val(r['p99_ttft_ms'], 1)} |"
                        row += f" {fmt_val(r['mean_tpot_ms'], 1)} |"
                        row += f" {fmt_val(r['mean_itl_ms'], 1)} |"
                    else:
                        row += " — | — | — | — |"
                lines.append(row)

            lines.append("")

    return "\n".join(lines)


def render_speedup_table(perf_results: list[dict]) -> str:
    """Render a compact speedup summary vs baseline (thresh=none)."""
    lines = []
    lines.append("## Speedup vs Baseline (threshold=none)")
    lines.append("")

    models = sorted(set(r["model"] for r in perf_results))

    for model in models:
        model_results = [r for r in perf_results if r["model"] == model]
        lines.append(f"### {model}")
        lines.append("")

        baseline_keys = {"none", "pfnone/dcnone"}
        thresholds = sorted(
            set(
                r["threshold"]
                for r in model_results
                if r["threshold"] not in baseline_keys
            ),
            key=_threshold_sort_key,
        )
        if not thresholds:
            lines.append("_No non-baseline thresholds found._")
            lines.append("")
            continue

        workloads = sorted(
            set((r["isl"], r["osl"], r["concurrency"]) for r in model_results)
        )

        header = "| Workload (ISL/OSL/Conc) |"
        separator = "|---|"
        for thresh in thresholds:
            header += f" thresh={thresh} output tok/s | speedup |"
            separator += "---|---|"
        header += " baseline output tok/s |"
        separator += "---|"
        lines.append(header)
        lines.append(separator)

        for isl, osl, conc in workloads:
            row = f"| {fmt_int(isl)}/{osl}/{conc} |"
            baseline = [
                r for r in model_results
                if r["threshold"] in baseline_keys and r["isl"] == isl
                and r["osl"] == osl and r["concurrency"] == conc
            ]
            base_throughput = baseline[0]["output_throughput"] if baseline else None

            for thresh in thresholds:
                match = [
                    r for r in model_results
                    if r["threshold"] == thresh and r["isl"] == isl
                    and r["osl"] == osl and r["concurrency"] == conc
                ]
                if match:
                    t = match[0]["output_throughput"]
                    row += f" {fmt_val(t, 1)} |"
                    if base_throughput and base_throughput > 0:
                        speedup = t / base_throughput
                        row += f" **{speedup:.2f}x** |"
                    else:
                        row += " — |"
                else:
                    row += " — | — |"

            row += f" {fmt_val(base_throughput, 1) if base_throughput else '—'} |"
            lines.append(row)

        lines.append("")

    return "\n".join(lines)


def main():
    acc_results = collect_accuracy()
    perf_results = collect_perf()

    parts = []
    parts.append("# Skip-Softmax Threshold Results Summary")
    parts.append("")
    parts.append("> Auto-generated by `summarize_results.py`")
    parts.append("")

    if acc_results:
        parts.append(render_accuracy_table(acc_results))
    if perf_results:
        parts.append(render_perf_table(perf_results))
        parts.append(render_speedup_table(perf_results))

    output = "\n".join(parts)
    print(output)

    # Also write to file
    out_path = RESULTS_DIR / "SUMMARY.md"
    with open(out_path, "w") as f:
        f.write(output + "\n")
    print(f"\n---\nWritten to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
