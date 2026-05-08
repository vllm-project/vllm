#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate an interactive MoE HTML report from EPLB JSONL data.

Produces a single-file HTML viewer that shows per-step, per-layer token
distribution across EP ranks and hot/cold experts.

Usage:
    ./moe_report.py data.jsonl -o report.html
"""

# === FOR AI AGENTS ===
# This script follows the conventions in .cursor/rules/script-conventions.mdc.
# If you modify this script, keep all comments up to date, comment any new
# logic you add, and preserve the colored output / session summary behaviour
# described in that rule.

import argparse
import json
import os
import sys
import time
from pathlib import Path


SCRIPT_NAME = "moe_report"


def _log(msg: str) -> None:
    sys.stdout.write(f"\033[1;36m[{SCRIPT_NAME}]\033[0m {msg}\n")
    sys.stdout.flush()


def _err(msg: str) -> None:
    sys.stderr.write(
        f"\033[0;31m[{SCRIPT_NAME}] ERROR:\033[0m {msg}\n"
    )
    sys.stderr.flush()


def _ok(msg: str) -> None:
    sys.stdout.write(f"\033[0;32m[{SCRIPT_NAME}]\033[0m {msg}\n")
    sys.stdout.flush()


class _Parser(argparse.ArgumentParser):
    """Print the full help on any usage error (same output as -h)."""

    def error(self, message: str) -> None:
        self.print_help(sys.stderr)
        self.exit(2, f"\nerror: {message}\n")


def _step_imbalance(rec: dict) -> float | None:
    """Per-step imbalance, computed identically to ``computeImbalance`` in
    the HTML viewer (moe_report_template.html: ``computeImbalance``):

        for each layer:
            row  = tokens_per_rank for that layer
            mean = sum(row) / num_ranks
            mx   = max(row)
            ratio = mx / mean        (skipped when mean == 0)
        imbalance = sum(ratios) / num_layers

    Returns ``None`` when the record lacks the ``tokens`` matrix (older
    JSONLs / non-stats records). The aggregate "average across steps" is
    then computed by ``main()`` as the arithmetic mean of these per-step
    values, which is the same number the user sees in the HTML stats bar
    averaged step-over-step.
    """
    tokens = rec.get("tokens")
    nr = rec.get("num_ranks")
    nl = rec.get("num_layers")
    if not tokens or not nr or not nl:
        return None
    total = 0.0
    for row in tokens:
        sm = sum(row)
        if sm <= 0:
            continue
        mean = sm / nr
        if mean <= 0:
            continue
        total += max(row) / mean
    return total / nl


def _is_non_empty_step(rec: dict) -> bool:
    """A step is non-empty if at least one expert received at least one
    token. Matches the definition used by ``variance.py`` so both tools
    filter records identically.
    """
    el = rec.get("expert_load")
    if not el:
        return False
    return any(any(v > 0 for v in row) for row in el)


def _parse_stats_jsonl(jsonl_path: Path) -> tuple[
    list[dict],
    dict | None,
    list[dict],
    int,
]:
    """Single-pass JSONL parser that recognises three record types:

    * ``eplb_load_stats`` — the per-step stats vLLM writes; non-empty ones
      are returned as the main step list (with extra ``bench_run_index`` /
      ``bench_run_label`` fields injected based on the most recent
      preceding ``bench_run_start`` marker).
    * ``session_metadata`` — one-time marker (per JSONL) written by
      ``nvtx.sh`` before launching ``vllm serve``; carries server argv,
      schedule path, host, etc.
    * ``bench_run_start`` — marker written by ``nvtx.sh`` before each
      ``vllm bench serve`` invocation; collected in order so the HTML
      viewer can show "Run #N (label)" next to each step.

    Records of any other type are silently ignored — keeps the parser
    forward-compatible with future marker kinds.
    """
    steps: list[dict] = []
    bench_runs: list[dict] = []
    session_meta: dict | None = None
    skipped_empty = 0
    current_run_pos: int | None = None
    current_run_index: int | None = None
    current_run_label: str | None = None

    with open(jsonl_path) as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{jsonl_path}:{i}: invalid JSON: {exc}"
                ) from exc
            rt = rec.get("record_type")
            if rt == "session_metadata":
                # First marker wins: in case the same JSONL was somehow
                # appended-to twice, the original session is the source of
                # truth for "what server produced this file".
                if session_meta is None:
                    session_meta = rec
            elif rt == "bench_run_start":
                bench_runs.append(rec)
                current_run_pos = len(bench_runs) - 1
                current_run_index = rec.get("run_index")
                current_run_label = rec.get("label")
            elif rt == "eplb_load_stats":
                if not _is_non_empty_step(rec):
                    skipped_empty += 1
                    continue
                # Stitch the marker context onto the step so the HTML viewer
                # doesn't have to re-derive the run for each step at render
                # time.
                rec["bench_run"] = current_run_pos
                rec["bench_run_index"] = current_run_index
                rec["bench_run_label"] = current_run_label
                steps.append(rec)
            # Unknown record_type → ignored by design.

    return steps, session_meta, bench_runs, skipped_empty


def _parse_perf_data(path: Path) -> list[list[dict]]:
    """Parse a text file containing one or more ``Serving Benchmark Result``
    blocks (the stdout of ``vllm bench serve`` runs concatenated together).

    Each block is delimited by lines that start with ``=`` — the opening
    bar carries the literal ``Serving Benchmark Result`` title, the closing
    bar is just equals signs. Inside, each row is one of:

    * **metric** — ``"<label>:<padding><value>"`` (the only thing we need
      to align across runs with arrows).
    * **header** — section dividers like
      ``---------------Time to First Token----------------`` or the
      surrounding ``=`` bars; preserved verbatim, not arrow-merged.

    Lines outside any block (CLI banner, vLLM warning spam, etc.) are
    ignored. The function returns a list of blocks; each block is a list
    of row dicts in original order. The number of blocks may be 1, 2, or
    more — formatting handles all cases.
    """
    blocks: list[list[dict]] = []
    current: list[dict] | None = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        is_eq_bar = stripped.startswith("=") and set(stripped) <= set("= ")
        is_title_bar = (
            stripped.startswith("=")
            and "Serving Benchmark Result" in stripped
        )
        if is_title_bar:
            current = [{"kind": "header", "text": line}]
            blocks.append(current)
            continue
        if current is None:
            continue
        if is_eq_bar:
            # Closing `=` bar. Capture and end the block.
            current.append({"kind": "header", "text": line})
            current = None
            continue
        if stripped.startswith("-") and set(stripped) - set("- ") and ":" not in line:
            # Sub-section header like `---------------Time to First Token-----`.
            # Excludes things like `--metric: 1.23` (have `:`), and pure dash
            # bars (no letters → set difference is empty).
            current.append({"kind": "header", "text": line})
            continue
        if ":" in line:
            label, _, val = line.partition(":")
            current.append({
                "kind": "metric",
                "label": label,
                "value": val.strip(),
                "raw": line,
            })
            continue
        # Anything else inside a block is rare (blank line) — skip silently.
    return blocks


# Sentinel chars wrapping the label and the best-value column for any
# row whose label is in ``_PERF_HIGHLIGHT_MAX_LABELS``. Picked so the JS
# side can swap them for `<span class="hl-bold">…</span>` AFTER the
# normal HTML-escape pass — the chosen control chars are not affected by
# the `&<>` escape and are extremely unlikely to appear in benchmark
# stdout, so we don't need a fancier marker scheme.
_HL_OPEN = "\x01"
_HL_CLOSE = "\x02"

# Metrics whose best run should be highlighted (highest value wins). Add
# more here when needed; throughput-style metrics belong in this set.
# For latency-style metrics ("smaller is better") a separate min-set
# would be needed — none requested yet.
_PERF_HIGHLIGHT_MAX_LABELS = {
    "Total token throughput (tok/s)",
}


def _format_perf_table(blocks: list[list[dict]]) -> str:
    """Render parsed blocks as a single column-aligned text view.

    For metric rows shared across blocks, values are printed as
    ``v0 -> v1 -> ... -> vN`` with each column ljust'd to its widest
    value so the arrows stack into clean columns. For header rows, the
    text from the first block is reused verbatim (sub-section dividers
    look identical between runs of the same benchmark).

    Single-block input is returned as-is (no arrows, since there is
    nothing to compare against). Empty input returns an empty string.
    Missing values for a metric in some block (uneven runs) are left
    blank — they still produce a column so arrow alignment with neighbours
    is preserved.
    """
    if not blocks:
        return ""

    n_runs = len(blocks)

    # Map label -> list of values per block (filled as we walk each block).
    # Ordering is taken from the master (first) block.
    label_to_values: dict[str, list[str]] = {}
    for blk in blocks:
        seen_in_block: set[str] = set()
        for row in blk:
            if row["kind"] != "metric":
                continue
            label = row["label"]
            label_to_values.setdefault(label, [""] * n_runs)
            seen_in_block.add(label)
        # Now fill in actual values (keyed by block index).
    # Rewalk to populate (cleaner than the seen tracking above).
    for i, blk in enumerate(blocks):
        for row in blk:
            if row["kind"] != "metric":
                continue
            vals = label_to_values.setdefault(row["label"], [""] * n_runs)
            vals[i] = row["value"]

    # Column widths for run 0..n_runs-1.
    col_widths = [0] * n_runs
    for vals in label_to_values.values():
        for i, v in enumerate(vals):
            col_widths[i] = max(col_widths[i], len(v))

    out_lines: list[str] = []
    master = blocks[0]
    for row in master:
        if row["kind"] == "header":
            out_lines.append(row["text"])
            continue
        label = row["label"]
        vals = label_to_values.get(label, [""] * n_runs)
        if n_runs == 1:
            # No arrows — keep the original line verbatim.
            out_lines.append(row["raw"])
            continue
        # Reuse the original "<label>:<padding>" prefix so label-to-value
        # column alignment matches the original `vllm bench serve` output.
        raw = row["raw"]
        after_colon_idx = raw.find(":") + 1
        after_colon = raw[after_colon_idx:]
        n_spaces = len(after_colon) - len(after_colon.lstrip())

        # Highlight the best (max) value's column AND the metric label
        # for selected throughput-style metrics. The wrap chars are
        # zero-width visually here; the HTML viewer swaps them for
        # `<span class="hl-bold">…</span>` to render bold-white.
        highlight = label in _PERF_HIGHLIGHT_MAX_LABELS
        best_idx = -1
        if highlight:
            parsed: list[tuple[float, bool]] = []
            for v in vals:
                try:
                    parsed.append((float(v), True))
                except ValueError:
                    parsed.append((float("-inf"), False))
            if any(ok for _, ok in parsed):
                best_idx = max(range(len(parsed)), key=lambda i: parsed[i][0])

        cols = [v.ljust(col_widths[i]) for i, v in enumerate(vals)]
        if highlight and best_idx >= 0:
            cols[best_idx] = _HL_OPEN + cols[best_idx] + _HL_CLOSE
        prefix_label = (
            _HL_OPEN + label + ":" + _HL_CLOSE if highlight else label + ":"
        )
        prefix = prefix_label + " " * n_spaces
        out_lines.append(prefix + " -> ".join(cols).rstrip())
    return "\n".join(out_lines)


def _parse_vllm_argv(vllm_argv: list[str] | None) -> dict:
    """Pull a few well-known flags out of a `vllm serve` argv list so the
    HTML viewer's "Server" row can show real `tp/pp/dp` numbers instead of
    the hard-coded "1" defaults inherited from the legacy template.

    Recognises both short (`-tp 1`) and long (`--tensor-parallel-size 1`)
    forms, with both space- and equals-separated values. Anything we don't
    recognise is ignored.
    """
    parsed: dict = {}
    if not vllm_argv:
        return parsed

    # Map every variant of a flag to a canonical output field.
    aliases = {
        "tp": [
            "-tp", "--tp",
            "--tensor-parallel-size", "--tensor_parallel_size",
        ],
        "pp": [
            "-pp", "--pp",
            "--pipeline-parallel-size", "--pipeline_parallel_size",
        ],
        "dp": [
            "-dp", "--dp",
            "--data-parallel-size", "--data_parallel_size",
        ],
        "model": [
            "--model",
        ],
        "port": [
            "--port", "-p",
        ],
        "stream_interval": [
            "--stream-interval", "--stream_interval",
        ],
    }

    flag_to_field: dict[str, str] = {}
    for field, names in aliases.items():
        for name in names:
            flag_to_field[name] = field

    int_fields = {"tp", "pp", "dp", "port", "stream_interval"}

    # vLLM serve has the model as the first positional after `serve`. Try
    # to honour that as a fallback if `--model` was not used.
    if "serve" in vllm_argv:
        i = vllm_argv.index("serve") + 1
        if i < len(vllm_argv) and not vllm_argv[i].startswith("-"):
            parsed["model"] = vllm_argv[i]

    bool_flags_present: set[str] = set()
    bool_flag_names = {
        "--enable-expert-parallel": "enable_expert_parallel",
        "--enable-eplb": "enable_eplb",
        "--enforce-eager": "enforce_eager",
        "--no-async-scheduling": "no_async_scheduling",
        "--language-model-only": "language_model_only",
    }

    i = 0
    while i < len(vllm_argv):
        tok = vllm_argv[i]
        # Boolean toggles: presence is the value.
        if tok in bool_flag_names:
            bool_flags_present.add(bool_flag_names[tok])
            i += 1
            continue
        # Equals form: --tp=8
        if "=" in tok:
            key, _, val = tok.partition("=")
            field = flag_to_field.get(key)
            if field:
                if field in int_fields:
                    try:
                        parsed[field] = int(val)
                    except ValueError:
                        parsed[field] = val
                else:
                    parsed[field] = val
            i += 1
            continue
        # Space form: --tp 8
        field = flag_to_field.get(tok)
        if field and i + 1 < len(vllm_argv):
            val = vllm_argv[i + 1]
            if field in int_fields:
                try:
                    parsed[field] = int(val)
                except ValueError:
                    parsed[field] = val
            else:
                parsed[field] = val
            i += 2
            continue
        i += 1

    for f in bool_flags_present:
        parsed[f] = True
    return parsed


def main() -> None:
    t0 = time.time()
    parser = _Parser(
        description=(
            "Generate an interactive MoE analysis report from EPLB JSONL."
        ),
        epilog=(
            "example:\n"
            "  %(prog)s data.jsonl -o report.html"
        ),
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog, width=80,
        ),
    )
    parser.add_argument(
        "jsonl",
        type=Path,
        help=(
            "Path to eplb_data.jsonl produced by vLLM with "
            "expert_load_stats_path=<path>."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output HTML file (default: moe_report.html next to the JSONL)."
        ),
    )
    parser.add_argument(
        "--perf-data",
        type=Path,
        default=None,
        help=(
            "Optional path to a text file containing one or more "
            "`Serving Benchmark Result` blocks (e.g. captured stdout from "
            "consecutive `vllm bench serve` runs). The script merges them "
            "into a column-aligned table where each metric is shown as "
            "`v0 -> v1 -> ...` so per-run differences are easy to eyeball "
            "in the HTML report."
        ),
    )
    args = parser.parse_args()

    jsonl_path: Path = args.jsonl
    if not jsonl_path.exists():
        parser.error(f"File not found: {jsonl_path}")
    if args.output is not None and args.output.suffix.lower() != ".html":
        parser.error(
            f"-o/--output must be a .html file, got: {args.output.name}"
        )
    if args.perf_data is not None and not args.perf_data.is_file():
        parser.error(f"--perf-data: file not found: {args.perf_data}")

    perf_data_text: str | None = None
    if args.perf_data is not None:
        perf_blocks = _parse_perf_data(args.perf_data)
        if not perf_blocks:
            _err(
                f"No `Serving Benchmark Result` blocks found in "
                f"{args.perf_data}; --perf-data ignored."
            )
        else:
            perf_data_text = _format_perf_table(perf_blocks)
            _log(
                f"Parsed {len(perf_blocks)} benchmark block(s) from "
                f"{args.perf_data}"
            )

    try:
        steps, session_meta, bench_runs, skipped = _parse_stats_jsonl(jsonl_path)
    except ValueError as exc:
        parser.error(str(exc))

    if not steps:
        parser.error(f"No non-empty records found in {jsonl_path}")

    # Enrich session_meta with parsed flags from vllm_argv so the HTML
    # viewer can render real tp/pp/dp/etc. instead of the legacy hard-coded
    # "1" defaults that the template falls back to when meta is missing.
    if session_meta is not None:
        session_meta["parsed"] = _parse_vllm_argv(session_meta.get("vllm_argv"))

    _log(
        f"Loaded {len(steps)} step(s) from {jsonl_path}"
        + (f" (skipped {skipped} empty)" if skipped else "")
        + (
            f", session_metadata: {'yes' if session_meta else 'no'}, "
            f"bench_run_start markers: {len(bench_runs)}"
        )
    )

    # Per-step imbalance for every step we are about to embed (same formula
    # as the HTML viewer's computeImbalance — see _step_imbalance docstring).
    # Used both for the new "Avg imbalance (all steps)" pill in the HTML
    # header and for the per-step table printed at the end of this script.
    step_imbalances: list[tuple[int, float]] = []
    for s in steps:
        imb = _step_imbalance(s)
        if imb is not None:
            step_imbalances.append((s.get("step", -1), imb))
    avg_imbalance: float | None
    if step_imbalances:
        avg_imbalance = sum(v for _, v in step_imbalances) / len(step_imbalances)
    else:
        avg_imbalance = None

    template_path = Path(__file__).parent / "moe_report_template.html"
    if not template_path.exists():
        parser.error(f"Template not found: {template_path}")
    template = template_path.read_text()

    output_path = args.output or jsonl_path.parent / "moe_report.html"

    steps_json = json.dumps(steps, separators=(",", ":"))
    html = template.replace("/*STEPS_DATA*/[]", steps_json)

    # NSYS_DATA / SCHEDULE_META placeholders are kept renderable for the
    # template — the viewer treats null as "no nsys timing / no offline
    # schedule metadata" and silently hides the corresponding rows.
    html = html.replace("/*NSYS_DATA*/null", "null")
    html = html.replace("/*SCHEDULE_META*/null", "null")

    selected_json = json.dumps(None)
    html = html.replace("/*SELECTED_STEP*/null", selected_json)

    # SESSION_META: how the server was launched (parsed from session_metadata
    # marker, if any). BENCH_RUNS: ordered list of every `vllm bench serve`
    # invocation observed via bench_run_start markers. Both default to "no
    # metadata" so the template stays renderable on legacy JSONLs.
    session_meta_json = json.dumps(session_meta, separators=(",", ":"))
    html = html.replace("/*SESSION_META*/null", session_meta_json)

    bench_runs_json = json.dumps(bench_runs, separators=(",", ":"))
    html = html.replace("/*BENCH_RUNS*/[]", bench_runs_json)

    # Aggregate-across-steps imbalance (same formula as the per-step pill in
    # the stats bar, just averaged across every step embedded in the report).
    # The viewer renders this in its header as a constant context number so
    # users can compare per-step pills against the run average.
    avg_imb_json = json.dumps(avg_imbalance)
    html = html.replace("/*AVG_IMBALANCE*/null", avg_imb_json)

    # Per-step imbalance breakdown — same numbers we print at the end of the
    # script. Embedded so the HTML viewer can render the (step, imbalance)
    # table behind a [per-step] toggle next to the Avg imbalance pill, with
    # the explicit (a + b + ...) / N derivation underneath. List of two-tuples
    # `[[step, imbalance], ...]`; empty list when no `tokens` data was
    # available (toggle is then hidden by the template).
    per_step_imb_payload = [[s, v] for s, v in step_imbalances]
    per_step_imb_json = json.dumps(per_step_imb_payload, separators=(",", ":"))
    html = html.replace("/*PER_STEP_IMBALANCE*/[]", per_step_imb_json)

    # --perf-data payload: a multi-line aligned text rendering of all
    # `Serving Benchmark Result` blocks merged together with arrows. The
    # template wraps the string in a <pre> when present, hides the row
    # entirely when null. json.dumps handles all the escaping for us
    # (newlines, quotes, etc.) so the embedded JS string is safe.
    perf_data_json = json.dumps(perf_data_text)
    html = html.replace("/*PERF_DATA*/null", perf_data_json)

    # Honour $TZ from the user's environment so generated_at / started_at
    # render in their wall-clock timezone (e.g. Europe/Helsinki) instead of
    # the raw UTC ISO timestamp written by nvtx.sh / generate_static_mapping.
    # Empty / unset → JS falls back to the browser default (which is also
    # the user's local TZ in practice, but explicit is better).
    report_tz = os.environ.get("TZ", "").strip() or None
    tz_json = json.dumps(report_tz)
    html = html.replace("/*REPORT_TIMEZONE*/null", tz_json)

    # Report title comes from the output file's stem so each report is
    # self-identifying when multiple tabs are open. Fallback-safe: if the
    # placeholder is missing for some reason, the default text stays.
    title = output_path.stem
    html = html.replace("/*REPORT_TITLE*/EPLB MoE Analysis", title)

    output_path.write_text(html)

    elapsed = time.time() - t0
    size_kb = output_path.stat().st_size / 1024
    _ok(
        f"Report written to {output_path} "
        f"({size_kb:.0f} KB, {elapsed:.1f}s)"
    )

    # Per-step imbalance table + the explicit aggregation formula. Printed
    # last so the user can eyeball whether the HTML header value matches
    # what the data implies, and to make the (a + b + ...) / N derivation
    # auditable without opening the HTML / re-reading the JSONL.
    if step_imbalances:
        sys.stdout.write("\nPer-step imbalance (avg over layers, max/mean):\n")
        sys.stdout.write(f"  {'step':>10}  imbalance\n")
        for step, imb in step_imbalances:
            sys.stdout.write(f"  {step:>10}  {imb:.2f}x\n")
        terms = " + ".join(f"{imb:.2f}" for _, imb in step_imbalances)
        n = len(step_imbalances)
        sys.stdout.write(
            f"\n  ({terms}) / {n} = {avg_imbalance:.2f}x   "
            f"(avg imbalance across {n} step{'s' if n != 1 else ''})\n"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _err("Interrupted.")
        sys.exit(130)

