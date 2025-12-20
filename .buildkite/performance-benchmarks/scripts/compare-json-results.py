# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import html as _html
import json
import os
from dataclasses import dataclass
from importlib import util
from typing import List, Tuple

import pandas as pd

pd.options.display.float_format = "{:.2f}".format
plotly_found = util.find_spec("plotly.express") is not None

DEFAULT_INFO_COLS = [
    "Model",
    "Dataset Name",
    "Input Len",
    "Output Len",
    "TP Size",
    "PP Size",
    "# of max concurrency.",
    "qps",
]


# -----------------------------
# Core data compare
# -----------------------------
def compare_data_columns(
    files: List[str],
    name_column: str,
    data_column: str,
    info_cols: List[str],
    drop_column: str,
    debug: bool = False,
):
    """
    Align concatenation by keys derived from info_cols instead of row order.
    - Pick one canonical key list: subset of info_cols present in ALL files.
    - For each file: set index to those keys, aggregate duplicates
      (mean for metric, first for names).
    - Concat along axis=1 (indexes align), then reset_index so callers can
      group by columns.
    - If --debug, add a <file_label>_name column per file.
    """
    print("\ncompare_data_column:", data_column)

    frames = []
    raw_data_cols = []
    compare_frames = []

    # 1) choose a canonical key list from info_cols that exists in ALL files
    cols_per_file = []
    for f in files:
        try:
            df_tmp = pd.read_json(f, orient="records")
        except Exception as err:
            raise ValueError(f"Failed to read {f}") from err
        cols_per_file.append(set(df_tmp.columns))

    key_cols = [c for c in info_cols if all(c in cset for cset in cols_per_file)]
    if not key_cols:
        # soft fallback: use any info_cols present in the first file
        key_cols = [c for c in info_cols if c in list(cols_per_file[0])]
    if not key_cols:
        raise ValueError(
            "No common key columns found from info_cols across the input files."
        )

    # 2) build a single "meta" block (keys as columns) once, aligned by the key index
    meta_added = False

    for file in files:
        df = pd.read_json(file, orient="records")

        # Keep rows that actually have the compared metric (same as original behavior)
        if drop_column in df.columns:
            df = df.dropna(subset=[drop_column], ignore_index=True)

        # Stabilize numeric key columns (harmless if missing)
        for c in (
            "Input Len",
            "Output Len",
            "TP Size",
            "PP Size",
            "# of max concurrency.",
            "qps",
        ):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Ensure all key columns exist
        for c in key_cols:
            if c not in df.columns:
                df[c] = pd.NA

        # Set index = key_cols and aggregate duplicates → unique MultiIndex
        df_idx = df.set_index(key_cols, drop=False)

        # meta (key columns), unique per key
        meta = df_idx[key_cols]
        if not meta.index.is_unique:
            meta = meta.groupby(level=key_cols, dropna=False).first()

        # metric series for this file, aggregated to one row per key
        file_label = "/".join(file.split("/")[:-1]) or os.path.basename(file)
        s = df_idx[data_column]
        if not s.index.is_unique:
            s = s.groupby(level=key_cols, dropna=False).mean()
        s.name = file_label  # column label like original

        # add meta once (from first file) so keys are the leftmost columns
        if not meta_added:
            frames.append(meta)
            meta_added = True

        # debug: aligned test-name column per file
        if debug and name_column in df_idx.columns:
            name_s = df_idx[name_column]
            if not name_s.index.is_unique:
                name_s = name_s.groupby(level=key_cols, dropna=False).first()
            name_s.name = f"{file_label}_name"
            frames.append(name_s)

        frames.append(s)
        raw_data_cols.append(file_label)
        compare_frames.append(s)

        # ratio columns: fileN / file1 (throughput) or file1 / fileN (latency)
        if len(compare_frames) >= 2:
            base = compare_frames[0]
            current = compare_frames[-1]
            if "P99" in data_column or "Median" in data_column:
                ratio = base / current  # for latency: larger means better
            else:
                ratio = current / base  # for throughput: larger means better
            ratio = ratio.mask(base == 0)
            ratio.name = f"Ratio 1 vs {len(compare_frames)}"
            frames.append(ratio)

    concat_df = pd.concat(frames, axis=1)

    # NOTE: meta already contains key columns as normal columns, so we can drop the index cleanly.
    concat_df = concat_df.reset_index(drop=True)

    # Ensure key/info columns appear first (in your info_cols order)
    front = [c for c in info_cols if c in concat_df.columns]
    rest = [c for c in concat_df.columns if c not in front]
    concat_df = concat_df[front + rest]

    print(raw_data_cols)
    return concat_df, raw_data_cols


# -----------------------------
# Split helper (restored)
# -----------------------------
def split_json_by_tp_pp(
    input_file: str = "benchmark_results.json", output_root: str = "."
) -> List[str]:
    """
    Split a benchmark JSON into separate folders by (TP Size, PP Size).

    Creates: <output_root>/tp{TP}_pp{PP}/benchmark_results.json
    Returns: list of file paths written.
    """
    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)

    # If the JSON is a dict with a list under common keys, use that list
    if isinstance(data, dict):
        for key in ("results", "serving_results", "benchmarks", "data"):
            if isinstance(data.get(key), list):
                data = data[key]
                break

    df = pd.DataFrame(data)

    # Keep only "serving" tests
    name_col = next(
        (c for c in ["Test name", "test_name", "Test Name"] if c in df.columns), None
    )
    if name_col:
        df = df[df[name_col].astype(str).str.contains(r"serving", case=False, na=False)].copy()

    # Handle alias column names
    rename_map = {
        "tp_size": "TP Size",
        "tensor_parallel_size": "TP Size",
        "pp_size": "PP Size",
        "pipeline_parallel_size": "PP Size",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # Ensure TP/PP columns exist (default to 1 if missing)
    if "TP Size" not in df.columns:
        df["TP Size"] = 1
    if "PP Size" not in df.columns:
        df["PP Size"] = 1

    df["TP Size"] = pd.to_numeric(df["TP Size"], errors="coerce").fillna(1).astype(int)
    df["PP Size"] = pd.to_numeric(df["PP Size"], errors="coerce").fillna(1).astype(int)

    saved_paths: List[str] = []
    for (tp, pp), group_df in df.groupby(["TP Size", "PP Size"], dropna=False):
        folder_name = os.path.join(output_root, f"tp{int(tp)}_pp{int(pp)}")
        os.makedirs(folder_name, exist_ok=True)
        filepath = os.path.join(folder_name, "benchmark_results.json")
        group_df.to_json(filepath, orient="records", indent=2, force_ascii=False)
        print(f"Saved: {filepath}")
        saved_paths.append(filepath)

    return saved_paths


# -----------------------------
# Styling helpers
# -----------------------------
def _find_concurrency_col(df: pd.DataFrame) -> str:
    for c in [
        "# of max concurrency.",
        "# of max concurrency",
        "Max Concurrency",
        "max_concurrency",
        "Concurrency",
    ]:
        if c in df.columns:
            return c
    for c in df.columns:
        if df[c].dtype.kind in "iu" and df[c].nunique() > 1 and df[c].min() >= 1:
            return c
    return "# of max concurrency."


def _highlight_threshold(df: pd.DataFrame, threshold: float) -> "pd.io.formats.style.Styler":
    """Highlight numeric per-configuration columns with value <= threshold."""
    conc_col = _find_concurrency_col(df)
    key_cols = [c for c in ["Model", "Dataset Name", "Input Len", "Output Len", conc_col] if c in df.columns]
    conf_cols = [c for c in df.columns if c not in key_cols and not str(c).startswith("Ratio")]
    conf_cols = [c for c in conf_cols if pd.api.types.is_numeric_dtype(df[c])]
    return df.style.map(
        lambda v: "background-color:#e6ffe6;font-weight:bold;"
        if pd.notna(v) and v <= threshold
        else "",
        subset=conf_cols,
    )


def highlight_ratio_columns(styler: "pd.io.formats.style.Styler"):
    """Highlight entire columns whose header contains 'Ratio'."""
    ratio_cols = [c for c in styler.data.columns if "ratio" in str(c).lower()]
    if not ratio_cols:
        return styler

    # highlight cells
    styler = styler.apply(
        lambda _: ["background-color: #fff3b0"] * len(styler.data),
        subset=ratio_cols,
        axis=0,
    )

    # highlight headers
    styler = styler.set_table_styles(
        [
            {"selector": f"th.col_heading.level0.col{i}", "props": [("background-color", "#fff3b0")]}
            for i, col in enumerate(styler.data.columns)
            if col in ratio_cols
        ],
        overwrite=False,
    )
    return styler


# -----------------------------
# Plot helper
# -----------------------------
def _add_limit_line(fig, y_value: float, label: str):
    fig.add_hline(
        y=y_value,
        line_dash="dash",
        line_color="red" if "ttft" in label.lower() else "blue",
        annotation_text=f"{label}: {y_value} ms",
        annotation_position="top left",
    )
    # If plotly is available, add a legend entry
    if plotly_found:
        import plotly.graph_objects as go

        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(dash="dash", color="red" if "ttft" in label.lower() else "blue"),
                name=label,
            )
        )


# -----------------------------
# Refactored "main"
# -----------------------------
@dataclass(frozen=True)
class MetricPlan:
    data_cols: List[str]
    drop_column: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", action="append", type=str, help="input file name")
    parser.add_argument("--debug", action="store_true", help="show all information for debugging")
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="plot perf diagrams or not --no-plot --plot",
    )
    parser.add_argument(
        "-x",
        "--xaxis",
        type=str,
        default="# of max concurrency.",
        help="column name to use as X Axis in comparison graph",
    )
    parser.add_argument(
        "-l",
        "--latency",
        type=str,
        default="p99",
        help="take median|p99 for latency like TTFT/TPOT",
    )
    parser.add_argument("--ttft-max-ms", type=float, default=3000.0, help="Reference limit for TTFT plots (ms)")
    parser.add_argument("--tpot-max-ms", type=float, default=100.0, help="Reference limit for TPOT plots (ms)")
    return parser


def choose_metrics(latency: str) -> MetricPlan:
    latency = (latency or "").lower()
    drop_column = "P99"
    if "median" in latency:
        return MetricPlan(
            data_cols=["Output Tput (tok/s)", "Median TTFT (ms)", "Median"],
            drop_column=drop_column,
        )
    return MetricPlan(
        data_cols=["Output Tput (tok/s)", "P99 TTFT (ms)", "P99"],
        drop_column=drop_column,
    )


def prepare_input_files(args, info_cols: List[str]) -> Tuple[List[str], List[str]]:
    if not args.file:
        raise ValueError("No input files provided. Use -f/--file.")
    if len(args.file) == 1:
        files = split_json_by_tp_pp(args.file[0], output_root="splits")
        info_cols = [c for c in info_cols if c not in ("TP Size", "PP Size")]
    else:
        files = args.file
    return files, info_cols


def get_y_axis_col(info_cols: List[str], xaxis: str) -> str:
    y_axis_index = info_cols.index(xaxis) if xaxis in info_cols else 6
    return info_cols[y_axis_index]


def get_group_cols(output_df: pd.DataFrame, info_cols: List[str]) -> List[str]:
    filtered_info_cols = info_cols[:4]
    group_cols = [c for c in filtered_info_cols if c in output_df.columns]
    if not group_cols:
        raise ValueError(
            f"No valid group-by columns. Expected subset: {filtered_info_cols}, "
            f"but DataFrame has: {list(output_df.columns)}"
        )
    return group_cols


def group_suffix(group_cols: List[str], name) -> str:
    name_vals = name if isinstance(name, tuple) else (name,)
    return " , ".join(f"{col} : [ {val} ] " for col, val in zip(group_cols, name_vals))


def group_filename(name, prefix: str = "perf_comparison_") -> str:
    name_vals = name if isinstance(name, tuple) else (name,)
    safe = ",".join(map(str, name_vals)).replace(",", "_").replace("/", "-")
    return f"{prefix}{safe}.html"


def render_metric_table_html(display_group: pd.DataFrame, metric_label: str, suffix: str, args) -> str:
    title = (
        f'<div style="font-size: 1.25em; font-weight: 600; margin: 12px 0;">'
        f'{_html.escape(metric_label)}'
        f' — {_html.escape(suffix)}'
        f"</div>\n"
    )

    metric_name = metric_label.lower()

    if "ttft" in metric_name:
        styler = _highlight_threshold(display_group, args.ttft_max_ms)
    elif ("tpot" in metric_name) or ("median" in metric_name) or ("p99" in metric_name):
        styler = _highlight_threshold(display_group, args.tpot_max_ms)
    else:
        styler = display_group.style

    # format numbers + highlight ratios
    styler = styler.format(
        {c: "{:.2f}" for c in display_group.select_dtypes("number").columns},
        na_rep="—",
    )
    styler = highlight_ratio_columns(styler)

    return title + styler.to_html(table_attributes='border="1" class="dataframe"')


def maybe_write_plot(
    main_fh,
    sub_fh,
    group_df: pd.DataFrame,
    raw_data_cols: List[str],
    metric_label: str,
    y_axis_col: str,
    args,
):
    if not (args.plot and plotly_found):
        return

    import plotly.express as px

    df = group_df[raw_data_cols].sort_values(by=y_axis_col)
    df_melted = df.melt(
        id_vars=y_axis_col,
        var_name="Configuration",
        value_name=metric_label,
    )

    fig = px.line(
        df_melted,
        x=y_axis_col,
        y=metric_label,
        color="Configuration",
        title=f"{metric_label} vs {y_axis_col}",
        markers=True,
    )

    metric_name = metric_label.lower()
    if "ttft" in metric_name:
        _add_limit_line(fig, args.ttft_max_ms, "TTFT limit")
    elif ("tpot" in metric_name) or ("median" in metric_name) or ("p99" in metric_name):
        _add_limit_line(fig, args.tpot_max_ms, "TPOT limit")

    html = fig.to_html(full_html=True, include_plotlyjs="cdn")
    main_fh.write(html)
    sub_fh.write(html)


def write_report(files: List[str], info_cols: List[str], plan: MetricPlan, args):
    name_column = "Test name"
    y_axis_col = get_y_axis_col(info_cols, args.xaxis)

    print("comparing : " + ", ".join(files))

    with open("perf_comparison.html", "w") as main_fh:
        for metric_label in plan.data_cols:
            output_df, raw_data_cols = compare_data_columns(
                files,
                name_column,
                metric_label,
                info_cols,
                plan.drop_column,
                debug=args.debug,
            )

            raw_data_cols = list(raw_data_cols)
            raw_data_cols.insert(0, y_axis_col)

            group_cols = get_group_cols(output_df, info_cols)

            output_df_sorted = output_df.sort_values(by=args.xaxis)
            for name, group_df in output_df_sorted.groupby(group_cols, dropna=False):
                suffix = group_suffix(group_cols, name)
                sub_path = group_filename(name)

                # drop group columns from display only
                display_group = group_df.drop(columns=group_cols, errors="ignore")

                html = render_metric_table_html(display_group, metric_label, suffix, args)

                main_fh.write(html)
                with open(sub_path, "a+") as sub_fh:
                    sub_fh.write(html)
                    maybe_write_plot(
                        main_fh,
                        sub_fh,
                        group_df=group_df,
                        raw_data_cols=raw_data_cols,
                        metric_label=metric_label,
                        y_axis_col=y_axis_col,
                        args=args,
                    )


def main():
    args = build_parser().parse_args()

    info_cols = list(DEFAULT_INFO_COLS)
    plan = choose_metrics(args.latency)

    files, info_cols = prepare_input_files(args, info_cols)
    write_report(files, info_cols, plan, args)


if __name__ == "__main__":
    main()

