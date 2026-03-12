# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import html as _html
import json
import os
from dataclasses import dataclass
from importlib import util
from pathlib import Path

import pandas as pd
import regex as re

pd.options.display.float_format = "{:.2f}".format
plotly_found = util.find_spec("plotly.express") is not None

DEFAULT_INFO_COLS = [
    "Model",
    "Dataset Name",
    "Input Len",
    "Output Len",
    #    "TP Size",
    #    "PP Size",
    "# of max concurrency.",
    "qps",
]

# Safety net: if any DataFrame leaks into to_html(), keep precision at 2.
pd.set_option("display.precision", 2)
pd.set_option("display.float_format", lambda x: f"{x:.2f}")


# -----------------------------
# Core data compare
# -----------------------------
def compare_data_columns(
    files: list[str],
    name_column: str,
    data_column: str,
    info_cols: list[str],
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
    raw_data_cols: list[str] = []
    compare_frames = []

    cols_per_file: list[set] = []
    for f in files:
        try:
            df_tmp = pd.read_json(f, orient="records")
        except Exception as err:
            raise ValueError(f"Failed to read {f}") from err
        cols_per_file.append(set(df_tmp.columns))

    key_cols = [c for c in info_cols if all(c in cset for cset in cols_per_file)]
    if not key_cols:
        key_cols = [c for c in info_cols if c in list(cols_per_file[0])]
    if not key_cols:
        raise ValueError(
            "No common key columns found from info_cols across the input files."
        )

    meta_added = False

    for file in files:
        df = pd.read_json(file, orient="records")

        if drop_column in df.columns:
            df = df.dropna(subset=[drop_column], ignore_index=True)

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

        for c in key_cols:
            if c not in df.columns:
                df[c] = pd.NA

        df_idx = df.set_index(key_cols, drop=False)

        meta = df_idx[key_cols]
        if not meta.index.is_unique:
            meta = meta.groupby(level=key_cols, dropna=False).first()

        file_label = "/".join(file.split("/")[:-1]) or os.path.basename(file)
        s = df_idx[data_column]
        if not s.index.is_unique:
            s = s.groupby(level=key_cols, dropna=False).mean()
        s.name = file_label

        if not meta_added:
            frames.append(meta)
            meta_added = True

        if debug and name_column in df_idx.columns:
            name_s = df_idx[name_column]
            if not name_s.index.is_unique:
                name_s = name_s.groupby(level=key_cols, dropna=False).first()
            name_s.name = f"{file_label}_name"
            frames.append(name_s)

        frames.append(s)
        raw_data_cols.append(file_label)
        compare_frames.append(s)

        if len(compare_frames) >= 2:
            base = compare_frames[0]
            current = compare_frames[-1]
            if "P99" in data_column or "Median" in data_column:
                ratio = base / current
            else:
                ratio = current / base
            ratio = ratio.mask(base == 0)
            ratio.name = f"Ratio 1 vs {len(compare_frames)}"
            frames.append(ratio)

    concat_df = pd.concat(frames, axis=1).reset_index(drop=True)

    front = [c for c in info_cols if c in concat_df.columns]
    rest = [c for c in concat_df.columns if c not in front]
    concat_df = concat_df[front + rest]

    print(raw_data_cols)
    return concat_df, raw_data_cols


# -----------------------------
# Split helper
# -----------------------------
def split_json_by_tp_pp(
    input_file: str = "benchmark_results.json", output_root: str = "."
) -> list[str]:
    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for key in ("results", "serving_results", "benchmarks", "data"):
            if isinstance(data.get(key), list):
                data = data[key]
                break

    df = pd.DataFrame(data)

    name_col = next(
        (c for c in ["Test name", "test_name", "Test Name"] if c in df.columns), None
    )
    if name_col:
        df = df[
            df[name_col].astype(str).str.contains(r"serving", case=False, na=False)
        ].copy()

    rename_map = {
        "tp_size": "TP Size",
        "tensor_parallel_size": "TP Size",
        "pp_size": "PP Size",
        "pipeline_parallel_size": "PP Size",
    }
    df.rename(
        columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True
    )

    if "TP Size" not in df.columns:
        df["TP Size"] = 1
    if "PP Size" not in df.columns:
        df["PP Size"] = 1

    df["TP Size"] = pd.to_numeric(df["TP Size"], errors="coerce").fillna(1).astype(int)
    df["PP Size"] = pd.to_numeric(df["PP Size"], errors="coerce").fillna(1).astype(int)

    saved_paths: list[str] = []
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


def _highlight_threshold(
    df: pd.DataFrame, threshold: float
) -> pd.io.formats.style.Styler:
    conc_col = _find_concurrency_col(df)
    key_cols = [
        c
        for c in ["Model", "Dataset Name", "Input Len", "Output Len", conc_col]
        if c in df.columns
    ]
    conf_cols = [
        c for c in df.columns if c not in key_cols and not str(c).startswith("Ratio")
    ]
    conf_cols = [c for c in conf_cols if pd.api.types.is_numeric_dtype(df[c])]

    return df.style.map(
        lambda v: "background-color:#e6ffe6;font-weight:bold;"
        if pd.notna(v) and v <= threshold
        else "",
        subset=conf_cols,
    )


def highlight_ratio_columns(styler: pd.io.formats.style.Styler):
    ratio_cols = [c for c in styler.data.columns if "ratio" in str(c).lower()]
    if not ratio_cols:
        return styler

    styler = styler.apply(
        lambda _: ["background-color: #fff3b0"] * len(styler.data),
        subset=ratio_cols,
        axis=0,
    )

    styler = styler.set_table_styles(
        [
            {
                "selector": f"th.col_heading.level0.col{i}",
                "props": [("background-color", "#fff3b0")],
            }
            for i, col in enumerate(styler.data.columns)
            if col in ratio_cols
        ],
        overwrite=False,
    )
    return styler


def _apply_two_decimals(
    styler: pd.io.formats.style.Styler,
) -> pd.io.formats.style.Styler:
    df = styler.data
    num_cols = df.select_dtypes("number").columns
    if len(num_cols) == 0:
        return styler
    return styler.format({c: "{:.2f}" for c in num_cols}, na_rep="")


# -----------------------------
# Export helpers (Excel + CSV)
# -----------------------------
def _sanitize_sheet_name(name: str) -> str:
    """
    Excel sheet constraints:
      - max 31 chars
      - cannot contain: : \ / ? * [ ]
      - cannot be empty
    """
    name = "sheet" if name is None else str(name)
    name = re.sub(r"[:\\/?*\[\]]", "_", name)
    name = name.strip().strip("'")
    name = re.sub(r"\s+", " ", name)
    if not name:
        name = "sheet"
    return name[:31]


def _group_to_sheet_base(group_cols: list[str], gkey_tuple) -> str:
    d = dict(zip(group_cols, gkey_tuple))
    model = d.get("Model", "model")
    model_short = str(model).split("/")[-1]
    ilen = d.get("Input Len", "")
    olen = d.get("Output Len", "")
    lens = f"_{ilen}x{olen}" if ilen != "" and olen != "" else ""
    return _sanitize_sheet_name(f"{model_short}{lens}")


def _write_tables_to_excel_sheet(
    writer: pd.ExcelWriter, sheet: str, blocks: list[tuple[str, pd.DataFrame]]
):
    startrow = 0
    for title, df in blocks:
        pd.DataFrame([[title]]).to_excel(
            writer, sheet_name=sheet, index=False, header=False, startrow=startrow
        )
        startrow += 1
        df.to_excel(writer, sheet_name=sheet, index=False, startrow=startrow)
        startrow += len(df) + 3


def _safe_filename(s: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", str(s).strip())
    return s[:180] if len(s) > 180 else s


# -----------------------------
# vLLM environment export helper
# -----------------------------
def _parse_vllm_env_txt(env_path: Path) -> pd.DataFrame:
    """Parse vllm_env.txt into a flat table (Section, Key, Value).

    Supports:
      - section headers as standalone lines (no ':' or '=')
      - key-value lines like 'OS: Ubuntu ...'
      - env var lines like 'HF_HOME=/data/hf'
    """
    lines = env_path.read_text(encoding="utf-8", errors="replace").splitlines()
    section = "General"
    rows: list[dict] = []

    def set_section(s: str):
        nonlocal section
        s = (s or "").strip()
        if s:
            section = s

    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        # divider lines like =====
        if set(stripped) <= {"="}:
            continue

        # section header heuristic: short standalone line
        if ":" not in stripped and "=" not in stripped and len(stripped) <= 64:
            if stripped.lower().startswith("collecting environment information"):
                continue
            set_section(stripped)
            continue

        # env var style: KEY=VALUE (and not a URL with :)
        if "=" in stripped and ":" not in stripped:
            k, v = stripped.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k:
                rows.append({"Section": section, "Key": k, "Value": v})
            continue

        # key: value
        if ":" in stripped:
            k, v = stripped.split(":", 1)
            k = k.strip()
            v = v.strip()
            if k:
                rows.append({"Section": section, "Key": k, "Value": v})
            continue

    return pd.DataFrame(rows, columns=["Section", "Key", "Value"])


def _load_env_df_for_inputs(args, files: list[str]) -> pd.DataFrame | None:
    """Load vllm_env.txt next to the *original* input JSON file.

    Note: when only one -f is provided, the script may split JSON into ./splits/...,
    but vllm_env.txt typically lives next to the original benchmark_results.json.
    """
    base_dir: Path | None = None
    if getattr(args, "file", None):
        base_dir = Path(args.file[0]).resolve().parent
    elif files:
        base_dir = Path(files[0]).resolve().parent
    if base_dir is None:
        return None

    env_path = base_dir / "vllm_env.txt"
    if not env_path.exists():
        return None
    df = _parse_vllm_env_txt(env_path)
    return df


# -----------------------------
# Valid max concurrency summary helpers
# -----------------------------
def _config_value_columns(df: pd.DataFrame, conc_col: str) -> list[str]:
    key_cols = [
        c
        for c in ["Model", "Dataset Name", "Input Len", "Output Len"]
        if c in df.columns
    ]
    exclude = set(key_cols + [conc_col, "qps", "QPS"])

    cols: list[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        lc = str(c).lower()
        if lc.startswith("ratio"):
            continue
        if lc.endswith("_name") or lc == "test name" or lc == "test_name":
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _max_concurrency_ok(
    df: pd.DataFrame, conc_col: str, cfg_col: str, threshold: float
):
    if df is None or conc_col not in df.columns or cfg_col not in df.columns:
        return pd.NA

    d = df[[conc_col, cfg_col]].copy()
    d[conc_col] = pd.to_numeric(d[conc_col], errors="coerce")
    d[cfg_col] = pd.to_numeric(d[cfg_col], errors="coerce")
    d = d.dropna(subset=[conc_col, cfg_col])

    if d.empty:
        return pd.NA

    ok = d[d[cfg_col] <= threshold]
    if ok.empty:
        return pd.NA

    return ok[conc_col].max()


def _value_at_concurrency(df: pd.DataFrame, conc_col: str, cfg_col: str, conc_value):
    if (
        df is None
        or conc_col not in df.columns
        or cfg_col not in df.columns
        or pd.isna(conc_value)
    ):
        return pd.NA

    d = df[[conc_col, cfg_col]].copy()
    d[conc_col] = pd.to_numeric(d[conc_col], errors="coerce")
    d[cfg_col] = pd.to_numeric(d[cfg_col], errors="coerce")

    conc_value = pd.to_numeric(conc_value, errors="coerce")
    if pd.isna(conc_value):
        return pd.NA

    hit = d[d[conc_col] == conc_value]
    if hit.empty:
        return pd.NA
    return hit[cfg_col].iloc[0]


def build_valid_max_concurrency_summary_html(
    tput_group_df: pd.DataFrame | None,
    ttft_group_df: pd.DataFrame | None,
    tpot_group_df: pd.DataFrame | None,
    conc_col: str,
    args,
) -> str:
    if ttft_group_df is None and tpot_group_df is None:
        return ""

    ttft_cols = (
        _config_value_columns(ttft_group_df, conc_col)
        if ttft_group_df is not None
        else []
    )
    tpot_cols = (
        _config_value_columns(tpot_group_df, conc_col)
        if tpot_group_df is not None
        else []
    )
    tput_cols = (
        _config_value_columns(tput_group_df, conc_col)
        if tput_group_df is not None
        else []
    )

    if ttft_group_df is not None and tpot_group_df is not None:
        cfg_cols = [c for c in ttft_cols if c in tpot_cols]
        if tput_group_df is not None:
            cfg_cols = [c for c in cfg_cols if c in tput_cols] or cfg_cols
    else:
        cfg_cols = ttft_cols or tpot_cols

    if not cfg_cols:
        cfg_cols = sorted(set(ttft_cols) | set(tpot_cols) | set(tput_cols), key=str)

    rows = []
    for cfg in cfg_cols:
        ttft_max = (
            _max_concurrency_ok(ttft_group_df, conc_col, cfg, args.ttft_max_ms)
            if ttft_group_df is not None
            else pd.NA
        )
        tpot_max = (
            _max_concurrency_ok(tpot_group_df, conc_col, cfg, args.tpot_max_ms)
            if tpot_group_df is not None
            else pd.NA
        )
        both = (
            pd.NA
            if (pd.isna(ttft_max) or pd.isna(tpot_max))
            else min(ttft_max, tpot_max)
        )

        tput_at_both = (
            _value_at_concurrency(tput_group_df, conc_col, cfg, both)
            if tput_group_df is not None
            else pd.NA
        )
        ttft_at_both = (
            _value_at_concurrency(ttft_group_df, conc_col, cfg, both)
            if ttft_group_df is not None
            else pd.NA
        )
        tpot_at_both = (
            _value_at_concurrency(tpot_group_df, conc_col, cfg, both)
            if tpot_group_df is not None
            else pd.NA
        )

        rows.append(
            {
                "Configuration": cfg,
                f"Max {conc_col} (TTFT ≤ {args.ttft_max_ms:g} ms)": ttft_max,
                f"Max {conc_col} (TPOT ≤ {args.tpot_max_ms:g} ms)": tpot_max,
                f"Max {conc_col} (Both)": both,
                "Output Tput @ Both (tok/s)": tput_at_both,
                "TTFT @ Both (ms)": ttft_at_both,
                "TPOT @ Both (ms)": tpot_at_both,
            }
        )

    summary_df = pd.DataFrame(rows)

    for c in summary_df.columns:
        if c == "Configuration":
            continue
        summary_df[c] = pd.to_numeric(summary_df[c], errors="coerce")

    both_col = f"Max {conc_col} (Both)"

    formatters = {}
    for c in summary_df.columns:
        if c == "Configuration":
            continue
        formatters[c] = lambda v: "" if pd.isna(v) else f"{float(v):.2f}"

    styler = summary_df.style.format(formatters)

    def _green(v):
        return "background-color:#e6ffe6;font-weight:bold;" if pd.notna(v) else ""

    if both_col in summary_df.columns:
        styler = styler.map(_green, subset=[both_col])

    title = (
        '<div style="font-size: 1.15em; font-weight: 700; margin: 12px 0 6px 0;">'
        "Valid Max Concurrency Summary"
        "</div>\n"
    )
    return title + styler.to_html(table_attributes='border="1" class="dataframe"')


def build_valid_max_concurrency_summary_df(
    tput_group_df: pd.DataFrame | None,
    ttft_group_df: pd.DataFrame | None,
    tpot_group_df: pd.DataFrame | None,
    conc_col: str,
    args,
) -> pd.DataFrame | None:
    if ttft_group_df is None and tpot_group_df is None:
        return None

    ttft_cols = (
        _config_value_columns(ttft_group_df, conc_col)
        if ttft_group_df is not None
        else []
    )
    tpot_cols = (
        _config_value_columns(tpot_group_df, conc_col)
        if tpot_group_df is not None
        else []
    )
    tput_cols = (
        _config_value_columns(tput_group_df, conc_col)
        if tput_group_df is not None
        else []
    )

    if ttft_group_df is not None and tpot_group_df is not None:
        cfg_cols = [c for c in ttft_cols if c in tpot_cols]
        if tput_group_df is not None:
            cfg_cols = [c for c in cfg_cols if c in tput_cols] or cfg_cols
    else:
        cfg_cols = ttft_cols or tpot_cols

    if not cfg_cols:
        cfg_cols = sorted(set(ttft_cols) | set(tpot_cols) | set(tput_cols), key=str)

    rows = []
    for cfg in cfg_cols:
        ttft_max = (
            _max_concurrency_ok(ttft_group_df, conc_col, cfg, args.ttft_max_ms)
            if ttft_group_df is not None
            else pd.NA
        )
        tpot_max = (
            _max_concurrency_ok(tpot_group_df, conc_col, cfg, args.tpot_max_ms)
            if tpot_group_df is not None
            else pd.NA
        )
        both = (
            pd.NA
            if (pd.isna(ttft_max) or pd.isna(tpot_max))
            else min(ttft_max, tpot_max)
        )

        tput_at_both = (
            _value_at_concurrency(tput_group_df, conc_col, cfg, both)
            if tput_group_df is not None
            else pd.NA
        )
        ttft_at_both = (
            _value_at_concurrency(ttft_group_df, conc_col, cfg, both)
            if ttft_group_df is not None
            else pd.NA
        )
        tpot_at_both = (
            _value_at_concurrency(tpot_group_df, conc_col, cfg, both)
            if tpot_group_df is not None
            else pd.NA
        )

        rows.append(
            {
                "Configuration": cfg,
                f"Max {conc_col} (TTFT ≤ {args.ttft_max_ms:g} ms)": ttft_max,
                f"Max {conc_col} (TPOT ≤ {args.tpot_max_ms:g} ms)": tpot_max,
                f"Max {conc_col} (Both)": both,
                "Output Tput @ Both (tok/s)": tput_at_both,
                "TTFT @ Both (ms)": ttft_at_both,
                "TPOT @ Both (ms)": tpot_at_both,
            }
        )

    df = pd.DataFrame(rows)
    for c in df.columns:
        if c != "Configuration":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


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
    if plotly_found:
        import plotly.graph_objects as go

        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(
                    dash="dash",
                    color="red" if "ttft" in label.lower() else "blue",
                ),
                name=label,
            )
        )


# -----------------------------
# Refactored main + group-first report
# -----------------------------
@dataclass(frozen=True)
class MetricPlan:
    data_cols: list[str]
    drop_column: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file", action="append", type=str, help="input file name"
    )
    parser.add_argument(
        "--debug", action="store_true", help="show all information for debugging"
    )
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
    parser.add_argument(
        "--ttft-max-ms",
        type=float,
        default=3000.0,
        help="Reference limit for TTFT plots (ms)",
    )
    parser.add_argument(
        "--tpot-max-ms",
        type=float,
        default=100.0,
        help="Reference limit for TPOT plots (ms)",
    )

    # ---- NEW: export options ----
    parser.add_argument(
        "--excel-out",
        type=str,
        default="perf_comparison.xlsx",
        help="Write one sheet per (Model, Dataset, Input Len, Output Len).",
    )
    parser.add_argument(
        "--csv-out-dir",
        type=str,
        default="",
        help="If set, write per-group per-metric CSVs into this directory.",
    )

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


def prepare_input_files(args, info_cols: list[str]) -> tuple[list[str], list[str]]:
    if not args.file:
        raise ValueError("No input files provided. Use -f/--file.")

    if len(args.file) == 1:
        files = split_json_by_tp_pp(args.file[0], output_root="splits")
        info_cols = [c for c in info_cols if c not in ("TP Size", "PP Size")]
    else:
        files = args.file

    return files, info_cols


def get_y_axis_col(info_cols: list[str], xaxis: str) -> str:
    y_axis_index = info_cols.index(xaxis) if xaxis in info_cols else 6
    return info_cols[y_axis_index]


def get_group_cols(output_df: pd.DataFrame, info_cols: list[str]) -> list[str]:
    filtered_info_cols = info_cols[:4]
    group_cols = [c for c in filtered_info_cols if c in output_df.columns]
    if not group_cols:
        raise ValueError(
            f"No valid group-by columns. Expected subset: {filtered_info_cols}, "
            f"but DataFrame has: {list(output_df.columns)}"
        )
    return group_cols


def normalize_group_key(name):
    return name if isinstance(name, tuple) else (name,)


def group_filename(name, prefix: str = "perf_comparison_") -> str:
    name_vals = normalize_group_key(name)
    safe = ",".join(map(str, name_vals)).replace(",", "_").replace("/", "-")
    return f"{prefix}{safe}.html"


def build_group_suffix(group_cols: list[str], name) -> str:
    name_vals = normalize_group_key(name)
    return " , ".join(f"{col} : [ {val} ] " for col, val in zip(group_cols, name_vals))


def render_metric_table_html(
    display_group: pd.DataFrame,
    metric_label: str,
    group_suffix: str,
    args,
) -> str:
    title = (
        f'<div style="font-size: 1.25em; font-weight: 600; margin: 12px 0;">'
        f"{_html.escape(metric_label)}"
        f" — {_html.escape(group_suffix)}"
        f"</div>\n"
    )

    metric_name = metric_label.lower()
    if "ttft" in metric_name:
        styler = _highlight_threshold(display_group, args.ttft_max_ms)
    elif ("tpot" in metric_name) or ("median" in metric_name) or ("p99" in metric_name):
        styler = _highlight_threshold(display_group, args.tpot_max_ms)
    else:
        styler = display_group.style

    styler = _apply_two_decimals(styler)
    styler = highlight_ratio_columns(styler)

    return title + styler.to_html(table_attributes='border="1" class="dataframe"')


def maybe_write_plot(
    main_fh,
    sub_fh,
    group_df: pd.DataFrame,
    raw_data_cols: list[str],
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

    fig.update_traces(hovertemplate="%{y:.2f}<extra></extra>")
    fig.update_yaxes(tickformat=".2f")

    metric_name = metric_label.lower()
    if "ttft" in metric_name:
        _add_limit_line(fig, args.ttft_max_ms, "TTFT limit")
    elif ("tpot" in metric_name) or ("median" in metric_name) or ("p99" in metric_name):
        _add_limit_line(fig, args.tpot_max_ms, "TPOT limit")

    html = fig.to_html(full_html=True, include_plotlyjs="cdn")
    main_fh.write(html)
    sub_fh.write(html)


def build_group_keys(
    df: pd.DataFrame, group_cols: list[str], sort_cols: list[str] | None = None
):
    if sort_cols:
        df = df.sort_values(by=sort_cols)
    gb = df.groupby(group_cols, dropna=False)
    return [k for k, _ in gb]


def write_report_group_first(
    files: list[str], info_cols: list[str], plan: MetricPlan, args
):
    name_column = "Test name"
    y_axis_col = get_y_axis_col(info_cols, args.xaxis)

    print("comparing : " + ", ".join(files))

    metric_cache: dict[str, tuple[pd.DataFrame, list[str]]] = {}
    group_cols_canonical: list[str] | None = None

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
        if group_cols_canonical is None:
            group_cols_canonical = group_cols
        else:
            group_cols_canonical = [c for c in group_cols_canonical if c in group_cols]

        metric_cache[metric_label] = (
            output_df.sort_values(by=args.xaxis),
            raw_data_cols,
        )

    if not group_cols_canonical:
        raise ValueError("No canonical group columns found across metrics.")

    first_metric = plan.data_cols[0]
    first_df_sorted, _ = metric_cache[first_metric]
    group_keys = build_group_keys(
        first_df_sorted, group_cols_canonical, sort_cols=[args.xaxis]
    )

    metric_groupbys = {
        metric_label: df.groupby(group_cols_canonical, dropna=False)
        for metric_label, (df, _) in metric_cache.items()
    }

    csv_dir = Path(args.csv_out_dir) if args.csv_out_dir else None
    if csv_dir:
        csv_dir.mkdir(parents=True, exist_ok=True)

    excel_path = args.excel_out or "perf_comparison.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as xw:
        # ---- Environment sheet (first) ----
        env_sheet = _sanitize_sheet_name("Environment")
        env_df = _load_env_df_for_inputs(args, files)
        if env_df is None or env_df.empty:
            pd.DataFrame(
                [
                    {
                        "Section": "Environment",
                        "Key": "vllm_env.txt",
                        "Value": "NOT FOUND (or empty)",
                    }
                ]
            ).to_excel(xw, sheet_name=env_sheet, index=False)
        else:
            env_df.to_excel(xw, sheet_name=env_sheet, index=False)
        with open("perf_comparison.html", "w", encoding="utf-8") as main_fh:
            main_fh.write('<meta charset="utf-8">\n')
            for gkey in group_keys:
                gkey_tuple = normalize_group_key(gkey)
                suffix = build_group_suffix(group_cols_canonical, gkey_tuple)
                sub_path = group_filename(gkey_tuple)
                group_header = (
                    '<div style="font-size: 1.4em; font-weight: 700; '
                    'margin: 18px 0 10px 0;">'
                    f"{_html.escape(suffix)}"
                    "</div>\n"
                )

                main_fh.write(group_header)

                sheet = _group_to_sheet_base(group_cols_canonical, gkey_tuple)
                sheet_base = sheet
                dedup_i = 1
                while sheet in xw.sheets:
                    dedup_i += 1
                    sheet = _sanitize_sheet_name(f"{sheet_base}_{dedup_i}")

                excel_blocks: list[tuple[str, pd.DataFrame]] = []

                with open(sub_path, "w", encoding="utf-8") as sub_fh:
                    sub_fh.write('<meta charset="utf-8">\n')
                    sub_fh.write(group_header)
                    tput_group_df = None
                    ttft_group_df = None
                    tpot_group_df = None
                    conc_col = args.xaxis

                    for metric_label in plan.data_cols:
                        gb = metric_groupbys[metric_label]
                        df_sorted, raw_data_cols = metric_cache[metric_label]

                        try:
                            group_df = gb.get_group(gkey)
                        except KeyError:
                            missing = (
                                '<div style="font-size: 1.1em; font-weight: 600; '
                                'margin: 10px 0;">'
                                f"{_html.escape(metric_label)} — missing for this group"
                                "</div>\n"
                            )
                            main_fh.write(missing)
                            sub_fh.write(missing)
                            continue

                        if conc_col not in group_df.columns:
                            conc_col = _find_concurrency_col(group_df)

                        mn = metric_label.lower().strip()
                        if "tok/s" in mn:
                            tput_group_df = group_df
                        elif "ttft" in mn:
                            ttft_group_df = group_df
                        elif mn in ("p99", "median") or "tpot" in mn:
                            tpot_group_df = group_df

                        display_group = group_df.drop(
                            columns=group_cols_canonical, errors="ignore"
                        )

                        html = render_metric_table_html(
                            display_group, metric_label, suffix, args
                        )
                        main_fh.write(html)
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

                        excel_blocks.append(
                            (metric_label, display_group.reset_index(drop=True))
                        )
                        if csv_dir:
                            fn = _safe_filename(
                                f"{sheet}__{metric_label}".replace(" ", "_").replace(
                                    "/", "_"
                                )
                            )
                            display_group.to_csv(csv_dir / f"{fn}.csv", index=False)

                    summary_html = build_valid_max_concurrency_summary_html(
                        tput_group_df=tput_group_df,
                        ttft_group_df=ttft_group_df,
                        tpot_group_df=tpot_group_df,
                        conc_col=conc_col,
                        args=args,
                    )
                    if summary_html:
                        main_fh.write(summary_html)
                        sub_fh.write(summary_html)

                    summary_df = build_valid_max_concurrency_summary_df(
                        tput_group_df=tput_group_df,
                        ttft_group_df=ttft_group_df,
                        tpot_group_df=tpot_group_df,
                        conc_col=conc_col,
                        args=args,
                    )
                    if summary_df is not None:
                        excel_blocks.append(
                            ("Valid Max Concurrency Summary", summary_df)
                        )
                        if csv_dir:
                            fn = _safe_filename(
                                f"{sheet}__Valid_Max_Concurrency_Summary"
                            )
                            summary_df.to_csv(csv_dir / f"{fn}.csv", index=False)

                _write_tables_to_excel_sheet(xw, sheet, excel_blocks)

    print(f"Wrote Excel: {excel_path}")
    if csv_dir:
        print(f"Wrote CSVs under: {csv_dir}")


def main():
    args = build_parser().parse_args()
    info_cols = list(DEFAULT_INFO_COLS)
    plan = choose_metrics(args.latency)
    files, info_cols = prepare_input_files(args, info_cols)
    write_report_group_first(files, info_cols, plan, args)


if __name__ == "__main__":
    main()
