# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
import os
from importlib import util

import pandas as pd

plotly_found = util.find_spec("plotly.express") is not None


def compare_data_columns(
    files, name_column, data_column, info_cols, drop_column, debug=False
):
    """
    Align concatenation by keys derived from info_cols instead of row order.
    - Pick one canonical key list: subset of info_cols present in ALL files.
    - For each file: set index to those keys, aggregate duplicates
    - (mean for metric, first for names).
    - Concat along axis=1 (indexes align), then reset_index so callers can
    - group by columns.
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

        # (NEW) debug: aligned test-name column per file
        if debug and name_column in df_idx.columns:
            name_s = df_idx[name_column]
            if not name_s.index.is_unique:
                name_s = name_s.groupby(level=key_cols, dropna=False).first()
            name_s.name = f"{file_label}_name"
            frames.append(name_s)

        frames.append(s)
        raw_data_cols.append(file_label)
        compare_frames.append(s)

        # Generalize ratio: for any file N>=2, add ratio (fileN / file1)
        if len(compare_frames) >= 2:
            base = compare_frames[0]
            current = compare_frames[-1]
            ratio = current / base
            ratio = ratio.mask(base == 0)  # avoid inf when baseline is 0
            ratio.name = f"Ratio 1 vs {len(compare_frames)}"
            frames.append(ratio)

    # 4) concat on columns with aligned MultiIndex;
    # then reset_index to return keys as columns
    concat_df = pd.concat(frames, axis=1)
    concat_df = concat_df.reset_index(drop=True).reset_index()
    if "index" in concat_df.columns:
        concat_df = concat_df.drop(columns=["index"])

    # Ensure key/info columns appear first (in your info_cols order)
    front = [c for c in info_cols if c in concat_df.columns]
    rest = [c for c in concat_df.columns if c not in front]
    concat_df = concat_df[front + rest]

    print(raw_data_cols)
    return concat_df, raw_data_cols


def split_json_by_tp_pp(
    input_file: str = "benchmark_results.json", output_root: str = "."
) -> list[str]:
    """
    Split a benchmark JSON into separate folders by (TP Size, PP Size).

    Creates: <output_root>/tp{TP}_pp{PP}/benchmark_results.json
    Returns: list of file paths written.
    """
    # Load JSON data into DataFrame
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
        df = df[
            df[name_col].astype(str).str.contains(r"serving", case=False, na=False)
        ].copy()

    # Handle alias column names
    rename_map = {
        "tp_size": "TP Size",
        "tensor_parallel_size": "TP Size",
        "pp_size": "PP Size",
        "pipeline_parallel_size": "PP Size",
    }
    df.rename(
        columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True
    )

    # Ensure TP/PP columns exist (default to 1 if missing)
    if "TP Size" not in df.columns:
        df["TP Size"] = 1
    if "PP Size" not in df.columns:
        df["PP Size"] = 1

    # make sure TP/PP are numeric ints with no NaN
    df["TP Size"] = (
        pd.to_numeric(df.get("TP Size", 1), errors="coerce").fillna(1).astype(int)
    )
    df["PP Size"] = (
        pd.to_numeric(df.get("PP Size", 1), errors="coerce").fillna(1).astype(int)
    )

    # Split into separate folders
    saved_paths: list[str] = []
    for (tp, pp), group_df in df.groupby(["TP Size", "PP Size"], dropna=False):
        folder_name = os.path.join(output_root, f"tp{int(tp)}_pp{int(pp)}")
        os.makedirs(folder_name, exist_ok=True)
        filepath = os.path.join(folder_name, "benchmark_results.json")
        group_df.to_json(filepath, orient="records", indent=2, force_ascii=False)
        print(f"Saved: {filepath}")
        saved_paths.append(filepath)

    return saved_paths


if __name__ == "__main__":
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
    args = parser.parse_args()

    drop_column = "P99"
    name_column = "Test name"
    info_cols = [
        "Model",
        "Dataset Name",
        "Input Len",
        "Output Len",
        "TP Size",
        "PP Size",
        "# of max concurrency.",
        "qps",
    ]
    data_cols_to_compare = ["Output Tput (tok/s)", "Median TTFT (ms)", "Median"]
    html_msgs_for_data_cols = [
        "Compare Output Tokens /n",
        "Median TTFT /n",
        "Median TPOT /n",
    ]

    if len(args.file) == 1:
        files = split_json_by_tp_pp(args.file[0], output_root="splits")
        info_cols = [c for c in info_cols if c not in ("TP Size", "PP Size")]
    else:
        files = args.file
    print("comparing : " + ", ".join(files))
    debug = args.debug
    plot = args.plot
    # For Plot feature, assign y axis from one of info_cols
    y_axis_index = info_cols.index(args.xaxis) if args.xaxis in info_cols else 6
    with open("perf_comparison.html", "w") as text_file:
        for i in range(len(data_cols_to_compare)):
            output_df, raw_data_cols = compare_data_columns(
                files,
                name_column,
                data_cols_to_compare[i],
                info_cols,
                drop_column,
                debug=debug,
            )

            # For Plot feature, insert y axis from one of info_cols
            raw_data_cols.insert(0, info_cols[y_axis_index])

            filtered_info_cols = info_cols[:-2]
            existing_group_cols = [
                c for c in filtered_info_cols if c in output_df.columns
            ]
            if not existing_group_cols:
                raise ValueError(
                    f"No valid group-by columns  "
                    f"Expected subset: {filtered_info_cols}, "
                    f"but DataFrame has: {list(output_df.columns)}"
                )
            output_df_sorted = output_df.sort_values(by=existing_group_cols)
            output_groups = output_df_sorted.groupby(existing_group_cols, dropna=False)
            for name, group in output_groups:
                html = group.to_html()
                text_file.write(html_msgs_for_data_cols[i])
                text_file.write(html)

                if plot and plotly_found:
                    import plotly.express as px

                    df = group[raw_data_cols]
                    df_sorted = df.sort_values(by=info_cols[y_axis_index])
                    # Melt DataFrame for plotting
                    df_melted = df_sorted.melt(
                        id_vars=info_cols[y_axis_index],
                        var_name="Configuration",
                        value_name=data_cols_to_compare[i],
                    )
                    title = data_cols_to_compare[i] + " vs " + info_cols[y_axis_index]
                    # Create Plotly line chart
                    fig = px.line(
                        df_melted,
                        x=info_cols[y_axis_index],
                        y=data_cols_to_compare[i],
                        color="Configuration",
                        title=title,
                        markers=True,
                    )
                    # Export to HTML
                    text_file.write(fig.to_html(full_html=True, include_plotlyjs="cdn"))
