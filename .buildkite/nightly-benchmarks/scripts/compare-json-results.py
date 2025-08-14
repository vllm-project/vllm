# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
import os

import pandas as pd


def compare_data_columns(
    files, name_column, data_column, info_cols, drop_column, debug=False
):
    print("\ncompare_data_column: " + data_column)
    frames = []
    raw_data_cols = []
    compare_frames = []
    for file in files:
        data_df = pd.read_json(file)
        serving_df = data_df.dropna(subset=[drop_column], ignore_index=True)
        # Show all info columns in the first couple columns
        if not frames:
            for col in info_cols:
                if col not in serving_df.columns:
                    print(f"Skipping missing column: {col}")
                    continue
                frames.append(serving_df[col])
        # only show test name under debug mode
        if debug is True:
            serving_df = serving_df.rename(columns={name_column: file + "_name"})
            frames.append(serving_df[file + "_name"])

        file = "/".join(file.split("/")[:-1])
        serving_df = serving_df.rename(columns={data_column: file})
        frames.append(serving_df[file])
        raw_data_cols.append(file)
        compare_frames.append(serving_df[file])
        if len(compare_frames) >= 2:
            # Compare numbers among two files
            ratio_df = compare_frames[1] / compare_frames[0]
            frames.append(ratio_df)
            compare_frames.pop(1)

    concat_df = pd.concat(frames, axis=1)
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
        help="column name to use as X Axis in comparision graph",
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

                if plot is True:
                    import pandas as pd
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
