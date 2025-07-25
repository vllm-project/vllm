# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse

import pandas as pd


def compare_data_columns(
    files, name_column, data_column, info_cols, drop_column, debug=False
):
    print("\ncompare_data_column: " + data_column)
    frames = []
    compare_frames = []
    for file in files:
        data_df = pd.read_json(file)
        serving_df = data_df.dropna(subset=[drop_column], ignore_index=True)
        if debug is True or not frames:
            serving_df = serving_df.rename(columns={name_column: file + "_name"})
            frames.append(serving_df[file + "_name"])
            for col in info_cols:
                frames.append(serving_df[col])
        serving_df = serving_df.rename(columns={data_column: file})
        frames.append(serving_df[file])
        compare_frames.append(serving_df[file])
        if len(compare_frames) >= 2:
            # Compare numbers among two files
            ratio_df = compare_frames[1] / compare_frames[0]
            frames.append(ratio_df)
            compare_frames.pop(1)

    concat_df = pd.concat(frames, axis=1)
    return concat_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file", action="append", type=str, help="input file name"
    )
    parser.add_argument(
        "--debug", action="store_true", help="show all information for debugging"
    )
    args = parser.parse_args()
    files = args.file
    print("comparing : " + ", ".join(files))

    drop_column = "P99"
    name_column = "Test name"
    info_cols = ["# of max concurrency.", "Total input tokens", "Total output tokens"]
    data_cols_to_compare = ["Output Tput (tok/s)", "Median TTFT (ms)", "Median"]
    html_msgs_for_data_cols = [
        "Compare Output Tokens /n",
        "Median TTFT /n",
        "Median TPOT /n",
    ]
    debug = args.debug
    with open("perf_comparison.html", "w") as text_file:
        for i in range(len(data_cols_to_compare)):
            output_df = compare_data_columns(
                files,
                name_column,
                data_cols_to_compare[i],
                info_cols,
                drop_column,
                debug=debug,
            )
            print(output_df)
            html = output_df.to_html()
            text_file.write(html_msgs_for_data_cols[i])
            text_file.write(html)
