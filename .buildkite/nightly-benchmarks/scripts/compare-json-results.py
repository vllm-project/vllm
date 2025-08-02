# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse

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
    return concat_df , raw_data_cols


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file", action="append", type=str, help="input file name"
    )
    parser.add_argument(
        "--debug", action="store_true", help="show all information for debugging"
    )
    parser.add_argument(
        "--plot", action="store_true", help="plot perf diagrams"
    )
    args = parser.parse_args()
    files = args.file
    print("comparing : " + ", ".join(files))

    drop_column = "P99"
    name_column = "Test name"
    info_cols = ["Model", "Dataset Name", "Input Len", "Output Len", "# of max concurrency.", "qps"]
    data_cols_to_compare = ["Output Tput (tok/s)", "Median TTFT (ms)", "Median"]
    html_msgs_for_data_cols = [
        "Compare Output Tokens /n",
        "Median TTFT /n",
        "Median TPOT /n",
    ]
    debug = args.debug
    plot = args.plot
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


            print(output_df)
            html = output_df.to_html()
            text_file.write(html_msgs_for_data_cols[i])
            text_file.write(html)

            if plot is True:
                import pandas as pd
                import plotly.express as px

                raw_data_cols.insert(0, info_cols[1])
                df = output_df[raw_data_cols]
                df_sorted = df.sort_values(by=info_cols[1])
                print(df_sorted)
                # Melt DataFrame for plotting
                df_melted = df_sorted.melt(id_vars=info_cols[1], var_name="Configuration", value_name=data_cols_to_compare[i])
                title = data_cols_to_compare[i] + " vs " + info_cols[1]
                # Create Plotly line chart
                fig = px.line(df_melted, x=info_cols[1], y=data_cols_to_compare[i], color="Configuration",
                    title=title, markers=True)
                # Export to HTML
                text_file.write(fig.to_html(full_html=True, include_plotlyjs='cdn'))

