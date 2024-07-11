import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=
        'Parse command line arguments for summary-nightly-results script.')
    parser.add_argument('--results-folder',
                        type=str,
                        required=True,
                        help='The folder where the results are stored.')
    parser.add_argument('--description',
                        type=str,
                        required=True,
                        help='Description of the results.')

    args = parser.parse_args()
    return args


def main(args):
    bar_colors = ['#56B4E9', '#009E73', '#D55E00', '#E69F00']
    results_folder = Path(args.results_folder)

    results = []

    # collect results
    for test_file in results_folder.glob("*_nightly_results.json"):
        with open(test_file, "r") as f:
            results = results + json.loads(f.read())

    # generate markdown table
    df = pd.DataFrame.from_dict(results)

    md_table = tabulate(df, headers='keys', tablefmt='pipe', showindex=False)

    with open(args.description, "r") as f:
        description = f.read()

    description = description.format(
        nightly_results_benchmarking_table=md_table)

    with open("nightly_results.md", "w") as f:
        f.write(description)

    plt.rcParams.update({'font.size': 20})

    # plot results
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.subplots_adjust(hspace=1)
    methods = ["vllm", "trt", "lmdeploy", "tgi"]
    for i, model in enumerate(["llama8B", "llama70B", "mixtral8x7B"]):
        for j, metric in enumerate(["TTFT", "ITL"]):
            means, stds = [], []
            for method in methods:
                target = df['Test name'].str.contains(model)
                target = target & df['Engine'].str.contains(method)
                filtered_df = df[target]

                if filtered_df.empty:
                    means.append(0.)
                    stds.append(0.)
                else:
                    means.append(filtered_df[f"Mean {metric} (ms)"].values[0])
                    std = filtered_df[f"Std {metric} (ms)"].values[0]
                    success = filtered_df["Successful req."].values[0]
                    stds.append(std / math.sqrt(success))

            print(model, metric)
            print(means, stds)

            ax = axes[i, j + 1]

            bars = ax.bar(
                ["vllm", "trt", "lmdeploy", "tgi"],
                means,
                yerr=stds,
                capsize=10,
            )
            for idx, bar in enumerate(bars):
                bar.set_color(bar_colors[idx])
            ax.set_ylim(bottom=0)

            ax.set_ylabel(f"{metric} (ms)")
            ax.set_title(f"{model} {metric}")
            ax.grid(axis='y')

        metric = "Tput"
        j = 0
        if True:
            tputs = []
            for method in methods:
                target = df['Test name'].str.contains(model)
                target = target & df['Engine'].str.contains(method)
                filtered_df = df[target]

                if filtered_df.empty:
                    tputs.append(0.)
                else:
                    input_tput = filtered_df["Input Tput (tok/s)"].values[0]
                    output_tput = filtered_df["Output Tput (tok/s)"].values[0]
                    tputs.append(input_tput + output_tput)

            print(model, metric)
            print(tputs)

            ax = axes[i, j]

            bars = ax.bar(
                ["vllm", "trt", "lmdeploy", "tgi"],
                tputs,
            )
            for idx, bar in enumerate(bars):
                bar.set_color(bar_colors[idx])

            ax.set_ylim(bottom=0)

            ax.set_ylabel("Tput (token/s)")
            ax.set_title(f"{model} {metric}")
            ax.grid(axis='y')

    fig.tight_layout()
    fig.savefig("nightly_results.png", bbox_inches='tight', dpi=400)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
