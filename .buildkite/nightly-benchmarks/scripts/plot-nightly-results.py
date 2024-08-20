import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
import numpy as np


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
    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        help="The dataset used for the benchmark.")

    args = parser.parse_args()
    return args

    
def get_perf(df, method, model, metric):
    
    means = []
    
    for qps in [2,4,8,16,"inf"]:
        target = df['Test name'].str.contains(model)
        target = target & df['Engine'].str.contains(method)
        target = target & df['Test name'].str.contains("qps_" + str(qps))
        filtered_df = df[target]

        if filtered_df.empty:
            means.append(0.)
        else:
            means.append(filtered_df[metric].values[0])

    return np.array(means)

def get_perf_w_std(df, method, model, metric):
    
    if metric in ["TTFT", "ITL"]:
        mean = get_perf(df, method, model, "Mean " + metric + " (ms)")
        std = get_perf(df, method, model, "Std " + metric + " (ms)")
        success = get_perf(df, method, model, "Successful req.")
        if std is not None:
            std = std / np.sqrt(success)

    else:
        assert metric == "Tput"
        mean = get_perf(df, method, model, "Input Tput (tok/s)") + get_perf(df, method, model, "Output Tput (tok/s)")
        std = None

    return mean, std


def main(args):
    results_folder = Path(args.results_folder)

    results = []

    # collect results
    for test_file in results_folder.glob("*_nightly_results.json"):
        with open(test_file, "r") as f:
            results = results + json.loads(f.read())

    # generate markdown table
    df = pd.DataFrame.from_dict(results)
    df = df[df["Test name"].str.contains(args.dataset)]

    md_table = tabulate(df, headers='keys', tablefmt='pipe', showindex=False)

    with open(args.description, "r") as f:
        description = f.read()

    description = description.format(
        nightly_results_benchmarking_table=md_table)

    with open("nightly_results.md", "w") as f:
        f.write(description)

    plt.rcParams.update({'font.size': 18})
    plt.set_cmap("cividis")

    # plot results
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.subplots_adjust(hspace=1)
    methods = ["vllm", "trtllm", "sglang", "lmdeploy", "tgi"]
    for i, model in enumerate(["llama8B", "llama70B", "mixtral8x7B"]):
        for j, metric in enumerate(["TTFT", "ITL", "Tput"]):
            ax = axes[i, j]
            ax.set_ylim(bottom=0)
            if metric in ["TTFT", "ITL"]:
                ax.set_ylabel(f"{metric} (ms)")
            else:
                ax.set_ylabel(f"{metric} (tok/s)")
            ax.set_xlabel("QPS")
            ax.set_title(f"{model} {metric}")
            ax.grid(axis='y')
            print("New line")
            print(model, metric)
            for k, method in enumerate(methods):
                mean, std = get_perf_w_std(df, method, model, metric)
                print(method, mean, std)
                ax.errorbar(["2", "4", "8", "16", "inf"], 
                            mean, 
                            yerr=std, 
                            capsize=10, 
                            label=method,)

            ax.legend()


    fig.tight_layout()
    fig.savefig(f"nightly_results_{args.dataset}.png", bbox_inches='tight', dpi=400)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
