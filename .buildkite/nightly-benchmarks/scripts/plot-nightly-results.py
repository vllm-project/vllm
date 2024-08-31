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
    args = parser.parse_args()
    return args

    
def get_perf(df, method, model, metric):
    
    means = []
    
    for qps in [2,4,8,16,32,"inf"]:
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
    
    if metric in ["TTFT", "Latency"]:
        mean = get_perf(df, method, model, "Mean " + metric + " (ms)")
        mean = mean.tolist()
        std = get_perf(df, method, model, "Std " + metric + " (ms)")
        if std.mean() == 0:
            std = None
        success = get_perf(df, method, model, "Successful req.")
        if std is not None:
            std = std / np.sqrt(success)
            std = std.tolist()

    else:
        assert metric == "Tput"
        mean = get_perf(df, method, model, "Input Tput (tok/s)") + get_perf(df, method, model, "Output Tput (tok/s)")
        mean = mean.tolist()
        std = None

    return mean, std


def main(args):
    results_folder = Path(args.results_folder)

    results = []

    # collect results
    for test_file in results_folder.glob("*_nightly_results.json"):
        with open(test_file, "r") as f:
            results = results + json.loads(f.read())
    
    # results = []
    # for step in [0,1,10,11,12]:
        
    #     with open(results_folder / f"step{step}.txt", "r") as f:
    #         temp_results = json.loads(f.read())
    #         # print(len(temp_results))
    #         for i in temp_results:
    #             i['step'] = step
    #         results = results + temp_results
    #     print(len(results))
            
    # generate markdown table
    df = pd.DataFrame.from_dict(results)
    # print(df)
    # df = df[df["Test name"].str.contains(args.dataset)]
    table = tabulate(df, headers='keys', tablefmt='pipe', showindex=False)
    with open(f"nightly_results.md", "w") as f:
        f.write(table)
    

    plt.rcParams.update({'font.size': 18})
    plt.set_cmap("cividis")

    # plot results
    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    fig.subplots_adjust(hspace=1)
    # methods = [0,1,10,11,12]
    methods = ['vllm', 'trt', 'sglang', 'lmdeploy']
    formal_name = ['vLLM', 'TRT-LLM', 'SGLang', 'lmdeploy']
    for model in ["llama70B"]:
        for i, dataset in enumerate(["sharegpt", "sonnet_128_2048", "sonnet_2048_128"]):
            for j, metric in enumerate(["Tput", "Latency"]):
                
                my_df = df[df["Test name"].str.contains(dataset)]
                my_dataset_name = {
                    "sharegpt": "ShareGPT",
                    "sonnet_128_2048": "Decode-heavy",
                    "sonnet_2048_128": "Prefill-heavy",
                }[dataset]
                
                ax = axes[i,j]
                if metric in ["TTFT", "Latency"]:
                    ax.set_ylabel(f"{metric} (ms)")
                else:
                    ax.set_ylabel(f"Thoughput (tokens/s)")
                
                if metric == "Tput":
                    ax.set_title(f"{model} {my_dataset_name} Thoughput")
                else:
                    ax.set_title(f"{model} {my_dataset_name} {metric}")
                ax.grid(axis='y')
                print(model, metric)
                
                tput = {}
                for k, method in enumerate(methods):
                    mean, std = get_perf_w_std(my_df, method, model, metric)
                    label = formal_name[k]
                    
                    if metric == "Latency":
                        ax.errorbar(range(len(mean)),
                                    mean, 
                                    yerr=std, 
                                    capsize=10, 
                                    capthick=4,
                                    label=label,
                                    lw=4,)
                        ax.set_ylim(bottom=0)
                        ax.set_xticks(range(len(mean)))
                        ax.set_xticklabels(["2", "4", "8", "16", "32", "inf"])
                        ax.set_xlabel("QPS")
                    else:
                        tput[method] = mean[-1]
                        
                if metric == "Latency":
                    ax.legend()   
                else:
                    ax.bar(formal_name, tput.values())

            


    fig.tight_layout()
    fig.savefig(f"nightly_results.png", bbox_inches='tight', dpi=400)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
