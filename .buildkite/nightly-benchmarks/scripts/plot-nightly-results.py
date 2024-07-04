
import json
import os
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import math

import pandas as pd
from tabulate import tabulate

def parse_arguments():
    parser = argparse.ArgumentParser(description='Parse command line arguments for summary-nightly-results script.')
    parser.add_argument('--results-folder', type=str, required=True, help='The folder where the results are stored.')
    parser.add_argument('--description', type=str, required=True, help='Description of the results.')
    
    args = parser.parse_args()
    return args

    
def main(args):
    results_folder = Path(args.results_folder)
    
    results = []

    # collect results
    for test_file in results_folder.glob("*_nightly_results.json"):
        with open(test_file, "r") as f:
            results = results + json.loads(f.read())
            
            
    # generate markdown table            
    df = pd.DataFrame.from_dict(results)

    md_table = tabulate(df,
                        headers='keys',
                        tablefmt='pipe',
                        showindex=False)
                        
    with open(args.description, "r") as f:
        description = f.read()
        
    description = description.format(
        nightly_results_benchmarking_table=md_table
    )
    
    with open("nightly_results.md", "w") as f:
        f.write(description)
        

    plt.rcParams.update({'font.size': 20})
        
    # plot results
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    methods = ["vllm", "trt", "lmdeploy", "tgi"]
    for i, model in enumerate(["llama8B", "llama70B", "mixtral8x7B"]):
        for j, metric in enumerate(["TTFT", "ITL"]):
            means, stds = [], []
            for method in methods:
                target = df['Test name'].str.contains(model)
                target = target & df['Test name'].str.contains(method)
                filtered_df = df[target]
                
                if filtered_df.empty:
                    means.append(0.)
                    stds.append(0.)
                else:
                    means.append(filtered_df[f"Mean {metric} (ms)"].values[0])
                    std = filtered_df[f"Std {metric} (ms)"].values[0]
                    success = filtered_df["Successful req."].values[0]
                    stds.append(std / math.sqrt(success))
                    
            ax = axes[i, j]
            
            ax.errorbar(
                ["vllm", "trt", "lmdeploy", "tgi"], 
                means, 
                yerr=stds,
                fmt='o', capsize=5)
            ax.set_ylim(bottom=0)

            for i, (method, mean, std) in enumerate(zip(method, means, stds)):
                ax.text(
                    i - 0.2, mean,  # Adjust position above the error bar
                    f'{mean:.0f}', 
                    ha='center', 
                    va='bottom'
                )
            
            ax.set_xlabel("Method")
            ax.set_ylabel(f"{metric} (ms)")
            ax.set_title(f"{model} {metric} comparison")
    
    fig.savefig("nightly_results.png")

if __name__ == '__main__':
    args = parse_arguments()
    main(args)