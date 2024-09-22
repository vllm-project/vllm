import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


def get_perf(df, method, model, metric):

    means = []

    for qps in [2, 4, 8, 16, 32, "inf"]:
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

    if metric in ["TTFT", "Latency", "TPOT", "ITL"]:
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
        # assert metric == "Tput"
        # mean = get_perf(df, method, model, "Input Tput (tok/s)") + \
        # get_perf(df, method, model, "Output Tput (tok/s)")
        # mean = get_perf(df, method, model, 'Tput (req/s)')
        mean = get_perf(df, method, model, metric)
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

    for result in results:
        if 'sonnet_512_16' in result['Test name']:
            result['Test name'] = result['Test name'].replace(
                'sonnet_512_16', 'prefill_heavy')
        if 'sonnet_512_256' in result['Test name']:
            result['Test name'] = result['Test name'].replace(
                'sonnet_512_256', 'decode_heavy')

    # generate markdown table
    df = pd.DataFrame.from_dict(sorted(results, key=lambda x: x['Test name']))
    df['Input tokens per request'] = df['Total input tokens'] / df[
        'Successful req.']
    df['Output tokens per request'] = df['Total output tokens'] / df[
        'Successful req.']
    df['Throughput (req/s)'] = df['Tput (req/s)']

    df2 = df.copy()

    drop_keys = []
    for key in df:
        if key not in [
                'Test name', 'Mean TTFT (ms)', 'Std TTFT (ms)',
                'Mean TPOT (ms)', 'Std TPOT (ms)', 'Throughput (req/s)',
                'Input tokens per request', 'Output tokens per request'
        ]:
            drop_keys.append(key)
    df = df.drop(columns=drop_keys)
    table = tabulate(df, headers='keys', tablefmt='pipe', showindex=False)

    with open("nightly_results.md", "w") as f:
        f.write(table)

    for qps in [2, 4, 8, 16, 32, "inf"]:
        for dataset in ["sharegpt", "decode_heavy", "prefill_heavy"]:
            subset_df = df['Test name'].str.contains(dataset)
            subset_df = subset_df & df['Test name'].str.contains("qps_" +
                                                                 str(qps))
            subset_df = df[subset_df]
            print((subset_df['Output tokens per request'].max() -
                   subset_df['Output tokens per request'].min()))

    df = df2
    df['Throughput'] = df['Throughput (req/s)']

    plt.rcParams.update({'font.size': 24})
    plt.set_cmap("cividis")

    # plot results
    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    fig.subplots_adjust(hspace=1)
    # methods = ['vllm', 'sglang', 'lmdeploy', 'trt']
    # formal_name = ['vLLM', 'SGLang', 'lmdeploy',  'TRT-LLM']
    methods = ['vllm_053post1', 'vllm_055'][::-1]
    formal_name = ['vLLM v0.5.3', 'vLLM v0.6.0'][::-1]
    # for model in ["llama70B"]:
    for i, model in enumerate(["llama8B", "llama70B"]):
        # for i, dataset in enumerate(["sharegpt",
        # "decode_heavy", "prefill_heavy"]):
        for dataset in ["sharegpt"]:
            for j, metric in enumerate(["TPOT", "Throughput"][::-1]):

                my_df = df[df["Test name"].str.contains(dataset)]
                # my_dataset_name = {
                #     "sharegpt": "ShareGPT",
                #     "sonnet_512_256": "Decode-heavy",
                #     "sonnet_512_16": "Prefill-heavy",
                # }[dataset]
                # my_dataset_name = dataset

                ax = axes[i, j]
                if metric in ["TTFT", "Latency", "TPOT", "ITL"]:
                    ax.set_ylabel(f"{metric} (ms)")
                else:
                    ax.set_ylabel("Throughput (req/s)")

                # if metric == "Tput":
                #     ax.set_title(f"{my_dataset_name} Throughput")
                # else:
                #     ax.set_title(f"{my_dataset_name} {metric}")
                # ax.grid(axis='y')
                # print(model, metric)

                tput = {}
                for k, method in enumerate(methods):
                    mean, std = get_perf_w_std(my_df, method, model, metric)
                    label = formal_name[k]

                    # print(method, metric, mean, std)

                    if "Tput" not in metric and "Throughput" not in metric:
                        ax.errorbar(
                            range(len(mean)),
                            mean,
                            yerr=std,
                            capsize=10,
                            capthick=4,
                            label=label,
                            lw=4,
                        )

                        ax.set_xticks(range(len(mean)))
                        ax.set_xticklabels(["2", "4", "8", "16", "32", "inf"])
                        ax.set_xlabel("QPS")
                    else:
                        tput[method] = mean[-1]

                if "Tput" not in metric and "Throughput" not in metric:
                    # ax.legend(framealpha=0.5)
                    ax.set_ylim(bottom=0)
                else:
                    for _ in range(len(formal_name)):
                        ax.bar(1 - _, tput[methods[_]])
                    ax.set_xticks(range(len(formal_name)))
                    ax.set_xticklabels(formal_name[::-1])
                    ax.set_ylim(bottom=0)
                    # ax.bar(formal_name, tput.values())

    fig.tight_layout()
    fig.savefig("nightly_results.png", bbox_inches='tight', dpi=400)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
