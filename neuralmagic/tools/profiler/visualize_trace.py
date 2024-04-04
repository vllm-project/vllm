import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd


def trim_string_back(string: str, width: int):
    if len(string) > width:
        offset = len(string) - width + 3
        string = string[:-offset]
        if len(string) > 3:
            string = string + "..."
    return string


def abbreviate_known_names(name: str):
    abbreviations = {
        "MergedColumnParallelLinear": "MCPLinear",
        "QKVParallelLinear": "QKVPLinear",
        "RowParallelLinear": "RPLinear",
        "weight=": "w=",
        "bfloat16": "bf16",
        "float16": "f16",
    }
    for key, value in abbreviations.items():
        name = name.replace(key, value)
    return name


def shorten_plot_legend_strings(legend, max_char_len: int):
    for t in legend.get_texts():
        t.set_text(
            trim_string_back(abbreviate_known_names(t.get_text()),
                             max_char_len))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--json-trace",
        type=str,
        required=True,
        help="json trace file output by examples/offline_profile.py")
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="Output figure file, should be a image file such as pdf, "
        "jpeg, png, etc., defaults to <json_trace>.pdf")
    parser.add_argument("--level",
                        type=str,
                        default="module",
                        choices=["module", "kernel"])
    parser.add_argument("--top_k",
                        type=int,
                        default=9,
                        help="Only graph the top `top_k` entries by time.")
    parser.add_argument("--ignore_sampler",
                        action='store_true',
                        help="Ignore everything under the \"Sampler\" module")

    args = parser.parse_args()

    ignore_sampler = args.ignore_sampler
    make_names_unique = False
    top_k = args.top_k

    if args.level == "module":
        depth = -2
        make_names_unique = True
    elif args.level == "kernel":
        depth = -1
    else:
        raise Exception(f"Unexpected level value ({args.level})")

    if ignore_sampler:
        print("WARNING: ignoring Sampler time so the pct_cuda_time will not "
              "add up to 100%")

    json_trace = args.json_trace
    output = args.output if args.output else json_trace.strip(".json") + ".pdf"

    with open(json_trace, "r") as f:
        profile_data = json.load(f)

    prefill_entries_and_traces = []
    decode_entries_and_traces = []

    def largest_dist_from_leaf(node, depth=0):
        if len(node["children"]) == 0:
            return depth
        return max([
            largest_dist_from_leaf(child, depth=depth + 1)
            for child in node["children"]
        ])

    def get_entries_at_depth(depth,
                             entries_and_traces,
                             node,
                             curr_depth=0,
                             trace=()):
        if ignore_sampler and node["entry"]["name"] == "Sampler":
            return

        if (depth >= 0 and depth == curr_depth) or (
                depth < 0
                and largest_dist_from_leaf(node) == (abs(depth) - 1)):
            entries_and_traces.append((node["entry"], trace))
        trace = (node["entry"]["name"], ) + trace
        for child in node["children"]:
            get_entries_at_depth(depth,
                                 entries_and_traces,
                                 child,
                                 curr_depth=curr_depth + 1,
                                 trace=trace)

    for root in profile_data["prefill"]["summary_stats"]:
        get_entries_at_depth(depth, prefill_entries_and_traces, root)
    for root in profile_data["decode"]["summary_stats"]:
        get_entries_at_depth(depth, decode_entries_and_traces, root)

    def attempt_to_make_names_unique(entries_and_traces):
        names, non_unique_names = (set(), set())

        def all_the_same(items) -> bool:
            return all(i == items[0] for i in items)

        for entry, _ in entries_and_traces:
            if entry["name"] in names:
                non_unique_names.add(entry["name"])
            else:
                names.add(entry["name"])

        for name in non_unique_names:
            entries_and_traces_with_name = [
                (entry, trace) for entry, trace in entries_and_traces
                if entry["name"] == name
            ]

            zipped_traces = list(
                zip(*[trace for _, trace in entries_and_traces_with_name]))
            first_trace_difference = next(
                (i for i, trace_eles in enumerate(zipped_traces)
                 if not all_the_same(trace_eles)), None)

            if first_trace_difference is None:
                # can't create a unique name, leave them names as the
                # are they will get aggregated by the pivot_table call
                continue

            for entry, trace in entries_and_traces_with_name:
                entry["name"] = " <- ".join((entry["name"], ) +
                                            trace[:first_trace_difference + 1])

    if make_names_unique:
        attempt_to_make_names_unique(prefill_entries_and_traces)
        attempt_to_make_names_unique(decode_entries_and_traces)

    def keep_only_top_entries(df, metric, top_k=9):
        df.loc[df.nsmallest(len(df) - top_k + 1, metric).index,
               ["name"]] = "others"

    prefill_df = pd.DataFrame(
        [entry for entry, _ in prefill_entries_and_traces])
    prefill_df["phase"] = "prefill"
    decode_df = pd.DataFrame([entry for entry, _ in decode_entries_and_traces])
    decode_df["phase"] = "decode"

    if top_k:
        keep_only_top_entries(prefill_df, "cuda_time_us", top_k)
        keep_only_top_entries(decode_df, "cuda_time_us", top_k)

    df = pd.concat([prefill_df, decode_df])
    df["cuda_time_ms"] = df["cuda_time_us"] / 1000

    fig, axes = plt.subplots(2, figsize=(5, 8), sharex=True)

    def plot_metric(metric: str, ax, add_totals=False):
        pivoted_df = df.pivot_table(index="phase",
                                    columns="name",
                                    values=metric,
                                    aggfunc="sum")
        pivoted_df.plot.bar(stacked=True, legend=False, ax=ax)
        ax.set_ylabel(metric)

        if add_totals:
            ax.bar_label(ax.containers[-1])

    plot_metric("cuda_time_ms", ax=axes[0], add_totals=True)
    plot_metric("pct_cuda_time", ax=axes[1])

    handles, labels = plt.gca().get_legend_handles_labels()
    legend = fig.legend(handles,
                        labels,
                        loc='center left',
                        bbox_to_anchor=(0.93, 0.5))
    shorten_plot_legend_strings(legend, 50)

    context = profile_data["context"]
    plt.suptitle(
        f"{context['model']}\n"
        f"Batch={context['batch_size']}, "
        f"PromptLen={context['prompt_len']}, "
        f"NumGpus={context['tensor_parallel_size']}"
        f"{', Sparsity ' +  context['sparsity'] if context['sparsity'] else ''}"
    )
    plt.savefig(output, bbox_inches='tight')
    print("Created: ", output)
