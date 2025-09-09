# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd

## JSON parsing utils ####


def largest_dist_from_leaf(node: dict, depth: int = 0):
    if len(node["children"]) == 0:
        return depth
    return max([
        largest_dist_from_leaf(child, depth=depth + 1)
        for child in node["children"]
    ])


def get_entries_at_depth(depth: int,
                         entries_and_traces: list[tuple[Any, Any]],
                         node: dict,
                         curr_depth: int = 0,
                         trace=()):
    # assert that the query is at kernel or module level
    assert depth == -1 or depth == -2

    if curr_depth == 0 and largest_dist_from_leaf(node) <= (abs(depth) - 1):
        # The tree is not tall enough!
        entries_and_traces.append((node["entry"], trace))
        return

    if largest_dist_from_leaf(node) == (abs(depth) - 1):
        entries_and_traces.append((node["entry"], trace))

    trace = (node["entry"]["name"], ) + trace
    for child in node["children"]:
        get_entries_at_depth(depth,
                             entries_and_traces,
                             child,
                             curr_depth=curr_depth + 1,
                             trace=trace)


def fold_nodes(root: dict, nodes_to_fold: list[str]):

    stack: list[dict] = [root]
    while len(stack) != 0:
        node = stack.pop()
        if node['entry']['name'] in nodes_to_fold:
            node["children"] = []
            continue
        for child in node["children"]:
            stack.append(child)
    return root


## Operation name cleanup utils ####


def trim_string_back(string: str, width: int) -> str:
    if len(string) > width:
        offset = len(string) - width + 3
        string = string[:-offset]
        if len(string) > 3:
            string = string + "..."
    return string


def shorten_plot_legend_strings(legend, max_char_len: int):
    for t in legend.get_texts():
        t.set_text(
            trim_string_back(abbreviate_known_names(t.get_text()),
                             max_char_len))


def abbreviate_known_names(name: str) -> str:
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
        entries_and_traces_with_name = [(entry, trace)
                                        for entry, trace in entries_and_traces
                                        if entry["name"] == name]

        zipped_traces = list(
            zip(*[trace for _, trace in entries_and_traces_with_name]))
        first_trace_difference = next(
            (i for i, trace_eles in enumerate(zipped_traces)
             if not all_the_same(trace_eles)), None)

        if first_trace_difference is None:
            # can't create a unique name, leave the names as they
            # are they will get aggregated by the pivot_table call
            continue

        for entry, trace in entries_and_traces_with_name:
            entry["name"] = " <- ".join((entry["name"], ) +
                                        trace[:first_trace_difference + 1])


## Operation grouping utils ####
'''
    Group operations in the given dataframe by some high-level ops like,
    - gemms
    - attention
    - rms_norm 
    etc.
'''


def group_trace_by_operations(trace_df: pd.DataFrame) -> pd.DataFrame:

    def is_rms_norm(op_name: str):
        if "rms_norm_kernel" in op_name:
            return True

    def is_attention_block(op_name: str):
        if "flash_fwd" in op_name or \
            "reshape_and_cache_flash_kernel" in op_name:
            return True

    def is_quant(op_name: str):
        if "scaled_fp8_quant" in op_name or \
           "scaled_int8_quant" in op_name:
            return True

    # LoRA ops
    def is_sgmv_shrink(op_name: str):
        return "sgmv_shrink" in op_name

    def is_sgmv_expand(op_name: str):
        return "sgmv_expand" in op_name

    def is_bgmv_shrink(op_name: str):
        return "bgmv_shrink" in op_name

    def is_bgmv_expand(op_name: str):
        return "bgmv_expand" in op_name

    def is_cutlass_gemm_op(op_name: str):
        return "void cutlass::Kernel" in op_name or \
           "void cutlass::device_kernel" in op_name

    def is_gemm_op(op_name: str):
        if is_quant(op_name):
            return False
        return is_cutlass_gemm_op(op_name) or \
           "xmma_gemm" in op_name  or \
           "gemv2T_kernel" in op_name or \
           "splitKreduce" in op_name or \
           "s16816gemm" in op_name

    def is_elementwise_op(op_name: str):
        return "elementwise_kernel" in op_name

    def is_mem_op(op_name: str):
        return "memcpy" in op_name.lower() or \
               "memset" in op_name.lower()

    def is_vocab_embedding_op(op_name: str):
        return "vocabparallelembed" in op_name.lower()

    # nccl ops
    def is_nccl_op(op_name: str):
        return "nccl" in op_name.lower()

    def is_nccl_all_reduce(op_name: str):
        return is_nccl_op(op_name) and \
                ("all_reduce" in op_name.lower() or \
                "allreduce" in op_name.lower())

    def is_nccl_gather(op_name: str):
        return is_nccl_op(op_name) and \
                "gather" in op_name.lower()

    def is_nccl_broadcast(op_name: str):
        return is_nccl_op(op_name) and \
                "broadcast" in op_name.lower()

    # Reduce ops types
    def is_cross_device_reduce_1stage(op_name: str):
        return "cross_device_reduce_1stage" in op_name

    def is_cross_device_reduce_2stage(op_name: str):
        return "cross_device_reduce_2stage" in op_name

    def is_custom_ar_all_reduce(op_name: str):
        return "_C_custom_ar::all_reduce" in op_name

    def is_reduce_kernel(op_name: str):
        return "reduce_kernel" in op_name

    headers = list(trace_df)
    ops = copy.deepcopy(headers)

    attention_ops = list(filter(lambda x: is_attention_block(x), ops))
    ops = list(filter(lambda x: x not in attention_ops, ops))

    quant_ops = list(filter(lambda x: is_quant(x), ops))
    ops = list(filter(lambda x: x not in quant_ops, ops))

    sgmv_shrink_ops = list(filter(lambda x: is_sgmv_shrink(x), ops))
    ops = list(filter(lambda x: x not in sgmv_shrink_ops, ops))
    sgmv_expand_ops = list(filter(lambda x: is_sgmv_expand(x), ops))
    ops = list(filter(lambda x: x not in sgmv_expand_ops, ops))
    bgmv_shrink_ops = list(filter(lambda x: is_bgmv_shrink(x), ops))
    ops = list(filter(lambda x: x not in bgmv_shrink_ops, ops))
    bgmv_expand_ops = list(filter(lambda x: is_bgmv_expand(x), ops))
    ops = list(filter(lambda x: x not in bgmv_expand_ops, ops))

    cutlass_gemm_ops = list(filter(lambda x: is_cutlass_gemm_op(x), ops))
    ops = list(filter(lambda x: x not in cutlass_gemm_ops, ops))

    gemm_ops = list(filter(lambda x: is_gemm_op(x), ops))
    ops = list(filter(lambda x: x not in gemm_ops, ops))

    rms_norm_ops = list(filter(lambda x: is_rms_norm(x), ops))
    ops = list(filter(lambda x: x not in rms_norm_ops, ops))

    vocab_embed_ops = list(filter(lambda x: is_vocab_embedding_op(x), ops))
    ops = list(filter(lambda x: x not in vocab_embed_ops, ops))

    mem_ops = list(filter(lambda x: is_mem_op(x), ops))
    ops = list(filter(lambda x: x not in mem_ops, ops))

    elementwise_ops = list(filter(lambda x: is_elementwise_op(x), ops))
    ops = list(filter(lambda x: x not in elementwise_ops, ops))

    nccl_all_reduce_ops = list(filter(lambda x: is_nccl_all_reduce(x), ops))
    ops = list(filter(lambda x: x not in nccl_all_reduce_ops, ops))

    nccl_gather_ops = list(filter(lambda x: is_nccl_gather(x), ops))
    ops = list(filter(lambda x: x not in nccl_gather_ops, ops))

    nccl_broadcast_ops = list(filter(lambda x: is_nccl_broadcast(x), ops))
    ops = list(filter(lambda x: x not in nccl_broadcast_ops, ops))

    nccl_other_ops = list(filter(lambda x: is_nccl_op(x), ops))
    ops = list(filter(lambda x: x not in nccl_other_ops, ops))

    cross_device_reduce_1stage_ops = list(
        filter(lambda x: is_cross_device_reduce_1stage(x), ops))
    ops = list(filter(lambda x: x not in cross_device_reduce_1stage_ops, ops))

    cross_device_reduce_2stage_ops = list(
        filter(lambda x: is_cross_device_reduce_2stage(x), ops))
    ops = list(filter(lambda x: x not in cross_device_reduce_2stage_ops, ops))

    custom_ar_all_reduce_ops = list(
        filter(lambda x: is_custom_ar_all_reduce(x), ops))
    ops = list(filter(lambda x: x not in custom_ar_all_reduce_ops, ops))

    reduce_kernel_ops = list(filter(lambda x: is_reduce_kernel(x), ops))
    ops = list(filter(lambda x: x not in reduce_kernel_ops, ops))

    if len(attention_ops):
        trace_df['attention'] = trace_df[attention_ops].agg("sum", axis=1)
    if len(quant_ops):
        trace_df['quant_ops'] = trace_df[quant_ops].agg("sum", axis=1)

    if len(sgmv_shrink_ops):
        trace_df['sgmv_shrink_ops'] = trace_df[sgmv_shrink_ops].agg("sum",
                                                                    axis=1)
    if len(sgmv_expand_ops):
        trace_df['sgmv_expand_ops'] = trace_df[sgmv_expand_ops].agg("sum",
                                                                    axis=1)
    if len(bgmv_shrink_ops):
        trace_df['bgmv_shrink_ops'] = trace_df[bgmv_shrink_ops].agg("sum",
                                                                    axis=1)
    if len(bgmv_expand_ops):
        trace_df['bgmv_expand_ops'] = trace_df[bgmv_expand_ops].agg("sum",
                                                                    axis=1)

    if len(cutlass_gemm_ops):
        trace_df['cutlass_gemm_ops'] = trace_df[cutlass_gemm_ops].agg("sum",
                                                                      axis=1)

    if len(gemm_ops):
        trace_df['gemm_ops'] = trace_df[gemm_ops].agg("sum", axis=1)
    if len(rms_norm_ops):
        trace_df['rms_norm_ops'] = trace_df[rms_norm_ops].agg("sum", axis=1)
    if len(vocab_embed_ops):
        trace_df['vocab_embed_ops'] = trace_df[vocab_embed_ops].agg("sum",
                                                                    axis=1)
    if len(mem_ops):
        trace_df['mem_ops'] = trace_df[mem_ops].agg("sum", axis=1)
    if len(elementwise_ops):
        trace_df['elementwise_ops'] = trace_df[elementwise_ops].agg("sum",
                                                                    axis=1)

    if len(nccl_all_reduce_ops):
        trace_df['nccl_all_reduce_ops'] = trace_df[nccl_all_reduce_ops].agg(
            "sum", axis=1)
    if len(nccl_gather_ops):
        trace_df['nccl_gather_ops'] = trace_df[nccl_gather_ops].agg("sum",
                                                                    axis=1)
    if len(nccl_broadcast_ops):
        trace_df['nccl_broadcast_ops'] = trace_df[nccl_broadcast_ops].agg(
            "sum", axis=1)
    if len(nccl_other_ops):
        trace_df['nccl_other_ops'] = trace_df[nccl_other_ops].agg("sum",
                                                                  axis=1)

    if len(cross_device_reduce_1stage_ops):
        trace_df['cross_device_reduce_1stage_ops'] = trace_df[
            cross_device_reduce_1stage_ops].agg("sum", axis=1)
    if len(cross_device_reduce_2stage_ops):
        trace_df['cross_device_reduce_2stage_ops'] = trace_df[
            cross_device_reduce_2stage_ops].agg("sum", axis=1)
    if len(custom_ar_all_reduce_ops):
        trace_df['custom_ar_all_reduce_ops'] = trace_df[
            custom_ar_all_reduce_ops].agg("sum", axis=1)
    if len(reduce_kernel_ops):
        trace_df['reduce_kernel_ops'] = trace_df[reduce_kernel_ops].agg("sum",
                                                                        axis=1)

    trace_df.drop(attention_ops + quant_ops + sgmv_shrink_ops +
                  sgmv_expand_ops + bgmv_shrink_ops + bgmv_expand_ops +
                  cutlass_gemm_ops + gemm_ops + rms_norm_ops +
                  vocab_embed_ops + mem_ops + elementwise_ops +
                  nccl_all_reduce_ops + nccl_gather_ops + nccl_broadcast_ops +
                  nccl_other_ops + cross_device_reduce_1stage_ops +
                  cross_device_reduce_2stage_ops + custom_ar_all_reduce_ops +
                  reduce_kernel_ops,
                  axis=1,
                  inplace=True)
    return trace_df


## Data plotting utils ####


def plot_trace_df(traces_df: pd.DataFrame,
                  plot_metric: str,
                  plot_title: str,
                  output: Optional[Path] = None):

    def get_phase_description(traces_df: pd.DataFrame, phase: str) -> str:
        phase_df = traces_df.query(f'phase == "{phase}"')
        descs = phase_df['phase_desc'].to_list()
        assert all([desc == descs[0] for desc in descs])
        return descs[0]

    phases = traces_df['phase'].unique()
    phase_descs = [get_phase_description(traces_df, p) for p in phases]
    traces_df = traces_df.pivot_table(index="phase",
                                      columns="name",
                                      values=plot_metric,
                                      aggfunc="sum")

    traces_df = group_trace_by_operations(traces_df)

    # Make the figure
    fig_size_x = max(5, len(phases))
    fig, ax = plt.subplots(1, figsize=(fig_size_x, 8), sharex=True)

    # Draw the stacked bars
    ops = list(traces_df)
    bottom = [0] * len(phases)
    for op in ops:
        values = [traces_df[op][phase] for phase in phases]
        values = list(map(lambda x: 0.0 if math.isnan(x) else x, values))
        ax.bar(phase_descs, values, label=op, bottom=bottom)
        bottom = [bottom[j] + values[j] for j in range(len(phases))]

    # Write the values as text on the bars
    for bar in ax.patches:
        if bar.get_height() != 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 2 + bar.get_y(),
                    f"{round(bar.get_height(), 2)}",
                    ha='center',
                    color='w',
                    weight='bold',
                    size=5)

    # Setup legend
    handles, labels = plt.gca().get_legend_handles_labels()
    legend = fig.legend(handles,
                        labels,
                        loc='center left',
                        bbox_to_anchor=(1, 1))
    shorten_plot_legend_strings(legend, 50)

    # Setup labels and title
    plt.setp(ax.get_xticklabels(), rotation=90)
    ax.set_ylabel(plot_metric)
    plt.suptitle(plot_title)

    plt.savefig(output, bbox_inches='tight')
    print("Created: ", output)


def main(
        json_trace: Path,
        output_directory: Path,
        depth: int,  # Fetch/Plot operations at this depth of the Json tree
        plot_metric: str,
        make_names_unique: bool,
        top_k: int,
        json_nodes_to_fold: list[str]):

    def prepare_data(profile_json: dict, step_keys: list[str]) -> pd.DataFrame:

        def get_entries_and_traces(key: str):
            entries_and_traces: list[tuple[Any, Any]] = []
            for root in profile_json[key]["summary_stats"]:
                # Fold nodes in the traces as per user request. i.e. simply
                # make the requested nodes leaf-nodes.
                root = fold_nodes(root, json_nodes_to_fold)
                get_entries_at_depth(depth, entries_and_traces, root)
            return entries_and_traces

        def keep_only_top_entries(df: pd.DataFrame,
                                  metric: str,
                                  top_k: int = 9) -> pd.DataFrame:
            df.loc[df.nsmallest(len(df) - top_k + 1, metric).index,
                   ["name"]] = "others"
            return df

        def get_phase_description(key: str) -> str:
            num_running_seqs = profile_json[key]['metadata'][
                'num_running_seqs']
            if num_running_seqs is not None:
                return f"{key}-seqs-{num_running_seqs}"
            else:
                return key

        # Get data for each key
        traces = list(map(lambda x: get_entries_and_traces(x), step_keys))

        # Attempt some cleanup
        if make_names_unique:
            for trace in traces:
                attempt_to_make_names_unique(trace)

        # To pandas dataframe
        trace_dfs = list(
            map(lambda t: pd.DataFrame([entry for entry, _ in t]).fillna(0),
                traces))

        # Respect top_k
        if top_k:
            trace_dfs = list(
                map(
                    lambda trace_df: keep_only_top_entries(
                        trace_df, "cuda_time_us", top_k), trace_dfs))

        # Fill in information about the step-keys
        for trace_df, step_key in zip(trace_dfs, step_keys):
            trace_df['phase'] = step_key
            trace_df['phase_desc'] = get_phase_description(step_key)

        # Combine all data frames so they can be put in a single plot
        traces_df = pd.concat(trace_dfs)

        # Add a derived metric `cuda_time_ms`
        traces_df["cuda_time_ms"] = traces_df["cuda_time_us"] / 1000
        traces_df = traces_df.fillna(0)

        return traces_df

    def make_plot_title_suffix(profile_json: dict) -> str:
        context = profile_json["context"]
        sparsity = context.get('sparsity', None)
        run_type = \
            f'Run {context["num_steps"]} steps' if context['num_steps'] else \
                (f'Complete {context["complete_num_requests_per_step"]} per '
                 f'step; Run till completion')
        return (f"{context['engine_args']['model']}\n"
                f"Batch={context['batch_size']}, "
                f"PromptLen={context['prompt_len']}, "
                f"NumGpus={context['engine_args']['tensor_parallel_size']}"
                f"{', Sparsity ' + sparsity if sparsity else ''}\n"
                f"Run Type: {run_type}")

    profile_json = None
    with open(json_trace) as f:
        profile_json = json.load(f)
    assert profile_json is not None

    # Get all `llm.generate.step()` profile
    step_traces = list(profile_json.keys())
    assert (step_traces[0] == 'context')
    step_traces = step_traces[1:]  # have only prefill and decodes
    prefills = list(filter(lambda x: "prefill" in x, step_traces))
    all_decodes = list(filter(lambda x: "decode" in x, step_traces))
    assert len(prefills) + len(all_decodes) == len(step_traces)
    assert len(prefills) == 1

    decodes = all_decodes[::args.step_plot_interval]
    if decodes[-1] != all_decodes[-1]:
        # Always have the last decode
        decodes.append(all_decodes[-1])

    prefill_traces = prepare_data(profile_json, prefills)
    decode_traces = prepare_data(profile_json, decodes)

    plot_title_suffix = make_plot_title_suffix(profile_json)

    plot_trace_df(prefill_traces, plot_metric, "prefill " + plot_title_suffix,
                  output_directory / Path("prefill.png"))
    plot_trace_df(decode_traces, plot_metric, "decodes " + plot_title_suffix,
                  output_directory / Path("decode_steps.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--json-trace",
                        type=str,
                        required=True,
                        help="json trace file output by \
                              examples/offline_inference/profiling.py")
    parser.add_argument("--output-directory",
                        type=str,
                        required=False,
                        help="Directory to output plots")
    parser.add_argument("--level",
                        type=str,
                        default="module",
                        choices=["module", "kernel"])
    parser.add_argument("--top-k",
                        type=int,
                        default=12,
                        help="Only graph the top `top_k` entries by time.")
    parser.add_argument("--fold-json-node",
                        nargs='+',
                        default=['Sampler', 'LogitsProcessor'],
                        help='Do not plot the children of these nodes. Let, \
                              the node represent the aggregate of all its \
                              children')
    parser.add_argument("--plot-metric",
                        type=str,
                        default="cuda_time_ms",
                        help='Metric to plot. some options are cuda_time_ms, \
                                pct_cuda_time')
    parser.add_argument(
        "--step-plot-interval",
        type=int,
        default=4,
        help="For every `step_plot_interval` steps, plot 1 step")

    args = parser.parse_args()

    # Prepare/Extract relevant args
    make_names_unique = False
    if args.level == "module":
        depth = -2
        make_names_unique = True
    elif args.level == "kernel":
        depth = -1
    else:
        raise Exception(f"Unexpected level value ({args.level})")

    output_directory = args.output_directory if args.output_directory else Path(
        args.json_trace).parent

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    main(Path(args.json_trace), output_directory, depth, args.plot_metric,
         make_names_unique, args.top_k, args.fold_json_node)
