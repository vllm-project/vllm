import argparse
import json
from typing import Dict

from vllm.profiler.layerwise_profile import ModelStatsEntry, SummaryStatsEntry
from vllm.profiler.utils import TablePrinter, indent_string


def flatten_entries(entry_cls, profile_dict: Dict):
    entries_and_depth = []

    def get_entries(node, curr_depth=0):
        entries_and_depth.append((entry_cls(**node["entry"]), curr_depth))

        for child in node["children"]:
            get_entries(
                child,
                curr_depth=curr_depth + 1,
            )

    for root in profile_dict:
        get_entries(root)

    return entries_and_depth


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--json-trace",
                        type=str,
                        required=True,
                        help="json trace file output by "
                        "examples/offline_inference/profiling.py")
    parser.add_argument("--phase",
                        type=str,
                        required=True,
                        help="The phase to print the table for. This is either"
                        "prefill or decode_n, where n is the decode step "
                        "number")
    parser.add_argument("--table",
                        type=str,
                        choices=["summary", "model"],
                        default="summary",
                        help="Which table to print, the summary table or the "
                        "layerwise model table")

    args = parser.parse_args()

    with open(args.json_trace) as f:
        profile_data = json.load(f)

    assert args.phase in profile_data, \
       (f"Cannot find phase {args.phase} in profile data. Choose one among"
        f'{[x for x in profile_data.keys() if "prefill" in x or "decode" in x]}') #noqa

    if args.table == "summary":
        entries_and_depths = flatten_entries(
            SummaryStatsEntry, profile_data[args.phase]["summary_stats"])
        column_widths = dict(name=80,
                             cuda_time_us=12,
                             pct_cuda_time=12,
                             invocations=15)
    elif args.table == "model":
        entries_and_depths = flatten_entries(
            ModelStatsEntry, profile_data[args.phase]["model_stats"])
        column_widths = dict(name=60,
                             cpu_time_us=12,
                             cuda_time_us=12,
                             pct_cuda_time=12,
                             trace=60)

    # indent entry names based on the depth
    entries = []
    for entry, depth in entries_and_depths:
        entry.name = indent_string(
            entry.name,
            indent=depth,
            indent_style=lambda indent: "|" + "-" * indent + " ")
        entries.append(entry)

    TablePrinter(type(entries[0]), column_widths).print_table(entries)
