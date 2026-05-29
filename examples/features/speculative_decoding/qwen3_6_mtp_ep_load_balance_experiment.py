# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
from pathlib import Path

from mtp_ep_experiment_analysis import analyze_experiment
from mtp_ep_experiment_runtime import (
    collect_experiment,
    collect_one_condition,
    default_output_dir,
)
from mtp_ep_load_balance_utils import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_DATASET,
    DEFAULT_DATASET_CONFIG,
    DEFAULT_DATASET_SPLIT,
    DEFAULT_DRAFT_LENGTHS,
    DEFAULT_LAYERS,
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_NUM_EXPERTS,
)


def add_common_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--dataset-split", default=DEFAULT_DATASET_SPLIT)
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_BATCH_SIZES),
    )
    parser.add_argument(
        "--draft-lengths",
        nargs="+",
        type=int,
        default=list(DEFAULT_DRAFT_LENGTHS),
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=list(DEFAULT_LAYERS),
    )
    parser.add_argument("--num-experts", type=int, default=DEFAULT_NUM_EXPERTS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Runtime output directory. Defaults to results/qwen3_6_mtp_ep_<UTC timestamp>.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect and analyze the Qwen3.6 MTP-EP speedup, step-time, and "
            "expert-load experiment."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser(
        "collect",
        help="Run all experiment conditions once and save raw data.",
    )
    add_common_runtime_args(collect_parser)
    collect_parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force eager mode during collection to avoid compile/cudagraph cold start effects.",
    )
    collect_parser.add_argument(
        "--warmup-rounds",
        type=int,
        default=1,
        help="Number of unmeasured warmup generate rounds per condition.",
    )
    collect_one_parser = subparsers.add_parser(
        "collect-one",
        help=argparse.SUPPRESS,
    )
    collect_one_parser.add_argument("--model", default=DEFAULT_MODEL)
    collect_one_parser.add_argument("--dataset", default=DEFAULT_DATASET)
    collect_one_parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    collect_one_parser.add_argument("--dataset-split", default=DEFAULT_DATASET_SPLIT)
    collect_one_parser.add_argument("--batch-size", type=int, required=True)
    collect_one_parser.add_argument("--draft-length", type=int, required=True)
    collect_one_parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS
    )
    collect_one_parser.add_argument(
        "--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN
    )
    collect_one_parser.add_argument("--tensor-parallel-size", type=int, default=2)
    collect_one_parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.85
    )
    collect_one_parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=list(DEFAULT_LAYERS),
    )
    collect_one_parser.add_argument(
        "--num-experts", type=int, default=DEFAULT_NUM_EXPERTS
    )
    collect_one_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    collect_one_parser.add_argument(
        "--prompt-cache-path",
        type=Path,
        default=None,
    )
    collect_one_parser.add_argument(
        "--warmup-rounds",
        type=int,
        default=1,
    )
    collect_one_parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Generate CSV tables, plots, and 实验报告.md from raw data.",
    )
    analyze_parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Experiment root directory containing collect_manifest.json and raw/*.npz.",
    )
    analyze_parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation.",
    )
    analyze_parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip 实验报告.md generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "collect":
        args.batch_sizes = tuple(args.batch_sizes)
        args.draft_lengths = tuple(args.draft_lengths)
        args.layers = tuple(args.layers)
        if 0 not in args.draft_lengths:
            raise ValueError("draft_length=0 must be present as the baseline.")
        output_dir = args.output_dir or default_output_dir()
        collect_experiment(args, output_dir, Path(__file__).resolve())
        return

    if args.command == "collect-one":
        args.layers = tuple(args.layers)
        collect_one_condition(args, args.output_dir)
        return

    analyze_experiment(
        args.input_dir,
        skip_plots=args.skip_plots,
        skip_report=args.skip_report,
    )


if __name__ == "__main__":
    main()
