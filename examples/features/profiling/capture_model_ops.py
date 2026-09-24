# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extract the operators a model executes, without weights or an accelerator.

Builds the model on `torch.device("meta")` from its HF config and drives vLLM's
real forward path over it, so a model far larger than the available hardware --
or one whose weights are not to hand -- still yields its ordered operator
sequence, custom kernels included.

    python examples/features/profiling/capture_model_ops.py \
        --model Qwen/Qwen3-30B-A3B

Pass `--verify-against <device>` to also run on hardware and assert the two
Chakra execution traces agree, as a regression check:

    python examples/features/profiling/capture_model_ops.py \
        --model Qwen/Qwen2.5-0.5B-Instruct --verify-against cuda
"""

import argparse
import sys
from pathlib import Path

from vllm.profiler.op_capture import (
    BatchSpec,
    capture_model_ops,
    compare_devices,
    format_diff,
    format_report,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser


def create_parser() -> FlexibleArgumentParser:
    parser = FlexibleArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--shapes", action="store_true", help="Show operand shapes.")
    parser.add_argument(
        "--trace", type=Path, default=None, help="Write a Chakra execution trace here."
    )
    parser.add_argument(
        "--verify-against",
        default=None,
        metavar="DEVICE",
        help="Also capture on this device and compare execution traces.",
    )

    batch_group = parser.add_argument_group("Batch parameters")
    batch_group.add_argument("--num-reqs", type=int, default=1)
    batch_group.add_argument("--num-tokens", type=int, default=8)
    batch_group.add_argument("--num-computed-tokens", type=int, default=0)

    return parser


def main(args: argparse.Namespace) -> int:
    batch = BatchSpec(
        num_reqs=args.num_reqs,
        num_tokens=args.num_tokens,
        num_computed_tokens=args.num_computed_tokens,
    )
    if args.verify_against is not None:
        diff = compare_devices(args.model, args.verify_against, batch=batch)
        print(format_diff(diff))
        return 0 if diff.equal else 1

    capture = capture_model_ops(args.model, batch=batch, trace_path=args.trace)
    print(format_report(capture, show_shapes=args.shapes))
    return 0


if __name__ == "__main__":
    sys.exit(main(create_parser().parse_args()))
