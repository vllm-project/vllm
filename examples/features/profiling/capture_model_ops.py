# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extract the operators a model executes, without weights or an accelerator.

Builds the model on `torch.device("meta")` from its HF config and drives vLLM's
real forward path over it, so a model far larger than the available hardware --
or one whose weights are not to hand -- still yields its ordered operator
sequence, custom kernels included. Every engine argument is accepted.

    python examples/features/profiling/capture_model_ops.py \
        --model Qwen/Qwen3-30B-A3B --max-model-len 8192

Pass `--batch REQS,TOKENS[,COMPUTED[,MM_ITEMS]]` more than once to capture
several batches -- a prefill and a long-context decode, say -- on one built model
and see which operators only some of them reach. `MM_ITEMS` runs a multimodal
model's encoder on that many dummy items, as the model runner profiles it:

    python examples/features/profiling/capture_model_ops.py \
        --model Qwen/Qwen3-30B-A3B --max-model-len 8192 \
        --batch 1,512 --batch 8,8,4096

    python examples/features/profiling/capture_model_ops.py \
        --model Qwen/Qwen2.5-VL-3B-Instruct --batch 1,64 --batch 1,2048,0,1

Pass `--output-dir` to also write each capture as files (`ops.txt`,
`ops.sequence.txt`, `capture.json`, `report.txt`), and
`--hf-overrides '{"quantization_config": null}'` to capture a quantized
checkpoint's unquantized path. With `--tensor-parallel-size N` every rank is
captured in its own process, collectives included.

Pass `--verify-against <device>` to also run on hardware and assert the two
Chakra execution traces agree, as a regression check:

    python examples/features/profiling/capture_model_ops.py \
        --model Qwen/Qwen2.5-0.5B-Instruct --verify-against cuda

Pass `--keep-going` to list every op standing between a model and this platform
in one run, rather than stopping at the first.
"""

import argparse
import sys
from pathlib import Path

from vllm.engine.arg_utils import EngineArgs
from vllm.profiler.op_capture import (
    BatchSpec,
    capture_batches,
    capture_model_ops,
    capture_ranks,
    compare_devices,
    format_batches,
    format_diff,
    format_report,
    write_capture_files,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser


def parse_batch(value: str) -> BatchSpec:
    fields = [int(field) for field in value.split(",")]
    if not 2 <= len(fields) <= 4:
        raise argparse.ArgumentTypeError("expected REQS,TOKENS[,COMPUTED[,MM_ITEMS]]")
    return BatchSpec(*fields)


def create_parser() -> FlexibleArgumentParser:
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    capture_group = parser.add_argument_group("Capture parameters")
    capture_group.add_argument(
        "--batch",
        type=parse_batch,
        action="append",
        metavar="REQS,TOKENS[,COMPUTED[,MM_ITEMS]]",
        help="Batch to capture; repeat for several. Default: one 8-token prefill.",
    )
    capture_group.add_argument(
        "--shapes", action="store_true", help="Show operand shapes."
    )
    capture_group.add_argument(
        "--trace", type=Path, default=None, help="Write a Chakra execution trace here."
    )
    capture_group.add_argument(
        "--output-dir", type=Path, default=None, help="Also write capture files here."
    )
    capture_group.add_argument(
        "--keep-going",
        action="store_true",
        help="Record past ops that cannot run here and report them all.",
    )
    capture_group.add_argument(
        "--verify-against",
        default=None,
        metavar="DEVICE",
        help="Also capture on this device and compare execution traces.",
    )
    return parser


def main(args: argparse.Namespace) -> int:
    engine_args = EngineArgs.from_cli_args(args)
    batches = args.batch or [BatchSpec()]
    if args.verify_against is not None:
        equal = True
        for batch in batches:
            diff = compare_devices(
                args.model, args.verify_against, batch=batch, engine_args=engine_args
            )
            print(format_diff(diff))
            equal &= diff.equal
        return 0 if equal else 1

    if engine_args.tensor_parallel_size > 1:
        ranks = capture_ranks(
            args.model, batches, engine_args=engine_args, keep_going=args.keep_going
        )
    elif len(batches) == 1:
        ranks = [
            [
                capture_model_ops(
                    args.model,
                    batch=batches[0],
                    trace_path=args.trace,
                    engine_args=engine_args,
                    keep_going=args.keep_going,
                )
            ]
        ]
    else:
        ranks = [
            capture_batches(
                args.model, batches, engine_args=engine_args, keep_going=args.keep_going
            )
        ]
    for rank, captures in enumerate(ranks):
        output_dir = args.output_dir
        if len(ranks) > 1:
            print(f"===== Rank {rank} of {len(ranks)} =====", end="\n\n")
            if output_dir is not None:
                output_dir /= f"rank{rank}"
        for index, capture in enumerate(captures):
            print(format_report(capture, show_shapes=args.shapes), end="\n\n")
            if output_dir is not None:
                subdir = output_dir if len(captures) == 1 else output_dir / str(index)
                print(f"Wrote {write_capture_files(capture, subdir)}", end="\n\n")
        if len(captures) > 1:
            print(format_batches(captures), end="\n\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(create_parser().parse_args()))
