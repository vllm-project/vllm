# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import time
from collections import defaultdict

from tabulate import tabulate
from tqdm import tqdm

from vllm.logprobs import (
    FlattenLogprobs,
    Logprob,
    LogprobsOnePosition,
    PromptLogprobs,
    SampleLogprobs,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser

NS_PER_S = 1.0e9

CNTS_PER_GENERATION: dict[int, int] = defaultdict(int)
TOTAL_NS_PER_GENERATION: dict[int, int] = defaultdict(int)
LAST_START_NS: int


def reset_gc() -> None:
    # TODO(Jialin): Switch to use freeze_gc_heap after PR #27896 landed
    gc.unfreeze()
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)
    gc.freeze()
    CNTS_PER_GENERATION.clear()
    TOTAL_NS_PER_GENERATION.clear()


def gc_handle(phase: str, info: dict[str, int]) -> None:
    """
    Handles a GC event (e.g. GC start or GC finish)
    """
    global LAST_START_NS
    generation = info.get("generation")
    if generation is None:
        return
    if phase == "start":
        CNTS_PER_GENERATION[generation] += 1
        LAST_START_NS = time.perf_counter_ns()
    elif phase == "stop":
        TOTAL_NS_PER_GENERATION[generation] += time.perf_counter_ns() - LAST_START_NS


def create_logprobs(top_logprob: int) -> dict[int, Logprob]:
    return {
        i: Logprob(logprob=0.1 * i, rank=i, decoded_token=str(i))
        for i in range(top_logprob)
    }


def process(
    prompt_logprobs: list[PromptLogprobs],
    sample_logprobs: list[SampleLogprobs],
    batch_size: int,
    input_length: int,
    output_length: int,
    prompt_logprob: int,
    sample_logprob: int,
) -> None:
    for i in range(batch_size):
        this_prompt_logprobs = prompt_logprobs[i]
        this_sample_logprobs = sample_logprobs[i]
        for _ in range(input_length):
            this_prompt_logprobs.append(create_logprobs(prompt_logprob))
        for _ in range(output_length):
            this_sample_logprobs.append(create_logprobs(sample_logprob))


def gc_metrics(generation: int) -> list[int | float]:
    return [
        CNTS_PER_GENERATION[generation],
        TOTAL_NS_PER_GENERATION[generation] / NS_PER_S,
    ]


def main(args) -> None:
    gc.callbacks.append(gc_handle)
    rows = []
    with tqdm(total=len(args.batch_sizes) * 2, desc="Processing Data") as pbar:
        for batch_size in args.batch_sizes:
            # Logprobs using list[dict[int, Logprob]]
            reset_gc()
            ns = time.perf_counter_ns()
            for _ in range(args.num_batches):
                prompt_logprobs: list[list[LogprobsOnePosition | None]] = [
                    [None] for _ in range(batch_size)
                ]
                sample_logprobs: list[list[LogprobsOnePosition]] = [
                    [] for _ in range(batch_size)
                ]
                process(
                    prompt_logprobs=prompt_logprobs,
                    sample_logprobs=sample_logprobs,
                    batch_size=batch_size,
                    input_length=args.input_length,
                    output_length=args.output_length,
                    prompt_logprob=args.prompt_logprob,
                    sample_logprob=args.sample_logprob,
                )
            ns = time.perf_counter_ns() - ns
            rows.append(
                [
                    batch_size,
                    "list[dict]",
                    *gc_metrics(0),
                    *gc_metrics(1),
                    *gc_metrics(2),
                    ns / NS_PER_S,
                ]
            )
            pbar.update(1)

            # Logprobs using FlattenLogprobs
            reset_gc()
            ns = time.perf_counter_ns()
            for _ in range(args.num_batches):
                prompt_logprobs: list[FlattenLogprobs] = [
                    FlattenLogprobs() for _ in range(batch_size)
                ]
                for this_prompt_logprobs in prompt_logprobs:
                    this_prompt_logprobs.append(None)
                sample_logprobs: list[FlattenLogprobs] = [
                    FlattenLogprobs() for _ in range(batch_size)
                ]
                process(
                    prompt_logprobs=prompt_logprobs,
                    sample_logprobs=sample_logprobs,
                    batch_size=batch_size,
                    input_length=args.input_length,
                    output_length=args.output_length,
                    prompt_logprob=args.prompt_logprob,
                    sample_logprob=args.sample_logprob,
                )
            ns = time.perf_counter_ns() - ns
            rows.append(
                [
                    batch_size,
                    "FlattenLogprobs",
                    *gc_metrics(0),
                    *gc_metrics(1),
                    *gc_metrics(2),
                    ns / NS_PER_S,
                ]
            )
            pbar.update(1)

    print(
        tabulate(
            rows,
            headers=[
                "Batch\nsize",
                "Logprob Type",
                "GC0\ncnt",
                "GC0\ntotal (s)",
                "GC1\ncnt",
                "GC1\ntotal (s)",
                "GC2\ncnt",
                "GC2\ntotal (s)",
                "Runtime\ntotal (s)",
            ],
            tablefmt="grid",
            floatfmt=".3f",
        )
    )


def invoke_main() -> None:
    parser = FlexibleArgumentParser(
        description="Benchmark the performance of logprobs processing."
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="*",
        default=[10, 100, 1000],  # [10, 100, 1000, 2000],
        help="Number of blocks to allocate",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=10,
        help="Number of full batches to process",
    )
    parser.add_argument(
        "--input-length",
        type=int,
        default=1000,
        help="Number of input tokens per request",
    )
    parser.add_argument(
        "--output-length",
        type=int,
        default=2000,
        help="Number of output tokens per request",
    )
    parser.add_argument(
        "--prompt-logprob",
        type=int,
        default=2,
        help="Number of logprobs to compute per prompt token",
    )
    parser.add_argument(
        "--sample-logprob",
        type=int,
        default=2,
        help="Number of logprobs to compute per sample token",
    )
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
