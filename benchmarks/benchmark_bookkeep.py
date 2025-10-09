# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
from random import randint, shuffle

import numpy as np
from benchmark_utils import TimeCollector, throughput_change
from tabulate import tabulate
from tqdm import trange

from vllm.utils import FlexibleArgumentParser

"""
Example Usage:

buck run @mode/opt scripts/jialino/llm/python:np_update
"""


def update_one_by_one(
    num_tokens_no_spec_np: np.ndarray[np.int32, np.dtype[np.int32]],
    num_tokens_np: np.ndarray[np.int32, np.dtype[np.int32]],
    update_tags: list[bool],
    update_values: list[int],
) -> None:
    for i in range(len(update_values)):
        if update_tags[i]:
            start_idx = num_tokens_no_spec_np[i]
            end_idx = start_idx + update_values[i]
            num_tokens_no_spec_np[i] = end_idx
            num_tokens_np[i] = end_idx


def update_by_batch(
    num_tokens_no_spec_np: np.ndarray[np.int32, np.dtype[np.int32]],
    num_tokens_np: np.ndarray[np.int32, np.dtype[np.int32]],
    update_tags: list[bool],
    update_values: list[int],
) -> None:
    num_tokens_no_spec = num_tokens_no_spec_np.tolist()
    num_tokens_indices_to_update: list[int] = []
    num_tokens_values_to_update: list[int] = []
    for i in range(len(update_values)):
        if update_tags[i]:
            start_idx = num_tokens_no_spec[i]
            end_idx = start_idx + update_values[i]
            num_tokens_indices_to_update.append(i)
            num_tokens_values_to_update.append(end_idx)
    if num_tokens_indices_to_update:
        num_tokens_no_spec_np[num_tokens_indices_to_update] = (
            num_tokens_values_to_update
        )
        num_tokens_np[num_tokens_indices_to_update] = num_tokens_values_to_update


def main(args) -> None:
    testsets = []
    for num_element in args.num_elements:
        update_element = num_element
        while update_element > 0:
            testsets.append((num_element, update_element))
            update_element = update_element // 10
        testsets.append((num_element, 0))
    testsets.sort()

    data_rows = []
    TIME_SCALE = TimeCollector.US

    for i in trange(len(testsets), desc="Testsets"):
        num_element, update_element = testsets[i]
        num_tokens_no_spec_np = np.empty((num_element,), dtype=np.int32)
        num_tokens_np = np.empty((num_element,), dtype=np.int32)
        one_by_one_times = TimeCollector(TIME_SCALE)
        batch_times = TimeCollector(TIME_SCALE)
        # Only update update_element per iterations
        update_tags = [True] * update_element + [False] * (num_element - update_element)
        gc.collect()
        for _ in trange(args.num_iteration, desc="Iterations per testset"):
            shuffle(update_tags)
            update_values = [randint(0, 100) for _ in range(num_element)]
            with one_by_one_times:
                update_one_by_one(
                    num_tokens_no_spec_np, num_tokens_np, update_tags, update_values
                )
            with batch_times:
                update_by_batch(
                    num_tokens_no_spec_np, num_tokens_np, update_tags, update_values
                )

        one_by_one_avg = one_by_one_times.avg_v()
        batch_metric_avg = batch_times.avg_v()
        data_rows.append(
            [
                num_element,
                update_element,
                one_by_one_times.avg(),
                batch_times.avg(),
                throughput_change(batch_metric_avg, one_by_one_avg),
            ]
        )

    print(
        tabulate(
            data_rows,
            headers=[
                "Total\nElements",
                "Update\nElements",
                "One by One\nAvg (us)",
                "Batch\nAvg (us)",
                "Throughput\nChange",
            ],
            tablefmt="pipe",
            floatfmt=".3f",
            colalign=["right"] * len(data_rows[0]),
        )
    )


def invoke_main() -> None:
    parser = FlexibleArgumentParser(
        description="Benchmark the performance of Bookkeeping "
        "(i.e. GPUModelRunner._bookkeeping_sync)"
    )
    parser.add_argument(
        "--num-iteration",
        type=int,
        default=1000,
        help="Number of iterations to run to stabilize final data readings",
    )
    parser.add_argument(
        "--num_elements",
        type=int,
        nargs="+",
        default=[10, 100, 1000, 10000],
    )
    main(parser.parse_args())


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
