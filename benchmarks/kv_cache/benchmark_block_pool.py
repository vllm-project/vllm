# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import time
from typing import Optional

from tabulate import tabulate

from vllm.utils import FlexibleArgumentParser
from vllm.v1.core.block_pool import BlockPool


class Metric:
    def __init__(self) -> None:
        self.cnt: int = 0
        self.sum_v: int = 0
        self.max_v: Optional[int] = None

    def update(self, v: int) -> None:
        self.cnt += 1
        self.sum_v += v
        if self.max_v is None:
            self.max_v = v
        else:
            self.max_v = max(self.max_v, v)

    def avg_v(self) -> float:
        return self.sum_v * 1.0 / self.cnt


def main(args):
    rows = []
    for allocate_block in args.allocate_blocks:
        # Enforce a GC collect ahead to minimize the impact among runs
        gc.collect()
        block_pool = BlockPool(num_gpu_blocks=args.num_gpu_blocks, enable_caching=True)

        get_blocks_metric: Metric = Metric()
        free_blocks_metric: Metric = Metric()
        for _ in range(args.num_iteration):
            t1 = time.monotonic_ns()
            blocks = block_pool.get_new_blocks(allocate_block)
            t2 = time.monotonic_ns()
            block_pool.free_blocks(blocks)
            t3 = time.monotonic_ns()
            get_blocks_metric.update(t2 - t1)
            free_blocks_metric.update(t3 - t2)

        if get_blocks_metric.max_v is not None and free_blocks_metric.max_v is not None:
            rows.append(
                [
                    get_blocks_metric.cnt,
                    args.num_gpu_blocks,
                    allocate_block,
                    get_blocks_metric.avg_v() / 1000000,
                    get_blocks_metric.max_v / 1000000.0,
                    free_blocks_metric.avg_v() / 1000000,
                    free_blocks_metric.max_v / 1000000.0,
                ]
            )
        else:
            print(
                "No valid metrics found."
                f" {get_blocks_metric.max_v=} {free_blocks_metric.max_v=}"
            )

    print(
        tabulate(
            rows,
            headers=[
                "Iterations",
                "Total\nBlocks",
                "Allocated\nBlocks",
                "Get Blocks\nAvg (ms)",
                "Get Blocks\nMax (ms)",
                "Free Blocks\nAvg (ms)",
                "Free Blocks\nMax (ms)",
            ],
            tablefmt="grid",
            floatfmt=".6f",
        )
    )


def invoke_main() -> None:
    parser = FlexibleArgumentParser(
        description="Benchmark the performance of BlockPool for KV Cache."
    )
    parser.add_argument("--num-gpu-blocks", type=int, default=100000)
    parser.add_argument(
        "--num-iteration",
        type=int,
        default=1000,
        help="Number of iterations to run to stablize final data readings",
    )
    parser.add_argument(
        "--allocate-blocks",
        type=int,
        nargs="*",
        default=[10, 50, 100, 500, 1000],
        help="Number of blocks to allocate",
    )
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
