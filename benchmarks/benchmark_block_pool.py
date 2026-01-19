# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc

from benchmark_utils import TimeCollector
from tabulate import tabulate

from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.core.block_pool import BlockPool


def main(args):
    rows = []
    for allocate_block in args.allocate_blocks:
        # Enforce a GC collect ahead to minimize the impact among runs
        gc.collect()
        block_pool = BlockPool(num_gpu_blocks=args.num_gpu_blocks, enable_caching=True)

        get_blocks_times = TimeCollector(TimeCollector.US)
        free_blocks_times = TimeCollector(TimeCollector.US)
        for _ in range(args.num_iteration):
            with get_blocks_times:
                blocks = block_pool.get_new_blocks(allocate_block)
            with free_blocks_times:
                block_pool.free_blocks(blocks)

        rows.append(
            [get_blocks_times.cnt, args.num_gpu_blocks, allocate_block]
            + get_blocks_times.dump_avg_max()
            + free_blocks_times.dump_avg_max()
        )

    print(
        tabulate(
            rows,
            headers=[
                "Iterations",
                "Total\nBlocks",
                "Allocated\nBlocks",
                "Get Blocks\nAvg (us)",
                "Get Blocks\nMax (us)",
                "Free Blocks\nAvg (us)",
                "Free Blocks\nMax (us)",
            ],
            tablefmt="grid",
            floatfmt=".3f",
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
        help="Number of iterations to run to stabilize final data readings",
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
