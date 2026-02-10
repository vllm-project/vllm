# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import statistics
import time

import torch

from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.sample.ops.topk_topp_sampler import _apply_exponential_by_generators


@torch.inference_mode()
def main(
    device: str,
    batch_size: int,
    vocab_size: int,
    num_generators: int,
    layout: str,
    seed: int,
    num_warmup_iters: int,
    num_iters: int,
) -> None:
    torch.set_default_device(device)
    set_random_seed(seed)

    probs = torch.empty((batch_size, vocab_size), dtype=torch.float32, device=device)
    generators = [torch.Generator(device=device) for _ in range(num_generators)]
    for idx, generator in enumerate(generators):
        generator.manual_seed(seed + idx)

    generators_by_row: dict[int, torch.Generator] = {}
    if layout == "contiguous":
        block = (batch_size + num_generators - 1) // num_generators
        for gen_idx, generator in enumerate(generators):
            start = gen_idx * block
            end = min(start + block, batch_size)
            for row in range(start, end):
                generators_by_row[row] = generator
    elif layout == "non-contiguous":
        for row in range(batch_size):
            generators_by_row[row] = generators[row % num_generators]
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    def run_naive() -> None:
        q = torch.empty_like(probs)
        for row, generator in generators_by_row.items():
            q[row].exponential_(generator=generator)

    def run_grouped() -> None:
        q = torch.empty_like(probs)
        _apply_exponential_by_generators(q, generators_by_row)

    def run_benchmark(fn, num_iters: int) -> tuple[float, float]:
        times: list[float] = []
        if device == "cuda":
            torch.accelerator.synchronize()
        for _ in range(num_iters):
            start_time = time.perf_counter()
            fn()
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        avg = statistics.mean(times)
        std = statistics.stdev(times) if len(times) > 1 else 0.0
        return avg, std

    print("Warming up for run_naive...")
    run_benchmark(run_naive, num_warmup_iters)
    naive_cost_avg, naive_cost_std = run_benchmark(run_naive, num_iters)

    print("Warming up for run_grouped...")
    run_benchmark(run_grouped, num_warmup_iters)
    grouped_cost_avg, grouped_cost_std = run_benchmark(run_grouped, num_iters)

    print(
        f"{device} {layout} naive cost avg:   {naive_cost_avg * 1e6:.3f} us \t "
        f"{device} {layout} naive cost std:   {naive_cost_std * 1e6:.3f} us \t\n"
        f"{device} {layout} grouped cost avg: {grouped_cost_avg * 1e6:.3f} us \t "
        f"{device} {layout} grouped cost std: {grouped_cost_std * 1e6:.3f} us \t "
    )


def run_main():
    global args
    parser = FlexibleArgumentParser(
        description="Benchmark per-request generator sampling overhead."
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--num-generators", type=int, default=128)
    parser.add_argument(
        "--layout",
        type=str,
        choices=["contiguous", "non-contiguous"],
        default="non-contiguous",
        help="How generator instances are assigned to rows.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-warmup-iters", type=int, default=16)
    parser.add_argument("--num-iters", type=int, default=256)
    args = parser.parse_args()
    if args.num_generators > args.batch_size:
        raise ValueError("--num-generators cannot exceed --batch-size.")

    main(
        device=args.device,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        num_generators=args.num_generators,
        layout=args.layout,
        seed=args.seed,
        num_warmup_iters=args.num_warmup_iters,
        num_iters=args.num_iters,
    )


if __name__ == "__main__":
    run_main()
