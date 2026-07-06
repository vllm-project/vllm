# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import statistics
import time
from collections.abc import Callable
from functools import partial
from types import SimpleNamespace

import numpy as np
import torch

from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.worker.gpu.sample.sampler import Sampler


def _array_state(values: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(np=values)


def _noop(*args, **kwargs) -> None:
    pass


def make_noop_sampler(batch_size: int, vocab_size: int) -> Sampler:
    sampler = Sampler.__new__(Sampler)
    sampler.sampling_states = SimpleNamespace(
        vocab_size=vocab_size,
        temperature=_array_state(np.ones(batch_size, dtype=np.float32)),
        top_k=_array_state(np.full(batch_size, vocab_size, dtype=np.int32)),
        top_p=_array_state(np.ones(batch_size, dtype=np.float32)),
        min_p=_array_state(np.zeros(batch_size, dtype=np.float32)),
        apply_temperature=_noop,
        apply_min_p=_noop,
        apply_top_k_top_p=lambda logits, *args: logits,
    )
    sampler.logit_bias_state = SimpleNamespace(
        use_logit_bias=np.zeros(batch_size, dtype=bool),
        apply_logit_bias=_noop,
    )
    sampler.penalties_state = SimpleNamespace(
        use_penalty=np.zeros(batch_size, dtype=bool),
        apply_penalties=_noop,
    )
    sampler.bad_words_state = SimpleNamespace(
        num_bad_words=_array_state(np.zeros(batch_size, dtype=np.int32)),
        apply_bad_words=_noop,
    )
    return sampler


def benchmark(
    fn: Callable[[], torch.Tensor],
    warmup_iters: int,
    benchmark_iters: int,
    repeats: int,
) -> float:
    for _ in range(warmup_iters):
        fn()
    torch.accelerator.synchronize()

    measurements = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(benchmark_iters):
            output = fn()
        torch.accelerator.synchronize()
        measurements.append((time.perf_counter() - start) * 1000 / benchmark_iters)
        del output
    return statistics.median(measurements)


def copy_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(logits, dtype=torch.float32).copy_(logits)


def apply_noop_sampling_params(
    sampler: Sampler,
    logits: torch.Tensor,
    idx_mapping_np: np.ndarray,
    unused: torch.Tensor,
) -> torch.Tensor:
    return sampler.apply_sampling_params(
        logits,
        unused,
        idx_mapping_np,
        unused,
        unused,
        unused,
        skip_top_k_top_p=True,
    )


def main(args) -> None:
    device = torch.device(args.device)
    torch.accelerator.set_device_index(device)
    dtype = getattr(torch, args.dtype)

    print(
        f"{'batch':>8} {'copy (ms)':>12} {'noop (ms)':>12} "
        f"{'speedup':>10} {'FP32 avoided':>14}"
    )
    for batch_size in args.batch_sizes:
        logits = torch.randn(
            batch_size,
            args.vocab_size,
            dtype=dtype,
            device=device,
        )
        sampler = make_noop_sampler(batch_size, args.vocab_size)
        idx_mapping_np = np.arange(batch_size, dtype=np.int32)
        unused = torch.empty(0, dtype=torch.int32, device=device)

        copy_path = partial(copy_logits, logits)
        noop_path = partial(
            apply_noop_sampling_params,
            sampler,
            logits,
            idx_mapping_np,
            unused,
        )

        copy_ms = benchmark(
            copy_path,
            args.warmup_iters,
            args.benchmark_iters,
            args.repeats,
        )
        noop_ms = benchmark(
            noop_path,
            args.warmup_iters,
            args.benchmark_iters,
            args.repeats,
        )
        avoided_mib = batch_size * args.vocab_size * 4 / (1 << 20)
        print(
            f"{batch_size:8d} {copy_ms:12.4f} {noop_ms:12.4f} "
            f"{copy_ms / noop_ms:9.2f}x {avoided_mib:11.1f} MiB"
        )


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the MRV2 no-op sampling-parameter fast path."
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--benchmark-iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    main(parser.parse_args())
