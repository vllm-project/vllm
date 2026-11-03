# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the GPU no-repeat n-gram kernel with MinerU's Python algorithm."""

import argparse
import time

import torch

from vllm.v1.worker.gpu.sample.no_repeat_ngram import apply_no_repeat_ngram


def build_mineru_caches(output_ids: list[list[int]], ngram_size: int):
    caches = []
    for tokens in output_ids:
        cache: dict[tuple[int, ...], list[int]] = {}
        for output_len in range(ngram_size, len(tokens)):
            history = tokens[:output_len]
            prefix = tuple(history[-ngram_size:-1])
            cache.setdefault(prefix, []).append(history[-1])
        caches.append(cache)
    return caches


def mineru_apply(output_ids, caches, logits: torch.Tensor, ngram_size: int):
    for row, (tokens, cache) in enumerate(zip(output_ids, caches)):
        previous_prefix = tuple(tokens[-ngram_size:-1])
        cache.setdefault(previous_prefix, []).append(tokens[-1])
        current_prefix = tuple(tokens[-ngram_size + 1 :])
        for token in cache.get(current_prefix, []):
            logits[row, token] = -float("inf")


def gpu_apply(case: tuple[torch.Tensor, ...]) -> None:
    logits, mapping, tokens, zeros, lengths, sizes, windows, whitelist = case
    logits.zero_()
    apply_no_repeat_ngram(
        logits,
        mapping,
        tokens,
        zeros,
        lengths,
        sizes,
        windows,
        whitelist,
        zeros,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--history-length", type=int, default=2048)
    parser.add_argument("--ngram-size", type=int, default=100)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)

    for batch_size in args.batch_sizes:
        tokens = torch.randint(
            0,
            args.vocab_size,
            (batch_size, args.history_length),
            dtype=torch.int32,
            device=device,
        )
        tokens[:, -args.ngram_size + 1 :] = tokens[:, : args.ngram_size - 1]
        output_ids = tokens.cpu().tolist()
        mineru_caches = build_mineru_caches(output_ids, args.ngram_size)
        mapping = torch.arange(batch_size, dtype=torch.int32, device=device)
        zeros = torch.zeros(batch_size, dtype=torch.int32, device=device)
        lengths = torch.full(
            (batch_size,), args.history_length, dtype=torch.int32, device=device
        )
        sizes = torch.full(
            (batch_size,), args.ngram_size, dtype=torch.int32, device=device
        )
        windows = lengths.clone()
        whitelist = torch.zeros((batch_size, 1), dtype=torch.int32, device=device)
        logits = torch.zeros((batch_size, args.vocab_size), device=device)

        gpu_case = (
            logits,
            mapping,
            tokens,
            zeros,
            lengths,
            sizes,
            windows,
            whitelist,
        )
        gpu_apply(gpu_case)
        torch.accelerator.synchronize(device)
        expected = torch.zeros_like(logits)
        validation_caches = build_mineru_caches(output_ids, args.ngram_size)
        mineru_apply(output_ids, validation_caches, expected, args.ngram_size)
        torch.testing.assert_close(logits, expected)
        started = time.perf_counter()
        for _ in range(args.iterations):
            gpu_apply(gpu_case)
        torch.accelerator.synchronize(device)
        gpu_ms = (time.perf_counter() - started) * 1000 / args.iterations

        started = time.perf_counter()
        for _ in range(args.iterations):
            logits.zero_()
            mineru_apply(output_ids, mineru_caches, logits, args.ngram_size)
        torch.accelerator.synchronize(device)
        mineru_ms = (time.perf_counter() - started) * 1000 / args.iterations
        print(
            f"batch={batch_size} history={args.history_length} "
            f"ngram={args.ngram_size} gpu_ms={gpu_ms:.4f} "
            f"mineru_ms={mineru_ms:.4f} speedup={mineru_ms / gpu_ms:.2f}x"
        )


if __name__ == "__main__":
    main()
