# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the per-token request mapping kernel.

The kernel builds ``token -> request index`` for sparse attention metadata.
This compares the previous early-exit binary search with the fixed-iteration
search used by ``compute_token_to_req_indices``.

Timing captures several launches in one CUDA graph and reports the per-kernel
time, so host launch overhead does not dominate.
"""

import argparse
import gc
import statistics
import time
from functools import partial

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.metadata import compute_token_to_req_indices


@triton.jit
def _token_request_baseline(query_start_loc, token, num_reqs):
    lo = tl.full(token.shape, 0, tl.int32)
    hi = tl.full(token.shape, num_reqs, tl.int32)
    while tl.sum((lo < hi).to(tl.int32), 0) > 0:
        mid = (lo + hi) // 2
        end = tl.load(query_start_loc + mid + 1, mid < num_reqs, other=0x7FFFFFFF)
        right = (lo < hi) & (end <= token)
        lo = tl.where(right, mid + 1, lo)
        hi = tl.where((lo < hi) & ~right, mid, hi)
    return tl.minimum(lo, num_reqs - 1)


@triton.jit(do_not_specialize=["num_reqs", "num_mapped", "num_tokens"])
def _baseline_kernel(query_start_loc, output, num_reqs, num_mapped, num_tokens):
    token = tl.program_id(0) * 256 + tl.arange(0, 256)
    req = _token_request_baseline(query_start_loc, token, num_reqs)
    tl.store(output + token, tl.where(token < num_mapped, req, 0), token < num_tokens)


def run_baseline(query_start_loc, out, num_tokens):
    _baseline_kernel[((num_tokens + 255) // 256,)](
        query_start_loc,
        out,
        query_start_loc.shape[0] - 1,
        num_tokens,
        num_tokens,
        num_warps=4,
    )
    return out[:num_tokens]


def reference(query_start_loc, num_tokens):
    tokens = torch.arange(num_tokens, device=query_start_loc.device, dtype=torch.int64)
    req = torch.searchsorted(query_start_loc.long(), tokens, right=True) - 1
    req = req.clamp(0, query_start_loc.numel() - 2)
    return req.to(torch.int32)


def make_graph(fn, reps):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(reps):
            fn()
    torch.accelerator.synchronize()
    return graph


def timed(fn, warmup_ms, rep_ms, rounds, graph_reps):
    graph = make_graph(fn, graph_reps)
    medians = []
    for _ in range(rounds):
        gc.collect()
        torch.accelerator.synchronize()
        median = triton.testing.do_bench(
            graph.replay,
            warmup=warmup_ms,
            rep=rep_ms,
            return_mode="median",
        )
        medians.append(float(median) / graph_reps)
    return statistics.median(medians)


def burn(seconds):
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    start = time.time()
    while time.time() - start < seconds:
        a = a @ a
    torch.accelerator.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--warmup", type=float, default=100.0, help="ms")
    parser.add_argument("--rep", type=float, default=300.0, help="ms")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--graph-reps", type=int, default=100)
    args = parser.parse_args()

    burn(3.0)
    print(torch.cuda.get_device_name(0))
    print(f"torch {torch.__version__}, triton {triton.__version__}")

    if args.quick:
        shapes = [(128, 5), (1024, 5), (1024, 64)]
    else:
        shapes = [
            (num_reqs, tokens_per_req)
            for num_reqs in (1, 8, 32, 128, 512, 1024)
            for tokens_per_req in (1, 5, 64, 512)
        ]

    print("num_reqs,tokens_per_req,num_tokens,baseline_us,new_us,speedup,exact")
    for num_reqs, tokens_per_req in shapes:
        num_tokens = num_reqs * tokens_per_req
        qsl = (
            torch.arange(num_reqs + 1, device="cuda", dtype=torch.int32)
            * tokens_per_req
        ).contiguous()
        out = torch.empty(num_tokens, device="cuda", dtype=torch.int32)
        expected = reference(qsl, num_tokens)

        baseline_fn = partial(run_baseline, qsl, out, num_tokens)
        new_fn = partial(compute_token_to_req_indices, qsl, out, num_tokens, num_tokens)

        exact = torch.equal(baseline_fn(), expected) and torch.equal(new_fn(), expected)
        base_ms = timed(
            baseline_fn,
            args.warmup,
            args.rep,
            args.rounds,
            args.graph_reps,
        )
        new_ms = timed(
            new_fn,
            args.warmup,
            args.rep,
            args.rounds,
            args.graph_reps,
        )
        print(
            f"{num_reqs},{tokens_per_req},{num_tokens},"
            f"{base_ms * 1e3:.2f},{new_ms * 1e3:.2f},"
            f"{base_ms / new_ms:.2f},{exact}",
            flush=True,
        )


if __name__ == "__main__":
    main()
