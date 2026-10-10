# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare Cake with the existing top-k/top-p paths on captured decode logits.

Example:
    python benchmarks/kernels/benchmark_cake_sampling.py \
        --logits flash_next_logits.pt --output microbench.json

Masking mutates its input, so graph timings restore logits before each timed
replay. Direct sampling uses FlashInfer's CUPTI helper. Both flush L2 before
timing; compilation, allocations, input restoration and transfers are excluded.
Requires FlashInfer >= 0.7.1 and cupti-python with CUDA 13 or newer.

"""

import argparse
import json
import statistics
import warnings
from functools import partial
from pathlib import Path

import flashinfer
import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.v1.sample.ops.topk_topp_cake import (
    CAKE_MASK_MAX_ROWS,
    apply_top_k_top_p_cake,
    cake_eligible,
    cake_sample,
)
from vllm.v1.sample.ops.topk_topp_sampler import (
    apply_top_k_top_p_pytorch,
    flashinfer_sample,
)
from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton


def mask_time_us(fn, logits, flush, repeats):
    work = logits.clone()
    for _ in range(5):
        work.copy_(logits)
        fn(work)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        work.copy_(logits)
        fn(work)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph):
        fn(work)
    events = []
    for _ in range(repeats):
        work.copy_(logits)
        flush.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        events.append((start, end))
    torch.accelerator.synchronize()
    return statistics.median(s.elapsed_time(e) * 1000 for s, e in events)


def sample_time_us(fn, logits, k, p, repeats):
    for _ in range(5):
        fn(logits, k, p)
    torch.accelerator.synchronize()
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="CUPTI is not installed.*")
        warnings.filterwarnings("error", message="cold_l2_cache=True but no GPU.*")
        times = bench_gpu_time_with_cupti(
            fn,
            use_cuda_graph=True,
            cold_l2_cache=True,
            dry_run_time_ms=250,
            repeat_iters=repeats,
            input_args=(logits, k, p),
        )
    return statistics.median(times) * 1000


def correctness(logits, k, p, top_k):
    reference = apply_top_k_top_p_pytorch(logits.clone(), k, p)
    baseline = apply_top_k_top_p_triton(logits.clone(), k, p)
    candidate = apply_top_k_top_p_cake(logits.clone(), k, p, top_k)
    expected = reference.double().softmax(-1)
    result = {}
    for name, actual in (("triton", baseline), ("cake", candidate)):
        tv = (actual.double().softmax(-1) - expected).abs().sum(-1) / 2
        result[name] = {
            "mask_diff_rows": int(
                ((actual > -torch.inf) != (reference > -torch.inf)).any(-1).sum()
            ),
            "tv_mean": float(tv.mean()),
            "tv_max": float(tv.max()),
        }
        assert torch.isfinite(actual).any(-1).all(), "empty retained support"
    # The reference retains ties at the top-k boundary; Cake keeps exactly k.
    # Report these differences instead of claiming token identity across backends.
    tokens = cake_sample(logits, k, p, top_k).long()
    assert (candidate > -torch.inf).gather(1, tokens[:, None]).all()
    # Independently check Cake's exact-k, lower-index tie contract.
    probs = flashinfer.sampling.softmax(logits)
    values, ids = probs.sort(descending=True, stable=True)
    values, ids = values[:, :top_k].double(), ids[:, :top_k]
    values /= values.sum(-1, keepdim=True)
    keep = values.cumsum(-1) - values < p[:, None]
    values = values.masked_fill(~keep, 0)
    values /= values.sum(-1, keepdim=True)
    exact = torch.zeros_like(logits, dtype=torch.float64).scatter_(1, ids, values)
    tv = (candidate.double().softmax(-1) - exact).abs().sum(-1) / 2
    result["cake_exact_k_tv_max"] = float(tv.max())
    assert tv.max() < 2e-5, "Cake disagrees with its exact-k/top-p contract"
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logits", type=Path)
    parser.add_argument("--vocab-size", type=int, default=248320)
    parser.add_argument(
        "--rows", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256]
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(123)
    if args.logits:
        captured = torch.load(args.logits, map_location="cuda", weights_only=True)
        source = str(args.logits)
    else:
        captured = torch.randn(max(args.rows), args.vocab_size, device="cuda") * 3
        source = "synthetic; shape validation only"
    captured = captured.float().contiguous()
    vocab = captured.shape[-1]
    assert cake_eligible(captured, args.top_k), "Cake pipeline is unavailable"
    flush = torch.empty(128 * 1024**2, device="cuda", dtype=torch.int8)
    result = {
        "metadata": {
            "source": source,
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "flashinfer": flashinfer.__version__,
            "dtype": str(captured.dtype),
            "vocab_size": vocab,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "rounds": args.rounds,
            "repeats": args.repeats,
            "mask_timing": "CUDA graph events; restore and 128 MiB L2 flush excluded",
            "sample_timing": "FlashInfer CUPTI helper; CUDA graph; cold L2",
        },
        "cases": [],
    }
    for rows in args.rows:
        logits = captured[torch.arange(rows, device="cuda") % len(captured)].clone()
        k = torch.full((rows,), args.top_k, device="cuda", dtype=torch.int32)
        p = torch.full((rows,), args.top_p, device="cuda")
        checks = correctness(logits, k, p, args.top_k)
        masks = {
            "triton": partial(apply_top_k_top_p_triton, k=k, p=p),
            "cake": partial(apply_top_k_top_p_cake, k=k, p=p, k_max=args.top_k),
        }
        samples = {
            "flashinfer": flashinfer_sample,
            "cake": partial(cake_sample, k_max=args.top_k),
        }
        timings = {
            "mask_triton": [],
            "mask_cake": [],
            "sample_flashinfer": [],
            "sample_cake": [],
        }
        for round_idx in range(args.rounds):
            for name in list(masks)[:: 1 if round_idx % 2 == 0 else -1]:
                timings[f"mask_{name}"].append(
                    mask_time_us(masks[name], logits, flush, args.repeats)
                )
            for name in list(samples)[:: 1 if round_idx % 2 == 0 else -1]:
                timings[f"sample_{name}"].append(
                    sample_time_us(samples[name], logits, k, p, args.repeats)
                )
        case = {
            "rows": rows,
            "mask_dispatch": "cake" if rows <= CAKE_MASK_MAX_ROWS else "triton",
            "correctness": checks,
            "timings_us": timings,
            "median_us": {
                name: statistics.median(values) for name, values in timings.items()
            },
        }
        result["cases"].append(case)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(case), flush=True)


if __name__ == "__main__":
    main()
