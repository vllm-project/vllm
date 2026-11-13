# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare native and padded SM90 BF16 sparse FlashMLA prefill.

Use a FlashMLA build containing the ETAP kernel. Allocations, input staging,
and correctness checks are outside timing; both arms use preallocated outputs
and CUDA graphs. Cold L2 is the default. Real heads define the FLOP count.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import os
import statistics
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cupti


def bench_graph_events(fn, repeats: int, warm_l2: bool) -> list[float]:
    """Flush Hopper's L2 before the timed events in each graph replay."""
    flush = (
        None if warm_l2 else torch.empty(128 * 1024**2, device="cuda", dtype=torch.int8)
    )
    for _ in range(10):
        fn()
    torch.accelerator.synchronize()
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        if flush is not None:
            flush.zero_()
        start.record()
        fn()
        end.record()
    for _ in range(10):
        graph.replay()
    torch.accelerator.synchronize()
    samples = []
    for _ in range(repeats):
        graph.replay()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--heads", type=int, nargs="+", choices=[8, 16, 32], default=[8, 16, 32]
    )
    parser.add_argument(
        "--dims", type=int, nargs="+", choices=[512, 576], default=[512, 576]
    )
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 8, 32, 64, 128, 256, 1024]
    )
    parser.add_argument("--topk", type=int, nargs="+", default=[128, 640, 2048])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--kv-tokens", type=int, default=8192)
    parser.add_argument(
        "--lengths",
        nargs="+",
        choices=["full", "ramp", "one", "empty"],
        default=["full"],
    )
    parser.add_argument("--include-mma", action="store_true")
    parser.add_argument("--warm-l2", action="store_true")
    parser.add_argument("--timer", choices=["cupti", "graph-events"], default="cupti")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(topk <= 0 or topk % 128 for topk in args.topk):
        parser.error("The padded SM90 baseline requires top-k to be a multiple of 128.")

    props = torch.cuda.get_device_properties(0)
    if props.major != 9:
        raise RuntimeError("This benchmark targets SM90.")
    import vllm._flashmla_C as flashmla_library

    from vllm.v1.attention.ops.flashmla import flash_mla_sparse_fwd

    library_path = Path(flashmla_library.__file__)
    torch.manual_seed(0)
    metadata = {
        "gpu": props.name,
        "sm_count": props.multi_processor_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flashinfer": importlib.metadata.version("flashinfer-python"),
        "flashmla_library": str(library_path),
        "flashmla_library_sha256": hashlib.sha256(
            library_path.read_bytes()
        ).hexdigest(),
        "arguments": vars(args) | {"output": str(args.output)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    arms = ["padded", "native"] + (["mma"] if args.include_mma else [])
    fields = [
        "tokens",
        "heads",
        "dim",
        "topk",
        "lengths",
        "arm",
        "median_us",
        "min_us",
        "max_us",
        "tflops",
        "native_speedup",
        "old_native",
    ]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        cases = itertools.product(
            args.dims, args.heads, args.topk, args.tokens, args.lengths
        )
        for dim, heads, topk, tokens, lengths in cases:
            q = torch.zeros((tokens, 64, dim), device="cuda", dtype=torch.bfloat16)
            q[:, :heads].normal_()
            kv = torch.randn(
                (args.kv_tokens, 1, dim), device="cuda", dtype=torch.bfloat16
            )
            indices = torch.randint(
                args.kv_tokens, (tokens, 1, topk), device="cuda", dtype=torch.int32
            )
            out = torch.empty((tokens, 64, 512), device="cuda", dtype=torch.bfloat16)
            valid = None
            attended_keys = tokens * topk
            if lengths != "full":
                if lengths == "ramp":
                    counts = [min(t + 1, topk) for t in range(tokens)]
                else:
                    counts = [int(lengths == "one")] * tokens
                attended_keys = sum(counts)
                valid = torch.tensor(counts, device="cuda", dtype=torch.int32)
                indices.masked_fill_(
                    torch.arange(topk, device="cuda")[None, None]
                    >= valid[:, None, None],
                    -1,
                )

            def run(
                arm,
                q=q,
                kv=kv,
                indices=indices,
                heads=heads,
                dim=dim,
                out=out,
                valid=valid,
            ):
                os.environ["USE_MMA_FOR_SMALL_N_HEAD"] = "1" if arm == "mma" else "0"
                width = 64 if arm == "padded" else heads
                return flash_mla_sparse_fwd(
                    q[:, :width],
                    kv,
                    indices,
                    dim**-0.5,
                    out=out[:, :width],
                    topk_length=valid,
                )

            reference, _, reference_lse = run("padded")
            reference = reference[:, :heads].clone()
            reference_lse = reference_lse[:, :heads].clone()
            for arm in arms[1:]:
                result, _, lse = run(arm)
                torch.testing.assert_close(result, reference, atol=2e-3, rtol=2e-2)
                torch.testing.assert_close(lse, reference_lse, atol=2e-4, rtol=2e-4)
            torch.accelerator.synchronize()

            timings = {arm: [] for arm in arms}
            for round_id in range(args.rounds):
                order = arms if round_id % 2 == 0 else arms[::-1]
                for arm in order:
                    if args.timer == "cupti":
                        samples = bench_gpu_time_with_cupti(
                            lambda arm=arm, run=run: run(arm),
                            dry_run_iters=10,
                            repeat_iters=args.repeats,
                            use_cuda_graph=True,
                            cold_l2_cache=not args.warm_l2,
                        )
                        samples = [ms * 1000 for ms in samples]
                    else:
                        samples = bench_graph_events(
                            lambda arm=arm, run=run: run(arm),
                            args.repeats,
                            args.warm_l2,
                        )
                    timings[arm].append(statistics.median(samples))
            medians = {arm: statistics.median(us) for arm, us in timings.items()}
            speedup = medians["padded"] / medians["native"]
            flops = 2 * heads * attended_keys * (dim + 512)
            for arm in arms:
                writer.writerow(
                    {
                        "tokens": tokens,
                        "heads": heads,
                        "dim": dim,
                        "topk": topk,
                        "lengths": lengths,
                        "arm": arm,
                        "median_us": medians[arm],
                        "min_us": min(timings[arm]),
                        "max_us": max(timings[arm]),
                        "tflops": flops / (medians[arm] * 1e6),
                        "native_speedup": speedup,
                        "old_native": tokens
                        > props.multi_processor_count * (2 if heads == 32 else 1),
                    }
                )
            handle.flush()
            print(
                f"T={tokens} H={heads} D={dim} K={topk} lengths={lengths}: "
                f"padded={medians['padded']:.3f}us "
                f"native={medians['native']:.3f}us speedup={speedup:.3f}",
                flush=True,
            )
            del q, kv, indices, out, reference, reference_lse, result, lse
    os.environ.pop("USE_MMA_FOR_SMALL_N_HEAD", None)


if __name__ == "__main__":
    main()
