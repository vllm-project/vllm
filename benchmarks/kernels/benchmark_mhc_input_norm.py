# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare mHC input norm with a git baseline in repeated CUDA graphs."""

import argparse
import ast
import contextlib
import hashlib
import json
import statistics
import subprocess
import tempfile
from pathlib import Path

import torch

from vllm.model_executor.kernels.mhc import tilelang_kernels, warmup
from vllm.model_executor.kernels.mhc.warmup import MHCPreNormKernel


def load_baseline(ref):
    directory = tempfile.TemporaryDirectory()
    namespace = vars(tilelang_kernels).copy()
    source_hashes = {}
    for filename, name in (
        ("tilelang_kernels.py", "mhc_pre_big_fuse_with_norm_tilelang"),
        ("warmup.py", "MHCPreNormKernel"),
    ):
        path = f"vllm/model_executor/kernels/mhc/{filename}"
        source = subprocess.check_output(["git", "show", f"{ref}:{path}"], text=True)
        source_hashes[path] = hashlib.sha256(source.encode()).hexdigest()
        source_path = Path(directory.name) / filename
        source_path.write_text(source)
        node = next(
            node
            for node in ast.parse(source).body
            if getattr(node, "name", None) == name
        )
        if filename == "warmup.py":
            baseline_function = namespace["mhc_pre_big_fuse_with_norm_tilelang"]
            namespace = vars(warmup).copy()
            namespace["mhc_pre_big_fuse_with_norm_tilelang"] = baseline_function
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), str(source_path), "exec"),
            namespace,
        )
    return namespace["MHCPreNormKernel"](), directory, source_hashes


@contextlib.contextmanager
def select_overlap_norm(kernel):
    previous = warmup.MHC_PRE_NORM_KERNEL
    warmup.MHC_PRE_NORM_KERNEL = kernel
    try:
        yield
    finally:
        warmup.MHC_PRE_NORM_KERNEL = previous


def benchmark_case(args, rows, scope, baseline, candidate, rng):
    residual = torch.randn(
        rows,
        args.hc,
        args.hidden,
        device="cuda",
        dtype=torch.bfloat16,
        generator=rng,
    )
    weight = torch.randn(
        args.hidden, device="cuda", dtype=torch.bfloat16, generator=rng
    )
    pre_mix = torch.rand(rows, args.hc, device="cuda", generator=rng)
    mix_size = args.hc * (args.hc + 2)
    mixes = torch.full((4, rows, mix_size), float("nan"), device="cuda")
    sqrsum = torch.full((4, rows), float("nan"), device="cuda")
    scale = torch.tensor([0.5, 0.25, 1.0], device="cuda")
    base = torch.randn(mix_size, device="cuda", generator=rng)
    post = torch.full((rows, args.hc), float("nan"), device="cuda")
    comb = torch.full((rows, args.hc * args.hc), float("nan"), device="cuda")
    next_pre = torch.full_like(post, float("nan"))
    normalized = {
        name: torch.empty(rows, args.hidden, device="cuda", dtype=torch.bfloat16)
        for name in ("baseline", "candidate")
    }
    fields = dict(
        hidden_size=args.hidden,
        hc_mult=args.hc,
        rms_numel=args.hc * args.hidden,
        use_pre_mix_in=True,
        save_pre_mix=True,
        split_mode="input",
        write_aux=False,
        rms_eps=1e-6,
        hc_pre_eps=1e-6,
        hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=20,
        norm_eps=1e-6,
    )
    projection = None
    if scope == "overlap":
        from vllm.models.deepseek_v41.nvidia.ops.mhc import mhc_pre_delayed_overlap

        projection = (
            torch.randn(mix_size, args.hc * args.hidden, device="cuda", generator=rng)
            * 0.02
        )
    side = {name: torch.cuda.Stream() for name in normalized}
    outputs = {}

    def run(name):
        kernel = baseline if name == "baseline" else candidate
        if scope == "norm":
            output = normalized[name]
            kernel(
                mixes,
                sqrsum,
                scale,
                base,
                residual,
                post,
                comb,
                output,
                weight,
                pre_mix,
                next_pre,
                output,
                **fields,
            )
            outputs[name] = (output,)
        else:
            with select_overlap_norm(kernel):
                out = mhc_pre_delayed_overlap(
                    residual,
                    projection,
                    scale,
                    base,
                    1e-6,
                    1e-6,
                    1e-6,
                    2.0,
                    20,
                    pre_mix=pre_mix,
                    norm_weight=weight,
                    norm_eps=1e-6,
                    stream=side[name],
                )
                consumed = out[2].clone()
                torch.cuda.current_stream().wait_stream(side[name])
                outputs[name] = (*out, consumed)

    for name in normalized:
        run(name)
    torch.accelerator.synchronize()
    for old, new in zip(outputs["baseline"], outputs["candidate"], strict=True):
        assert torch.equal(old.view(torch.uint8), new.view(torch.uint8))
    calls = args.calls if scope == "norm" else min(args.calls, 16)
    graphs = {}
    for name in normalized:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(calls):
                run(name)
        graphs[name] = graph
        graph.replay()
    torch.accelerator.synchronize()
    for old, new in zip(outputs["baseline"], outputs["candidate"], strict=True):
        assert torch.equal(old.view(torch.uint8), new.view(torch.uint8))
    samples = {name: [] for name in normalized}
    for trial in range(args.trials):
        order = list(normalized) if trial % 2 == 0 else list(reversed(normalized))
        for name in order:
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / calls)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    key = candidate.dispatch(
        num_tokens=rows, input_is_disjoint=True, n_splits=4, **fields
    )
    return {
        "scope": scope,
        "rows": rows,
        "dispatch_tiles": key.h_tiles,
        "calls_per_graph": calls,
        "original_bytes": "exact",
        "baseline_us": medians["baseline"],
        "candidate_us": medians["candidate"],
        "speedup": medians["baseline"] / medians["candidate"],
        "samples_us": samples,
    }


def benchmark(args):
    baseline, baseline_directory, source_hashes = load_baseline(args.baseline_ref)
    candidate = MHCPreNormKernel()
    rng = torch.Generator(device="cuda").manual_seed(args.seed)
    scopes = ("norm", "overlap") if args.scope == "both" else (args.scope,)
    results = [
        benchmark_case(args, rows, scope, baseline, candidate, rng)
        for rows in args.rows
        for scope in scopes
    ]
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "settings": vars(args),
                "baseline_source_sha256": source_hashes,
                "cache_policy": (
                    "Repeated fixed inputs; no cache flush or residency instrumentation"
                ),
                "timing_method": "CUDA events; alternating trial order",
                "overlap_scope": (
                    "Direct overlap owner, with input clone before coefficient join; "
                    "no model or serving dispatch"
                ),
                "results": results,
            },
            indent=2,
        )
    )
    baseline_directory.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="origin/main")
    parser.add_argument("--scope", choices=("norm", "overlap", "both"), default="norm")
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 9, 16, 17, 40])
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--hc", type=int, default=4)
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--calls", type=int, default=64)
    parser.add_argument("--trials", type=int, default=9)
    options = parser.parse_args()
    assert min(options.rows) > 0 and options.calls > 0 and options.trials > 0
    benchmark(options)
