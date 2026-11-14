# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare mHC post dispatch against a previous git revision in CUDA graphs."""

import argparse
import ast
import json
import statistics
import subprocess
import tempfile
from pathlib import Path

import torch

from vllm.model_executor.kernels.mhc import tilelang_kernels
from vllm.model_executor.kernels.mhc.tilelang_kernels import MhcPostTileLangKernel


def load_baseline(ref):
    path = "vllm/model_executor/kernels/mhc/tilelang_kernels.py"
    source = subprocess.check_output(["git", "show", f"{ref}:{path}"], text=True)
    node = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "mhc_post_tilelang"
    )
    namespace = vars(tilelang_kernels).copy()
    directory = tempfile.TemporaryDirectory()
    source_path = Path(directory.name) / "tilelang_kernels.py"
    source_path.write_text(source)
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(source_path), "exec"),
        namespace,
    )
    return namespace["mhc_post_tilelang"], directory


def benchmark(args):
    baseline, baseline_directory = load_baseline(args.baseline_ref)
    candidate = MhcPostTileLangKernel()
    rng = torch.Generator(device="cuda").manual_seed(18)
    results = []
    for rows in args.rows:
        residual = torch.randn(
            rows,
            args.hc,
            args.hidden,
            device="cuda",
            dtype=torch.bfloat16,
            generator=rng,
        )
        x = torch.randn(
            rows, args.hidden, device="cuda", dtype=torch.bfloat16, generator=rng
        )
        post = torch.randn(rows, args.hc, device="cuda", generator=rng)
        comb = torch.randn(rows, args.hc, args.hc, device="cuda", generator=rng)
        outputs = {
            variant: torch.empty_like(residual) for variant in ("baseline", "candidate")
        }

        def run(variant, comb=comb, residual=residual, post=post, x=x, outputs=outputs):
            kernel = baseline if variant == "baseline" else candidate
            kernel(comb, residual, post, x, outputs[variant], args.hc, args.hidden)

        for variant in outputs:
            run(variant)
        torch.accelerator.synchronize()
        assert torch.equal(
            outputs["baseline"].view(torch.uint8),
            outputs["candidate"].view(torch.uint8),
        )
        # Reuse the output address so allocation placement cannot favor an arm.
        outputs["candidate"] = outputs["baseline"]
        graphs = {}
        for variant in outputs:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(args.calls):
                    run(variant)
            graphs[variant] = graph
            graph.replay()
        torch.accelerator.synchronize()

        samples = {variant: [] for variant in outputs}
        for trial in range(args.trials):
            order = list(outputs) if trial % 2 == 0 else list(reversed(outputs))
            for variant in order:
                start, end = torch.cuda.Event(True), torch.cuda.Event(True)
                start.record()
                graphs[variant].replay()
                end.record()
                end.synchronize()
                samples[variant].append(start.elapsed_time(end) * 1000 / args.calls)
        baseline_us = statistics.median(samples["baseline"])
        candidate_us = statistics.median(samples["candidate"])
        key = candidate.dispatch(
            num_tokens=rows, hidden_size=args.hidden, hc_mult=args.hc
        )
        results.append(
            {
                "rows": rows,
                "dispatch_tiles": key.h_tiles,
                "baseline_us": baseline_us,
                "candidate_us": candidate_us,
                "speedup": baseline_us / candidate_us,
                "samples_us": samples,
            }
        )
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "settings": vars(args),
                "results": results,
            },
            indent=2,
        )
    )
    baseline_directory.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="origin/main")
    parser.add_argument(
        "--rows", type=int, nargs="+", default=[1, 8, 16, 40, 64, 65, 128]
    )
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--hc", type=int, default=4)
    parser.add_argument("--calls", type=int, default=256)
    parser.add_argument("--trials", type=int, default=7)
    benchmark(parser.parse_args())
