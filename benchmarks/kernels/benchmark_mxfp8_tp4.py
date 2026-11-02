# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate MXFP8 TP4 linears and time HIP graph replay over rotating weights.

Example (record the baseline and candidate SHAs):
    baseline=$(git rev-parse origin/main)
    git rev-parse "$baseline" HEAD
    git show "$baseline":vllm/model_executor/kernels/linear/mxfp8/rocm_native.py \
        > /tmp/baseline_rocm_native.py
    .venv/bin/python benchmarks/kernels/benchmark_mxfp8_tp4.py \
        --baseline /tmp/baseline_rocm_native.py

Reports cold (>400 MiB rotating weights) and warm timings separately. Each timing
is the median of 30 device-event samples; Python wall time is never ranked.
"""

import argparse
import importlib.util
import json
import math
import statistics
from functools import partial
from pathlib import Path

import torch

from vllm.model_executor.kernels.linear.mxfp8 import rocm_native
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    dequant_mxfp8_to_bf16,
    mxfp8_e4m3_quantize,
)
from vllm.triton_utils import triton

SHAPES = [
    (m, n, k)
    for n, k in [(2560, 6144), (6144, 2048), (1536, 6144), (6144, 768)]
    for m in [1, 8, 16]
]


def time_graph(calls):
    for call in calls:
        call()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(2):
            for call in calls:
                call()
    for _ in range(5):
        graph.replay()
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(30)
    ]
    for start, end in events:
        start.record()
        for _ in range(5):
            graph.replay()
        end.record()
        # Bound outstanding HIP graph nodes on short-K cases with many slots.
        end.synchronize()
    torch.accelerator.synchronize()
    return statistics.median(
        start.elapsed_time(end) / (10 * len(calls)) for start, end in events
    )


def launch(module, xq, xs, w, ws, out, config):
    bm, bn, bk, nw, ns = config
    m, k = xq.shape
    n = w.shape[0]
    return module._mxfp8_linear_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        xq,
        xs,
        w,
        ws,
        out,
        m,
        n,
        k,
        *xq.stride(),
        *xs.stride(),
        *w.stride(),
        *ws.stride(),
        *out.stride(),
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        num_warps=nw,
        num_stages=ns,
    )


def emit(**data):
    print(json.dumps(data), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--shape", type=int, choices=range(len(SHAPES)))
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--asm-dir", type=Path)
    args = parser.parse_args()
    torch.accelerator.set_device_index(args.device)
    torch.manual_seed(42)
    variants = {"candidate": rocm_native}
    if args.baseline:
        spec = importlib.util.spec_from_file_location(
            "vllm.model_executor.kernels.linear.mxfp8.baseline_rocm_native",
            args.baseline,
        )
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        variants = {"baseline": baseline, **variants}
    emit(
        torch=torch.__version__,
        triton=triton.__version__,
        device=str(torch.cuda.get_device_properties(args.device)),
    )
    for index, (m, n, k) in enumerate(SHAPES):
        if args.shape is not None and index != args.shape:
            continue
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.5
        w, ws = mxfp8_e4m3_quantize(
            torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05
        )
        xq, xs = mxfp8_e4m3_quantize(x)
        reference = (
            dequant_mxfp8_to_bf16(xq, xs).float()
            @ dequant_mxfp8_to_bf16(w, ws).float().T
        )
        out = torch.empty(m, n, device="cuda", dtype=x.dtype)
        slots = math.ceil(400 * 1024**2 / (n * k))
        weights = [(w.clone(), ws.clone()) for _ in range(slots)]
        calls = {}
        for name, module in variants.items():
            configs = [module._select_cfg(m, n, k)]
            if args.tune:
                configs += [
                    (16, 16, 1024, 2, 2),
                    (16, 16, 1024, 2, 3),
                    (16, 16, 1024, 2, 4),
                    (16, 16, 512, 2, 2),
                    (16, 32, 256, 4, 3),
                    (16, 16, 256, 2, 3),
                    (128, 128, 128, 8, 2),
                ]
            for config in dict.fromkeys(configs):
                if k % config[2]:
                    continue
                fn = partial(launch, module, xq, xs, config=config, out=out)
                obj = fn(w=w, ws=ws)
                rel = ((out.float() - reference).norm() / reference.norm()).item()
                snr = -20 * math.log10(rel) if rel else float("inf")
                assert rel < 0.005 and snr >= 30, (name, config, rel, snr)
                emit(
                    shape=[m, n, k],
                    variant=name,
                    config=config,
                    rel=rel,
                    snr_db=snr,
                    registers=obj.n_regs,
                    spills=obj.n_spills,
                    lds=obj.metadata.shared,
                )
                if args.asm_dir:
                    args.asm_dir.mkdir(parents=True, exist_ok=True)
                    (args.asm_dir / f"{name}-{m}-{n}-{k}-{config}.amdgcn").write_text(
                        obj.asm["amdgcn"]
                    )
                if args.profile:
                    for _ in range(10):
                        fn(w=w, ws=ws)
                    torch.accelerator.synchronize()
                    continue
                calls[f"{name}-kernel-{config}"] = [
                    partial(fn, w=wi, ws=si) for wi, si in weights
                ]
            if not args.profile and not args.tune:
                wrapper = partial(module._mxfp8_dot_scaled_linear, x)
                actual = wrapper(w, ws)
                assert ((actual.float() - reference).norm() / reference.norm()) < 0.005
                calls[f"{name}-wrapper"] = [
                    partial(wrapper, wi, si) for wi, si in weights
                ]
        for repeat in range(2):
            order = list(calls.items())
            if repeat:
                order.reverse()
            for name, functions in order:
                cold = time_graph(functions)
                warm = time_graph([functions[0]] * 32)
                emit(
                    shape=[m, n, k],
                    variant=name,
                    repeat=repeat,
                    cold_ms=cold,
                    warm_ms=warm,
                    slots=slots,
                )


if __name__ == "__main__":
    main()
