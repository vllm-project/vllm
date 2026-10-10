# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time Kimi-K3 K1 against its numerical reference chain on MI355X.

Example:
    .venv/bin/python benchmarks/kernels/benchmark_kimi_k3_mono.py \
        --rows 8 16 32 64 --output /tmp/kimi-k1.json

Uses the numerical suite's random weights. Graph replay excludes allocations,
JIT and state reset. The reference is not the production model's dispatch;
use serving benchmarks to assess end-to-end gains.

"""

import argparse
import json
import statistics
from pathlib import Path

import torch

from tests.models.kimi_k3.test_amd_mono import _kda_inputs, _kda_reference
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.models.kimi_k3.amd.mono.attention import kda
from vllm.models.kimi_k3.amd.mono.common.tiles import token_tiles
from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated


def measure(rows, length, warmup, iterations, repeats, flush_l2_mb):
    inputs = _kda_inputs(4, 1, False, rows, qlen=length)
    key = kda.KdaPreBuild(
        tokens=rows,
        qlen=length,
        nblocks=4,
        delta=True,
        state_len=inputs["conv_state"].size(-1),
        state_rows=rows if length == 8 and rows in (32, 64) else 0,
    )
    baseline = {
        name: inputs[name].clone()
        for name in ("prefix", "blocks", "conv_state", "rstate")
    }
    scratch = torch.zeros(kda.scratch_bytes(key), dtype=torch.uint8, device="cuda")
    epoch = torch.ones(1, dtype=torch.int32, device="cuda")
    out = torch.empty(rows, kda.PROJ, dtype=torch.bfloat16, device="cuda")
    eviction = (
        torch.empty(flush_l2_mb * 1024 * 1024, dtype=torch.uint8, device="cuda")
        if flush_l2_mb
        else None
    )
    with set_current_vllm_config(VllmConfig()):
        norm = FusedRMSNormGated(kda.HD, activation="sigmoid").cuda()
    norm.weight.data.copy_(inputs["on_w"])

    def reset():
        for name, value in baseline.items():
            inputs[name].copy_(value)
        epoch.add_(1)

    def mono():
        kda.kda_pre(
            key,
            prefix=inputs["prefix"],
            delta=inputs["delta"],
            blocks=inputs["blocks"],
            ares_nw=inputs["ares_nw"],
            ares_qk=inputs["ares_qk"],
            in_nw=inputs["in_nw"],
            w_in=inputs["w_in"],
            w_fb=inputs["w_fb"],
            conv_w=inputs["conv_w"],
            conv_state=inputs["conv_state"],
            a_log=inputs["a_log"],
            dt_bias=inputs["dt_bias"],
            on_w=inputs["on_w"],
            rstate=inputs["rstate"],
            st_idx=inputs["st_idx"],
            num_acc=inputs["num_acc"],
            core_out=out,
            scratch=scratch,
            layer=0,
            epoch=epoch,
        )

    variants = {
        "reference": lambda: _kda_reference(
            inputs, 4, rows, -1, norm=norm, inplace=True
        ),
        "mono": mono,
    }
    graphs = {}
    for name, fn in variants.items():
        for _ in range(warmup):
            reset()
            fn()
        torch.accelerator.synchronize()
        reset()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        graphs[name] = graph
    samples = {name: [] for name in variants}
    for repeat in range(repeats):
        order = list(variants)
        if repeat % 2:
            order.reverse()
        for name in order:
            events = []
            for _ in range(iterations):
                reset()
                if eviction is not None:
                    eviction.zero_()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                graphs[name].replay()
                end.record()
                events.append((start, end))
            torch.accelerator.synchronize()
            samples[name].append(
                statistics.median(a.elapsed_time(b) * 1000 for a, b in events)
            )
    return {"rows": rows, "sequence_length": length, "median_us_per_run": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--sequence-length", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--flush-l2-mb", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.repeats) < 1 or args.flush_l2_mb < 0:
        parser.error("counts must be positive and flush-l2-mb nonnegative")
    for rows in args.rows:
        token_tiles(rows, args.sequence_length, tile_tokens=16)
    result = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(),
        "results": [],
    }
    for rows in args.rows:
        measured = measure(
            rows,
            args.sequence_length,
            args.warmup,
            args.iterations,
            args.repeats,
            args.flush_l2_mb,
        )
        result["results"].append(measured)
        print(json.dumps(measured), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
