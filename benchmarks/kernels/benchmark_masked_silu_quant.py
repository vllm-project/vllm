# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare separately built persistent MoE activation kernels on CUDA.

Run each library in its own process to avoid duplicate operator registration::

    python benchmarks/kernels/benchmark_masked_silu_quant.py \
        --library /path/to/_C_stable_libtorch.abi3.so --sweep full \
        --capacity 2048 --output stock.jsonl

Repeat with the patched library, with and without --limit 10. Alternate library
order and increment --round to reverse shape traversal between rounds. Timing
uses CUPTI, CUDA graphs, cold L2, preallocated buffers, and untimed correctness
checks. Hashes compare valid outputs between unclamped builds with the same
round/seed; the clamped kernel intentionally computes a different function.
"""

import argparse
import hashlib
import json
import statistics
from functools import partial
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cupti


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--library", required=True)
    p.add_argument("--limit", type=float)
    p.add_argument("--output", required=True)
    p.add_argument("--sweep", choices=["pilot", "full"], default="pilot")
    p.add_argument("--round", type=int, default=0)
    p.add_argument("--capacity", type=int, default=0)
    p.add_argument("--experts", type=int, default=32)
    a = p.parse_args()
    torch.ops.load_library(a.library)
    op = torch.ops._C.persistent_masked_m_silu_mul_quant
    has_clamp = "clamp_limit" in str(op.default._schema)
    torch.manual_seed(1234)
    # Paired pilot spans both launch configurations and all scale formats.
    cases = [
        (a.experts, h, n, fmt, "uniform")
        for h in [128, 2048, 4096, 7168]
        for n in [1, 64, 256]
        for fmt in ["float", "ceil", "packed"]
    ]
    if a.sweep == "full":
        cases = [
            (e, h, n, fmt, distribution)
            for e in [1, 32, 256]
            for h in [128, 256, 2048, 4096, 7168]
            for n in [1, 16, 64, 256]
            for fmt in ["float", "ceil", "packed"]
            for distribution in ["uniform", "skew"]
        ]
    # Rotate traversal order to avoid consistently favoring the same cases.
    if a.round % 2:
        cases.reverse()
    metadata = dict(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        library=a.library,
        sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),
        limit=a.limit,
        round=a.round,
        cold_l2=True,
        cuda_graph=True,
        sweep=a.sweep,
        capacity=a.capacity or "max(n,16)",
        shape_count=len(cases),
    )
    with Path(a.output).open("x") as out:
        out.write(json.dumps({"metadata": metadata}) + "\n")
        for e, h, n, fmt, distribution in cases:
            capacity = max(n, a.capacity or 16)
            y = torch.empty(
                e, capacity, h * 2, device="cuda", dtype=torch.bfloat16
            ).uniform_(-30, 30)
            q = torch.empty(e, capacity, h, device="cuda", dtype=torch.float8_e4m3fn)
            groups = h // 128
            packed = fmt == "packed"
            g = (groups + 3) // 4 if packed else groups
            s = torch.empty_strided(
                (e, capacity, g),
                (capacity * g, 1, capacity),
                dtype=torch.int32 if packed else torch.float32,
                device="cuda",
            )
            counts = torch.full((e,), n, device="cuda", dtype=torch.int32)
            if distribution == "skew" and e > 1:
                counts[::3] = 0
                counts[1::3] = min(n, 1)

            args = (y, counts, q, s, fmt != "float")
            if has_clamp:
                args += (a.limit,)
            invoke = partial(op, *args)

            invoke()
            gate, up = y[:, :n, :h], y[:, :n, h:]
            if a.limit is not None:
                gate, up = gate.clamp(max=a.limit), up.clamp(-a.limit, a.limit)
            expected = (torch.nn.functional.silu(gate.float()).bfloat16() * up).float()
            if packed:
                scale = (
                    (
                        s[:, :n]
                        .reshape(-1)
                        .clone(memory_format=torch.contiguous_format)
                        .view(torch.uint8)
                        .view(e, n, g * 4)
                        .int()
                        << 23
                    ).view(torch.float32)
                )[..., :groups]
            else:
                scale = s[:, :n]
            actual = q[:, :n].float() * scale.repeat_interleave(128, dim=-1)
            valid = torch.arange(n, device="cuda")[None, :] < counts[:, None]
            norm = (
                expected.reshape(e, n, groups, 128)
                .abs()
                .amax(-1)
                .repeat_interleave(128, -1)
                .clamp_min(1e-10)
            )
            err = ((actual - expected).abs() / norm)[valid].max().item()
            assert err < 0.075, (e, h, n, fmt, err)
            for _ in range(25):
                invoke()
            times = bench_gpu_time_with_cupti(
                invoke,
                use_cuda_graph=True,
                cold_l2_cache=True,
                dry_run_time_ms=25,
                repeat_time_ms=100,
            )
            us = statistics.median(times) * 1000
            tokens = int(counts.sum())
            row = dict(
                E=e,
                H=h,
                n=n,
                format=fmt,
                distribution=distribution,
                q_sha256=hashlib.sha256(
                    q[:, :n].float()[valid].cpu().numpy().tobytes()
                ).hexdigest(),
                s_sha256=hashlib.sha256(
                    scale[valid].cpu().numpy().tobytes()
                ).hexdigest(),
                capacity=capacity,
                us=us,
                gbps=(tokens * (h * 5 + g * 4) + e * 4) / (us * 1000),
                max_normalized_error=err,
            )
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(json.dumps(row), flush=True)
            del (
                invoke,
                args,
                y,
                q,
                s,
                counts,
                gate,
                up,
                expected,
                scale,
                actual,
                valid,
                norm,
            )


if __name__ == "__main__":
    main()
