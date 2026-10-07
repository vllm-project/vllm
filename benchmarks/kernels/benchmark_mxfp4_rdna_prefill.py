# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Break down the MXFP4 prefill (M > 8) linear time on RDNA.

For every MXFP4 linear shape in a checkpoint, times the pieces of the current
M > 8 path (weight dequant, bf16 GEMM) and the whole path, then sums over the
model's layers. ``gemm`` alone (bf16 weight already in memory) is the bound for
a kernel that dequantizes inside the GEMM. ``act_qdq`` is what the emulation
kernel additionally spends on W4A4 checkpoints; the fallback does not run it.

Usage:
    python benchmarks/kernels/benchmark_mxfp4_rdna_prefill.py \
        --model amd/Qwen3.8-27B-Quark-AWQ-MXFP4 --m 16 64 256 1024 2048
"""

import argparse
import collections
import glob
import json
import os

import torch
import torch.nn.functional as F

from vllm.model_executor.kernels.linear.mxfp4.rdna3_hip import (
    _rdna3_mxfp4_w4a8_apply_impl,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    dequant_mxfp4,
    quant_dequant_mxfp4,
)
from vllm.triton_utils import triton

PIECES = ["act_qdq", "dequant", "gemm", "fallback"]


def checkpoint_shapes(model: str) -> collections.Counter:
    """Count (N, K) of MXFP4 linears from the safetensors headers."""
    path = model
    if not os.path.isdir(path):
        from huggingface_hub import snapshot_download

        path = snapshot_download(model, allow_patterns=["*.safetensors", "*.json"])
    shapes: dict[str, list[int]] = {}
    for f in glob.glob(os.path.join(path, "*.safetensors")):
        with open(f, "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            header = json.loads(fh.read(n))
        for name, info in header.items():
            if name != "__metadata__":
                shapes[name] = info
    counts: collections.Counter = collections.Counter()
    for name, info in shapes.items():
        if not name.endswith(".weight") or "mtp" in name:
            continue
        if name + "_scale" in shapes and info["dtype"] == "U8":
            n_out, k_half = info["shape"]
            counts[(n_out, 2 * k_half)] += 1
    return counts


def bench(fn) -> float:
    return triton.testing.do_bench(fn, quantiles=[0.5])


def time_shape(N: int, K: int, M: int) -> dict[str, float]:
    dev, dt = "cuda", torch.bfloat16
    w = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
    s = torch.randint(118, 128, (N, K // 32), dtype=torch.uint8, device=dev)
    w_bf16 = dequant_mxfp4(w, s, dt)
    x = torch.randn(M, K, dtype=dt, device=dev)
    return {
        "act_qdq": bench(lambda: quant_dequant_mxfp4(x)),
        "dequant": bench(lambda: dequant_mxfp4(w, s, dt)),
        "gemm": bench(lambda: F.linear(x, w_bf16)),
        "fallback": bench(lambda: _rdna3_mxfp4_w4a8_apply_impl(x, w, s, None)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="amd/Qwen3.8-27B-Quark-AWQ-MXFP4")
    p.add_argument("--m", nargs="+", type=int, default=[16, 64, 256, 1024, 2048])
    args = p.parse_args()

    counts = checkpoint_shapes(args.model)
    if not counts:
        raise SystemExit("no MXFP4 linears found")
    print(torch.cuda.get_device_properties(0).gcnArchName, args.model)
    for (N, K), c in sorted(counts.items()):
        print(f"  N={N:>6} K={K:>6}  x{c}")

    for M in args.m:
        total = dict.fromkeys(PIECES, 0.0)
        print(f"\nM={M}  (ms per call; model = sum over all MXFP4 linears)")
        print(f"  {'N':>6} {'K':>6}  " + "".join(f"{k:>10}" for k in PIECES))
        for (N, K), c in sorted(counts.items()):
            t = time_shape(N, K, M)
            for k in PIECES:
                total[k] += c * t[k]
            print(f"  {N:>6} {K:>6}  " + "".join(f"{t[k]:>10.3f}" for k in PIECES))
            torch.cuda.empty_cache()
        print("  model         " + "".join(f"{total[k]:>10.1f}" for k in PIECES))
        f = total["fallback"]
        print(
            f"  fallback = {f:.1f} ms; gemm alone = {total['gemm']:.1f} ms "
            f"({total['gemm'] / f:.0%}); max gain from in-GEMM dequant "
            f"{f / total['gemm']:.2f}x"
        )


if __name__ == "__main__":
    main()
