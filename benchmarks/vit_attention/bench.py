# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark for the ViT TRITON_ATTN kernel (_fwd_kernel from
vllm.v1.attention.ops.triton_prefill_attention) at a parametrizable shape,
default = Gemma3-4B SigLIP ViT.

Calls `triton.testing.do_bench`, which clears the L2 cache before every
measurement iteration to approximate the kernel's cold-cache behavior inside
a real transformer block (where surrounding qkv-proj / MLP / norm work
evicts q/k/v/output between layer calls).

Per-call shape (one SigLIP layer of Gemma3): B=1, S=4096, num_q_heads=
num_kv_heads=16, head_dim=72, dtype=bf16, is_causal=False. 27 layers / image.

Tuning knobs are passed as CLI flags (NOT env vars) and forwarded directly
to the kernel launch — production code remains unchanged.

Output: JSON line to stdout.
"""

import argparse
import json
import os
import site
import sys

# Without amd_smi importable, vllm.platforms falls back to UnspecifiedPlatform
# and get_block_size returns the wrong default for ROCm (BLOCK_M=64 instead
# of the cuda_alike+capability(80) branch's 128). The TheRock ROCm SDK ships
# the amd_smi Python package under a non-standard share/ path; add it to
# sys.path if present, BEFORE importing anything from vllm.
for _site in site.getsitepackages():
    _amdsmi = os.path.join(_site, "_rocm_sdk_core", "share", "amd_smi")
    if os.path.isdir(_amdsmi) and _amdsmi not in sys.path:
        sys.path.insert(0, _amdsmi)
        break

import torch  # noqa: E402

from vllm.triton_utils import triton  # noqa: E402
from vllm.utils.math_utils import RCP_LN2  # noqa: E402
from vllm.v1.attention.ops.triton_prefill_attention import (  # noqa: E402
    _fwd_kernel,
    get_block_size,
    get_num_warps,
)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq", type=int, default=4096)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--head-dim", type=int, default=72)
    p.add_argument("--num-layers", type=int, default=27)
    p.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    p.add_argument(
        "--bm", type=int, default=None, help="BLOCK_M (default: get_block_size)"
    )
    p.add_argument("--bn", type=int, default=None, help="BLOCK_N (default: BLOCK_M)")
    p.add_argument(
        "--nw", type=int, default=None, help="num_warps (default: get_num_warps)"
    )
    p.add_argument("--ns", type=int, default=1, help="num_stages")
    p.add_argument("--we", type=int, default=None, help="waves_per_eu (default: none)")
    p.add_argument("--warmup-ms", type=int, default=200)
    p.add_argument("--rep-ms", type=int, default=600)
    return p.parse_args()


def main() -> int:
    args = _parse()
    device = "cuda"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.dtype
    ]
    torch.manual_seed(0)

    B, S, H, D = args.batch, args.seq, args.heads, args.head_dim
    q = torch.randn(B * S, H, D, dtype=dtype, device=device)
    k = torch.randn(B * S, H, D, dtype=dtype, device=device)
    v = torch.randn(B * S, H, D, dtype=dtype, device=device)
    o = torch.empty_like(q)
    cu = torch.tensor([i * S for i in range(B + 1)], dtype=torch.int32, device=device)
    seqlen = cu[1:] - cu[:-1]

    BLOCK_M = args.bm if args.bm is not None else get_block_size(dtype)
    BLOCK_N = args.bn if args.bn is not None else BLOCK_M
    num_warps = args.nw if args.nw is not None else get_num_warps(D)

    sm_scale = (1.0 / (D**0.5)) * RCP_LN2
    grid = (B, H, triton.cdiv(S, BLOCK_M))
    kv_group_num = H // H

    extra_kwargs: dict = {}
    if args.we is not None:
        extra_kwargs["waves_per_eu"] = args.we

    def _fn():
        _fwd_kernel[grid](
            q,
            k,
            v,
            sm_scale,
            cu[:-1],
            seqlen,
            o,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            v.stride(0),
            v.stride(1),
            o.stride(0),
            o.stride(1),
            kv_group_num=kv_group_num,
            BLOCK_M=BLOCK_M,
            BLOCK_DMODEL=triton.next_power_of_2(D),
            BLOCK_N=BLOCK_N,
            IS_CAUSAL=False,
            SLIDING_WINDOW_Q=0,
            SLIDING_WINDOW_K=0,
            num_warps=num_warps,
            num_stages=args.ns,
            Lk=D,
            **extra_kwargs,
        )

    # do_bench clears the L2 cache before each measurement iteration.
    all_times = triton.testing.do_bench(
        _fn,
        warmup=args.warmup_ms,
        rep=args.rep_ms,
        return_mode="all",
    )
    all_times = sorted(all_times)
    n = len(all_times)
    mean = sum(all_times) / n
    median = all_times[n // 2]
    p10 = all_times[max(0, int(n * 0.10))]
    p90 = all_times[min(n - 1, int(n * 0.90))]
    fastest = all_times[0]

    result = {
        "per_call_ms_mean": mean,
        "per_call_ms_median": median,
        "per_call_ms_min": fastest,
        "per_call_ms_p10": p10,
        "per_call_ms_p90": p90,
        "samples": n,
        "num_layers": args.num_layers,
        "total_per_image_ms_median": median * args.num_layers,
        "config": {
            "B": B,
            "S": S,
            "H": H,
            "D": D,
            "dtype": args.dtype,
            "BLOCK_M": BLOCK_M,
            "BLOCK_N": BLOCK_N,
            "num_warps": num_warps,
            "num_stages": args.ns,
            "waves_per_eu": args.we,
        },
    }
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
