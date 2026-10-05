#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark for the ViT prefill attention backends at a parametrizable
shape, default = Gemma3-4B SigLIP ViT.

Calls `triton.testing.do_bench`, which clears the L2 cache before every
measurement iteration to approximate the kernel's cold-cache behavior inside
a real transformer block (where surrounding qkv-proj / MLP / norm work
evicts q/k/v/output between layer calls).

Per-call shape (one SigLIP layer of Gemma3): B=1, S=4096, num_q_heads=
num_kv_heads=16, head_dim=72, dtype=bf16, is_causal=False. 27 layers / image.

--backend selects the attention implementation, matching the values accepted
by --mm-encoder-attn-backend in vllm serve (TRITON_ATTN, TORCH_SDPA,
FLASH_ATTN, ROCM_AITER_FA).  The default "all" runs every backend in sequence
and emits one JSON line per backend.

TRITON_ATTN launches _fwd_kernel with the configuration context_attention_fwd
picks on this device. The Triton tuning knobs (--bm/--bn/--nw/--ns/--we)
override that configuration and are ignored for other backends.

Output: one JSON line per backend to stdout.
"""

import argparse
import json
import sys

import torch

from vllm.triton_utils import triton
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.ops import triton_prefill_attention as prefill_ops
from vllm.v1.attention.ops.triton_prefill_attention import _fwd_kernel

_SUPPORTED_BACKENDS = [
    "TRITON_ATTN",
    "TORCH_SDPA",
    "FLASH_ATTN",
    "ROCM_AITER_FA",
]


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq", type=int, default=4096)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--head-dim", type=int, default=72)

    p.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    p.add_argument(
        "--backend",
        default="all",
        choices=_SUPPORTED_BACKENDS + ["all"],
        help=(
            "Attention backend (same values as --mm-encoder-attn-backend), "
            "or 'all' to run every backend in sequence (default)"
        ),
    )
    # TRITON_ATTN-only tuning knobs
    p.add_argument("--bm", type=int, default=None, help="BLOCK_M (TRITON_ATTN only)")
    p.add_argument("--bn", type=int, default=None, help="BLOCK_N (TRITON_ATTN only)")
    p.add_argument("--nw", type=int, default=None, help="num_warps (TRITON_ATTN only)")
    p.add_argument("--ns", type=int, default=None, help="num_stages (TRITON_ATTN only)")
    p.add_argument(
        "--we", type=int, default=None, help="waves_per_eu (TRITON_ATTN only)"
    )
    p.add_argument("--warmup-ms", type=int, default=200)
    p.add_argument("--rep-ms", type=int, default=600)
    return p.parse_args()


class _LaunchCapture:
    """Records the launch in place of ``_fwd_kernel``."""

    args: tuple
    kwargs: dict

    def __getitem__(self, grid):
        return self._record

    def _record(self, *args, **kwargs) -> None:
        self.args, self.kwargs = args, kwargs


def _bench_one(
    backend_name: str,
    args: argparse.Namespace,
    dtype: torch.dtype,
    q4: torch.Tensor,
    k4: torch.Tensor,
    v4: torch.Tensor,
    cu: torch.Tensor,
    max_seqlen: torch.Tensor,
) -> None:
    """Benchmark one backend and write a JSON result line to stdout."""
    B, S, H, D = args.batch, args.seq, args.heads, args.head_dim
    backend = AttentionBackendEnum[backend_name]
    scale = 1.0 / (D**0.5)

    if backend == AttentionBackendEnum.TRITON_ATTN:
        # Flat (B*S, H, D) layout expected by _fwd_kernel
        q = q4.view(B * S, H, D)
        k = k4.view(B * S, H, D)
        v = v4.view(B * S, H, D)
        o = torch.empty_like(q)
        seqlen = cu[1:] - cu[:-1]

        # Take the launch context_attention_fwd would make, so the defaults
        # are whatever vLLM runs on this device; the knobs override it.
        launch = _LaunchCapture()
        prefill_ops._fwd_kernel = launch
        try:
            prefill_ops.context_attention_fwd(
                q, k, v, o, cu[:-1], seqlen, S, is_causal=False, softmax_scale=scale
            )
        finally:
            prefill_ops._fwd_kernel = _fwd_kernel

        kwargs = launch.kwargs
        for knob, key in (
            (args.bm, "BLOCK_M"),
            (args.bn, "BLOCK_N"),
            (args.nw, "num_warps"),
            (args.ns, "num_stages"),
            (args.we, "waves_per_eu"),
        ):
            if knob is not None:
                kwargs[key] = knob
        grid = (B, H, triton.cdiv(S, kwargs["BLOCK_M"]))

        def _fn():
            _fwd_kernel[grid](*launch.args, **kwargs)

    elif backend in (
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.ROCM_AITER_FA,
    ):
        from vllm.v1.attention.backends.fa_utils import (
            get_flash_attn_version,  # noqa: E402
        )
        from vllm.v1.attention.ops.vit_attn_wrappers import (
            vit_flash_attn_wrapper,  # noqa: E402
        )

        fa_version = get_flash_attn_version(head_size=D)
        is_rocm_aiter = backend == AttentionBackendEnum.ROCM_AITER_FA

        def _fn():
            vit_flash_attn_wrapper(
                q=q4,
                k=k4,
                v=v4,
                batch_size=B,
                is_rocm_aiter=is_rocm_aiter,
                fa_version=fa_version,
                scale=scale,
                cu_seqlens=cu,
                max_seqlen=max_seqlen,
            )

    elif backend == AttentionBackendEnum.TORCH_SDPA:
        from vllm.v1.attention.ops.vit_attn_wrappers import (
            vit_torch_sdpa_wrapper,  # noqa: E402
        )

        def _fn():
            vit_torch_sdpa_wrapper(
                q=q4,
                k=k4,
                v=v4,
                scale=scale,
                cu_seqlens=cu,
            )

    else:
        raise ValueError(f"Unsupported backend: {backend_name}")

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

    config: dict = {
        "B": B,
        "S": S,
        "H": H,
        "D": D,
        "dtype": args.dtype,
        "backend": backend_name,
    }
    if backend == AttentionBackendEnum.TRITON_ATTN:
        config.update(
            {
                key: kwargs.get(key)
                for key in (
                    "BLOCK_M",
                    "BLOCK_DMODEL",
                    "BLOCK_DMODEL_TAIL",
                    "BLOCK_N",
                    "HEAD_STRIDE_ALIGNED_8",
                    "num_warps",
                    "num_stages",
                    "waves_per_eu",
                )
            }
        )

    result = {
        "per_call_ms_mean": mean,
        "per_call_ms_median": median,
        "per_call_ms_min": fastest,
        "per_call_ms_p10": p10,
        "per_call_ms_p90": p90,
        "samples": n,
        "config": config,
    }
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")
    sys.stdout.flush()


def main() -> int:
    args = _parse()
    device = "cuda"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.dtype
    ]
    torch.manual_seed(0)

    B, S, H, D = args.batch, args.seq, args.heads, args.head_dim

    # Wrappers (FLASH_ATTN, ROCM_AITER_FA, TRITON_ATTN) expect (B, S, H, D).
    # TORCH_SDPA expects the same via vit_torch_sdpa_wrapper.
    q4 = torch.randn(B, S, H, D, dtype=dtype, device=device)
    k4 = torch.randn(B, S, H, D, dtype=dtype, device=device)
    v4 = torch.randn(B, S, H, D, dtype=dtype, device=device)

    # cu_seqlens / max_seqlen used by FA and Triton backends
    cu = torch.tensor([i * S for i in range(B + 1)], dtype=torch.int32, device=device)
    max_seqlen = torch.tensor(S, dtype=torch.int32)

    backends = _SUPPORTED_BACKENDS if args.backend == "all" else [args.backend]
    for backend_name in backends:
        try:
            _bench_one(backend_name, args, dtype, q4, k4, v4, cu, max_seqlen)
        except Exception as exc:
            if args.backend == "all":
                sys.stderr.write(f"[skip] {backend_name}: {exc}\n")
            else:
                raise

    return 0


if __name__ == "__main__":
    sys.exit(main())
