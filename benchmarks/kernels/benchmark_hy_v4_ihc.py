# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark HY V4 Triton iHC kernels on CUDA or ROCm against eager PyTorch."""

import os
import subprocess
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from statistics import median

import torch
import torch.nn.functional as F

import vllm
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser

if current_platform.is_rocm():
    from vllm.models.hy_v4.amd.triton_ihc import (
        triton_ihc_post,
        triton_ihc_pre,
    )
else:
    from vllm.models.hy_v4.nvidia.triton_ihc import (
        triton_ihc_post,
        triton_ihc_pre,
    )


def eager_pre(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, hc_mult, hidden_size = x.shape
    x_flat = x.flatten(1).float()
    reciprocal_rms = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x_flat, weight) * reciprocal_rms
    pre = torch.sigmoid(mixes[:, :hc_mult] * scale[0] + base[:hc_mult])
    post = torch.sigmoid(mixes[:, hc_mult:] * scale[1] + base[hc_mult:])
    pre = pre + hc_eps
    post = magnitude * post + hc_eps
    output = torch.sum(pre.unsqueeze(-1) * x.float(), dim=1)
    return output.to(x.dtype).reshape(num_tokens, hidden_size), post


def eager_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
) -> torch.Tensor:
    return (post.float().unsqueeze(-1) * x.float().unsqueeze(-2) + residual.float()).to(
        x.dtype
    )


def _timer(method: str):
    """Return a platform-appropriate GPU timer."""
    if current_platform.is_rocm():
        return _torch_event_timer

    if method == "cupti":
        try:
            from flashinfer.testing import bench_gpu_time_with_cupti
        except ImportError:
            return _torch_event_timer
        return partial(
            bench_gpu_time_with_cupti,
            use_cuda_graph=True,
            cold_l2_cache=True,
        )
    if method == "cudagraph":
        try:
            from flashinfer.testing import bench_gpu_time_with_cudagraph
        except ImportError:
            return _torch_event_timer
        return partial(bench_gpu_time_with_cudagraph, cold_l2_cache=True)
    raise ValueError(f"unknown timing method: {method}")


def _torch_event_timer(fn):
    for _ in range(10):
        fn()
    torch.accelerator.synchronize()
    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)
    start.record()
    for _ in range(20):
        fn()
    end.record()
    end.synchronize()
    return [start.elapsed_time(end) / 20]


@torch.inference_mode()
def run_benchmark(
    token_counts: list[int],
    hidden_size: int,
    dtype: torch.dtype,
    method: str,
) -> None:
    hc_mult = 4
    device = torch.device("cuda")
    torch.manual_seed(0)
    weight = torch.randn(
        2 * hc_mult,
        hc_mult * hidden_size,
        device=device,
        dtype=torch.float32,
    )
    scale = torch.randn(2, device=device, dtype=torch.float32) * 0.01
    base = torch.randn(2 * hc_mult, device=device, dtype=torch.float32)
    timer = _timer(method)
    git_branch = subprocess.run(
        ["git", "branch", "--show-current"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(f"device: {current_platform.get_device_name()}")
    print(f"branch: {git_branch}; commit: {git_commit}")
    print(
        f"vllm: {vllm.__version__}; torch: {torch.__version__}; "
        f"platform: {'ROCm' if current_platform.is_rocm() else 'CUDA'}"
    )
    if current_platform.is_rocm():
        flashinfer_version = "not used"
        timing_description = "PyTorch CUDA/HIP events"
    else:
        try:
            flashinfer_version = version("flashinfer-python")
        except PackageNotFoundError:
            flashinfer_version = "unavailable"
        timing_description = (
            f"{method}; FlashInfer timing when available, PyTorch events otherwise"
        )
    print(
        f"triton: {triton.__version__}; flashinfer: {flashinfer_version}; "
        f"dtype: {dtype}; timing: {timing_description}"
    )
    print(
        "env: VLLM_ENABLE_HPC_OPS="
        f"{os.getenv('VLLM_ENABLE_HPC_OPS', '<unset>')}; "
        "VLLM_BATCH_INVARIANT="
        f"{os.getenv('VLLM_BATCH_INVARIANT', '<unset>')}"
    )
    print(f"hidden_size: {hidden_size}; hc_mult: {hc_mult}")
    print(
        f"{'tokens':>8} {'op':>10} {'reference (us)':>15} "
        f"{'candidate (us)':>15} {'speedup':>9} {'GiB':>8} "
        f"{'reference GB/s':>15} {'candidate GB/s':>16}"
    )

    for num_tokens in token_counts:
        x = torch.randn(
            num_tokens,
            hc_mult,
            hidden_size,
            device=device,
            dtype=dtype,
        )
        block_output = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
        residual = torch.randn_like(x)
        eager_output, post = eager_pre(x, weight, scale, base, 2.0, 1e-6, 1e-5)
        triton_output, triton_post = triton_ihc_pre(
            x, weight, scale, base, 2.0, 1e-6, 1e-5
        )
        torch.testing.assert_close(triton_output, eager_output, atol=2e-2, rtol=1e-2)
        torch.testing.assert_close(triton_post, post, atol=2e-5, rtol=1e-5)
        torch.testing.assert_close(
            triton_ihc_post(block_output, residual, post),
            eager_post(block_output, residual, post),
            atol=0,
            rtol=0,
        )

        benchmarks = (
            (
                "pre",
                partial(eager_pre, x, weight, scale, base, 2.0, 1e-6, 1e-5),
                partial(triton_ihc_pre, x, weight, scale, base, 2.0, 1e-6, 1e-5),
                x.nbytes
                + weight.nbytes
                + scale.nbytes
                + base.nbytes
                + eager_output.nbytes
                + post.nbytes,
            ),
            (
                "post",
                partial(eager_post, block_output, residual, post),
                partial(triton_ihc_post, block_output, residual, post),
                block_output.nbytes + residual.nbytes + post.nbytes + residual.nbytes,
            ),
        )
        for op_name, reference_fn, candidate_fn, logical_bytes in benchmarks:
            reference_us = median(timer(reference_fn)) * 1e3
            candidate_us = median(timer(candidate_fn)) * 1e3
            speedup = reference_us / candidate_us
            logical_gib = logical_bytes / 2**30
            reference_gbps = logical_bytes / reference_us / 1e3
            candidate_gbps = logical_bytes / candidate_us / 1e3
            print(
                f"{num_tokens:>8} {op_name:>10} {reference_us:>15.1f} "
                f"{candidate_us:>15.1f} {speedup:>8.2f}x {logical_gib:>8.3f} "
                f"{reference_gbps:>15.1f} {candidate_gbps:>16.1f}"
            )


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument(
        "--token-counts",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 64, 256, 1024, 4096, 8192],
    )
    parser.add_argument("--hidden-size", type=int, choices=[4096, 6144], default=6144)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--method", choices=["cupti", "cudagraph"], default="cupti")
    args = parser.parse_args()
    run_benchmark(
        args.token_counts,
        args.hidden_size,
        getattr(torch, args.dtype),
        args.method,
    )
