# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the HY V4 Triton iHC pre/post/head kernels against eager PyTorch
and torch.compile of the eager code."""

import os
import subprocess
from functools import partial
from importlib.metadata import version
from statistics import median

import torch
import torch.nn.functional as F

import vllm
from vllm.models.hy_v4.nvidia.triton_ihc import (
    triton_ihc_head,
    triton_ihc_post,
    triton_ihc_pre,
)
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser


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


def eager_head(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    hc_eps: float,
    norm_eps: float,
) -> torch.Tensor:
    num_tokens, hc_mult, hidden_size = x.shape
    x_flat = x.flatten(1).float()
    reciprocal_rms = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x_flat, weight) * reciprocal_rms
    pre = torch.sigmoid(mixes * scale + base) + hc_eps
    output = torch.sum(pre.unsqueeze(-1) * x.float(), dim=1)
    return output.to(x.dtype).reshape(num_tokens, hidden_size)


def eager_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
) -> torch.Tensor:
    return (post.float().unsqueeze(-1) * x.float().unsqueeze(-2) + residual.float()).to(
        x.dtype
    )


def _timer(method: str):
    if method == "cupti":
        from flashinfer.testing import bench_gpu_time_with_cupti

        return partial(
            bench_gpu_time_with_cupti,
            use_cuda_graph=True,
            cold_l2_cache=True,
        )
    if method == "cudagraph":
        from flashinfer.testing import bench_gpu_time_with_cudagraph

        return partial(bench_gpu_time_with_cudagraph, cold_l2_cache=True)
    raise ValueError(f"unknown timing method: {method}")


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
    head_weight = weight[:hc_mult].contiguous()
    head_scale, head_base = scale[:1].contiguous(), base[:hc_mult].contiguous()
    timer = _timer(method)
    compiled_pre = torch.compile(eager_pre)
    compiled_post = torch.compile(eager_post)
    compiled_head = torch.compile(eager_head)

    properties = torch.cuda.get_device_properties(device)
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
    print(f"device: {properties.name}")
    print(f"branch: {git_branch}; commit: {git_commit}")
    print(
        f"vllm: {vllm.__version__}; torch: {torch.__version__}; "
        f"CUDA: {torch.version.cuda}"
    )
    print(
        f"triton: {triton.__version__}; flashinfer: {version('flashinfer-python')}; "
        f"dtype: {dtype}; method: {method}; cache: cold L2"
    )
    print(
        "env: VLLM_ENABLE_HPC_OPS="
        f"{os.getenv('VLLM_ENABLE_HPC_OPS', '<unset>')}; "
        "VLLM_BATCH_INVARIANT="
        f"{os.getenv('VLLM_BATCH_INVARIANT', '<unset>')}"
    )
    print(f"hidden_size: {hidden_size}; hc_mult: {hc_mult}")
    print(
        f"{'tokens':>8} {'op':>6} {'eager (us)':>12} {'compile (us)':>13} "
        f"{'triton (us)':>12} {'x eager':>9} {'x compile':>10} {'GiB':>8} "
        f"{'triton GB/s':>13}"
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
        head_args = (x, head_weight, head_scale, head_base, 1e-6, 1e-5)
        torch.testing.assert_close(
            triton_ihc_head(*head_args),
            eager_head(*head_args),
            atol=2e-2,
            rtol=1e-2,
        )

        benchmarks = (
            (
                "pre",
                partial(eager_pre, x, weight, scale, base, 2.0, 1e-6, 1e-5),
                partial(compiled_pre, x, weight, scale, base, 2.0, 1e-6, 1e-5),
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
                partial(compiled_post, block_output, residual, post),
                partial(triton_ihc_post, block_output, residual, post),
                block_output.nbytes + residual.nbytes + post.nbytes + residual.nbytes,
            ),
            (
                "head",
                partial(eager_head, *head_args),
                partial(compiled_head, *head_args),
                partial(triton_ihc_head, *head_args),
                x.nbytes
                + head_weight.nbytes
                + head_scale.nbytes
                + head_base.nbytes
                + eager_output.nbytes,
            ),
        )
        for op_name, eager_fn, compiled_fn, triton_fn, logical_bytes in benchmarks:
            eager_us = median(timer(eager_fn)) * 1e3
            compile_us = median(timer(compiled_fn)) * 1e3
            triton_us = median(timer(triton_fn)) * 1e3
            logical_gib = logical_bytes / 2**30
            triton_gbps = logical_bytes / triton_us / 1e3
            print(
                f"{num_tokens:>8} {op_name:>6} {eager_us:>12.1f} {compile_us:>13.1f} "
                f"{triton_us:>12.1f} {eager_us / triton_us:>8.2f}x "
                f"{compile_us / triton_us:>9.2f}x {logical_gib:>8.3f} "
                f"{triton_gbps:>13.1f}"
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
