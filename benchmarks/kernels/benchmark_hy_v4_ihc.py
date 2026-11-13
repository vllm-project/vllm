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
        triton_ihc_post_pre_rms_norm,
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


def eager_post_pre_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    attn_post: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_weight: torch.Tensor,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y = eager_post(x, residual, attn_post)
    output, post = eager_pre(y, weight, scale, base, magnitude, hc_eps, norm_eps)
    output_float = output.float()
    output = (
        output_float
        * torch.rsqrt(output_float.square().mean(-1, keepdim=True) + norm_eps)
        * norm_weight
    ).to(output.dtype)
    return output, post, y


def _timer(method: str):
    """Return the explicitly requested GPU timer or fail if unavailable."""
    if method == "torch_event":
        return _torch_event_timer

    if current_platform.is_rocm():
        raise RuntimeError(
            f"timing method {method!r} is CUDA-only; use --method torch_event on ROCm"
        )

    if method == "cupti":
        try:
            from flashinfer.testing import bench_gpu_time_with_cupti
        except ImportError as error:
            raise RuntimeError(
                "timing method 'cupti' requires flashinfer-python with "
                "bench_gpu_time_with_cupti"
            ) from error
        return partial(
            bench_gpu_time_with_cupti,
            use_cuda_graph=True,
            cold_l2_cache=True,
        )
    if method == "cudagraph":
        try:
            from flashinfer.testing import bench_gpu_time_with_cudagraph
        except ImportError as error:
            raise RuntimeError(
                "timing method 'cudagraph' requires flashinfer-python with "
                "bench_gpu_time_with_cudagraph"
            ) from error
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


def _git_metadata(arguments: list[str], *, empty: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return "<unknown>"
    return result.stdout.strip() or empty


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
    norm_weight = torch.randn(hidden_size, device=device, dtype=torch.float32)
    timer = _timer(method)
    properties = torch.cuda.get_device_properties(device)
    git_branch = _git_metadata(["branch", "--show-current"], empty="<detached>")
    git_commit = _git_metadata(["rev-parse", "--short", "HEAD"], empty="<unknown>")
    device_name = (
        current_platform.get_device_name()
        if current_platform.is_rocm()
        else properties.name
    )
    platform_description = (
        f"HIP: {torch.version.hip}"
        if current_platform.is_rocm()
        else f"CUDA: {torch.version.cuda}"
    )
    print(f"device: {device_name}")
    print(f"branch: {git_branch}; commit: {git_commit}")
    print(
        f"vllm: {vllm.__version__}; torch: {torch.__version__}; {platform_description}"
    )
    if method == "torch_event":
        flashinfer_version = "not used"
        event_backend = "HIP" if current_platform.is_rocm() else "CUDA"
        time_description = f"method: PyTorch {event_backend} events; cache: warm"
    else:
        try:
            flashinfer_version = version("flashinfer-python")
        except PackageNotFoundError:
            flashinfer_version = "unavailable"
        time_description = f"method: {method}; cache: cold L2"
    print(
        f"triton: {triton.__version__}; flashinfer: {flashinfer_version}; "
        f"dtype: {dtype}; {time_description}"
    )
    print(
        "env: VLLM_ENABLE_HPC_OPS="
        f"{os.getenv('VLLM_ENABLE_HPC_OPS', '<unset>')}; "
        "VLLM_BATCH_INVARIANT="
        f"{os.getenv('VLLM_BATCH_INVARIANT', '<unset>')}"
    )
    print(f"hidden_size: {hidden_size}; hc_mult: {hc_mult}")
    print(
        f"{'tokens':>8} {'op':>6} {'eager (us)':>12} "
        f"{'triton (us)':>12} {'speedup':>9} {'GiB':>8} "
        f"{'eager GB/s':>12} {'triton GB/s':>13}"
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
        benchmarks = [
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
        ]
        if current_platform.is_rocm():
            fused_expected = eager_post_pre_rms_norm(
                block_output,
                residual,
                post,
                weight,
                scale,
                base,
                norm_weight,
                2.0,
                1e-6,
                1e-5,
            )
            fused_actual = triton_ihc_post_pre_rms_norm(
                block_output,
                residual,
                post,
                weight,
                scale,
                base,
                norm_weight,
                2.0,
                1e-6,
                1e-5,
            )
            for actual, expected in zip(fused_actual, fused_expected):
                torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
            benchmarks.append(
                (
                    "fused",
                    partial(
                        eager_post_pre_rms_norm,
                        block_output,
                        residual,
                        post,
                        weight,
                        scale,
                        base,
                        norm_weight,
                        2.0,
                        1e-6,
                        1e-5,
                    ),
                    partial(
                        triton_ihc_post_pre_rms_norm,
                        block_output,
                        residual,
                        post,
                        weight,
                        scale,
                        base,
                        norm_weight,
                        2.0,
                        1e-6,
                        1e-5,
                    ),
                    block_output.nbytes
                    + residual.nbytes
                    + post.nbytes
                    + weight.nbytes
                    + scale.nbytes
                    + base.nbytes
                    + norm_weight.nbytes
                    + sum(tensor.nbytes for tensor in fused_expected),
                )
            )
        for op_name, eager_fn, triton_fn, logical_bytes in benchmarks:
            eager_us = median(timer(eager_fn)) * 1e3
            triton_us = median(timer(triton_fn)) * 1e3
            speedup = eager_us / triton_us
            logical_gib = logical_bytes / 2**30
            eager_gbps = logical_bytes / eager_us / 1e3
            triton_gbps = logical_bytes / triton_us / 1e3
            print(
                f"{num_tokens:>8} {op_name:>6} {eager_us:>12.1f} "
                f"{triton_us:>12.1f} {speedup:>8.2f}x {logical_gib:>8.3f} "
                f"{eager_gbps:>12.1f} {triton_gbps:>13.1f}"
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
    parser.add_argument(
        "--method",
        choices=["torch_event", "cupti", "cudagraph"],
        default="torch_event" if current_platform.is_rocm() else "cupti",
        help=(
            "timing backend; cupti/cudagraph require flashinfer-python on CUDA, "
            "while torch_event uses warm-cache PyTorch GPU events"
        ),
    )
    args = parser.parse_args()
    run_benchmark(
        args.token_counts,
        args.hidden_size,
        getattr(torch, args.dtype),
        args.method,
    )
