# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark HY V4 iHC pre / post / head: eager torch vs torch.compile vs the
fused Triton ops in ``vllm.models.hy_v4.nvidia.ops.ihc``.

Timing follows the kernel-microbenchmark skill: FlashInfer CUPTI with CUDA
graph replay and cold L2 when available, Triton's ``do_bench_cudagraph``
otherwise. Correctness against the eager reference is asserted before timing.
"""

import functools
import math
import statistics

import torch
import torch.nn.functional as F
from tabulate import tabulate

from vllm.models.hy_v4.nvidia.ops.ihc import ihc_head, ihc_post, ihc_pre
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, set_random_seed

HC = 4
NORM_EPS = 1e-5
HC_EPS = 1e-6
MAGNITUDE = 2.0


def eager_pre(x, weight, hc_scale, hc_base):
    shape, hc = x.size(), x.shape[1]
    x_flat = x.flatten(1).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + NORM_EPS)
    mixes = F.linear(x_flat, weight) * rsqrt
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + HC_EPS
    post = (
        MAGNITUDE * torch.sigmoid(mixes[..., hc:] * hc_scale[1] + hc_base[hc:]) + HC_EPS
    )
    y = torch.sum(pre.unsqueeze(-1) * x.reshape(shape), dim=1)
    return y.to(x.dtype), post


def eager_head(x, weight, hc_scale, hc_base):
    shape, x_dtype = x.size(), x.dtype
    xf = x.flatten(1).float()
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + NORM_EPS)
    mixes = F.linear(xf, weight) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + HC_EPS
    return torch.sum(pre.unsqueeze(-1) * xf.reshape(shape), dim=1).to(x_dtype)


def eager_post(x, residual, post):
    return (post.float().unsqueeze(-1) * x.float().unsqueeze(-2) + residual.float()).to(
        x.dtype
    )


def make_inputs(op: str, tokens: int, hidden: int, dtype: torch.dtype):
    set_random_seed(0)
    dev = "cuda"
    if op == "post":
        x = torch.randn(tokens, hidden, dtype=dtype, device=dev)
        residual = torch.randn(tokens, HC, hidden, dtype=dtype, device=dev)
        post = torch.rand(tokens, HC, dtype=torch.float32, device=dev) * MAGNITUDE
        return (x, residual, post)
    n_out = 2 * HC if op == "pre" else HC
    x = torch.randn(tokens, HC, hidden, dtype=dtype, device=dev)
    weight = torch.randn(n_out, HC * hidden, dtype=torch.float32, device=dev) * 6e-3
    scale = torch.full((2 if op == "pre" else 1,), 0.5, device=dev)
    base = torch.full((n_out,), -math.log(HC - 1.0), device=dev)
    base += torch.randn_like(base) * 0.3
    return (x, weight, scale, base)


def fused(op: str, args):
    if op == "pre":
        return ihc_pre(*args, NORM_EPS, HC_EPS, MAGNITUDE)
    if op == "head":
        return ihc_head(*args, NORM_EPS, HC_EPS)
    return ihc_post(*args)


EAGER = {"pre": eager_pre, "head": eager_head, "post": eager_post}


def time_us(fn) -> float:
    for _ in range(10):
        fn()
    torch.accelerator.synchronize()
    try:
        from flashinfer.testing import bench_gpu_time_with_cupti

        ms = bench_gpu_time_with_cupti(fn, use_cuda_graph=True, cold_l2_cache=True)
        return statistics.median(ms) * 1e3
    except ImportError:
        from triton.testing import do_bench_cudagraph

        return do_bench_cudagraph(fn) * 1e3


def bytes_moved(op: str, tokens: int, hidden: int, dtype: torch.dtype) -> int:
    es = torch.tensor([], dtype=dtype).element_size()
    if op == "post":
        return (tokens * hidden + 2 * tokens * HC * hidden) * es
    n_out = 2 * HC if op == "pre" else HC
    return (tokens * HC * hidden + tokens * hidden) * es + n_out * HC * hidden * 4


def main(args) -> None:
    dtype = STR_DTYPE_TO_TORCH_DTYPE[args.dtype]
    rows = []
    for op in args.ops:
        eager = EAGER[op]
        compiled = torch.compile(eager)
        for hidden in args.hidden_sizes:
            for tokens in args.tokens:
                inputs = make_inputs(op, tokens, hidden, dtype)
                ref = eager(*inputs)
                out = fused(op, inputs)
                if isinstance(ref, tuple):
                    torch.testing.assert_close(out[1], ref[1], atol=1e-3, rtol=1e-3)
                    ref, out = ref[0], out[0]
                torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)

                t_eager = time_us(functools.partial(eager, *inputs))
                t_comp = time_us(functools.partial(compiled, *inputs))
                t_fused = time_us(functools.partial(fused, op, inputs))
                gbs = bytes_moved(op, tokens, hidden, dtype) / (t_fused * 1e-6) / 1e9
                rows.append(
                    [
                        op,
                        hidden,
                        tokens,
                        t_eager,
                        t_comp,
                        t_fused,
                        t_eager / t_fused,
                        t_comp / t_fused,
                        gbs,
                    ]
                )
    print(
        f"GPU: {current_platform.get_device_name()}  dtype: {args.dtype}  "
        "(median us, CUDA graph, cold L2)"
    )
    print(
        tabulate(
            rows,
            headers=[
                "op",
                "hidden",
                "tokens",
                "eager us",
                "compile us",
                "triton us",
                "x eager",
                "x compile",
                "GB/s",
            ],
            floatfmt=(None, None, None, ".2f", ".2f", ".2f", ".2f", ".2f", ".0f"),
        )
    )


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark HY V4 fused iHC Triton ops.")
    parser.add_argument(
        "--ops",
        nargs="+",
        default=["pre", "post", "head"],
        choices=["pre", "post", "head"],
    )
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64, 256, 1024, 4096]
    )
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[4096, 6144])
    parser.add_argument("--dtype", choices=["bfloat16", "half"], default="bfloat16")
    main(parser.parse_args())
