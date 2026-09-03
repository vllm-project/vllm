# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph-timed Marlin small-M (decode) benchmark with a correctness gate.

Times `_C.marlin_gemm` on dense decode-shaped GEMMs by capturing N
back-to-back calls in one CUDA graph and timing graph replays with CUDA
events. Eager per-call timing is host-bound-flat at these sizes and is
inadmissible for kernel comparisons; it is reported only as a labeled
reference. The primary (cold) number rotates each call across enough
distinct weight clones to exceed L2, matching the production regime where
consecutive decode GEMMs read different layers' weights; a hot
(single-weight) number is also reported.

The default shapes are the dominant dense Marlin calls of
Nemotron-3.5-Lightning-30B-A3B-NVFP4 and its DSpark drafter (hidden 2688).
Each result is gated on the same mean-relative-diff-vs-dequantized-reference
tolerance as tests/kernels/quantization/test_marlin_gemm.py (< 0.04).

`--workspace-blocks-per-sm 1` shrinks the lock workspace so the kernel's
small-M wave-quantization launch (two blocks per SM) fails closed to the
single-wave launch, giving an in-binary A/B of that path.
"""

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    rand_marlin_weight_nvfp4_like,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    marlin_quant_fp8_torch,
)
from vllm.scalar_type import scalar_types
from vllm.utils.argparse_utils import FlexibleArgumentParser

# name, weight quant, group_size, K, N
SHAPES = [
    ("mamba_in_proj", "fp8", -1, 2688, 10304),
    ("mamba_out_proj", "fp8", -1, 4096, 2688),
    ("draft_gate_up", "nvfp4", 16, 2688, 12288),
    ("draft_down", "nvfp4", 16, 6144, 2688),
    ("shared_exp_up", "nvfp4", 16, 2688, 3712),
    ("shared_exp_down", "nvfp4", 16, 3712, 2688),
    ("lm_head", "nvfp4", 16, 2688, 131072),
]

WEIGHT_ROTATION_BYTES = 256 * 1024 * 1024


def compute_max_diff(output, output_ref):
    return (
        torch.mean(torch.abs(output - output_ref)) / torch.mean(torch.abs(output_ref))
    ).item()


def graph_time_us(fn, ncalls, nreplays):
    fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(ncalls):
            fn()
    for _ in range(3):
        graph.replay()
    torch.accelerator.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(nreplays):
        graph.replay()
    end.record()
    torch.accelerator.synchronize()
    return start.elapsed_time(end) * 1e3 / (ncalls * nreplays)


def eager_time_us(fn, iters=50):
    fn()
    torch.accelerator.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.accelerator.synchronize()
    return start.elapsed_time(end) * 1e3 / iters


def build_weights(wtype, group_size, size_k, size_n, device):
    b_weight = torch.randn((size_k, size_n), dtype=torch.bfloat16, device=device) / 10
    if wtype == "nvfp4":
        w_ref, qweight, scales, global_scale = rand_marlin_weight_nvfp4_like(
            b_weight.T, group_size
        )
    else:
        w_ref, qweight, scales = marlin_quant_fp8_torch(b_weight.T, group_size)
        global_scale = None
    wbytes = qweight.numel() * qweight.element_size()
    wbytes += scales.numel() * scales.element_size()
    num_clones = max(2, min(24, (WEIGHT_ROTATION_BYTES + wbytes - 1) // wbytes))
    qweights = [qweight] + [qweight.clone() for _ in range(num_clones - 1)]
    all_scales = [scales] + [scales.clone() for _ in range(num_clones - 1)]
    return w_ref, qweights, all_scales, global_scale


def make_fn(a, qweights, all_scales, global_scale, workspace, wtype, m, n, k, rotate):
    if wtype == "nvfp4":
        b_type = scalar_types.float4_e2m1f
    else:
        b_type = scalar_types.float8_e4m3fn
    num_weights = len(qweights) if rotate else 1
    state = {"i": 0}

    def fn():
        i = state["i"]
        state["i"] = (i + 1) % num_weights
        return ops.marlin_gemm(
            a,
            None,
            qweights[i],
            None,
            all_scales[i],
            None,
            global_scale,
            None,
            None,
            None,
            workspace,
            b_type,
            m,
            n,
            k,
            is_k_full=True,
            use_atomic_add=False,
            use_fp32_reduce=True,
            is_zp_float=False,
        )

    return fn


def main(args):
    torch.manual_seed(7)
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    print(
        f"device={torch.cuda.get_device_name(device)} "
        f"cc={torch.cuda.get_device_capability(device)} "
        f"sms={props.multi_processor_count} "
        f"workspace_blocks_per_sm={args.workspace_blocks_per_sm}"
    )
    workspace = marlin_make_workspace_new(device, args.workspace_blocks_per_sm)

    failures = 0
    for name, wtype, group_size, size_k, size_n in SHAPES:
        w_ref, qweights, all_scales, global_scale = build_weights(
            wtype, group_size, size_k, size_n, device
        )
        for m in args.batch_sizes:
            a = torch.randn((m, size_k), dtype=torch.bfloat16, device=device) / 10
            fn_cold = make_fn(
                a,
                qweights,
                all_scales,
                global_scale,
                workspace,
                wtype,
                m,
                size_n,
                size_k,
                rotate=True,
            )
            fn_hot = make_fn(
                a,
                qweights,
                all_scales,
                global_scale,
                workspace,
                wtype,
                m,
                size_n,
                size_k,
                rotate=False,
            )
            max_diff = compute_max_diff(fn_cold(), torch.matmul(a, w_ref))
            torch.accelerator.synchronize()
            ok = max_diff < 0.04
            failures += not ok
            cold_us = graph_time_us(fn_cold, args.ncalls, args.nreplays)
            hot_us = graph_time_us(fn_hot, args.ncalls, args.nreplays)
            eager_us = eager_time_us(fn_hot)
            print(
                f"{name:16s} {wtype:5s} g{group_size:<3d} K={size_k:<6d} "
                f"N={size_n:<6d} M={m} | cold {cold_us:8.2f} us/call | "
                f"hot {hot_us:8.2f} | eager(inadmissible) {eager_us:8.2f} | "
                f"maxdiff {max_diff:.5f} {'OK' if ok else 'FAIL'}"
            )
        del qweights, all_scales
        torch.accelerator.empty_cache()

    if failures:
        raise SystemExit(f"{failures} case(s) failed the correctness gate")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Graph-timed small-M Marlin benchmark with correctness gate"
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 3, 4, 8])
    parser.add_argument("--ncalls", type=int, default=100)
    parser.add_argument("--nreplays", type=int, default=30)
    parser.add_argument("--workspace-blocks-per-sm", type=int, default=2)
    main(parser.parse_args())
