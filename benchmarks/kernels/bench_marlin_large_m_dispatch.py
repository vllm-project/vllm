# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph-timed benchmark of the Marlin large-M dequant dispatch
(VLLM_MARLIN_LARGE_M_BF16) with a correctness gate.

Runs the real prepare_/apply_ path of dense FP8-channelwise and NVFP4-g16
Marlin layers at prefill-shaped M and reports Marlin-kernel time vs the
dequant + 16-bit GEMM dispatch. The dequant cost (including the NVFP4
nibble unpack) is charged inside the dispatched call by construction.
Both arms are timed by capturing N back-to-back calls in one CUDA graph
and timing replays with CUDA events, rotating each call across enough
distinct layer clones to exceed L2 (cold weights, the prefill regime).
The Marlin arm reuses the dispatched layer with its threshold forced
above every M, so both arms traverse the identical apply path.

The default shapes are the dense Marlin calls of
Nemotron-3.5-Lightning-30B-A3B-NVFP4 (hidden 2688); the default Ms cover
decode (M=8, stays on Marlin by the threshold floor), the cache-hit
recompute chunks (1952/2192) and the cache-miss chunk (6576). lm_head is
excluded: it exceeds the per-layer workspace cap and only sees small M
in serving. Every case is gated on the same
mean-relative-diff-vs-dequantized-reference tolerance as
tests/kernels/quantization/test_marlin_gemm.py (< 0.04).
"""

import os

import torch

from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    prepare_fp4_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear,
    prepare_fp8_layer_for_marlin,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser

# name, weight quant, K, N
SHAPES = [
    ("mamba_in_proj", "fp8", 2688, 10304),
    ("mamba_out_proj", "fp8", 4096, 2688),
    ("shared_exp_up", "nvfp4", 2688, 3712),
    ("shared_exp_down", "nvfp4", 3712, 2688),
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


def _nvfp4_dequant_ref(packed, group_scales, dtype):
    hi = (packed & 0b10000000) | ((packed & 0b01110000) >> 2)
    lo = packed << 4
    lo = (lo & 0b10000000) | ((lo & 0b01110000) >> 2)
    w = torch.stack(
        [
            lo.view(torch.float8_e4m3fn).to(dtype),
            hi.view(torch.float8_e4m3fn).to(dtype),
        ],
        dim=-1,
    ).view(packed.size(0), -1) * (2**6)
    n, k = w.shape
    w = w.view(n, k // 16, 16) * group_scales.to(dtype).unsqueeze(-1)
    return w.view(n, k)


def build_layer(wtype, size_k, size_n, device, dtype):
    """One prepared marlin layer (dispatch ctx attached) + [K, N] ref."""
    layer = torch.nn.Module()
    layer.input_size_per_partition = size_k
    layer.output_size_per_partition = size_n
    if wtype == "fp8":
        w = torch.randn((size_n, size_k), dtype=dtype, device=device) / 10
        scale = w.abs().amax(dim=1, keepdim=True) / 448
        w_fp8 = (w / scale).to(torch.float8_e4m3fn)
        w_ref = (w_fp8.to(dtype) * scale).T
        layer.weight = torch.nn.Parameter(w_fp8.T.contiguous(), requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(scale.view(-1), requires_grad=False)
        layer.orig_dtype = dtype
        prepare_fp8_layer_for_marlin(layer, size_k_first=True)
    else:
        packed = torch.randint(
            0, 256, (size_n, size_k // 2), dtype=torch.uint8, device=device
        )
        scales = torch.rand((size_n, size_k // 16), device=device) * 1.5 + 0.5
        scales = scales.to(torch.float8_e4m3fn)
        global_scale = torch.tensor(0.01, dtype=torch.float32, device=device)
        w_ref = _nvfp4_dequant_ref(
            packed, scales.to(torch.float32) * global_scale, dtype
        ).T
        layer.weight = torch.nn.Parameter(packed, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(scales, requires_grad=False)
        layer.weight_global_scale = torch.nn.Parameter(
            global_scale, requires_grad=False
        )
        layer.params_dtype = dtype
        prepare_fp4_layer_for_marlin(layer)
    assert layer.weight.marlin_large_m_ctx is not None
    return layer, w_ref


def make_fn(layers, apply_fn, a, size_n, size_k, rotate):
    num_layers = len(layers) if rotate else 1
    state = {"i": 0}

    def fn():
        i = state["i"]
        state["i"] = (i + 1) % num_layers
        return apply_fn(layers[i], a, size_n, size_k)

    return fn


def _apply_fp8(layer, x, n, k):
    return apply_fp8_marlin_linear(
        input=x,
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        workspace=layer.workspace,
        size_n=n,
        size_k=k,
        bias=None,
    )


def _apply_fp4(layer, x, n, k):
    return apply_fp4_marlin_linear(
        input=x,
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        weight_global_scale=layer.weight_global_scale,
        workspace=layer.workspace,
        size_n=n,
        size_k=k,
        bias=None,
    )


def main(args):
    os.environ["VLLM_MARLIN_LARGE_M_BF16"] = str(args.flag_value)
    torch.manual_seed(7)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    props = torch.cuda.get_device_properties(device)
    print(
        f"device={torch.cuda.get_device_name(device)} "
        f"cc={torch.cuda.get_device_capability(device)} "
        f"sms={props.multi_processor_count} "
        f"VLLM_MARLIN_LARGE_M_BF16={args.flag_value}"
    )

    failures = 0
    for name, wtype, size_k, size_n in SHAPES:
        wbytes = size_k * size_n * (1 if wtype == "fp8" else 0.5)
        num_clones = max(2, min(16, int(WEIGHT_ROTATION_BYTES // wbytes)))
        layers, w_ref = [], None
        for _ in range(num_clones):
            layer, w_ref = build_layer(wtype, size_k, size_n, device, dtype)
            layers.append(layer)
        apply_fn = _apply_fp8 if wtype == "fp8" else _apply_fp4
        threshold = layers[0].weight.marlin_large_m_ctx.threshold

        for m in args.batch_sizes:
            a = torch.randn((m, size_k), dtype=dtype, device=device) / 10
            fn = make_fn(layers, apply_fn, a, size_n, size_k, rotate=True)

            ref = torch.matmul(a, w_ref)
            dispatch_out = fn()
            for layer in layers:
                layer.weight.marlin_large_m_ctx.threshold = 1 << 30
            marlin_out = fn()
            torch.accelerator.synchronize()
            marlin_diff = compute_max_diff(marlin_out, ref)
            dispatch_diff = compute_max_diff(dispatch_out, ref)
            ok = marlin_diff < 0.04 and dispatch_diff < 0.04
            failures += not ok

            marlin_us = graph_time_us(fn, args.ncalls, args.nreplays)
            for layer in layers:
                layer.weight.marlin_large_m_ctx.threshold = threshold
            dispatch_us = graph_time_us(fn, args.ncalls, args.nreplays)
            routed = "dequant+gemm" if m >= threshold else "marlin(guard)"
            print(
                f"{name:16s} {wtype:5s} K={size_k:<5d} N={size_n:<6d} "
                f"M={m:<5d} | marlin {marlin_us:9.2f} us/call | "
                f"dispatch {dispatch_us:9.2f} ({routed}) | "
                f"ratio {marlin_us / dispatch_us:5.2f}x | "
                f"maxdiff {dispatch_diff:.5f} {'OK' if ok else 'FAIL'}"
            )
        del layers
        torch.accelerator.empty_cache()

    if failures:
        raise SystemExit(f"{failures} case(s) failed the correctness gate")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Graph-timed Marlin large-M dequant dispatch benchmark"
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[8, 512, 1024, 1952, 2192, 6576],
    )
    parser.add_argument("--ncalls", type=int, default=20)
    parser.add_argument("--nreplays", type=int, default=10)
    parser.add_argument("--flag-value", type=int, default=1)
    main(parser.parse_args())
