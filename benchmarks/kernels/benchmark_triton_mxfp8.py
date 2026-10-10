# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare Marlin W8A16 and Triton MXFP8 W8A8, including A quantization.

Requires FlashInfer's CUPTI timing dependencies. Synthetic numerical checks here
are not a replacement for checkpoint quality evaluation. Useful GEMM TFLOPS uses
2*M*N*K operations and excludes the activation quantization operation count.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from vllm.model_executor.kernels.linear.mxfp8.marlin import MarlinMxfp8LinearKernel
from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
    Mxfp8LinearLayerConfig,
)
from vllm.model_executor.kernels.linear.mxfp8.triton import TritonMxfp8LinearKernel
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype


def relative_rmse(actual, expected):
    return (
        (actual.float() - expected.float()).square().mean().sqrt()
        / expected.float().square().mean().sqrt().clamp_min(1e-12)
    ).item()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows", nargs="+", type=int, default=[1, 8, 16, 32, 64, 128, 1024, 8192]
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("triton-mxfp8.json"))
    args = parser.parse_args()
    if not current_platform.is_cuda() or not current_platform.is_device_capability(90):
        parser.error("This comparison requires an SM90 CUDA GPU")
    if args.repeats < 1 or any(m <= 0 for m in args.rows):
        parser.error("Use positive repeats and row counts")

    # Explicit import prevents silently falling back from CUPTI to event timing.
    from cupti import cupti  # noqa: F401
    from flashinfer.testing import bench_gpu_time_with_cupti

    torch.manual_seed(1729)
    flush = torch.empty(64 * 1024**2, device="cuda", dtype=torch.uint8)
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "timing": "CUPTI graphs with cold L2, and synchronized eager wall time",
        "scope": "Synthetic single-rank projection; not a model quality evaluation",
        "cases": [],
    }
    for n, k in (
        (1792, 5120),
        (4096, 1280),
        (5120, 1024),
        (576, 5120),
        (5120, 288),
        (25600, 6144),
    ):
        weight = torch.randn((n, k), device="cuda").to(torch.float8_e4m3fn)
        encoded = torch.randint(
            120, 126, (n, k // 32), device="cuda", dtype=torch.uint8
        )
        scales = encoded.view(torch.float8_e8m0fnu).float()
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(encoded.clone(), requires_grad=False)
        layer.input_size_per_partition = k
        layer.output_size_per_partition = n
        with set_default_torch_dtype(torch.bfloat16):
            kernel = MarlinMxfp8LinearKernel(Mxfp8LinearLayerConfig())
            kernel.process_weights_after_loading(layer)

        candidate = torch.nn.Module()
        candidate.weight = torch.nn.Parameter(weight, requires_grad=False)
        candidate.weight_scale = torch.nn.Parameter(encoded, requires_grad=False)
        candidate_kernel = TritonMxfp8LinearKernel(Mxfp8LinearLayerConfig())
        candidate_kernel.process_weights_after_loading(candidate)

        for m in args.rows:
            x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) / 8
            calls = {
                "marlin": (kernel.apply_weights, (layer, x)),
                "triton_w8a8": (candidate_kernel.apply_weights, (candidate, x)),
            }
            reference = (
                x[:16].float()
                @ (weight[:64].float() * scales[:64].repeat_interleave(32, dim=1)).T
            )
            groups = x[:16].float().reshape(-1, k // 32, 32)
            amax = groups.abs().amax(dim=-1).clamp_min(1e-10)
            a_scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
            quantized = (groups / a_scale[..., None]).to(torch.float8_e4m3fn)
            dequantized = (quantized.float() * a_scale[..., None]).reshape(-1, k)
            quantized_reference = (
                dequantized
                @ (weight[:64].float() * scales[:64].repeat_interleave(32, dim=1)).T
            )
            row = {"m": m, "n": n, "k": k, "reference_rmse": {}, "timings": []}
            for label, (fn, inputs) in calls.items():
                actual = fn(*inputs)
                error = relative_rmse(actual[:16, :64], reference)
                assert torch.isfinite(actual).all()
                assert error < (0.01 if label == "marlin" else 0.05), (label, error)
                row["reference_rmse"][label] = error
                if label == "triton_w8a8":
                    quantized_error = relative_rmse(
                        actual[:16, :64], quantized_reference
                    )
                    assert quantized_error < 0.006, quantized_error
                    row["quantized_reference_rmse"] = quantized_error
            for repeat in range(args.repeats):
                order = list(calls)
                if repeat % 2:
                    order.reverse()
                timing = {}
                for label in order:
                    fn, inputs = calls[label]
                    for _ in range(5):
                        fn(*inputs)
                    samples = bench_gpu_time_with_cupti(
                        fn,
                        input_args=inputs,
                        use_cuda_graph=True,
                        cold_l2_cache=True,
                        repeat_time_ms=100,
                    )
                    ms = statistics.median(samples)
                    timing[label] = {
                        "ms": ms,
                        "useful_tflops": 2 * m * n * k / (ms * 1e9),
                    }
                row["timings"].append(timing)
            row["eager_ms"] = {}
            for label, (fn, inputs) in calls.items():
                samples = []
                for _ in range(9):
                    flush.zero_()
                    torch.accelerator.synchronize()
                    start = time.perf_counter()
                    fn(*inputs)
                    torch.accelerator.synchronize()
                    samples.append((time.perf_counter() - start) * 1000)
                row["eager_ms"][label] = statistics.median(samples)
            report["cases"].append(row)
            args.output.write_text(json.dumps(report, indent=2))
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
