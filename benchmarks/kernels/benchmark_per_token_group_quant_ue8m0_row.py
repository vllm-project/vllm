# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare native CUDA and the normal row-quant op with cold-cache GPU timing.

Example: python benchmarks/kernels/benchmark_per_token_group_quant_ue8m0_row.py
    --rows 32 2048 3187 4096 --widths 5120 6144 17408 --output row-quant.json
"""

import argparse
import importlib.metadata
import json
import statistics
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.model_executor.layers.quantization.utils import fp8_utils  # noqa: F401
from vllm.platforms import current_platform


def benchmark(rows, width, args):
    x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    outputs = {}
    for name in ("native", "row"):
        quant = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        if args.layout == "row":
            scales = torch.empty(
                (rows, width // 128), device="cuda", dtype=torch.float32
            )
        else:
            stride = rows if args.layout == "col" else (rows + 3) // 4 * 4
            scales = torch.empty_strided(
                (rows, width // 128), (1, stride), device="cuda", dtype=torch.float32
            )
        outputs[name] = quant, scales
    operators = {
        "native": torch.ops._C.per_token_group_fp8_quant,
        "row": torch.ops.vllm.per_token_group_fp8_quant_ue8m0_row,
    }

    def call(name):
        operators[name](
            x,
            *outputs[name],
            128,
            1e-10,
            -448.0,
            448.0,
            args.ue8m0,
            args.layout != "row",
            args.layout == "tma",
        )

    for name in operators:
        call(name)
    q_ref, s_ref = outputs["native"]
    quant, scales = outputs["row"]
    assert torch.equal(quant.view(torch.uint8), q_ref.view(torch.uint8))
    assert torch.equal(
        scales.contiguous().view(torch.int32), s_ref.contiguous().view(torch.int32)
    )
    samples = {name: [] for name in operators}
    for index in range(3):
        order = list(operators) if index % 2 == 0 else list(reversed(operators))
        for name in order:
            times = bench_gpu_time_with_cupti(
                lambda name=name: call(name),
                use_cuda_graph=True,
                cold_l2_cache=True,
                dry_run_iters=10,
                repeat_iters=30,
            )
            samples[name].append(statistics.median(times) * 1000)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    io_bytes = rows * width * 3 + rows * (width // 128) * 4
    return {
        "rows": rows,
        "width": width,
        "layout": args.layout,
        "ue8m0": args.ue8m0,
        "bit_exact": True,
        "median_us": medians,
        "round_medians_us": samples,
        "estimated_bytes": io_bytes,
        "gb_s": {name: io_bytes / us / 1000 for name, us in medians.items()},
        "speedup": medians["native"] / medians["row"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows", type=int, nargs="+", default=[1, 32, 2048, 3187, 4096]
    )
    parser.add_argument("--widths", type=int, nargs="+", default=[5120, 6144, 17408])
    parser.add_argument("--layout", choices=["row", "col", "tma"], default="col")
    parser.add_argument("--ue8m0", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not current_platform.is_cuda():
        raise RuntimeError("CUDA is required")
    if any(value <= 0 for value in args.rows) or any(
        width <= 0 or width % 128 for width in args.widths
    ):
        parser.error("rows must be positive; widths must be positive multiples of128")
    torch.manual_seed(0)
    result = {
        "gpu": current_platform.get_device_name(),
        "torch": torch.__version__,
        "cupti_python": importlib.metadata.version("cupti-python"),
        "timer": "FlashInfer CUPTI, CUDA graphs, cold L2, preallocated outputs",
        "cases": [],
    }
    for rows in args.rows:
        for width in args.widths:
            case = benchmark(rows, width, args)
            result["cases"].append(case)
            print(json.dumps(case), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
