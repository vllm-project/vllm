# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare tiled MegaMoE staging against an explicit reference source file."""

import argparse
import hashlib
import importlib.util
import json
import statistics
from functools import partial
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.models.deepseek_v4.nvidia.ops.prepare_megamoe import prepare_megamoe_inputs

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--tokens", type=int, nargs="+", default=[6, 127, 128, 135, 434, 8192, 16384]
)
parser.add_argument("--hidden-size", type=int, default=5120)
parser.add_argument("--top-k", type=int, default=6)
args = parser.parse_args()
spec = importlib.util.spec_from_file_location("staging_reference", args.reference)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
torch.manual_seed(42)
results = []
for tokens in args.tokens:
    hidden, topk, block_m = args.hidden_size, args.top_k, 128
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    ids = torch.randint(384, (tokens, topk), device="cuda", dtype=torch.int32)
    weights = torch.randn(tokens, topk, device="cuda")
    padding = torch.arange(tokens, device="cuda") % 7 == 0
    for transposed_scales in (False, True):

        def buffers(
            hidden=hidden,
            tokens=tokens,
            transposed_scales=transposed_scales,
            block_m=block_m,
            x=x,
            ids=ids,
            weights=weights,
        ):
            sf = torch.empty(
                (hidden // 128, tokens)
                if transposed_scales
                else (tokens, hidden // 128),
                device="cuda",
                dtype=torch.int32,
            )
            if transposed_scales:
                sf = sf.t()
            shared = torch.full(
                (hidden // 128, ((tokens + block_m - 1) // block_m) * block_m),
                -1,
                device="cuda",
                dtype=torch.int32,
            ).t()
            return (
                torch.empty_like(x, dtype=torch.float8_e4m3fn),
                sf,
                torch.empty_like(ids, dtype=torch.int64),
                torch.empty_like(weights),
                shared,
            )

        expected, actual = buffers(), buffers()

        def call(
            fn, out, x=x, weights=weights, ids=ids, padding=padding, block_m=block_m
        ):
            fn(
                x,
                weights,
                ids,
                *out[:4],
                is_padding=padding,
                shared_x_sf=out[4],
                shared_block_m=block_m,
            )

        call(reference.prepare_megamoe_inputs, expected)
        call(prepare_megamoe_inputs, actual)
        torch.accelerator.synchronize()
        for a, b in zip(actual, expected):
            assert torch.equal(
                a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
            )
        row = {
            "tokens": tokens,
            "hidden": hidden,
            "transposed_scales": transposed_scales,
            "bitwise_equal": True,
        }
        for name, fn, out in (
            ("reference", reference.prepare_megamoe_inputs, expected),
            ("candidate", prepare_megamoe_inputs, actual),
        ):
            samples = bench_gpu_time_with_cupti(
                partial(call, fn, out),
                use_cuda_graph=True,
                cold_l2_cache=True,
            )
            us = statistics.median(samples) * 1000
            row[name + "_us"] = us
            byte_count = tokens * (hidden * 3 + hidden // 128 * 8 + topk * 20 + 1)
            row[name + "_gbps"] = byte_count / (us * 1000)
        results.append(row)
        print(json.dumps(row), flush=True)
args.output.write_text(
    json.dumps(
        {
            "gpu": torch.cuda.get_device_name(),
            "seed": 42,
            "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
            "scope": "input quantization, scale layouts, routing and padding staging",
            "timing": "CUPTI CUDA graph, cold L2; allocations and compilation excluded",
            "results": results,
        },
        indent=2,
    )
    + "\n"
)
