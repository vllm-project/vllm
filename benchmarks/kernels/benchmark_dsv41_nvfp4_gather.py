# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen worker counts for NVFP4 prefill cache gathering on identical inputs."""

import argparse
import hashlib
import inspect
import json
import statistics
from functools import partial
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.models.deepseek_v41.common.ops.cache_utils import (
    _dequantize_and_gather_k_nvfp4_kernel,
    dequantize_and_gather_k_cache_triton,
)


def launch(out, cache, lengths, gather_lengths, table, workers):
    if workers == 0:
        return dequantize_and_gather_k_cache_triton(
            out, cache, lengths, gather_lengths, table, 64, 7
        )
    _dequantize_and_gather_k_nvfp4_kernel[(len(lengths), workers)](
        out,
        out.stride(0),
        out.stride(1),
        cache,
        lengths,
        table,
        7,
        gather_lengths,
        max_blocks_per_seq=table.shape[1],
        head_dim=512,
        scale_dim=32,
        quant_block=16,
        cache_block_size=64,
        block_stride=cache.stride(0),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--kv-rows", type=int, nargs="+", default=[6144, 102400])
    parser.add_argument(
        "--workers", type=int, nargs="+", default=[128, 512, 1024, 2048, 0]
    )
    args = parser.parse_args()
    torch.manual_seed(42)
    rows = []
    for requests in args.requests:
        for width in args.kv_rows:
            assert width >= requests > 0
            blocks = (width + 63) // 64
            cache = torch.randint(
                256,
                (requests * blocks, 64, 288),
                dtype=torch.uint8,
                device="cuda",
            )
            # Each page stores packed values first, then finite FP8 scales.
            cache.view(requests * blocks, -1)[:, 64 * 256 :] = 0x38
            table = torch.randperm(
                requests * blocks,
                dtype=torch.int32,
                device="cuda",
            ).reshape(requests, blocks)
            lengths = width - torch.arange(requests, device="cuda", dtype=torch.int32)
            gathered = (lengths - 13).clamp(min=0)
            expected = torch.zeros(
                requests,
                width + 7,
                512,
                dtype=torch.bfloat16,
                device="cuda",
            )
            actual = torch.zeros_like(expected)
            launch(expected, cache, lengths, gathered, table, 128)
            torch.accelerator.synchronize()
            for workers in args.workers:
                fn = partial(launch, actual, cache, lengths, gathered, table, workers)
                actual.zero_()
                fn()
                torch.accelerator.synchronize()
                assert torch.equal(
                    actual.view(torch.uint8), expected.view(torch.uint8)
                ), (requests, width, workers)
                times = bench_gpu_time_with_cupti(
                    fn,
                    use_cuda_graph=True,
                    cold_l2_cache=True,
                )
                us = statistics.median(times) * 1000
                rows.append(
                    {
                        "requests": requests,
                        "max_kv_rows": width,
                        "workers": workers,
                        "bitwise_equal": True,
                        "median_us": us,
                        "effective_gbps": int(gathered.sum())
                        * (288 + 1024)
                        / (us * 1000),
                    }
                )
                print(json.dumps(rows[-1]), flush=True)
            del cache, expected, actual
    source = Path(inspect.getfile(_dequantize_and_gather_k_nvfp4_kernel.fn))
    args.output.write_text(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "seed": 42,
                "page_layout": "seeded random permutation, shared across worker counts",
                "source": str(source),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "scope": "NVFP4 gather: same math, different worker counts",
                "timing": "CUPTI graph, cold L2; no compilation or allocation",
                "results": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
