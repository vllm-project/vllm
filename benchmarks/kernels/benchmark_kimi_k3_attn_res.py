# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark vLLM Kimi-K3 AttnRes kernels on NVIDIA Blackwell.

The benchmark uses a zero delta so repeated in-place launches remain stable.

Example:

.. code-block:: console

    .venv/bin/python benchmarks/kernels/benchmark_kimi_k3_attn_res.py \
      --output /tmp/attn_res.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import torch

import vllm._custom_ops  # noqa: F401 - loads torch.ops._C
from vllm.models.kimi_k3.nvidia.ops.attn_res import _attn_res_kernel
from vllm.platforms import current_platform
from vllm.triton_utils import triton

HIDDEN_SIZE = 7168
MAX_BLOCKS = 8
EPS = 1e-5


@dataclass
class Inputs:
    prefix: torch.Tensor
    delta: torch.Tensor
    blocks: torch.Tensor
    norm_weight: torch.Tensor
    qk_weight: torch.Tensor
    output_norm_weight: torch.Tensor


@dataclass
class Result:
    num_tokens: int
    num_blocks: int
    backend: str
    median_us: float
    p20_us: float
    p80_us: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
    )
    parser.add_argument("--num-blocks", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--warmup-ms", type=int, default=50)
    parser.add_argument("--repeat-ms", type=int, default=200)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def make_inputs(num_tokens: int) -> Inputs:
    prefix = torch.randn(num_tokens, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    blocks = torch.randn(
        num_tokens,
        MAX_BLOCKS,
        HIDDEN_SIZE,
        device="cuda",
        dtype=torch.bfloat16,
    )
    norm_weight = torch.ones(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    qk_weight = (
        torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16) / HIDDEN_SIZE**0.5
    )
    output_norm_weight = 1 + 0.1 * torch.randn(
        HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
    )
    return Inputs(
        prefix=prefix,
        delta=torch.zeros_like(prefix),
        blocks=blocks,
        norm_weight=norm_weight,
        qk_weight=qk_weight,
        output_norm_weight=output_norm_weight,
    )


def launch_triton(inputs: Inputs, output: torch.Tensor, num_blocks: int) -> None:
    num_tokens = inputs.prefix.shape[0]
    if num_tokens >= 256 or num_blocks <= 1:
        block_l, num_warps = 1, 4
    else:
        block_l, num_warps = 4, 8
    _attn_res_kernel[(num_tokens,)](
        inputs.prefix,
        inputs.delta,
        inputs.blocks,
        inputs.norm_weight,
        inputs.qk_weight,
        inputs.output_norm_weight,
        output,
        inputs.prefix.stride(0),
        inputs.delta.stride(0),
        inputs.blocks.stride(0),
        inputs.blocks.stride(1),
        output.stride(0),
        num_blocks,
        HIDDEN_SIZE,
        -1,
        EPS,
        EPS,
        HAS_DELTA=True,
        WRITE_BLOCK=False,
        APPLY_OUTPUT_NORM=True,
        BLOCK_L=block_l,
        BLOCK_D=triton.next_power_of_2(HIDDEN_SIZE),
        num_warps=num_warps,
        num_stages=2,
        launch_pdl=current_platform.is_arch_support_pdl(),
    )


def launch_cuda(inputs: Inputs, output: torch.Tensor, num_blocks: int) -> None:
    torch.ops._C.kimi_k3_attn_res(
        inputs.prefix,
        inputs.delta,
        inputs.blocks,
        inputs.norm_weight,
        inputs.qk_weight,
        inputs.output_norm_weight,
        output,
        num_blocks,
        -1,
        EPS,
        EPS,
    )


def benchmark(
    function: Callable[[], None], warmup_ms: int, repeat_ms: int
) -> tuple[float, float, float]:
    median_ms, p20_ms, p80_ms = triton.testing.do_bench(
        function,
        warmup=warmup_ms,
        rep=repeat_ms,
        quantiles=[0.5, 0.2, 0.8],
    )
    return median_ms * 1000, p20_ms * 1000, p80_ms * 1000


def validate(inputs: Inputs, num_blocks: int) -> None:
    outputs = [torch.empty_like(inputs.prefix) for _ in range(2)]
    launch_triton(inputs, outputs[0], num_blocks)
    launch_cuda(inputs, outputs[1], num_blocks)
    torch.accelerator.synchronize()
    torch.testing.assert_close(outputs[1], outputs[0], atol=8e-2, rtol=3e-2)


def main() -> None:
    args = parse_args()
    if torch.cuda.get_device_capability()[0] != 10:
        raise SystemExit("The vLLM CUDA AttnRes kernel requires the SM100 family")
    if not hasattr(torch.ops._C, "kimi_k3_attn_res"):
        raise SystemExit("This vLLM build does not contain kimi_k3_attn_res")
    if any(not 0 <= blocks <= MAX_BLOCKS for blocks in args.num_blocks):
        raise SystemExit(f"--num-blocks values must be in [0, {MAX_BLOCKS}]")

    torch.manual_seed(0)
    check_inputs = make_inputs(5)
    for num_blocks in args.num_blocks:
        validate(check_inputs, num_blocks)
    print("correctness: all selected block counts passed", flush=True)

    results: list[Result] = []
    for num_blocks in args.num_blocks:
        for num_tokens in args.num_tokens:
            inputs = make_inputs(num_tokens)
            output = torch.empty_like(inputs.prefix)
            candidates = {
                "vllm_triton": partial(launch_triton, inputs, output, num_blocks),
                "vllm_cuda": partial(launch_cuda, inputs, output, num_blocks),
            }
            row: dict[str, float] = {}
            for backend, candidate in candidates.items():
                median_us, p20_us, p80_us = benchmark(
                    candidate, args.warmup_ms, args.repeat_ms
                )
                results.append(
                    Result(
                        num_tokens=num_tokens,
                        num_blocks=num_blocks,
                        backend=backend,
                        median_us=median_us,
                        p20_us=p20_us,
                        p80_us=p80_us,
                    )
                )
                row[backend] = median_us
            print(
                f"blocks={num_blocks} tokens={num_tokens:5d}  "
                f"triton={row['vllm_triton']:9.2f} us  "
                f"cuda={row['vllm_cuda']:9.2f} us",
                flush=True,
            )

    if args.output is not None:
        metadata = {
            "device": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "warmup_ms": args.warmup_ms,
            "repeat_ms": args.repeat_ms,
        }
        args.output.write_text(
            json.dumps(
                {"metadata": metadata, "results": [asdict(item) for item in results]},
                indent=2,
            )
            + "\n"
        )
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
