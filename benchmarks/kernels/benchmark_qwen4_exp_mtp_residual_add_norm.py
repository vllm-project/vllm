# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark Qwen4Exp MTP unit-injection combine and HC normalization."""

import argparse
import statistics
from functools import partial

import pandas as pd
import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.models.qwen4_exp.nvidia.ops.hc import (
    grouped_gemma_rmsnorm,
    hc_combine_norm,
)

HC_COUNT = 4
HIDDEN_SIZE = 2560
EPS = 1e-6


def _separate_add_norm(
    embedding: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual = (hidden + embedding.unsqueeze(1)).flatten(1)
    normalized = grouped_gemma_rmsnorm(residual, weight, EPS, HC_COUNT)
    return residual, normalized


_compiled_add_norm = torch.compile(_separate_add_norm, fullgraph=True)


def _unit_injection_combine_norm(
    embedding: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return hc_combine_norm(hidden.flatten(1), embedding, None, weight, EPS, HC_COUNT)


def _bench_us(fn) -> float:
    times = bench_gpu_time_with_cupti(
        fn,
        dry_run_time_ms=25,
        repeat_time_ms=50,
        cold_l2_cache=True,
        use_cuda_graph=True,
    )
    return statistics.median(times) * 1e3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=[1, 2, 16, 64, 256, 2048, 8192, 32768],
    )
    args = parser.parse_args()

    if not torch.accelerator.is_available() or torch.version.cuda is None:
        raise RuntimeError("CUDA is required for Qwen4Exp MTP kernel timing")

    torch.set_default_device("cuda")
    torch.manual_seed(0)
    rows = []
    for num_tokens in args.num_tokens:
        embedding = torch.randn(num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16)
        hidden = torch.randn(num_tokens, HC_COUNT, HIDDEN_SIZE, dtype=torch.bfloat16)
        weight = torch.randn(HC_COUNT * HIDDEN_SIZE, dtype=torch.bfloat16)

        expected, expected_norm = _separate_add_norm(embedding, hidden, weight)
        actual, actual_norm = _unit_injection_combine_norm(embedding, hidden, weight)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_norm, expected_norm)

        eager = partial(_separate_add_norm, embedding, hidden, weight)
        compiled = partial(_compiled_add_norm, embedding, hidden, weight)
        fused = partial(_unit_injection_combine_norm, embedding, hidden, weight)
        _compiled_add_norm(embedding, hidden, weight)
        torch.accelerator.synchronize()

        eager_us = _bench_us(eager)
        compiled_us = _bench_us(compiled)
        fused_us = _bench_us(fused)
        rows.append(
            {
                "tokens": num_tokens,
                "eager_separate_us": eager_us,
                "compiled_separate_us": compiled_us,
                "fused_unit_injection_us": fused_us,
                "vs_compiled": compiled_us / fused_us,
            }
        )

    metadata = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": "bfloat16",
        "hidden_size": HIDDEN_SIZE,
        "hc_count": HC_COUNT,
        "cold_l2": True,
        "cuda_graph": True,
    }
    print(pd.Series(metadata, name="value").to_string())
    print(
        pd.DataFrame(rows).to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )


if __name__ == "__main__":
    main()
