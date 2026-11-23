# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The LM head shard GEMM and the logits all-gather of decode steps on Hopper.

`HopperLogitsGather` is main's NVLink gather (`logits_gather.LogitsGather`),
which runs unchanged on SM90. `HopperLogitsHead` takes main's skinny GEMM at
the row counts where it beats cuBLAS on H200 (`SKINNY`); the split-K GEMM main
uses from four rows is SM100-only.
"""

import torch

from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)

from ..logits_gather import LogitsGather

# (block size, outputs per block, K unroll) by rows, by vocabulary shard
# (N, K), from an H200 sweep with the weights out of L2 (us, cuBLAS -> skinny).
# cuBLAS wins from 6 rows (TP4) and 4 rows (TP2).
SKINNY: dict[tuple[int, int], dict[int, tuple[int, int, int]]] = {
    # TP4: 1 83.1 -> 70.2, 2 84.5 -> 71.0, 4 83.5 -> 81.3
    (38720, 4096): {1: (128, 2, 1), 2: (128, 2, 1), 4: (64, 4, 2)},
    # TP2: 1 148.1 -> 137.7, 2 152.4 -> 138.8
    (77440, 4096): {1: (128, 2, 1), 2: (128, 2, 1)},
}


class HopperLogitsHead:
    """`x @ weight.T` for the row counts of `tokens` that `SKINNY` lists for
    this weight's shape; `supports` is False for the others (cuBLAS)."""

    def __init__(self, weight: torch.Tensor, tokens=()):
        self.weight = weight
        n, k = weight.shape
        self.configs = {}
        for m, (block, outputs, unroll) in SKINNY.get((n, k), {}).items():
            if m not in tokens:
                continue
            cfg = SkinnyGemmConfig(m, block, outputs, unroll, 8, k)
            shape_dynamic_skinny_gemm(weight.new_zeros(m, k), weight, cfg, None)
            self.configs[m] = cfg  # compiled before the first step

    def supports(self, rows: int) -> bool:
        return rows in self.configs

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return shape_dynamic_skinny_gemm(x, self.weight, self.configs[x.shape[0]], None)


class HopperLogitsGather(LogitsGather):
    """Main's `LogitsGather`: `[M, vocab shard]` BF16 logits of every rank
    stored over NVLink into each rank's `[M, vocab]` buffer. Eager calls only."""
