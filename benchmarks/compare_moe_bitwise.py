#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare HPC and Triton BF16 MoE outputs on identical tensors."""

from __future__ import annotations

import json
import os
from pathlib import Path

import hpc
import torch

from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

E = 256
HIDDEN = 3072
INTERMEDIATE = 192
TOPK = 8
BATCHES = (1, 2, 4)
ROUNDS = 5


def run_hpc(x, w1, w2, topk_weights, topk_ids):
    workspace = hpc.allocate_fuse_moe_bf16_workspace(x, w1, topk_ids, E)
    return hpc.fuse_moe_bf16(
        x,
        w1,
        w2,
        topk_ids,
        topk_weights,
        rank_ep=0,
        num_expert_total=E,
        workspace=workspace,
    )


def run_triton(x, w1, w2, topk_weights, topk_ids):
    quant_config = FusedMoEQuantConfig.make(None)
    return fused_experts(
        x,
        w1,
        w2,
        topk_weights,
        topk_ids,
        quant_config=quant_config,
    )


def stats(hpc_out: torch.Tensor, triton_out: torch.Tensor) -> dict[str, object]:
    hpc_f = hpc_out.float()
    triton_f = triton_out.float()
    diff = (hpc_f - triton_f).abs()
    denom = triton_f.abs().clamp_min(1e-12)
    return {
        "bitwise_equal": bool(torch.equal(hpc_out, triton_out)),
        "different_elements": int(torch.count_nonzero(hpc_out != triton_out)),
        "total_elements": hpc_out.numel(),
        "different_fraction": float(torch.count_nonzero(hpc_out != triton_out))
        / hpc_out.numel(),
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "max_relative_error": float((diff / denom).max()),
        "mean_relative_error": float((diff / denom).mean()),
        "allclose_1e-2": bool(torch.allclose(hpc_f, triton_f, rtol=1e-2, atol=1e-2)),
        "allclose_1e-1": bool(torch.allclose(hpc_f, triton_f, rtol=1e-1, atol=1e-1)),
    }


def main() -> None:
    os.environ.setdefault("VLLM_LOGGING_CONFIG_PATH", "")
    torch.manual_seed(20260812)
    torch.cuda.manual_seed_all(20260812)
    device = torch.device("cuda")

    # Keep one common set of weights and inputs for every backend and batch.
    w1 = (
        torch.randn((E, INTERMEDIATE * 2, HIDDEN), device=device, dtype=torch.bfloat16)
        / 10
    )
    w2 = (
        torch.randn((E, HIDDEN, INTERMEDIATE), device=device, dtype=torch.bfloat16) / 10
    )
    results: list[dict[str, object]] = []
    for batch in BATCHES:
        x = torch.randn((batch, HIDDEN), device=device, dtype=torch.bfloat16) / 10
        topk_ids = torch.stack(
            [torch.randperm(E, device=device)[:TOPK] for _ in range(batch)]
        ).to(torch.int32)
        topk_weights = torch.rand((batch, TOPK), device=device, dtype=torch.float32)
        topk_weights /= topk_weights.sum(dim=1, keepdim=True)

        # Warm up both paths with the exact tensors used for comparison.
        run_hpc(x, w1, w2, topk_weights, topk_ids)
        run_triton(x, w1, w2, topk_weights, topk_ids)
        torch.cuda.synchronize()

        round_stats = []
        hpc_out = triton_out = None
        for _ in range(ROUNDS):
            hpc_out = run_hpc(x, w1, w2, topk_weights, topk_ids)
            triton_out = run_triton(x, w1, w2, topk_weights, topk_ids)
            torch.cuda.synchronize()
            round_stats.append(stats(hpc_out, triton_out))
        assert hpc_out is not None and triton_out is not None
        results.append(
            {
                "batch": batch,
                "shape": {
                    "experts": E,
                    "hidden": HIDDEN,
                    "intermediate": INTERMEDIATE,
                    "topk": TOPK,
                },
                "rounds": round_stats,
                "deterministic_hpc": all(
                    torch.equal(hpc_out, run_hpc(x, w1, w2, topk_weights, topk_ids))
                    for _ in range(2)
                ),
                "deterministic_triton": all(
                    torch.equal(
                        triton_out,
                        run_triton(x, w1, w2, topk_weights, topk_ids),
                    )
                    for _ in range(2)
                ),
            }
        )
        print(json.dumps(results[-1], indent=2), flush=True)

    output = Path("benchmark_results/moe_bitwise_20260812.json")
    output.write_text(
        json.dumps(
            {
                "seed": 20260812,
                "device": torch.cuda.get_device_name(),
                "rounds": ROUNDS,
                "results": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
