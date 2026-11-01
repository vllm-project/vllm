# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure histogram and fused EPLB instrumentation cost, not serving throughput."""

import argparse
import json
from functools import partial

import torch

from vllm.distributed.expert_load import ExpertLoadLayer
from vllm.model_executor.layers.fused_moe.router.base_router import (
    eplb_map_to_physical_and_record,
)
from vllm.triton_utils import triton


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 64, 512, 4096])
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()
    if not 0 < args.top_k <= args.num_experts or min(args.tokens) < 1:
        parser.error("Require positive token counts and 0 < top-k <= num-experts")
    torch.manual_seed(0)
    experts = args.num_experts
    mapping = torch.arange(experts, device="cuda", dtype=torch.int32).view(-1, 1)
    replicas = torch.ones(experts, device="cuda", dtype=torch.int32)
    physical = torch.zeros(experts, device="cuda", dtype=torch.int32)
    logical = torch.zeros(experts, device="cuda", dtype=torch.int64)
    valid = torch.zeros((), device="cuda", dtype=torch.int32)
    record = torch.tensor(True, device="cuda")
    stats = ExpertLoadLayer(logical, valid, 0, 0, 1, False)
    for tokens in args.tokens:
        valid.fill_(tokens)
        for distribution in ("uniform", "skewed"):
            if distribution == "uniform":
                ids = (
                    torch.rand((tokens, experts), device="cuda")
                    .topk(args.top_k)
                    .indices.int()
                )
            else:
                ids = torch.arange(args.top_k, device="cuda", dtype=torch.int32).repeat(
                    tokens, 1
                )
            logical.zero_()
            stats.record(ids)
            expected = torch.bincount(ids.flatten().long(), minlength=experts)
            torch.testing.assert_close(logical, expected)
            histogram_ms = triton.testing.do_bench(partial(stats.record, ids))
            for record_eplb in (False, True):
                record.fill_(record_eplb)

                run = partial(
                    eplb_map_to_physical_and_record,
                    ids,
                    physical,
                    mapping,
                    replicas,
                    record,
                    valid,
                )
                run_with_stats = partial(run, expert_load_stats=stats)

                logical.zero_()
                torch.testing.assert_close(run_with_stats(), ids)
                torch.testing.assert_close(logical, expected)
                baseline_ms = triton.testing.do_bench(run)
                instrumented_ms = triton.testing.do_bench(run_with_stats)
                print(
                    json.dumps(
                        {
                            "tokens": tokens,
                            "experts": experts,
                            "top_k": args.top_k,
                            "distribution": distribution,
                            "record_eplb": record_eplb,
                            "histogram_ms": histogram_ms,
                            "eplb_ms": baseline_ms,
                            "eplb_with_stats_ms": instrumented_ms,
                            "eplb_overhead_percent": 100
                            * (instrumented_ms / baseline_ms - 1),
                        }
                    )
                )


if __name__ == "__main__":
    main()
