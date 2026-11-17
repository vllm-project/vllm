# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare Triton and native windowed MLA with unchanged standard FP8 KV."""

import argparse
import json
from dataclasses import asdict
from functools import partial
from pathlib import Path

import torch
from benchmark_mla_query_metadata import (
    Case,
    load_module,
    make_call,
    make_inputs,
    measure_graph,
)

from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops.windowed_flashmla import windowed_flash_mla


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--baseline-source")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = (
        [Case(**c) for c in json.loads(args.cases.read_text())]
        if args.cases
        else [Case(batch=b, splits=16) for b in (1, 4, 16)]
    )
    baseline = load_module(args.baseline_source, "windowed_mla_baseline")
    results = []
    for case in cases:
        if not (
            case.fp8
            and case.latent == 512
            and case.rope == 64
            and case.heads in (64, 128)
        ):
            raise ValueError(
                "This benchmark requires native 512/64 geometry, "
                "64/128 heads and FP8 KV."
            )
        set_random_seed(0)
        inputs = make_inputs(case)
        q, cache, table, lengths, scale = inputs
        starts = torch.arange(
            0, q.shape[0] + 1, case.query_len, device=q.device, dtype=torch.int32
        )
        old_call, old_out, _ = make_call(baseline, case, inputs, expanded=True)

        candidate = partial(
            windowed_flash_mla,
            q,
            cache.squeeze(2),
            table,
            lengths,
            starts,
            case.window,
            case.query_len,
            scale,
            0.0625,
            case.causal,
        )

        old_call()
        new_out = candidate()[0]
        torch.testing.assert_close(old_out, new_out, atol=0.02, rtol=0.02)
        difference = old_out.float() - new_out.float()
        old_time = measure_graph(old_call)
        new_time = measure_graph(candidate)
        row = dict(
            case=asdict(case),
            baseline=old_time,
            candidate=new_time,
            speedup=old_time["median_us"] / new_time["median_us"],
            max_abs_error=difference.abs().max().item(),
            relative_rms=(
                difference.square().mean().sqrt()
                / old_out.float().square().mean().sqrt()
            ).item(),
            persistent_kv_bytes=case.latent + case.rope,
            candidate_includes="temporary pack, indices, live planning, attention",
        )
        results.append(row)
        args.output.write_text(json.dumps(results, indent=2))
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
