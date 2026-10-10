# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eight-GPU AttnRes+FP8 projection comparison with rotated graph timing."""

import argparse
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.models.kimi_k3.amd.ops.attn_res import attn_res
from vllm.platforms import current_platform


def capture(fn, repeats):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeats):
            fn()
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 4, 16, 64, 256, 1024, 4096, 16384]
    )
    parser.add_argument("--blocks", type=int, nargs="+", default=[1, 4, 8])
    args = parser.parse_args()
    assert int(os.environ["WORLD_SIZE"]) == 8
    assert torch.accelerator.device_count() == 8
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    torch.set_num_threads(1)
    torch.manual_seed(1729 + rank)
    dist.init_process_group("gloo")
    args.output.mkdir(parents=True, exist_ok=True)
    dtype = current_platform.fp8_dtype()
    rows = []
    with set_current_vllm_config(VllmConfig()):
        weight = torch.randn(2112, 7168, device="cuda", dtype=torch.bfloat16) * 0.02
        weight, ws = rocm_aiter_ops.per_token_quant(weight, dtype)
        weight = rocm_aiter_ops.shuffle_weight(weight)
        for m in args.tokens:
            prefix = torch.randn(m, 7168, device="cuda", dtype=torch.bfloat16)
            delta = torch.zeros_like(prefix)
            blocks = torch.randn(m, 9, 7168, device="cuda", dtype=torch.bfloat16)
            norm = torch.ones(7168, device="cuda", dtype=torch.bfloat16)
            score = torch.randn_like(norm) / 7168**0.5
            for b in args.blocks:

                def produce(
                    fused,
                    prefix=prefix,
                    delta=delta,
                    blocks=blocks,
                    norm=norm,
                    score=score,
                    b=b,
                ):
                    result = attn_res(
                        prefix,
                        delta,
                        blocks,
                        norm,
                        score,
                        norm,
                        b,
                        -1,
                        1e-5,
                        1e-5,
                        quant_dtype=dtype if fused else None,
                    )
                    return (
                        result
                        if fused
                        else rocm_aiter_ops.per_token_quant(result, dtype)
                    )

                def segment(fused, produce=produce):
                    x, scale = produce(fused)
                    return rocm_aiter_ops.preshuffled_per_token_w8a8_gemm(
                        x, weight, scale, ws, output_dtype=torch.bfloat16
                    )

                expected, actual = segment(False), segment(True)
                relative_l2 = (
                    (actual.float() - expected.float()).norm() / expected.float().norm()
                ).item()
                assert relative_l2 < 0.002, relative_l2
                repeats = 20 if m < 1024 else 4
                functions = {
                    "baseline_producer": lambda produce=produce: produce(False),
                    "fused_producer": lambda produce=produce: produce(True),
                    "baseline_segment": lambda segment=segment: segment(False),
                    "fused_segment": lambda segment=segment: segment(True),
                }
                graphs = {k: capture(fn, repeats) for k, fn in functions.items()}
                samples = {k: [] for k in functions}
                names = list(functions)
                for sample in range(9):
                    for name in names[sample % 4 :] + names[: sample % 4]:
                        dist.barrier()
                        graphs[name].replay()
                        torch.accelerator.synchronize()
                        start, end = [
                            torch.cuda.Event(enable_timing=True) for _ in range(2)
                        ]
                        start.record()
                        graphs[name].replay()
                        end.record()
                        end.synchronize()
                        samples[name].append(start.elapsed_time(end) * 1000 / repeats)
                row = {
                    "tokens": m,
                    "blocks": b,
                    "relative_l2": relative_l2,
                    "samples_us": samples,
                    "median_us": {k: statistics.median(v) for k, v in samples.items()},
                }
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "rank": rank,
                            "tokens": m,
                            "blocks": b,
                            "median_us": row["median_us"],
                        }
                    ),
                    flush=True,
                )
                del graphs
    (args.output / f"rank-{rank}.json").write_text(
        json.dumps(
            {
                "rank": rank,
                "world_size": 8,
                "torch": torch.__version__,
                "rows": rows,
                "scope": "AttnRes + FP8 quant + TP8 MLA input projection",
            },
            indent=2,
        )
        + "\n"
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
