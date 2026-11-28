# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare native-head SM90 decode with the padded FlashMLA production call."""

import argparse
import csv
import statistics
import sys

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.models.deepseek_v41.common.ops import quantize_and_insert_k_cache
from vllm.models.deepseek_v41.nvidia.ops.small_head_sparse_decode import (
    small_head_sparse_decode,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.v1.attention.ops.flashmla import (
    flash_mla_with_kvcache,
    get_mla_metadata,
)


def make_cache(num_pages: int, page_size: int) -> torch.Tensor:
    page_bytes = round_up(page_size * 584, 576)
    backing = torch.empty(num_pages, page_bytes, dtype=torch.uint8, device="cuda")
    kv = torch.randn(num_pages * page_size, 512, dtype=torch.bfloat16, device="cuda")
    slots = torch.arange(kv.shape[0], dtype=torch.int64, device="cuda")
    quantize_and_insert_k_cache(kv, backing, slots, block_size=page_size)
    return backing.as_strided((num_pages, page_size, 584), (page_bytes, 584, 1))


def benchmark(tokens: int, heads: int, topk: int, rounds: int) -> list:
    swa_cache, extra_cache = make_cache(64, 64), make_cache(64, 64)
    q = torch.randn(tokens, 64, 512, dtype=torch.bfloat16, device="cuda")
    q[:, heads:] = 0
    out = torch.empty_like(q)
    swa_indices = torch.randint(
        0, 4096, (tokens, 1, 128), dtype=torch.int32, device="cuda"
    )
    swa_lens = torch.full((tokens,), 128, dtype=torch.int32, device="cuda")
    extra_indices = torch.randint(
        0, 4096, (tokens, 1, topk), dtype=torch.int32, device="cuda"
    )
    extra_lens = torch.full((tokens,), topk, dtype=torch.int32, device="cuda")
    sink = torch.zeros(64, device="cuda")
    sink[heads:] = -float("inf")
    scale = 512**-0.5
    metadata = get_mla_metadata()[0]

    def baseline(
        q,
        swa_cache,
        swa_indices,
        swa_lens,
        extra_cache,
        extra_indices,
        extra_lens,
        sink,
        out,
    ):
        return flash_mla_with_kvcache(
            q=q.unsqueeze(1),
            k_cache=swa_cache.unsqueeze(-2),
            block_table=None,
            head_dim_v=512,
            tile_scheduler_metadata=metadata,
            cache_seqlens=None,
            is_fp8_kvcache=True,
            indices=swa_indices,
            topk_length=swa_lens,
            softmax_scale=scale,
            attn_sink=sink,
            extra_k_cache=extra_cache.unsqueeze(-2) if topk else None,
            extra_indices_in_kvcache=extra_indices if topk else None,
            extra_topk_length=extra_lens if topk else None,
            out=out.unsqueeze(1),
        )[0]

    def candidate(
        q,
        swa_cache,
        swa_indices,
        swa_lens,
        extra_cache,
        extra_indices,
        extra_lens,
        sink,
        out,
    ):
        small_head_sparse_decode(
            q,
            swa_cache,
            swa_indices,
            swa_lens,
            extra_cache if topk else None,
            extra_indices if topk else None,
            extra_lens if topk else None,
            sink,
            scale,
            out,
            heads,
        )

    inputs = (
        q,
        swa_cache,
        swa_indices,
        swa_lens,
        extra_cache,
        extra_indices,
        extra_lens,
        sink,
        out,
    )
    expected = baseline(*inputs).squeeze(1)[:, :heads].clone()
    candidate(*inputs)
    torch.testing.assert_close(out[:, :heads], expected, atol=2e-2, rtol=2e-2)
    times: dict[str, list[float]] = {"flashmla": [], "native": []}
    for round_idx in range(rounds):
        arms = [("flashmla", baseline), ("native", candidate)]
        if round_idx % 2:
            arms.reverse()
        for name, fn in arms:
            samples = bench_gpu_time_with_cupti(
                fn,
                input_args=inputs,
                use_cuda_graph=True,
                cold_l2_cache=True,
                dry_run_time_ms=25,
                repeat_time_ms=100,
            )
            times[name].append(statistics.median(samples) * 1000)
    base_us, native_us = (statistics.median(times[n]) for n in times)
    return [tokens, heads, topk, base_us, native_us, base_us / native_us]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[*range(1, 17), 24, 32]
    )
    parser.add_argument(
        "--heads", type=int, nargs="+", choices=[8, 16], default=[8, 16]
    )
    parser.add_argument("--topk", type=int, nargs="+", default=[0, 64, 512])
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if not current_platform.is_device_capability_family(90):
        parser.error("requires SM90")
    torch.manual_seed(0)
    print(f"# {torch.cuda.get_device_name()}, torch={torch.__version__}", flush=True)
    writer = csv.writer(sys.stdout)
    writer.writerow(["tokens", "heads", "topk", "flashmla_us", "native_us", "speedup"])
    for heads in args.heads:
        for topk in args.topk:
            for tokens in args.tokens:
                writer.writerow(benchmark(tokens, heads, topk, args.rounds))
                sys.stdout.flush()


if __name__ == "__main__":
    main()
