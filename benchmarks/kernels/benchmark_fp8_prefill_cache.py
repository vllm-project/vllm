# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare native FP8 prefill with gather/cast plus BF16 attention.

Check whether avoiding repeated FP8 conversion offsets the extra page copy.
Uses generated tensors and illustrative power-of-two sizes. Planning is
outside the timed region; results measure a layer kernel path, not model
serving throughput. Measure serving latency and capacity separately.
"""

import argparse
import json
import statistics

import torch

from vllm.v1.attention.ops.fp8_prefill_cache import expand_cache, plan_expansion


def measure(call, repeats):
    for _ in range(5):
        call()
    torch.accelerator.synchronize()
    samples = []
    for _ in range(repeats):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-tokens", type=int, default=4096)
    parser.add_argument(
        "--kv-tokens", type=int, nargs="+", default=[4096, 16384, 65536]
    )
    parser.add_argument("--layout", choices=["HND", "NHD"], default="NHD")
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if (
        args.query_tokens < 1024
        or args.repeats < 1
        or min(args.kv_tokens) < args.query_tokens
    ):
        parser.error("require query >= 1024, KV >= query, repeats >= 1")
    import flashinfer

    torch.manual_seed(2718)
    page_size, heads, kv_heads, dim = 32, 16, 4, 256
    query = torch.randn(
        (args.query_tokens, heads, dim), device="cuda", dtype=torch.bfloat16
    )
    for length in args.kv_tokens:
        pages = (length + page_size - 1) // page_size
        cache = (
            torch.randn((pages, 2, kv_heads, page_size, dim), device="cuda") * 6
        ).to(torch.float8_e4m3fn)
        if args.layout == "NHD":
            cache = cache.transpose(2, 3).contiguous()
        indptr = torch.tensor([0, pages], dtype=torch.int32)
        indices = torch.arange(pages, dtype=torch.int32, device="cuda")
        plan = plan_expansion(
            indptr,
            indices,
            num_query_tokens=args.query_tokens,
            num_kv_heads=kv_heads,
            page_size=page_size,
            head_dim=dim,
            budget=1 << 30,
            layout=args.layout,
        )
        if plan is None:
            raise RuntimeError("Requested geometry exceeds the expansion budget")
        wrappers = []
        for dtype in (torch.float8_e4m3fn, torch.bfloat16):
            wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                torch.empty(128 << 20, dtype=torch.uint8, device="cuda"),
                args.layout,
                backend="fa2",
            )
            wrapper.plan(
                torch.tensor([0, args.query_tokens], dtype=torch.int32),
                indptr,
                indices,
                torch.tensor([length % page_size or page_size], dtype=torch.int32),
                heads,
                kv_heads,
                dim,
                page_size,
                causal=True,
                q_data_type=torch.bfloat16,
                kv_data_type=dtype,
            )
            wrappers.append(wrapper)

        def native(wrapper=wrappers[0], cache=cache):
            return wrapper.run(query, cache, q_scale=1.0, k_scale=0.125, v_scale=0.25)

        def expanded(wrapper=wrappers[1], cache=cache, plan=plan):
            gathered = expand_cache(cache[:, 0], cache[:, 1], plan)
            return wrapper.run(
                query, gathered, q_scale=1.0, k_scale=0.125, v_scale=0.25
            )

        reference, candidate = native().float(), expanded().float()
        relative_difference = ((candidate - reference).norm() / reference.norm()).item()
        print(
            json.dumps(
                {
                    "query_tokens": args.query_tokens,
                    "kv_tokens": length,
                    "layout": args.layout,
                    "repeats": args.repeats,
                    "native": measure(native, args.repeats),
                    "expansion_plus_attention": measure(expanded, args.repeats),
                    "scratch_capacity_bytes": plan.cache.untyped_storage().nbytes(),
                    "relative_difference": relative_difference,
                }
            )
        )


if __name__ == "__main__":
    main()
