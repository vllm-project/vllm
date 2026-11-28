# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure CPU time for DiffKV cache-view derivation versus identity-cache hits."""

import json
from functools import partial
from pathlib import Path

import torch
from torch.utils.benchmark import Timer

from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import canonicalize_singleton_dim_strides
from vllm.v1.attention.backends.flash_attn_diffkv import FlashAttentionDiffKVImpl


def main():
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument("--min-run-time", type=float, default=0.2)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    results = []
    print("dtype,kv_heads,path,before_us,cached_us,before_iqr_us,cached_iqr_us")
    for dtype in (torch.bfloat16, torch.uint8):
        for num_kv_heads in (1, 4, 8):
            impl = FlashAttentionDiffKVImpl(
                num_heads=16,
                head_size=192,
                scale=192**-0.5,
                num_kv_heads=num_kv_heads,
                alibi_slopes=None,
                sliding_window=None,
                kv_cache_dtype="auto",
            )
            kv_cache = torch.empty(
                (1000, 16, num_kv_heads, 320), dtype=dtype, device="cpu"
            ).transpose(1, 2)

            def derive_read_views(kv_cache=kv_cache):
                key, value = kv_cache.transpose(1, 2).split(192, dim=-1)
                return (
                    canonicalize_singleton_dim_strides(key),
                    canonicalize_singleton_dim_strides(value),
                )

            impl._get_diffkv_views(kv_cache)
            for path, before in (
                ("write", lambda kv_cache=kv_cache: kv_cache.transpose(1, 2)),
                ("read", derive_read_views),
            ):
                measurements = [
                    Timer(
                        stmt="fn()", globals={"fn": fn}, num_threads=1
                    ).blocked_autorange(min_run_time=args.min_run_time)
                    for fn in (before, partial(impl._get_diffkv_views, kv_cache))
                ]
                row = {
                    "dtype": str(dtype),
                    "kv_heads": num_kv_heads,
                    "path": path,
                    "before_us": measurements[0].median * 1e6,
                    "cached_us": measurements[1].median * 1e6,
                    "before_iqr_us": measurements[0].iqr * 1e6,
                    "cached_iqr_us": measurements[1].iqr * 1e6,
                    "before_samples": len(measurements[0].raw_times),
                    "cached_samples": len(measurements[1].raw_times),
                    "before_calls_per_sample": measurements[0].number_per_run,
                    "cached_calls_per_sample": measurements[1].number_per_run,
                }
                results.append(row)
                print(
                    f"{dtype},{num_kv_heads},{path},"
                    f"{row['before_us']:.3f},{row['cached_us']:.3f},"
                    f"{row['before_iqr_us']:.3f},{row['cached_iqr_us']:.3f}"
                )
    if args.output_json:
        args.output_json.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
