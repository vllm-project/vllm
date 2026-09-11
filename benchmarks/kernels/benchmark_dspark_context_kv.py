# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare DSpark KV insertion with the former dummy-query implementation."""

import argparse
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch

from vllm.models.deepseek_v4_1.nvidia.dspark import _insert_context_kv
from vllm.triton_utils import triton


def legacy_insert(attn, kv, positions, slots):
    cache = attn.swa_cache_layer.kv_cache
    block_size = attn.swa_cache_layer.block_size
    cos_sin = attn.rotary_emb.cos_sin_cache
    q = torch.zeros(
        kv.shape[0], attn.n_local_heads, 512, dtype=kv.dtype, device=kv.device
    )
    if cache.dtype == torch.uint8:
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q,
            kv,
            cache.view(cache.shape[0], -1),
            slots,
            positions,
            cos_sin,
            attn.padded_heads,
            1e-20,
            block_size,
        )
    elif cache.dtype == torch.bfloat16:
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(
            q,
            kv,
            cache,
            slots,
            positions,
            cos_sin,
            1e-20,
            block_size,
        )
    else:
        q_fp8 = torch.zeros_like(q, dtype=cache.dtype)
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert(
            q,
            kv,
            q_fp8,
            cache,
            slots,
            positions,
            cos_sin,
            attn._flashinfer_fp8_kv_scale,
            attn._flashinfer_fp8_q_scale_inv,
            1e-20,
            block_size,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 8, 32, 128, 512, 2048]
    )
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    results = []
    for dtype in (torch.uint8, torch.bfloat16, torch.float8_e4m3fn):
        for tokens in args.tokens:
            block_size = 256
            blocks = (tokens + block_size - 1) // block_size
            cache = torch.zeros(
                blocks,
                block_size,
                584 if dtype == torch.uint8 else 512,
                device="cuda",
                dtype=dtype,
            )
            kv = torch.randn(tokens, 512, device="cuda", dtype=torch.bfloat16)
            positions = torch.arange(tokens, device="cuda")
            slots = torch.arange(tokens, device="cuda")
            angles = torch.randn(tokens, 32, device="cuda")
            scale = torch.ones(1, device="cuda")
            attn = SimpleNamespace(
                swa_cache_layer=SimpleNamespace(kv_cache=cache, block_size=block_size),
                head_dim=512,
                n_local_heads=args.heads,
                padded_heads=args.heads,
                rotary_emb=SimpleNamespace(
                    cos_sin_cache=torch.cat((angles.cos(), angles.sin()), -1)
                ),
                _flashinfer_fp8_kv_scale=scale,
                _flashinfer_fp8_q_scale_inv=scale,
            )
            row = {"dtype": str(dtype), "tokens": tokens, "heads": args.heads}
            for name, function in (
                ("dummy_q", legacy_insert),
                ("kv_only", _insert_context_kv),
            ):
                median, p10, p90 = triton.testing.do_bench(
                    partial(function, attn, kv, positions, slots),
                    warmup=50,
                    rep=200,
                    quantiles=[0.5, 0.1, 0.9],
                )
                row[name] = {
                    "median_us": median * 1000,
                    "p10_us": p10 * 1000,
                    "p90_us": p90 * 1000,
                }
            row["speedup"] = row["dummy_q"]["median_us"] / row["kv_only"]["median_us"]
            results.append(row)
            print(json.dumps(row), flush=True)
    args.output.write_text(
        json.dumps({"gpu": torch.cuda.get_device_name(), "results": results}, indent=2)
    )


if __name__ == "__main__":
    main()
