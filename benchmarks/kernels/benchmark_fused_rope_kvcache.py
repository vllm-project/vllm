# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the fused ``rope + reshape_and_cache_flash`` CUDA kernel against
the unfused reference pipeline (``rotary_embedding`` then
``reshape_and_cache_flash``).

Sanity targets (decode + prefill bracket):
  * num_tokens=1     ≥ 2.0× speedup (launch-overhead bound)
  * num_tokens=2048  ≥ 1.3× speedup (HBM-bandwidth bound)
"""

import time

import torch
from tabulate import tabulate

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import set_random_seed

BLOCK_SIZE = 16
MAX_POS = 4096
SEED = 0


def _make_inputs(
    num_tokens,
    num_q_heads,
    num_kv_heads,
    head_size,
    dtype,
    kv_cache_dtype,
    num_blocks,
    device,
):
    rot_dim = head_size
    query = torch.randn(num_tokens, num_q_heads, head_size, dtype=dtype, device=device)
    key = torch.randn(num_tokens, num_kv_heads, head_size, dtype=dtype, device=device)
    value = torch.randn(num_tokens, num_kv_heads, head_size, dtype=dtype, device=device)
    cos_sin_cache = torch.randn(MAX_POS, rot_dim, dtype=dtype, device=device)
    positions = torch.randint(
        0, MAX_POS, (num_tokens,), dtype=torch.long, device=device
    )

    total_slots = num_blocks * BLOCK_SIZE
    slot_mapping = torch.randperm(total_slots, device=device)[:num_tokens].to(
        torch.long
    )

    cache_dtype = current_platform.fp8_dtype() if kv_cache_dtype != "auto" else dtype
    key_cache = torch.zeros(
        num_blocks,
        BLOCK_SIZE,
        num_kv_heads,
        head_size,
        dtype=cache_dtype,
        device=device,
    )
    value_cache = torch.zeros_like(key_cache)

    k_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    v_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    return (
        query,
        key,
        value,
        positions,
        cos_sin_cache,
        slot_mapping,
        key_cache,
        value_cache,
        k_scale,
        v_scale,
    )


@torch.inference_mode()
def time_us(fn, warmup_iters: int, num_iters: int) -> float:
    for _ in range(warmup_iters):
        fn()
    torch.accelerator.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        fn()
    torch.accelerator.synchronize()
    return (time.perf_counter() - start) / num_iters * 1e6  # microseconds


@torch.inference_mode()
def bench_one(
    num_tokens,
    head_config,
    dtype,
    kv_cache_dtype,
    is_neox,
    num_blocks,
    num_iters,
    warmup_iters,
    device,
):
    num_q_heads, num_kv_heads, head_size = head_config
    set_random_seed(SEED)

    inputs = _make_inputs(
        num_tokens,
        num_q_heads,
        num_kv_heads,
        head_size,
        dtype,
        kv_cache_dtype,
        num_blocks,
        device,
    )
    (
        query,
        key,
        value,
        positions,
        cos_sin_cache,
        slot_mapping,
        key_cache,
        value_cache,
        k_scale,
        v_scale,
    ) = inputs

    def fused():
        ops.fused_rope_and_reshape_cache_flash(
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            is_neox,
            key_cache,
            value_cache,
            slot_mapping,
            k_scale,
            v_scale,
            kv_cache_dtype,
        )

    def unfused():
        torch.ops._C.rotary_embedding(
            positions,
            query,
            key,
            head_size,
            cos_sin_cache,
            is_neox,
            0,
            False,
        )
        ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )

    return time_us(fused, warmup_iters, num_iters), time_us(
        unfused, warmup_iters, num_iters
    )


def main(args):
    device = torch.device("cuda")
    dtype_cache_combos = [
        (torch.bfloat16, "auto"),
        (torch.float16, "auto"),
        (torch.bfloat16, "fp8_e4m3"),
        (torch.float16, "fp8_e4m3"),
    ]
    head_configs = [
        ("MHA(32,32,128)", (32, 32, 128)),
        ("GQA(32,8,128)", (32, 8, 128)),
    ]
    num_tokens_list = args.num_tokens

    rows = []
    for dtype, kv_cache_dtype in dtype_cache_combos:
        for head_label, head_config in head_configs:
            for n_tok in num_tokens_list:
                fused_us, unfused_us = bench_one(
                    num_tokens=n_tok,
                    head_config=head_config,
                    dtype=dtype,
                    kv_cache_dtype=kv_cache_dtype,
                    is_neox=True,
                    num_blocks=args.num_blocks,
                    num_iters=args.iters,
                    warmup_iters=args.warmup,
                    device=device,
                )
                speedup = unfused_us / fused_us if fused_us > 0 else float("inf")
                rows.append(
                    [
                        str(dtype).removeprefix("torch."),
                        kv_cache_dtype,
                        head_label,
                        n_tok,
                        f"{unfused_us:.2f}",
                        f"{fused_us:.2f}",
                        f"{speedup:.2f}×",
                    ]
                )

    print(
        f"Device: {torch.cuda.get_device_name(0)}  "
        f"iters={args.iters}  warmup={args.warmup}\n"
    )
    print(
        tabulate(
            rows,
            headers=[
                "dtype",
                "cache",
                "heads",
                "N",
                "unfused (µs)",
                "fused (µs)",
                "speedup",
            ],
        )
    )

    # Sanity targets.
    issues = []
    for row in rows:
        n = row[3]
        sp = float(row[6].rstrip("×"))
        if n == 1 and sp < 2.0:
            issues.append(f"  N=1 ({row[0]}, {row[1]}, {row[2]}): {sp:.2f}× < 2.0×")
        elif n == 2048 and sp < 1.3:
            issues.append(f"  N=2048 ({row[0]}, {row[1]}, {row[2]}): {sp:.2f}× < 1.3×")
    if issues:
        print("\n[warn] sanity targets missed:")
        for line in issues:
            print(line)
    else:
        print("\n[ok] all rows meet sanity targets (≥2× at N=1, ≥1.3× at N=2048).")


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument("--num-tokens", type=int, nargs="+", default=[1, 8, 128, 2048])
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    main(args)
