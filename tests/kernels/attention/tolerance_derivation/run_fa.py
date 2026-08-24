# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A priori tolerance derivation for every flash-attention group in
tests/kernels/attention/test_rocm_aiter_fa.py.

All five groups share `ref_paged_attn`, so they share one derivation. See
apriori_bound.py for the term-by-term derivation and its validation (the
float64 replay reproduces the reference bit-exactly).

Per-element tolerance:

    atol = T0 + T1 + T2 + u * A_d

    T0  reference logit rounding   (einsum on bf16 operands returns bf16, so
                                    the scores are rounded before softmax)
    T1  reference P rounding
    T2  reference q*scale rounding (zero when head_size is a power of 4, since
                                    then head_size**-0.5 is a power of two)
    u*A kernel P rounding, deterministic, A_d = sum_j p_j |v_jd|

    T3 (output rounding, both sides) is proportional to |o_d| and is covered by
    rtol, not atol.

Nothing here reads the kernel output; the kernel is run only to check that the
derived tolerance actually covers the observed residual.
"""

import argparse
import json

import torch

from tests.kernels.attention.tolerance_derivation.fa_apriori import (
    U,
    analyze,
    ref_paged_attn,
)

NUM_BLOCKS = 2048
BLOCK_SIZE = 16


def build_case(
    *,
    seed,
    seq_lens,
    num_heads,
    head_size,
    dtype,
    num_blocks=NUM_BLOCKS,
    sliding_window=None,
    fp8_kv=False,
):
    import aiter
    from vllm.utils.torch_utils import set_random_seed
    from vllm.v1.attention.backends.rocm_aiter_fa import cp_mha_gather_cache

    set_random_seed(seed)

    num_q_heads, num_kv_heads = num_heads
    query_lens = [q for q, _ in seq_lens]
    kv_lens = [k for _, k in seq_lens]
    num_seqs = len(seq_lens)
    total_q, total_kv = sum(query_lens), sum(kv_lens)
    scale = head_size**-0.5

    query = torch.randn(total_q, num_q_heads, head_size, dtype=dtype)
    if fp8_kv:
        from vllm.platforms import current_platform

        FP8_DTYPE = current_platform.fp8_dtype()
        key_src = torch.clamp(
            torch.randn(num_blocks, BLOCK_SIZE, num_kv_heads, head_size), -1.0, 1.0
        ).to(FP8_DTYPE)
        value_src = torch.clamp(
            torch.randn(num_blocks, BLOCK_SIZE, num_kv_heads, head_size), -1.0, 1.0
        ).to(FP8_DTYPE)
        # The reference dequantizes these same fp8 tensors, so from the
        # derivation's point of view the inputs simply *are* these values.
        key_cache = key_src.to(dtype)
        value_cache = value_src.to(dtype)
    else:
        key_src = key_cache = torch.randn(
            num_blocks, BLOCK_SIZE, num_kv_heads, head_size, dtype=dtype
        )
        value_src = value_cache = torch.randn_like(key_cache)

    cu_q = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    cu_k = torch.tensor([0] + kv_lens, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    max_num_blocks = (max(kv_lens) + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks), dtype=torch.int32
    )

    token_to_batch = torch.empty(total_kv, dtype=torch.int32)
    idx = 0
    for b, kl in enumerate(kv_lens):
        token_to_batch[idx : idx + kl] = b
        idx += kl

    gathered_key = torch.empty(total_kv, num_kv_heads, head_size, dtype=dtype)
    gathered_value = torch.empty_like(gathered_key)
    cp_mha_gather_cache(
        key_cache=key_src,
        value_cache=value_src,
        key=gathered_key,
        value=gathered_value,
        block_tables=block_tables,
        k_scales=torch.ones(1, dtype=torch.float32),
        v_scales=torch.ones(1, dtype=torch.float32),
        cu_seqlens_kv=cu_k,
        token_to_batch=token_to_batch,
        seq_starts=torch.zeros(num_seqs, dtype=torch.int32),
        dequant=fp8_kv,
        kv_cache_layout="NHD",
        total_tokens=total_kv,
    )

    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    out = torch.empty_like(query)
    aiter.flash_attn_varlen_func(
        q=query,
        k=gathered_key,
        v=gathered_value,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(query_lens),
        max_seqlen_k=max(kv_lens),
        min_seqlen_q=1,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        alibi_slopes=None,
        return_lse=False,
        out=out,
    )

    ref = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        sliding_window=sliding_window,
    )
    return dict(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        kernel_out=out,
        ref_out=ref,
        sliding_window=sliding_window,
    )


DTYPES = (torch.bfloat16, torch.float16)
RTOL = {torch.bfloat16: 1e-2, torch.float16: 1e-3}


def groups():
    """Exact parametrization of each test in test_rocm_aiter_fa.py."""
    out = []

    # test_aiter_fa_head_sizes + test_aiter_mha_varlen_paged_kv: seed 0
    single_seq_num_heads = ((16, 16), (16, 4), (8, 8))
    for head_size in (64, 128, 256):
        for num_heads in single_seq_num_heads:
            for seq_lens in ((8, 512), (32, 1024)):
                for dtype in DTYPES:
                    out.append(
                        (
                            "FA_SINGLE_SEQ",
                            dtype,
                            f"nh={num_heads} hs={head_size} sl={seq_lens}",
                            dict(
                                seed=0,
                                seq_lens=[seq_lens],
                                num_heads=num_heads,
                                head_size=head_size,
                                dtype=dtype,
                            ),
                        )
                    )

    # test_aiter_mha_multi_batch: seed 42, fixed seq_lens, sweeps num_heads/head_size/dtype
    for num_heads in ((8, 8), (16, 4)):
        for head_size in (64, 128, 256):
            for dtype in DTYPES:
                out.append(
                    (
                        "FA_MULTI_BATCH",
                        dtype,
                        f"nh={num_heads} hs={head_size}",
                        dict(
                            seed=42,
                            seq_lens=[(4, 128), (2, 256), (8, 64)],
                            num_heads=num_heads,
                            head_size=head_size,
                            dtype=dtype,
                        ),
                    )
                )

    # test_aiter_mha_decode_single_token: seed 0, fixed shape. bf16 only (fp16 xfail).
    out.append(
        (
            "FA_DECODE",
            torch.bfloat16,
            "q=1 kv=512 hs=128",
            dict(
                seed=0,
                seq_lens=[(1, 512)],
                num_heads=(8, 8),
                head_size=128,
                dtype=torch.bfloat16,
            ),
        )
    )

    # test_aiter_fa_large_block_table_matches_reference: seed 0, num_blocks in {2048, 32768}
    for num_blocks in (2048, 32768):
        out.append(
            (
                "FA_DIRECT",
                torch.bfloat16,
                f"large_block_table nb={num_blocks}",
                dict(
                    seed=0,
                    seq_lens=[(10, 1328), (5, 18), (129, 463)],
                    num_heads=(8, 2),
                    head_size=128,
                    dtype=torch.bfloat16,
                    num_blocks=num_blocks,
                ),
            )
        )

    # test_aiter_fa_sliding_window_matches_reference: seed 0, window 256
    out.append(
        (
            "FA_DIRECT",
            torch.bfloat16,
            "sliding_window=256",
            dict(
                seed=0,
                seq_lens=[(8, 523), (24, 37), (3, 2011)],
                num_heads=(8, 2),
                head_size=128,
                dtype=torch.bfloat16,
                num_blocks=2048,
                sliding_window=256,
            ),
        )
    )

    # test_aiter_mha_varlen_fp8_kv: seed 10, fixed shape, both dtypes
    for dtype in DTYPES:
        out.append(
            (
                "FA_FP8_KV",
                dtype,
                "q=4 kv=128 hs=128",
                dict(
                    seed=10,
                    seq_lens=[(4, 128)],
                    num_heads=(8, 8),
                    head_size=128,
                    dtype=dtype,
                    fp8_kv=True,
                ),
            )
        )

    return out


def main(out: str | None = None, seeds: int = 10) -> list[dict]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=seeds)
    ap.add_argument(
        "--out",
        default=out or "apriori_fa_all.jsonl",
        help="JSONL output path (default: cwd/apriori_fa_all.jsonl)",
    )
    args = ap.parse_args()

    torch.set_default_device("cuda")
    rows = []
    with open(args.out, "w") as fh:
        for name, dtype, label, kw in groups():
            for s in range(args.seeds):
                kw2 = dict(kw)
                kw2["seed"] = kw["seed"] + s
                case = build_case(**kw2)
                sw = case.pop("sliding_window")
                r = analyze(
                    label=f"{name} {label} seed={kw2['seed']}",
                    dtype=dtype,
                    rtol=RTOL[dtype],
                    sliding_window=sw,
                    **case,
                )
                r["group"] = name
                r["cfg"] = label
                rows.append(r)
                fh.write(json.dumps(r) + "\n")
                fh.flush()

    print(f"\nmodel fidelity (sim vs reference), must be 0: "
          f"{max(r['model_gap'] for r in rows):.3e}")
    print(f"budget bounds real residual, elementwise:       "
          f"{int(sum(r['full_covers'] for r in rows))}/{len(rows)}"
          f"  worst ratio {max(r['full_worst_ratio'] for r in rows):.3f}")
    print(f"share of budget that is reference noise: "
          f"median {sorted(r['ref_share'] for r in rows)[len(rows) // 2]:.2f}")

    print("\n=== derived atol per group ===")
    print(f"{'group':16} {'dtype':16} {'derived':>10} {'measured':>10} {'ratio':>7}")
    agg = {}
    for r in rows:
        agg.setdefault((r["group"], r["dtype"]), []).append(r)
    for key in sorted(agg):
        g = agg[key]
        d = max(x["atol_full"] for x in g)
        m = max(x["atol_measured"] for x in g)
        print(f"{key[0]:16} {key[1]:16} {d:10.3e} {m:10.3e} {d / m:7.2f}")

    print("\n=== worst config per group ===")
    for key in sorted(agg):
        w = max(agg[key], key=lambda r: r["atol_full"])
        print(f"{key[0]:16} {key[1]:16} {w['atol_full']:10.3e}  {w['label']}")
    return rows


if __name__ == "__main__":
    main()
