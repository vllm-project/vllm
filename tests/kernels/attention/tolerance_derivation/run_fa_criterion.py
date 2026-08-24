# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Apply the upstream FlashAttention correctness criterion to AITER test 1.

Dao-AILab's flash-attention does NOT assert a fixed atol. It asserts that the
kernel's error against a high-precision golden is at most 2x the error of a
PyTorch baseline running at the SAME low precision:

    assert (out - out_ref).abs().max() <= 2 * (out_pt - out_ref).abs().max()

    tests/test_flash_attn.py, and in tests/cute/test_flash_attn.py with an
    added floor:
        fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max()
        rtol     = 2 (3 when softcap != 0)
        assert (out - out_ref).abs().max() <= rtol * (out_pt - out_ref).abs().max() + fwd_atol

where
    out_ref = attention_ref(..., upcast=True)    # fp32 golden, q/k/v upcast
                                                 # BEFORE the QK^T matmul
    out_pt  = attention_ref(..., upcast=False,   # same code at bf16/fp16, with
                            reorder_ops=True)    # ops deliberately reordered

The property that makes this work: reference rounding noise appears on BOTH
sides of the inequality, so it cancels out of the criterion instead of forcing
the tolerance open. A fixed atol tuned against a low-precision reference cannot
do that -- it has to be wide enough to swallow the reference's own error, which
is exactly what blinds it to real kernel bugs.

vLLM's `ref_paged_attn` is structurally `attention_ref(upcast=False)`: it is the
BASELINE, not the golden. The ROCm tests compare the kernel directly against it
with a fixed atol, so they inherit its noise as their sensitivity floor.

This script measures, for AITER test 1 (`test_aiter_fa_head_sizes` /
`test_aiter_mha_varlen_paged_kv`), what the upstream criterion says.
"""

import argparse
import json

import torch

from tests.kernels.attention.tolerance_derivation.fa_apriori import (
    build_single_seq,
    causal_window_mask,
    gather_kv,
)


def golden(
    query, key_cache, value_cache, query_lens, kv_lens, block_tables, scale, dtype, upcast_to
):
    """attention_ref equivalent. upcast_to=torch.float32 reproduces upstream's
    golden; passing the working dtype reproduces its `out_pt` baseline."""
    block_tables_np = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outs = []
    start = 0
    for i, (query_len, kv_len) in enumerate(zip(query_lens, kv_lens)):
        q = query[start : start + query_len].to(upcast_to)
        k = gather_kv(
            key_cache.to(upcast_to), block_tables_np, i, kv_len, block_size, num_kv_heads, head_size
        )
        v = gather_kv(
            value_cache.to(upcast_to), block_tables_np, i, kv_len, block_size, num_kv_heads, head_size
        )
        if q.shape[1] != k.shape[1]:
            rep = q.shape[1] // k.shape[1]
            k = torch.repeat_interleave(k, rep, dim=1)
            v = torch.repeat_interleave(v, rep, dim=1)

        scores = torch.einsum("qhd,khd->hqk", q * scale, k)
        mask = causal_window_mask(query_len, kv_len, None, scores.device)
        scores = scores.masked_fill(mask, float("-inf"))
        p = torch.softmax(scores, dim=-1)
        outs.append(torch.einsum("hqk,khd->qhd", p, v))
        start += query_len
    return torch.cat(outs, dim=0)


def main(out: str | None = None, seeds: int = 20) -> list[dict]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=seeds)
    ap.add_argument(
        "--out",
        default=out or "fa_criterion.jsonl",
        help="JSONL output path",
    )
    args = ap.parse_args()

    torch.set_default_device("cuda")

    configs = []
    # Union of the parametrization of both tests that share this builder:
    # test_aiter_fa_head_sizes uses (16,16),(16,4); test_aiter_mha_varlen_paged_kv
    # uses (8,8),(16,4). Sweep all three so both tests are fully covered.
    for head_size in (64, 128, 256):
        for num_heads in ((16, 16), (16, 4), (8, 8)):
            for seq_lens in ((8, 512), (32, 1024)):
                for dtype in (torch.bfloat16, torch.float16):
                    configs.append((head_size, num_heads, seq_lens, dtype))

    rows = []
    with open(args.out, "w") as fh:
        for head_size, num_heads, seq_lens, dtype in configs:
            for seed in range(args.seeds):
                case = build_single_seq(head_size, num_heads, seq_lens, dtype, seed)
                out = case["kernel_out"]
                out_vllm_ref = case["ref_out"]

                common = dict(
                    query=case["query"],
                    key_cache=case["key_cache"],
                    value_cache=case["value_cache"],
                    query_lens=case["query_lens"],
                    kv_lens=case["kv_lens"],
                    block_tables=case["block_tables"],
                    scale=case["scale"],
                    dtype=dtype,
                )
                out_ref = golden(**common, upcast_to=torch.float32)
                out_ref64 = golden(**common, upcast_to=torch.float64)

                err_kernel = (out.float() - out_ref).abs().amax().item()
                err_pt = (out_vllm_ref.float() - out_ref).abs().amax().item()
                # upstream's machine-computed atol floor: one ulp of the output
                # at its own magnitude, forced out by a round trip through +0.3
                fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().amax().item()

                ratio = err_kernel / max(err_pt, 1e-30)
                row = {
                    "head_size": head_size,
                    "num_heads": list(num_heads),
                    "seq_lens": list(seq_lens),
                    "dtype": str(dtype),
                    "seed": seed,
                    "err_kernel": err_kernel,
                    "err_pt": err_pt,
                    "ratio": ratio,
                    "fwd_atol": fwd_atol,
                    "fa_pass_strict": bool(err_kernel <= 2 * err_pt),
                    "fa_pass_floor": bool(err_kernel <= 2 * err_pt + fwd_atol),
                    # golden-precision sanity: fp32 vs fp64 golden should agree
                    "golden_gap": (out_ref.double() - out_ref64).abs().amax().item(),
                    # what a fixed atol against the vLLM reference would need
                    "residual_vs_vllm_ref": (out.float() - out_vllm_ref.float())
                    .abs()
                    .amax()
                    .item(),
                }
                rows.append(row)
                fh.write(json.dumps(row) + "\n")
                fh.flush()

    n = len(rows)
    ratios = sorted(r["ratio"] for r in rows)
    print(f"\ncases: {n}")
    print("\n=== upstream FlashAttention criterion: err_kernel <= 2 * err_baseline ===")
    print(f"strict form passes : {sum(r['fa_pass_strict'] for r in rows)}/{n}")
    print(f"with atol floor    : {sum(r['fa_pass_floor'] for r in rows)}/{n}")
    print(
        f"ratio err_kernel/err_baseline: min {ratios[0]:.3f}  "
        f"median {ratios[n // 2]:.3f}  p99 {ratios[int(0.99 * n)]:.3f}  max {ratios[-1]:.3f}"
    )
    print(f"\nfp32 vs fp64 golden gap (max): {max(r['golden_gap'] for r in rows):.3e}")

    print("\n=== per group ===")
    hdr = f"{'head_size':>9} {'num_heads':>10} {'seq_lens':>12} {'dtype':>16} "
    hdr += f"{'err_ker':>10} {'err_base':>10} {'ratio':>7} {'pass':>6}"
    print(hdr)
    groups = {}
    for r in rows:
        key = (r["head_size"], tuple(r["num_heads"]), tuple(r["seq_lens"]), r["dtype"])
        groups.setdefault(key, []).append(r)
    for key in sorted(groups):
        g = groups[key]
        worst = max(g, key=lambda r: r["ratio"])
        print(
            f"{key[0]:>9} {str(key[1]):>10} {str(key[2]):>12} {key[3]:>16} "
            f"{worst['err_kernel']:10.3e} {worst['err_pt']:10.3e} "
            f"{worst['ratio']:7.3f} {'ok' if all(x['fa_pass_strict'] for x in g) else 'FAIL':>6}"
        )
    return rows


if __name__ == "__main__":
    main()
