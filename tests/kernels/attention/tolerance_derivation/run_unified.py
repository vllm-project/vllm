# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A priori tolerance derivation for tests/kernels/attention/test_rocm_aiter_unified_attn.py.

The reference is `test_triton_unified_attention.ref_paged_attn`, which has the
same precision structure as the flash-attention one (einsum on working-precision
operands, so the logits are rounded before the softmax). As in apriori_bound.py
the reference error is therefore computed exactly in float64 rather than bounded,
and only the kernel is bounded.

Per-element budget:

    E_d = |ref_d - o_d|      reference error, exact
        + |o_q8_d - o_d|     fp8 query quantization, exact (fp8_query variants)
        + u * A_d            kernel P rounding, A_d = sum_j p_j |v_jd|
        + u * |o_d|          kernel output rounding

The fp8 KV variants need no quantization term: the reference dequantizes the
*same* fp8 tensors the kernel reads, and because k_scale/v_scale are 0.5/0.25
the dequantizing multiply is exact in bf16, so both sides see bit-identical
values and the e4m3 rounding cancels out of the residual entirely.

The fp8 query variants do need one, and it dominates: the kernel sees
`(q/0.75).to(e4m3) * 0.75` while the reference sees `q`. That difference is
deterministic and known before the kernel runs, so it is computed exactly
rather than bounded -- bounding it via the softmax sensitivity gives
2*u_e4m3*G*A with G ~ 12, i.e. a tolerance near 1.0, which is useless.
"""

import argparse
import importlib.util
import json
import math
import sys
from collections import defaultdict

import torch

from pathlib import Path

from tests.kernels.attention.tolerance_derivation.core import vllm_repo_root

U = {torch.bfloat16: 2.0**-8, torch.float16: 2.0**-11}
U_FP8_E4M3 = 2.0**-4
LAMBDA = 4.0


def load(path, name):
    root = vllm_repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def exact_paged(query, key_cache, value_cache, query_lens, kv_lens, block_tables, scale):
    """float64 attention over the paged layout, plus A_d and B_d."""
    bt = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape
    o_p, a_p, b_p = [], [], []
    start = 0
    for i, (ql, kl) in enumerate(zip(query_lens, kv_lens)):
        nb = (kl + block_size - 1) // block_size
        idx = bt[i, :nb]
        q = query[start : start + ql].double() * scale
        k = key_cache[idx].view(-1, num_kv_heads, head_size)[:kl].double()
        v = value_cache[idx].view(-1, num_kv_heads, head_size)[:kl].double()
        if q.shape[1] != k.shape[1]:
            rep = q.shape[1] // k.shape[1]
            k = torch.repeat_interleave(k, rep, dim=1)
            v = torch.repeat_interleave(v, rep, dim=1)
        s = torch.einsum("qhd,khd->hqk", q, k)
        mask = torch.triu(
            torch.ones(ql, kl, device=s.device), diagonal=kl - ql + 1
        ).bool()
        s = s.masked_fill(mask, float("-inf"))
        p = torch.softmax(s, dim=-1)
        o_p.append(torch.einsum("hqk,khd->qhd", p, v))
        a_p.append(torch.einsum("hqk,khd->qhd", p, v.abs()))
        b_p.append(torch.einsum("hqk,khd->qhd", p.square(), v.square()).sqrt())
        start += ql
    return (
        torch.cat(o_p, dim=0),
        torch.cat(a_p, dim=0),
        torch.cat(b_p, dim=0),
    )


def run_group(mod, group, cases, seeds, results, rtol_override=None):
    from vllm.utils.torch_utils import set_random_seed

    for cfg_label, maker in cases:
        for s in range(seeds):
            set_random_seed(s)
            case = maker()
            dtype = case["query_dtype"]
            u = U[dtype]
            rtol = rtol_override or (1e-2 if dtype is torch.bfloat16 else 1e-3)

            # Kernel first: ref_paged_attn scales the query IN PLACE, so the
            # reference mutates case["query"]. The test relies on this ordering
            # too; reversing it would silently feed a pre-scaled query to the
            # kernel.
            out = mod._run_aiter_unified_attention(case).clone()
            query0 = case["query"].clone()

            # Caches exactly as the reference dequantizes them.
            kc, vc = case["key_cache"], case["value_cache"]
            if kc.dtype != dtype:
                kc = kc.to(dtype) * case["k_scale"]
                vc = vc.to(dtype) * case["v_scale"]

            ref = mod._ref_output(case)

            o64, a, b = exact_paged(
                query0,
                kc,
                vc,
                case["query_lens"],
                case["kv_lens"],
                case["block_tables"],
                case["scale"],
            )

            # When q_descale is supplied the kernel does not merely dequantize
            # an fp8 query against a bf16 cache: it runs the whole attention in
            # e4m3, quantizing K and V as well. Verified directly -- at the
            # first query token of a causal sequence p = 1 exactly, so the
            # output must equal v_0, and the kernel's error there is exactly
            # v_0 - e4m3(v_0) even when the V cache is bf16.
            # The quantization of Q, K and V is deterministic and known before
            # the kernel runs, so its effect is computed exactly; only the
            # rounding of P is bounded, at e4m3 precision.
            q_quant_term = torch.zeros_like(o64)
            a_k, b_k, u_p = a, b, u
            if case["q_descale"] is not None:
                from vllm.platforms import current_platform

                fp8 = current_platform.fp8_dtype()
                q_eff = (
                    case["kernel_query"].double() * case["q_descale"].double().item()
                )
                o_k, a_k, b_k = exact_paged(
                    q_eff,
                    kc.to(fp8).to(dtype),
                    vc.to(fp8).to(dtype),
                    case["query_lens"],
                    case["kv_lens"],
                    case["block_tables"],
                    case["scale"],
                )
                q_quant_term = (o_k - o64).abs()
                u_p = U_FP8_E4M3

            err_ref = (ref.double() - o64).abs()
            kernel_budget = u_p * a_k + u * o64.abs()
            prob_budget = LAMBDA * u_p * b_k / math.sqrt(3.0) + u * o64.abs()

            full = err_ref + q_quant_term + kernel_budget
            full_prob = err_ref + q_quant_term + prob_budget
            resid = (out.double() - ref.double()).abs()
            allowed = rtol * ref.double().abs()

            results.append(
                dict(
                    group=group,
                    cfg=cfg_label,
                    dtype=str(dtype),
                    seed=s,
                    rtol=rtol,
                    out_scale=o64.abs().amax().item(),
                    err_ref_max=err_ref.amax().item(),
                    q_quant_max=q_quant_term.amax().item(),
                    kernel_budget_max=kernel_budget.amax().item(),
                    atol_full=(full - allowed).clamp_min(0.0).amax().item(),
                    atol_prob=(full_prob - allowed).clamp_min(0.0).amax().item(),
                    atol_measured=(resid - allowed).amax().item(),
                    covers=float((resid <= full).all().item()),
                    prob_covers=float((resid <= full_prob).all().item()),
                    worst_ratio=(resid / full.clamp_min(1e-300)).amax().item(),
                    ref_share=(err_ref.amax() / full.amax()).item(),
                )
            )


def main(out: str | None = None, seeds: int = 5) -> list[dict]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=seeds)
    ap.add_argument(
        "--out",
        default=out or "apriori_unified.jsonl",
        help="JSONL output path",
    )
    args = ap.parse_args()

    mod = load(
        str(
            vllm_repo_root()
            / "tests/kernels/attention/test_rocm_aiter_unified_attn.py"
        ),
        "uni",
    )
    torch.set_default_device("cuda")
    results = []

    def mk(**kw):
        return lambda: mod._make_case(**kw)

    def mkf(**kw):
        return lambda: mod._make_fp8_case(**kw)

    mixed = [
        (
            f"sl{i} hs={hs} bs={bs} {str(dt).split('.')[-1]}",
            mk(seq_lens=sl, head_size=hs, block_size=bs, dtype=dt),
        )
        for i, sl in enumerate(mod.MIXED_SEQ_LENS)
        for hs in mod.HEAD_SIZES
        for bs in mod.BLOCK_SIZES
        for dt in mod.DTYPES
    ]
    decode = [
        (
            f"sl{i} hs={hs} bs={bs}",
            mk(seq_lens=sl, head_size=hs, block_size=bs, dtype=torch.bfloat16),
        )
        for i, sl in enumerate(mod.DECODE_SEQ_LENS)
        for hs in mod.HEAD_SIZES
        for bs in mod.BLOCK_SIZES
    ]
    prefill = [
        (
            f"sl{i} hs=128 bs=16",
            mk(seq_lens=sl, head_size=128, block_size=16, dtype=torch.bfloat16),
        )
        for i, sl in enumerate(mod.PREFILL_SEQ_LENS)
    ]

    run_group(mod, "UNIFIED_MIXED_BATCH", mixed, args.seeds, results)
    run_group(mod, "UNIFIED_DECODE", decode, args.seeds, results)
    run_group(mod, "UNIFIED_PREFILL", prefill, args.seeds, results)

    for variant in ("fp8_kv", "fp8_query", "fp8_query_kv"):
        cases = [
            (
                f"sl{i} hs=128 bs={bs}",
                mkf(seq_lens=sl, head_size=128, block_size=bs, variant=variant),
            )
            for i, sl in enumerate(mod.FP8_SEQ_LENS)
            for bs in (16, 64)
        ]
        # fp8 query variants use a larger rtol in the tests.
        rtol_v = 1.5e-1 if "QUERY" in variant.upper() else 1e-2
        run_group(
            mod, f"UNIFIED_{variant.upper()}", cases, args.seeds, results, rtol_v
        )

    out_path = Path(args.out)
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print("\n=== elementwise soundness ===")
    print(
        f"deterministic: covered {int(sum(r['covers'] for r in results))}/{len(results)}"
        f"   worst ratio {max(r['worst_ratio'] for r in results):.3f}"
    )
    print(
        f"probabilistic: covered "
        f"{int(sum(r['prob_covers'] for r in results))}/{len(results)}"
    )

    print("\n=== derived atol per group ===")
    g = defaultdict(list)
    for r in results:
        g[(r["group"], r["dtype"])].append(r)
    print(
        f"{'group':24} {'dtype':16} {'determ':>10} {'probab':>10} {'measured':>10} "
        f"{'ref_sh':>7} {'qquant':>10}  worst cfg"
    )
    for k, v in sorted(g.items()):
        w = max(v, key=lambda r: r["atol_full"])
        print(
            f"{k[0]:24} {k[1]:16} "
            f"{max(x['atol_full'] for x in v):10.3e} "
            f"{max(x['atol_prob'] for x in v):10.3e} "
            f"{max(x['atol_measured'] for x in v):10.3e} "
            f"{max(x['ref_share'] for x in v):7.3f} "
            f"{max(x['q_quant_max'] for x in v):10.3e}  {w['cfg']}"
        )


    return results


if __name__ == "__main__":
    main()
