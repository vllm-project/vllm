# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A priori error bound for the ROCm AITER flash-attention paths.

The point of this script is to produce a tolerance that does NOT depend on
what the kernel currently outputs. It does three things:

1. Computes an "exact" attention output in float64.
2. Derives a per-element error bound from the algorithm, the unit roundoff,
   and the *inputs only* -- never from the kernel output.
3. Measures the true error of the bf16/fp16 reference AND of the kernel
   against the float64 result, and checks each against the derived bound.

Step 3 is the non-circular part. The reference is fully under our control, so
if its measured error exceeds the derived bound the derivation is wrong. If
the derivation survives that check and the *kernel* then exceeds the bound,
the kernel is wrong.

Derivation (flash-attention path, reference `ref_paged_attn`)
------------------------------------------------------------
The reference performs, in order:

    q      = query * scale                  rounded to working precision
    s      = einsum(q, k) -> float32        fp32 accumulate
    p      = softmax(s)                     in fp32
    p_hat  = p.to(v.dtype)                  rounded to working precision
    o      = einsum(p_hat, v)               fp32 accumulate, result rounded

Writing u for the unit roundoff of the working precision, and letting the
exact quantities carry no hat:

T1  P rounding.       p_hat_j = p_j (1 + d_j),  |d_j| <= u.
    Contribution  sum_j p_j d_j v_jd,  bounded by  u * A_d
    with  A_d = sum_j p_j |v_jd|.
    Note A_d >= |o_d| and A_d <= max_j |v_jd|, so this term is *not*
    proportional to the output element -- which is exactly why an atol is
    needed at all and why rtol alone cannot cover it.

T2  Scaling of q.     q is multiplied by `scale` in working precision, so
    q_d -> q_d (1 + a_d), |a_d| <= u. This perturbs the logits by
    ds_j = sum_d q_d a_d k_jd, bounded by u * G with
    G = max_j sum_d |q_d k_jd|.
    Softmax turns a logit perturbation into a relative probability
    perturbation of at most 2 max_j |ds_j| (the factor 2 covers the shift of
    the normalizing constant), so the contribution is bounded by
    2 * u * G * A_d.

T3  Output rounding.  u * |o_d|. This term IS proportional to the output
    element, so it is what rtol covers.

T4  fp32 accumulation of both einsums and the fp32 softmax. Probabilistically
    ~sqrt(n log n) * u_fp32; for n <= 4096 that is ~1e-5 relative, two to
    three decades below u_bf16. Computed and reported, not assumed.

Deterministic bound:   u * A_d * (1 + 2G) + u * |o_d|
Probabilistic bound:   lam * u * (B_d * (1 + 2G)) / sqrt(3) + u * |o_d|
    with B_d = sqrt(sum_j p_j^2 v_jd^2), i.e. the l2 concentration of T1 under
    the Higham & Mary model where the d_j are independent and mean zero.
    lam is the confidence multiplier (lam = 4 used below).

The kernel's internal sequence differs (online softmax, tiled reduction) but
it rounds P to the same working precision and accumulates in fp32, so the same
term structure applies to it. It applies `scale` in fp32, so T2 is a reference
-only term; it is kept in the shared bound because the test compares the two.
"""

import argparse
import json
import math

import torch

U = {
    torch.bfloat16: 2.0**-8,
    torch.float16: 2.0**-11,
}
U_FP32 = 2.0**-24
LAMBDA = 4.0


def ref_paged_attn(
    query,
    key_cache,
    value_cache,
    query_lens,
    kv_lens,
    block_tables,
    scale,
    sliding_window=None,
):
    """Verbatim copy of tests/kernels/attention/test_rocm_aiter_fa.py::ref_paged_attn.

    Inlined rather than imported: importing the test module drags in the vLLM
    package tree, whose `tokenizers/` directory shadows the real `tokenizers`
    package once the repo is on sys.path.
    """
    num_seqs = len(query_lens)
    block_tables_np = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len] * scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables_np[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)

        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        mask = torch.triu(
            torch.ones(query_len, kv_len), diagonal=kv_len - query_len + 1
        ).bool()
        if sliding_window is not None:
            window_mask = (
                torch.triu(
                    torch.ones(query_len, kv_len),
                    diagonal=kv_len - (query_len + sliding_window) + 1,
                )
                .bool()
                .logical_not()
            )
            mask |= window_mask
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)
        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


def gather_kv(cache, block_tables_np, seq_idx, kv_len, block_size, num_kv_heads, head_size):
    num_kv_blocks = (kv_len + block_size - 1) // block_size
    block_indices = block_tables_np[seq_idx, :num_kv_blocks]
    return cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]


def causal_window_mask(query_len, kv_len, sliding_window, device):
    mask = torch.triu(
        torch.ones(query_len, kv_len, device=device),
        diagonal=kv_len - query_len + 1,
    ).bool()
    if sliding_window is not None:
        window_mask = (
            torch.triu(
                torch.ones(query_len, kv_len, device=device),
                diagonal=kv_len - (query_len + sliding_window) + 1,
            )
            .bool()
            .logical_not()
        )
        mask |= window_mask
    return mask


def exact_and_bound(
    query,
    key_cache,
    value_cache,
    query_lens,
    kv_lens,
    block_tables,
    scale,
    sliding_window,
    u,
    dtype_of,
):
    """float64 output plus the derived per-element bound. Kernel never touched."""
    block_tables_np = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    o_parts, det_parts, prob_parts, a_parts, sim_parts = [], [], [], [], []
    t0_parts, t1_parts, t2_parts, t3_parts, derived_parts = [], [], [], [], []
    derived_prob_parts = []
    kernel_budget_parts = []
    g_max = 0.0
    start = 0
    for i, (query_len, kv_len) in enumerate(zip(query_lens, kv_lens)):
        q64 = query[start : start + query_len].double() * scale
        k64 = gather_kv(
            key_cache.double(), block_tables_np, i, kv_len, block_size, num_kv_heads, head_size
        )
        v64 = gather_kv(
            value_cache.double(), block_tables_np, i, kv_len, block_size, num_kv_heads, head_size
        )
        if q64.shape[1] != k64.shape[1]:
            rep = q64.shape[1] // k64.shape[1]
            k64 = torch.repeat_interleave(k64, rep, dim=1)
            v64 = torch.repeat_interleave(v64, rep, dim=1)

        s = torch.einsum("qhd,khd->hqk", q64, k64)
        mask = causal_window_mask(query_len, kv_len, sliding_window, s.device)
        s = s.masked_fill(mask, float("-inf"))
        p = torch.softmax(s, dim=-1)

        o = torch.einsum("hqk,khd->qhd", p, v64)
        # A_d = sum_j p_j |v_jd|  (worst-case scale of the P-rounding term)
        a = torch.einsum("hqk,khd->qhd", p, v64.abs())
        # B_d = sqrt(sum_j p_j^2 v_jd^2)  (l2 concentration of the same term)
        b = torch.einsum("hqk,khd->qhd", p.square(), v64.square()).sqrt()
        # G = max_j sum_d |q_d k_jd|  (logit perturbation from rounding q*scale)
        g = torch.einsum("qhd,khd->hqk", q64.abs(), k64.abs())
        g = g.masked_fill(mask, 0.0).amax().item()
        g_max = max(g_max, g)

        det = u * a * (1.0 + 2.0 * g) + u * o.abs()
        # Kernel-side bound. The kernel applies `scale` in fp32 so it carries no
        # T2; its output rounding (T3) is proportional to |o| and is therefore
        # covered by rtol, not atol. What is left for atol is its P rounding,
        # bounded probabilistically under the Higham & Mary model where the
        # per-weight rounding errors are independent and mean zero.
        prob = LAMBDA * u * b / math.sqrt(3.0)

        # Exact simulation of the reference's own rounding, in float64. Every
        # rounding the reference performs is deterministic given the inputs, so
        # this is not a bound -- it is the reference's error, computed without
        # ever running the kernel.
        # Faithful replay of the reference, dtype for dtype. Note that
        # `torch.einsum` on bf16/fp16 operands returns that same dtype, so the
        # logits are rounded to working precision *before* `.float()` and before
        # the softmax. That rounding (T0) is the reference's largest error
        # source and is an artifact of the reference, not of the kernel.
        qs = query[start : start + query_len] * scale
        ks = key_cache.new_empty(0)  # placeholder, k gathered below in dtype
        k_w = gather_kv(
            key_cache, block_tables_np, i, kv_len, block_size, num_kv_heads, head_size
        )
        v_w = gather_kv(
            value_cache, block_tables_np, i, kv_len, block_size, num_kv_heads, head_size
        )
        if qs.shape[1] != k_w.shape[1]:
            rep = qs.shape[1] // k_w.shape[1]
            k_w = torch.repeat_interleave(k_w, rep, dim=1)
            v_w = torch.repeat_interleave(v_w, rep, dim=1)
        s_sim = torch.einsum("qhd,khd->hqk", qs, k_w).float()
        s_sim = s_sim.masked_fill(mask, float("-inf"))
        p_sim = torch.softmax(s_sim, dim=-1).to(dtype_of)
        o_sim = torch.einsum("hqk,khd->qhd", p_sim, v_w).double()

        # T0 only: logits rounded to working precision, everything else exact.
        s_t0 = torch.einsum("qhd,khd->hqk", q64, k64).to(dtype_of).double()
        s_t0 = s_t0.masked_fill(mask, float("-inf"))
        t0_only = torch.einsum("hqk,khd->qhd", torch.softmax(s_t0, dim=-1), v64)
        t0_parts.append((t0_only - o).abs())

        # Isolate each rounding step by enabling exactly one at a time.
        # T2 only: q*scale rounded to working precision, everything else exact.
        s_t2 = torch.einsum("qhd,khd->hqk", qs.double(), k64)
        s_t2 = s_t2.masked_fill(mask, float("-inf"))
        t2_only = torch.einsum("hqk,khd->qhd", torch.softmax(s_t2, dim=-1), v64)
        # T1 only: P rounded, q and output exact.
        t1_only = torch.einsum("hqk,khd->qhd", p.to(dtype_of).double(), v64)
        # T3 only: output rounded, everything upstream exact.
        t3_only = o.to(dtype_of).double()

        t1_parts.append((t1_only - o).abs())
        t2_parts.append((t2_only - o).abs())
        t3_parts.append((t3_only - o).abs())

        # Derived atol for this element. The reference side is computed exactly
        # -- we own that code, so its rounding needs no bounding at all -- and
        # only the kernel side is bounded. Neither quantity touches the kernel
        # output, so this is an a priori tolerance.
        # Kernel side uses the DETERMINISTIC P-rounding bound u * A_d. The
        # probabilistic form is not available here: the kernel's tiled online
        # softmax correlates its rounding errors, so the independent mean-zero
        # model that licenses lambda*u*B/sqrt(3) does not hold for it.
        # This stays tight only because the q-scaling term is computed exactly
        # rather than bounded -- bounding it is what produced the (1+2G) blowup.
        derived_parts.append(
            (t0_only - o).abs() + (t1_only - o).abs() + (t2_only - o).abs() + u * a
        )
        # Per-element budget for the quantity the test asserts on. Only the
        # kernel is bounded; the reference contributes its *exact* error,
        # added by the caller. The sum of the isolated T-terms is deliberately
        # NOT used here: each T is a point evaluation of a signed error, so the
        # terms cancel against each other and their sum is not an upper bound
        # (measured elementwise, it under-predicts by up to 36x).
        #   u*A_d  P rounding, A_d = sum_j p_j |v_jd| >= |o_d|
        #   u*|o|  output rounding
        # The kernel applies `scale` and keeps the logits in fp32, so it has no
        # counterpart to the reference's T0/T2 terms.
        kernel_budget_parts.append(u * a + u * o.abs())
        derived_prob_parts.append(
            (t1_only - o).abs() + (t2_only - o).abs() + LAMBDA * u * b / math.sqrt(3.0)
        )

        o_parts.append(o)
        a_parts.append(a)
        det_parts.append(det)
        prob_parts.append(prob)
        sim_parts.append(o_sim)
        start += query_len

    return (
        torch.cat(o_parts, dim=0),
        torch.cat(det_parts, dim=0),
        torch.cat(prob_parts, dim=0),
        torch.cat(a_parts, dim=0),
        torch.cat(sim_parts, dim=0),
        torch.cat(t0_parts, dim=0),
        torch.cat(t1_parts, dim=0),
        torch.cat(t2_parts, dim=0),
        torch.cat(t3_parts, dim=0),
        torch.cat(derived_parts, dim=0),
        torch.cat(derived_prob_parts, dim=0),
        torch.cat(kernel_budget_parts, dim=0),
        g_max,
    )


def required_atol(actual, expected, rtol):
    """Smallest atol that makes assert_close(actual, expected, atol, rtol) pass."""
    diff = (actual.double() - expected.double()).abs()
    allowed_rel = rtol * expected.double().abs()
    return (diff - allowed_rel).amax().item()


def analyze(
    *,
    label,
    query,
    key_cache,
    value_cache,
    query_lens,
    kv_lens,
    block_tables,
    scale,
    kernel_out,
    ref_out,
    dtype,
    rtol,
    sliding_window=None,
):
    u = U[dtype]
    (
        o64,
        det,
        prob,
        a,
        o_sim,
        t0,
        t1,
        t2,
        t3,
        derived,
        derived_prob,
        kernel_budget,
        g,
    ) = exact_and_bound(
        query,
        key_cache,
        value_cache,
        query_lens,
        kv_lens,
        block_tables,
        scale,
        sliding_window,
        u,
        dtype,
    )

    err_ref = (ref_out.double() - o64).abs()
    err_ker = (kernel_out.double() - o64).abs()
    err_sim = (o_sim - o64).abs()
    # Fidelity of the first-principles simulation to the real reference. If this
    # is not tiny, the derivation has missed a rounding step.
    model_gap = (o_sim - ref_out.double()).abs().amax().item()

    # Bound on the residual the *test* asserts on: both sides carry the bound.
    det_pair = 2.0 * det
    prob_pair = math.sqrt(2.0) * prob

    resid = (kernel_out.double() - ref_out.double()).abs()
    # Per-element budget for |kernel - reference|. The reference term is exact
    # (the float64 replay reproduces the reference bit for bit, so there is
    # nothing to bound on that side); only the kernel is bounded.
    full = err_ref + kernel_budget
    # Scalar atol implied by that budget, given that the assertion already
    # grants rtol * |expected|. This is the number the test should use.
    atol_full = (full - rtol * ref_out.double().abs()).clamp_min(0.0).amax().item()

    return {
        "label": label,
        "dtype": str(dtype),
        "u": u,
        "G": g,
        "out_scale": o64.abs().amax().item(),
        "A_max": a.amax().item(),
        # measured against float64, per side
        "err_ref_max": err_ref.amax().item(),
        "err_ker_max": err_ker.amax().item(),
        # first-principles simulation of the reference, no kernel involved
        "err_sim_max": err_sim.amax().item(),
        "model_gap": model_gap,
        "model_gap_rel": model_gap / max(err_ref.amax().item(), 1e-300),
        # exact per-term contributions to the reference error, each computed
        # with exactly one rounding step enabled and the rest kept in float64
        "T0_logit_rounding": t0.amax().item(),
        "T1_P_rounding": t1.amax().item(),
        "T2_q_scaling": t2.amax().item(),
        "T3_out_rounding": t3.amax().item(),
        # the a priori tolerance, and whether the real residual respects it
        "atol_derived": derived.amax().item(),
        "derived_covers": float(
            (
                (kernel_out.double() - ref_out.double()).abs()
                - rtol * ref_out.double().abs()
                <= derived
            )
            .all()
            .item()
        ),
        "derived_headroom": derived.amax().item()
        / max(required_atol(kernel_out, ref_out, rtol), 1e-300),
        # element-by-element budget, no appeal to rtol anywhere
        "atol_full": atol_full,
        "full_covers": float((resid <= full).all().item()),
        "full_worst_ratio": (resid / full.clamp_min(1e-300)).amax().item(),
        # how much of the budget is the reference's own noise rather than the
        # kernel's -- i.e. how much of the tolerance is spent on the test's own
        # reference implementation
        "ref_share": (err_ref.amax() / full.amax()).item(),
        "kernel_budget_max": kernel_budget.amax().item(),
        "atol_derived_prob": derived_prob.amax().item(),
        "derived_prob_covers": float(
            (
                (kernel_out.double() - ref_out.double()).abs()
                - rtol * ref_out.double().abs()
                <= derived_prob
            )
            .all()
            .item()
        ),
        # derived, per side
        "bound_det_max": det.amax().item(),
        "bound_prob_max": prob.amax().item(),
        # is the derivation sound? ref is fully under our control
        "ref_over_det": (err_ref / det).amax().item(),
        "ref_over_prob": (err_ref / prob).amax().item(),
        # is the kernel behaving like a correct implementation?
        "ker_over_det": (err_ker / det).amax().item(),
        "ker_over_prob": (err_ker / prob).amax().item(),
        # what the test tolerance should be, derived
        "atol_det": det_pair.amax().item(),
        "atol_prob": prob_pair.amax().item(),
        # what the test tolerance would be if calibrated (the old method)
        "atol_measured": required_atol(kernel_out, ref_out, rtol),
        "fp32_accum_rel": math.sqrt(max(kv_lens) * math.log(max(max(kv_lens), 2))) * U_FP32,
    }


def build_single_seq(head_size, num_heads, seq_lens, dtype, seed):
    from vllm.utils.torch_utils import set_random_seed

    import aiter

    from vllm.v1.attention.backends.rocm_aiter_fa import cp_mha_gather_cache

    NUM_BLOCKS, BLOCK_SIZE = 2048, 16
    set_random_seed(seed)
    num_q_heads, num_kv_heads = num_heads
    query_len, kv_len = seq_lens
    scale = head_size**-0.5

    query = torch.randn(query_len, num_q_heads, head_size, dtype=dtype)
    key_cache = torch.randn(NUM_BLOCKS, BLOCK_SIZE, num_kv_heads, head_size, dtype=dtype)
    value_cache = torch.randn_like(key_cache)

    cu_q = torch.tensor([0, query_len], dtype=torch.int32).cumsum(0, dtype=torch.int32)
    cu_k = torch.tensor([0, kv_len], dtype=torch.int32).cumsum(0, dtype=torch.int32)
    max_num_blocks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables = torch.randint(0, NUM_BLOCKS, (1, max_num_blocks), dtype=torch.int32)

    gathered_key = torch.empty(kv_len, num_kv_heads, head_size, dtype=dtype)
    gathered_value = torch.empty_like(gathered_key)
    cp_mha_gather_cache(
        key_cache=key_cache,
        value_cache=value_cache,
        key=gathered_key,
        value=gathered_value,
        block_tables=block_tables,
        k_scales=torch.ones(1, dtype=torch.float32),
        v_scales=torch.ones(1, dtype=torch.float32),
        cu_seqlens_kv=cu_k,
        token_to_batch=torch.zeros(kv_len, dtype=torch.int32),
        seq_starts=torch.zeros(1, dtype=torch.int32),
        dequant=False,
        kv_cache_layout="NHD",
        total_tokens=kv_len,
    )

    out = torch.empty_like(query)
    aiter.flash_attn_varlen_func(
        q=query,
        k=gathered_key,
        v=gathered_value,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=query_len,
        max_seqlen_k=kv_len,
        min_seqlen_q=1,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        alibi_slopes=None,
        return_lse=False,
        out=out,
    )

    ref = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=[query_len],
        kv_lens=[kv_len],
        block_tables=block_tables,
        scale=scale,
    )
    return dict(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=[query_len],
        kv_lens=[kv_len],
        block_tables=block_tables,
        scale=scale,
        kernel_out=out,
        ref_out=ref,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default="/work/apriori_fa.jsonl")
    args = ap.parse_args()

    torch.set_default_device("cuda")

    configs = []
    for head_size in (64, 128, 256):
        for num_heads in ((16, 16), (16, 4)):
            for seq_lens in ((8, 512), (32, 1024)):
                for dtype in (torch.bfloat16, torch.float16):
                    configs.append((head_size, num_heads, seq_lens, dtype))

    rtol_for = {torch.bfloat16: 1e-2, torch.float16: 1e-3}
    results = []
    with open(args.out, "w") as fh:
        for head_size, num_heads, seq_lens, dtype in configs:
            for seed in range(args.seeds):
                case = build_single_seq(head_size, num_heads, seq_lens, dtype, seed)
                r = analyze(
                    label=f"single_seq hs={head_size} nh={num_heads} sl={seq_lens} seed={seed}",
                    dtype=dtype,
                    rtol=rtol_for[dtype],
                    **case,
                )
                results.append(r)
                fh.write(json.dumps(r) + "\n")
                fh.flush()

    print("\n=== model fidelity (first-principles sim vs real reference) ===")
    print("If the simulation reproduces the reference, the derivation is complete.")
    print(f"max |o_sim - o_ref|            = {max(r['model_gap'] for r in results):.4e}")
    print(f"max |o_sim - o_ref| / err_ref  = {max(r['model_gap_rel'] for r in results):.4f}")

    print("\n=== derivation soundness (reference vs float64, must be <= 1.0) ===")
    worst_ref_det = max(r["ref_over_det"] for r in results)
    worst_ref_prob = max(r["ref_over_prob"] for r in results)
    print(f"max err_ref / bound_det  = {worst_ref_det:.4f}")
    print(f"max err_ref / bound_prob = {worst_ref_prob:.4f}")

    print("\n=== kernel conformance (kernel vs float64, must be <= 1.0) ===")
    worst_ker_det = max(r["ker_over_det"] for r in results)
    worst_ker_prob = max(r["ker_over_prob"] for r in results)
    print(f"max err_ker / bound_det  = {worst_ker_det:.4f}")
    print(f"max err_ker / bound_prob = {worst_ker_prob:.4f}")

    print("\n=== kernel vs reference error against float64 ===")
    print("Ratio > 1 means the kernel is less accurate than the reference.")
    ratios = [r["err_ker_max"] / max(r["err_ref_max"], 1e-300) for r in results]
    print(f"max err_ker / err_ref = {max(ratios):.4f}   median = {sorted(ratios)[len(ratios)//2]:.4f}")

    print("\n=== exact per-term contribution to the reference error ===")
    print("Each term computed with one rounding enabled, the rest in float64.")
    print(f"{'term':22} {'max':>12} {'median':>12}")
    for k in ("T0_logit_rounding", "T1_P_rounding", "T2_q_scaling", "T3_out_rounding"):
        vals = sorted(r[k] for r in results)
        print(f"{k:22} {vals[-1]:12.4e} {vals[len(vals) // 2]:12.4e}")

    print("\n=== does the a priori tolerance cover the real residual? ===")
    n_cov = sum(r["derived_covers"] for r in results)
    n_cov_p = sum(r["derived_prob_covers"] for r in results)
    print(f"deterministic kernel term: covered {int(n_cov)}/{len(results)} cases")
    print(f"probabilistic kernel term: covered {int(n_cov_p)}/{len(results)} cases")
    hr = [r["derived_headroom"] for r in results]
    print(f"headroom over calibrated: min {min(hr):.2f}x  median {sorted(hr)[len(hr) // 2]:.2f}x  max {max(hr):.2f}x")

    print("\n=== derived vs calibrated tolerance, per group ===")
    print("\n=== element-wise soundness of the budget ===")
    print(f"covered {int(sum(r['full_covers'] for r in results))}/{len(results)}"
          f"   worst ratio {max(r['full_worst_ratio'] for r in results):.3f}")

    print(f"{'group':46} {'dtype':16} {'atol_full':>10} {'derived':>10} {'measured':>10} {'G':>8}")
    groups = {}
    for r in results:
        key = (r["label"].split(" seed=")[0], r["dtype"])
        g = groups.setdefault(key, [])
        g.append(r)
    for (lbl, dt), rows in sorted(groups.items()):
        print(
            f"{lbl:46} {dt:16} "
            f"{max(x['atol_full'] for x in rows):10.3e} "
            f"{max(x['atol_derived'] for x in rows):10.3e} "
            f"{max(x['atol_measured'] for x in rows):10.3e} "
            f"{max(x['G'] for x in rows):8.2f}"
        )


if __name__ == "__main__":
    main()
