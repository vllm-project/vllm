# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A priori tolerance derivation for the AITER MLA tests.

Same principle as apriori_bound.py, but the MLA references are structurally
better than the flash-attention one: all three carry out the whole computation
in float32, so the reference contributes essentially no rounding noise and the
tolerance is set almost entirely by the kernel.

Per-element budget for |kernel - reference|:

    E_d = |ref_d - o_d|            reference error, computed exactly in float64
        + u_p * A_d                kernel rounding of P, A_d = sum_j p_j |v_jd|
        + u_o * |o_d|              kernel rounding of the output

u_p is the precision the kernel holds the attention weights in (bf16 for the
two decode paths, e4m3 for the fp8 prefill path) and u_o the precision of the
kernel's output buffer. A_d >= |o_d| and is not proportional to o_d, which is
why this term needs atol rather than rtol.

The scalar the test should use is then

    atol = max_d ( E_d - rtol * |ref_d| )

Nothing in E_d reads the kernel output.
"""

import argparse
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from tests.kernels.attention.tolerance_derivation.core import vllm_repo_root

U_BF16 = 2.0**-8
U_FP16 = 2.0**-11
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


def budget_from_pv(q64, k64, v64, scale, mask, u_p, u_o, head_chunk=16):
    """float64 attention plus the kernel's per-element error budget.

    q64 [H, Sq, Dqk], k64 [S, Dqk], v64 [S, Dv] (K/V shared across heads, which
    is the MQA/absorbed-MLA layout) or [H, S, D] when per-head.
    Returns (o, budget) each [H, Sq, Dv].
    """
    outs, buds, probs = [], [], []
    H = q64.shape[0]
    for i in range(0, H, head_chunk):
        q = q64[i : i + head_chunk]
        k = k64 if k64.dim() == 2 else k64[i : i + head_chunk]
        v = v64 if v64.dim() == 2 else v64[i : i + head_chunk]
        s = (q @ k.transpose(-1, -2)) * scale
        if mask is not None:
            s = s.masked_fill(mask, float("-inf"))
        p = torch.softmax(s, dim=-1)
        o = p @ v
        # A_d = sum_j p_j |v_jd|: worst case, every rounding error aligns.
        a = p @ v.abs()
        # B_d = sqrt(sum_j p_j^2 v_jd^2): the same sum under the Higham & Mary
        # model where the per-weight rounding errors are independent and mean
        # zero, so they accumulate in l2 rather than l1. lambda = 4 sets the
        # confidence level. This is the right scale when attention is spread
        # over many keys; the two bounds coincide when it is concentrated.
        b = (p.square() @ v.square()).sqrt()
        outs.append(o)
        buds.append(u_p * a + u_o * o.abs())
        probs.append(LAMBDA * u_p * b / math.sqrt(3.0) + u_o * o.abs())
    return torch.cat(outs, dim=0), torch.cat(buds, dim=0), torch.cat(probs, dim=0)


def summarize(label, group, cfg, kernel, ref, o64, budget, prob, rtol, extra=None):
    kernel = kernel.double()
    ref = ref.double()
    err_ref = (ref - o64).abs()
    full = err_ref + budget
    full_prob = err_ref + prob
    resid = (kernel - ref).abs()
    atol_full = (full - rtol * ref.abs()).clamp_min(0.0).amax().item()
    atol_prob = (full_prob - rtol * ref.abs()).clamp_min(0.0).amax().item()
    measured = (resid - rtol * ref.abs()).amax().item()
    rec = dict(
        atol_prob=atol_prob,
        prob_covers=float((resid <= full_prob).all().item()),
        label=label,
        group=group,
        cfg=cfg,
        dtype="torch.bfloat16",
        rtol=rtol,
        out_scale=o64.abs().amax().item(),
        err_ref_max=err_ref.amax().item(),
        err_ker_max=(kernel - o64).abs().amax().item(),
        budget_max=budget.amax().item(),
        atol_full=atol_full,
        atol_measured=measured,
        headroom=atol_full / max(measured, 1e-300),
        covers=float((resid <= full).all().item()),
        worst_ratio=(resid / full.clamp_min(1e-300)).amax().item(),
        ref_share=(err_ref.amax() / full.amax()).item(),
    )
    if extra:
        rec.update(extra)
    return rec


# ------------------------------------------------------------- MLA decode
def run_mla_decode(seeds, results):
    mod = load(
        str(vllm_repo_root() / "tests/kernels/attention/test_rocm_aiter_mla_decode.py"),
        "mla_dec",
    )
    from vllm.utils.torch_utils import set_random_seed

    for nhead in mod.NUM_HEADS:
        for bs in mod.BATCH_SIZES:
            for kv in mod.KV_SEQ_LENS:
                for contig in (True, False):
                    group = "MLA_DECODE" + ("" if contig else "_NONCONTIG")
                    cfg = f"nhead={nhead} bs={bs} kv={kv}"
                    for s in range(seeds):
                        set_random_seed(s)
                        inp = mod._make_inputs(bs, nhead, kv, contiguous_indices=contig)
                        out = mod._run_kernel(inp).clone()
                        ref = mod._ref_output(inp)

                        kv_flat = inp["kv_buffer"].squeeze(1)
                        o_parts, b_parts, p_parts = [], [], []
                        for b in range(bs):
                            lo = inp["kv_indptr"][b].item()
                            hi = inp["kv_indptr"][b + 1].item()
                            idx = inp["kv_indices"][lo:hi]
                            k64 = kv_flat[idx].double()
                            v64 = k64[:, : mod.V_HEAD_DIM]
                            q64 = inp["q"][b].double().unsqueeze(1)  # [H,1,Dqk]
                            o, bud, prb = budget_from_pv(
                                q64, k64, v64, mod.SM_SCALE, None, U_BF16, U_BF16
                            )
                            o_parts.append(o.squeeze(1))
                            b_parts.append(bud.squeeze(1))
                            p_parts.append(prb.squeeze(1))
                        o64 = torch.stack(o_parts)
                        budget = torch.stack(b_parts)
                        prob = torch.stack(p_parts)

                        results.append(
                            summarize(
                                f"{group} {cfg} seed={s}",
                                group,
                                cfg,
                                out,
                                ref,
                                o64,
                                budget,
                                prob,
                                mod.RTOL,
                                extra={},
                            )
                        )


# --------------------------------------------------------- MLA h12 decode
def run_h12(seeds, results):
    import types

    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAImpl

    mod = load(
        str(
            vllm_repo_root()
            / "tests/kernels/attention/test_rocm_aiter_mla_head_padding.py"
        ),
        "mla_pad",
    )
    device = torch.device("cuda:0")
    rtol = 1e-2
    for s in range(seeds):
        torch.manual_seed(s)
        q = torch.randn(
            1, mod.NUM_HEADS, mod.QK_HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        kv_cache = torch.randn(
            mod.CONTEXT_LEN, 1, mod.QK_HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        impl = object.__new__(AiterMLAImpl)
        impl.num_heads = mod.NUM_HEADS
        impl.kv_lora_rank = mod.KV_LORA_RANK
        impl.scale = mod.SCALE
        impl.kv_cache_dtype = "auto"
        one = torch.ones(1, dtype=torch.float32, device=device)
        layer = types.SimpleNamespace(_q_scale=one, _k_scale=one)
        metadata = mod._make_h12_decode_metadata(device)
        out, _ = impl.forward_mqa(q, kv_cache, metadata, layer)

        key = kv_cache[:, 0].float().unsqueeze(0)
        value = key[..., : mod.KV_LORA_RANK]
        ref = F.scaled_dot_product_attention(
            q[0].float().unsqueeze(1), key, value, scale=mod.SCALE, enable_gqa=True
        ).squeeze(1)

        k64 = kv_cache[:, 0].double()
        v64 = k64[:, : mod.KV_LORA_RANK]
        q64 = q[0].double().unsqueeze(1)
        o64, budget, prob = budget_from_pv(
            q64, k64, v64, mod.SCALE, None, U_BF16, U_BF16
        )
        # The reference here is left in fp32 and the kernel output is upcast to
        # fp32 for the comparison, so the reference contributes no output
        # rounding at all.
        results.append(
            summarize(
                f"MLA_H12_DECODE seed={s}",
                "MLA_H12_DECODE",
                f"nheads={mod.NUM_HEADS} ctx={mod.CONTEXT_LEN}",
                out.float().unsqueeze(0),
                ref.unsqueeze(0),
                o64.squeeze(1).unsqueeze(0),
                budget.squeeze(1).unsqueeze(0),
                prob.squeeze(1).unsqueeze(0),
                rtol,
                extra={},
            )
        )


# -------------------------------------------------------- MLA fp8 prefill
def run_fp8_prefill(seeds, seq_lens, results):
    from vllm.platforms import current_platform
    from vllm.v1.worker.workspace import (
        init_workspace_manager,
        reset_workspace_manager,
    )

    mod = load(
        str(
            vllm_repo_root()
            / "tests/kernels/attention/test_rocm_aiter_mla_fp8_prefill.py"
        ),
        "mla_fp8_pre",
    )
    fp8 = current_platform.fp8_dtype()
    device = torch.device("cuda")
    init_workspace_manager(device)
    try:
        for seq_len in seq_lens:
            for s in range(seeds):
                torch.manual_seed(s)
                metadata, total_q = mod._build_prefill_metadata([seq_len], device)
                q = torch.randn(
                    total_q,
                    mod.NUM_HEADS,
                    mod.QK_HEAD_DIM,
                    dtype=torch.bfloat16,
                    device=device,
                )
                k = torch.randn_like(q)
                v = torch.randn(
                    total_q,
                    mod.NUM_HEADS,
                    mod.V_HEAD_DIM,
                    dtype=torch.bfloat16,
                    device=device,
                )
                out = torch.zeros(
                    total_q,
                    mod.NUM_HEADS * mod.V_HEAD_DIM,
                    dtype=torch.bfloat16,
                    device=device,
                )
                impl = mod._make_impl()
                impl._mla_fp8_prefill_attn(q, k, v, metadata, out)
                ref = mod._reference(q, k, v)

                # The reference quantizes q/k/v to fp8 and then works in fp32,
                # so the fp8 *input* quantization is common to both sides and
                # cancels. The inputs to the derivation are therefore the
                # dequantized fp8 values, exactly as the reference sees them.
                q64 = q.to(fp8).double().transpose(0, 1)
                k64 = k.to(fp8).double().transpose(0, 1)
                v64 = v.to(fp8).double().transpose(0, 1)
                mask = torch.ones(
                    seq_len, seq_len, dtype=torch.bool, device=device
                ).triu(1)
                # Only the kernel keeps P in fp8; that is the dominant term.
                o64, budget, prob = budget_from_pv(
                    q64, k64, v64, mod.SCALE, mask, U_FP8_E4M3, U_BF16, head_chunk=4
                )
                results.append(
                    summarize(
                        f"MLA_FP8_PREFILL seq_len={seq_len} seed={s}",
                        "MLA_FP8_PREFILL",
                        f"seq_len={seq_len}",
                        out.view(total_q, mod.NUM_HEADS, mod.V_HEAD_DIM)
                        .float()
                        .transpose(0, 1),
                        ref.float().transpose(0, 1),
                        o64,
                        budget,
                        prob,
                        mod.RTOL,
                        extra={},
                    )
                )
    finally:
        reset_workspace_manager()


def main(out: str | None = None, seeds: int = 10, only: str = "all") -> list[dict]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=seeds)
    ap.add_argument("--only", default=only)
    ap.add_argument(
        "--out",
        default=out or f"apriori_mla_{only}.jsonl",
        help="JSONL output path",
    )
    args = ap.parse_args()

    results = []
    if args.only in ("all", "decode"):
        run_mla_decode(args.seeds, results)
    if args.only in ("all", "h12"):
        run_h12(args.seeds, results)
    if args.only in ("all", "fp8"):
        run_fp8_prefill(args.seeds, [128, 512, 1024, 2048], results)

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
        g[r["group"]].append(r)
    print(
        f"{'group':22} {'determ':>10} {'probab':>10} "
        f"{'measured':>10} {'ref_share':>10} {'worst cfg'}"
    )
    for k, v in sorted(g.items()):
        w = max(v, key=lambda r: r["atol_full"])
        print(
            f"{k:22} "
            f"{max(x['atol_full'] for x in v):10.3e} "
            f"{max(x['atol_prob'] for x in v):10.3e} "
            f"{max(x['atol_measured'] for x in v):10.3e} "
            f"{max(x['ref_share'] for x in v):10.3f} {w['cfg']}"
        )
    return results


if __name__ == "__main__":
    main()
