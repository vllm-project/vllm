# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# The timed closures are built and run within one loop iteration (no late binding).
# ruff: noqa: B023
"""Single-GPU microbenchmarks of the kdamoe amd_kernels ops vs the stock sequences (CUDA
graph replay over rotating weight sets > LLC, min of 15 replays, us per layer):

  kda   f_b GEMM + AITER spec decode  vs  kda_decode_fb_gate_spec (akr_kda.run)
  moe   router GEMM + topk + hgemm down + aiter fused_moe (shipped CSV)
        vs  router(+aux) + down_quant + experts (reduce=True), and with the 16-slot
        sum left to the latent kernel (reduce=False + a separate sum kernel stand-in)

    HIP_VISIBLE_DEVICES=0 python tests/models/kimi_k3/akr/bench_akr_single.py [kda,moe]
"""

import os
import sys
import time
import types

import torch

os.environ.setdefault("AITER_SITUV2_A4W4", "1")

NH, D, LP = 12, 128, 1536
E, H, LAT, INTER, TOPK = 896, 7168, 3584, 384, 16
M_LIST = [int(t) for t in os.environ.get("TOKENS", "12,24,40,65,80,128,192").split(",")]
ROUNDS = 15


def graph_time(fns, n_layers, rounds=ROUNDS):
    """Us per layer: min over replays of a graph of fn(0..n_layers-1)."""
    for li in range(n_layers):
        fns(li)
    torch.accelerator.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fns(0)
        torch.accelerator.synchronize()
        with torch.cuda.graph(g, stream=s):
            for li in range(n_layers):
                fns(li)
    torch.accelerator.synchronize()
    for _ in range(3):
        g.replay()
    best = 1e9
    for _ in range(rounds):
        torch.accelerator.synchronize()
        t = time.perf_counter()
        g.replay()
        torch.accelerator.synchronize()
        best = min(best, (time.perf_counter() - t) * 1e6 / n_layers)
    return best


def bench_kda():
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.kimi_k3.amd import akr_kda
    from vllm.models.kimi_k3.amd.ops import kda_fb_gemm  # noqa: F401

    dev = torch.device("cuda")
    print(
        "| M (seq x T) | stock f_b + AITER us | akr kda_decode_fb_gate_spec us | "
        "speedup |"
    )
    print("|---|---|---|---|")
    for n_seq, t in [(3, 4), (6, 4), (10, 4), (13, 5), (20, 4), (32, 4), (48, 4)]:
        rows = n_seq * t
        if rows not in M_LIST and os.environ.get("TOKENS"):
            continue
        L = 6
        layers, projs, states, convs = [], [], [], []
        g = torch.Generator(device=dev).manual_seed(0)
        slots = 1 + n_seq * (t + 1)
        for _ in range(L):
            layers.append(
                types.SimpleNamespace(
                    local_num_heads=NH,
                    head_dim=D,
                    conv_size=4,
                    use_full_rank_gate=True,
                    gate_lower_bound=-5.0,
                    decode_conv1d_weight=(
                        torch.randn(3, 4, LP, device=dev, generator=g) * 0.5
                    ).contiguous(),
                    A_log=torch.randn(NH, device=dev, generator=g) * 0.5,
                    dt_bias=torch.randn(LP, device=dev, generator=g) * 0.5,
                    f_b_proj=types.SimpleNamespace(
                        weight=(
                            torch.randn(LP, D, device=dev, generator=g) * 0.1
                        ).bfloat16()
                    ),
                    o_norm=types.SimpleNamespace(
                        weight=(
                            1 + 0.1 * torch.randn(D, device=dev, generator=g)
                        ).bfloat16(),
                        eps=1e-6,
                    ),
                )
            )
            projs.append(
                (torch.randn(rows, 6304, device=dev, generator=g) * 0.5).bfloat16()
            )
            states.append(torch.randn(slots, NH, D, D, device=dev, generator=g) * 0.1)
            convs.append(
                (
                    torch.randn(slots, 3 + t, 3 * LP, device=dev, generator=g) * 0.5
                ).bfloat16()
            )
        idx = (1 + torch.arange(n_seq * t, device=dev, dtype=torch.int32)).view(
            n_seq, t
        )
        acc = torch.randint(
            1, t + 1, (n_seq,), device=dev, dtype=torch.int32, generator=g
        )
        cu = torch.arange(0, (n_seq + 1) * t, t, device=dev, dtype=torch.int32)
        m = types.SimpleNamespace(
            spec_state_indices_tensor=idx,
            spec_query_start_loc=cu,
            num_spec_decodes=n_seq,
            num_actual_tokens=rows,
            uniform_spec_sequence_length=t,
        )
        out = torch.empty(1, rows, NH, D, dtype=torch.bfloat16, device=dev)

        def stock(li):
            lay, proj = layers[li], projs[li]
            f_a = proj[:, 4 * LP : 4 * LP + D]
            g1 = torch.ops.vllm.kimi_k3_kda_fb_proj(f_a, lay.f_b_proj.weight).view(
                1, rows, NH, D
            )
            beta = proj[:, 4 * LP + D : 4 * LP + D + NH].unsqueeze(0)
            q, k, v = (
                x.reshape(1, rows, NH, D) for x in proj[:, : 3 * LP].split(LP, dim=-1)
            )
            rocm_aiter_ops.kda_fused_recurrent_spec_decode(
                q=q,
                k=k,
                v=v,
                g=g1,
                beta=beta,
                A_log=lay.A_log,
                dt_bias=lay.dt_bias,
                lower_bound=-5.0,
                initial_state=states[li],
                cu_seqlens=cu,
                ssm_state_indices=idx,
                num_accepted_tokens=acc,
                conv_state=convs[li].transpose(-1, -2),
                conv_weight=lay.decode_conv1d_weight,
                out_gate=proj[:, 3 * LP : 4 * LP].reshape(1, rows, NH, D),
                norm_weight=lay.o_norm.weight,
                norm_eps=1e-6,
                out=out,
            )

        def akr(li):
            akr_kda.run(
                layers[li],
                projs[li],
                convs[li].transpose(-1, -2),
                states[li],
                m,
                acc,
                out,
            )

        ts, ta = graph_time(stock, L), graph_time(akr, L)
        print(
            f"| {rows} ({n_seq}x{t}) | {ts:.1f} | {ta:.1f} | {ts / ta:.2f}x |",
            flush=True,
        )


def bench_moe():
    import aiter.fused_moe as fm
    from aiter import ActivationType, QuantType, biased_grouped_topk
    from amd_kernels.gemm import bf16_common as gl
    from amd_kernels.moe import latent_moe_common as lm

    from vllm.models.kimi_k3.amd import akr_moe

    dev = torch.device("cuda")
    L = 3
    g = torch.Generator(device=dev).manual_seed(0)
    mk = lambda *s, lo=0, hi=256: torch.randint(
        lo, hi, s, dtype=torch.uint8, device=dev, generator=g
    )  # noqa: E731
    ws = [
        lm.moe_weights(
            mk(E, 2 * INTER, LAT // 2),
            mk(E, 2 * INTER, LAT // 32, lo=118, hi=124),
            mk(E, LAT, INTER // 2),
            mk(E, LAT, 16, lo=118, hi=124),
        )
        for _ in range(L)
    ]
    gates = [
        (torch.randn(E, H, device=dev, generator=g) * 0.02).bfloat16() for _ in range(L)
    ]
    downs = [
        (torch.randn(LAT, H, device=dev, generator=g) * 0.02).bfloat16()
        for _ in range(L)
    ]
    bias = torch.randn(E, device=dev, generator=g) * 0.1
    if (
        os.environ.get("ROUTING", "uniform") == "zipf"
    ):  # skewed expert popularity (the real routing is skewed)
        pop = torch.randperm(E, device=dev, generator=g).float()
        bias = -float(os.environ.get("ZIPF_S", "0.35")) * torch.log1p(pop)
    fp4 = torch.float4_e2m1fn_x2
    stock_w = []
    for w in ws:
        w13, w2 = (
            w.w13.view(E, 2 * INTER, LAT // 2).view(fp4),
            w.w2.view(E, LAT, INTER // 2).view(fp4),
        )
        w13.is_shuffled = w2.is_shuffled = True
        stock_w.append(
            (
                w13,
                w2,
                w.s13.view(E, 2 * INTER, LAT // 32).view(torch.float8_e8m0fnu),
                w.s2.view(E, LAT, 16).view(torch.float8_e8m0fnu),
            )
        )
    print(
        "| M | stock router+topk+down+fused_moe us | akr router+down_quant+experts us "
        "| speedup | akr (slots, no sum) us |"
    )
    print("|---|---|---|---|---|")
    for m in M_LIST:
        x = (torch.randn(m, H, device=dev, generator=g) * 0.5).bfloat16()
        aux = akr_moe._aux(
            types.SimpleNamespace(gate=types.SimpleNamespace(weight=gates[0])),
            m,
            dev,
            TOPK,
        )

        def stock(li):
            logits = torch.mm(x, gates[li].t(), out_dtype=torch.float32)
            tw = torch.empty(m, TOPK, dtype=torch.float32, device=dev)
            ti = torch.empty(m, TOPK, dtype=torch.int32, device=dev)
            biased_grouped_topk(logits, bias, tw, ti, 1, 1, True, 1.0)
            y = x @ downs[li].t()
            w13, w2, s13, s2 = stock_w[li]
            fm.fused_moe(
                y,
                w13,
                w2,
                tw,
                ti,
                None,
                ActivationType.Situv2,
                QuantType.per_1x32,
                False,
                s13,
                s2,
                None,
                None,
                beta=4.0,
                linear_beta=25.0,
                gate_mode="separated",
                dtype=torch.bfloat16,
            )

        def akr(li, reduce=True):
            lm.router_sigmoid_bias_topk(
                x, gates[li], bias, 1.0, topk=TOPK, shuffled=False, aux=aux
            )
            codes, scales = gl.bf16_latent_down_mxfp4_quant(
                x, downs[li], shuffled=False
            )
            lm.mxfp4_moe_experts(
                codes,
                scales,
                ws[li],
                lm.moe_sort_from_aux(aux, weights=ws[li]),
                reduce=reduce,
            )

        _, ids = lm.router_sigmoid_bias_topk(
            x, gates[0], bias, 1.0, topk=TOPK, shuffled=False, aux=aux
        )
        uniq = int(torch.unique(ids).numel())
        ts, ta, tn = (
            graph_time(stock, L),
            graph_time(akr, L),
            graph_time(lambda li: akr(li, False), L),
        )
        print(
            f"| {m} ({uniq} experts) | {ts:.1f} | {ta:.1f} | "
            f"{ts / ta:.2f}x | {tn:.1f} |",
            flush=True,
        )


if __name__ == "__main__":
    which = (sys.argv[1] if len(sys.argv) > 1 else "kda,moe").split(",")
    if "kda" in which:
        bench_kda()
    if "moe" in which:
        bench_moe()
