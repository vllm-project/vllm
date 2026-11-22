# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# The timed closures are built and run within one loop iteration (no late binding).
# ruff: noqa: B023
"""Stock vs amd_kernels microbenchmarks of the single-GPU non-layer pieces of Kimi-K3.1
(CUDA-graph replay of several calls over rotated operands, min and median of 15 rounds,
us per call): Markov chain, top-p mask, drafter dense MLP (with its tile packer), final
attn_res + norm, drafter dense MLA decode (the lm_head GEMM runs inside the exchange:
benchmark_akr_nonlayer_8rank.py).

Run:
    HIP_VISIBLE_DEVICES=0 python benchmarks/kernels/benchmark_akr_nonlayer_1gpu.py \
        [markov topp mlp final mla]
"""

import statistics
import sys
from types import SimpleNamespace

import torch

VOCAB = 163840


def time_graph(calls, rounds=15, replays=5):
    for c in calls:
        c()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for c in calls:
            c()
    graph.replay()
    torch.accelerator.synchronize()
    ts = []
    for _ in range(rounds):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(replays):
            graph.replay()
        e.record()
        torch.accelerator.synchronize()
        ts.append(s.elapsed_time(e) * 1e3 / (replays * len(calls)))
    return min(ts), statistics.median(ts)


def row(name, stock, new):
    print(
        f"  {name:28s} stock {stock[0]:8.1f} / {stock[1]:8.1f}   akr {new[0]:8.1f} / "
        f"{new[1]:8.1f}   x{stock[0] / new[0]:.2f}  (min / median us)",
        flush=True,
    )


def bench_markov():
    sys.path.insert(0, "tests/kernels/models/amd")
    import test_akr_nonlayer_1gpu as T

    from vllm.models.kimi_k3.amd import akr_sampling
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    print("Markov chain (all draft steps after the base logits)")
    for name, reqs, k in (
        ("c16", 8, 4),
        ("c24", 13, 4),
        ("c32", 20, 3),
        ("c48", 48, 3),
    ):
        T.MAX_REQS = reqs
        s = T._speculator(1.0, 1, False, k)
        hidden = torch.zeros(T.MAX_REQS * k, 8, device="cuda")
        stock = time_graph(
            [lambda: DSparkSpeculator._sample_sequential(s, reqs, hidden)]
        )
        akr_sampling.setup_speculator(s)
        new = time_graph([lambda: DSparkSpeculator._sample_sequential(s, reqs, hidden)])
        s._akr_markov = None
        row(f"{name} {reqs} reqs x {k} steps", stock, new)


def bench_topp():
    from vllm.models.kimi_k3.amd import akr_sampling
    from vllm.v1.sample.ops import topk_topp_triton as tt

    print("top-p mask (apply_top_k_top_p_triton, p only)")
    for rows in (1, 8, 24, 64):
        logits = torch.randn(rows, VOCAB, device="cuda") * 3
        p = torch.full((rows,), 0.9, device="cuda")
        tt.split_top_p_override = None
        stock = time_graph(
            [lambda: tt.apply_top_k_top_p_triton(logits.clone(), None, p)]
        )
        akr_sampling.install_top_p(163840)
        new = time_graph([lambda: tt.apply_top_k_top_p_triton(logits.clone(), None, p)])
        tt.split_top_p_override = None
        copy = time_graph([lambda: logits.clone()])
        row(f"{rows} rows (copy {copy[0]:.1f} incl.)", stock, new)


def bench_mlp():
    from amd_kernels.decoder.bf16_dense_mlp_common import dense_mlp_workspace

    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.models.kimi_k3.amd import akr_draft

    print(
        "drafter dense MLP (gate_up + silu_and_mul + down) vs pack + dense_mlp "
        "(weights rotated over 4 copies)"
    )
    dev = torch.device("cuda")
    bf16 = torch.bfloat16
    for rows in (16, 32, 52, 60, 144, 160, 192, 224, 240, 256):
        x = torch.randn(rows, 7168, device=dev).to(bf16)
        act = SiluAndMul()
        gus = [(torch.randn(3584, 7168, device=dev) * 0.02).to(bf16) for _ in range(4)]
        dns = [(torch.randn(7168, 1792, device=dev) * 0.02).to(bf16) for _ in range(4)]
        stock = time_graph(
            [lambda g=g, d=d: act(x @ g.t()) @ d.t() for g, d in zip(gus, dns)]
        )
        ws = dense_mlp_workspace(dev, hidden=7168, inter=1792)
        stage = torch.zeros(
            akr_draft._x_numel(dns[0], akr_draft.MLP_MAX_ROWS), dtype=bf16, device=dev
        )
        mlps = []
        for g, d in zip(gus, dns):
            mlp = SimpleNamespace(
                gate_up_proj=SimpleNamespace(weight=SimpleNamespace(data=g)),
                down_proj=SimpleNamespace(weight=SimpleNamespace(data=d)),
            )
            mlps.append(akr_draft._Mlp(mlp, ws, stage))
        new = time_graph([lambda m=m: m(x) for m in mlps])
        row(f"{rows} rows", stock, new)


def bench_final():
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.models.kimi_k3.amd import akr_final
    from vllm.models.kimi_k3.amd.ops.attn_res import attn_res

    sys.path.insert(0, "tests/kernels/models/amd")
    import test_akr_nonlayer_1gpu as T

    print("final attn_res + final RMSNorm")
    dev, bf16 = torch.device("cuda"), torch.bfloat16
    for rows in (12, 24, 80, 192):
        nb = 8
        prefix = torch.randn(rows, 7168, device=dev).to(bf16)
        delta = torch.randn(rows, 7168, device=dev).to(bf16)
        bank = (torch.randn(rows, nb, 7168, device=dev) * 2).to(bf16)
        nw = (torch.rand(7168, device=dev) + 0.5).to(bf16)
        qk = (torch.randn(7168, device=dev) * 0.05).to(bf16)
        norm = RMSNorm(7168, eps=1e-5).to(bf16).to(dev)
        model = SimpleNamespace(
            norm=norm,
            output_attn_res_norm=SimpleNamespace(weight=nw, variance_epsilon=1e-6),
            output_attn_res_proj=SimpleNamespace(weight=qk.unsqueeze(0)),
            _akr_final=T._fake_comm(),
        )
        import aiter

        stock = time_graph(
            [
                lambda: aiter.rms_norm(
                    attn_res(
                        prefix.clone(), delta, bank, nw, qk, None, nb, -1, 1e-6, 0.0
                    ),
                    norm.weight,
                    1e-5,
                )
            ]
        )
        copy = time_graph([lambda: prefix.clone()])
        print(f"    (prefix.clone() in both columns: {copy[0]:.1f} us)")
        new = time_graph(
            [lambda: akr_final._launch(model, prefix.clone(), bank, delta, nb)]
        )
        row(f"{rows} rows", stock, new)


def bench_mla():
    sys.path.insert(0, "tests/kernels/models/amd")
    print(
        "see amd-kernels benchmarks/attention/fp8_dense_mla_decode_h96_kv512 (graph "
        "replay vs mla_decode_fwd)"
    )


BENCH = dict(
    markov=bench_markov,
    topp=bench_topp,
    mlp=bench_mlp,
    final=bench_final,
    mla=bench_mla,
)

if __name__ == "__main__":
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        for n in sys.argv[1:] or list(BENCH):
            BENCH[n]()
