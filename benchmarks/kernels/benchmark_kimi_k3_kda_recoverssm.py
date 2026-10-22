# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark for the Kimi-K3 KDA RecoverSSM kernels (SM10x).

Per decode step, RecoverSSM runs a verify in every KDA layer (the short
convolution, then the delta-rule recurrence over the speculative window from one
state slot per request) and, after sampling, one commit that folds the accepted
tokens of every layer into the state. This compares, at Kimi-K3 KDA shapes
(head_dim 128, conv width 4, gate_lower_bound -5, bfloat16 state):

* verify, per layer, under CUDA graph replay as decode runs it:
  - ``spec``: stock speculative decoding without RecoverSSM
    (``causal_conv1d_update`` + ``fused_recurrent_kda``, which stores the state
    after every draft token);
  - ``triton``: ``causal_conv1d_update`` + the Triton RecoverSSM verify;
  - ``cute``: the CuTe verify (convolution fused in);
* commit, per step for all ``--layers`` KDA layers: the Triton fold and the CuTe
  fold (both after the same Triton plan and conv-state compaction). Decode runs
  the commit eagerly after sampling; with async scheduling its host time overlaps
  the GPU, so the GPU time (from CUDA graph replay) is reported, and the host
  time per call separately.

``--heads`` is the per-GPU head count (Kimi-K3 has 96: TP1/2/4/8 give 96, 48,
24, 12). ``--accept`` sets the accepted tokens per request of the commit. The conv
state uses the dim-first (DS) layout, ``VLLM_SSM_CONV_STATE_LAYOUT=DS``.

Example:
    python benchmarks/kernels/benchmark_kimi_k3_kda_recoverssm.py \
        --batch 1 8 32 128 --window 4 8 --heads 96 12

"""

import argparse
import itertools
import os
import time
from types import SimpleNamespace

import torch

from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.models.kimi_k3.nvidia.ops.recoverssm import (
    KDARecoverSSMCommitContext,
    KDARecoverSSMRecords,
    kda_recoverssm_verify,
)
from vllm.models.kimi_k3.nvidia.ops.third_party.kda import fused_recurrent_kda
from vllm.models.kimi_k3.nvidia.recoverssm.kernels import (
    CuteRecoverSSMVerify,
    compile_state_commit,
)
from vllm.triton_utils import triton

D = 128
CONV_WIDTH = 4
LOWER_BOUND = -5.0
DTYPE = torch.bfloat16


def _bench_graph(fn, repeats: int) -> float:
    """Milliseconds per call under CUDA graph replay (``repeats`` calls per
    graph, one per KDA layer, so launch latency overlaps as in decode)."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeats):
            fn()
    total = triton.testing.do_bench(
        graph.replay, warmup=25, rep=200, return_mode="median"
    )
    return total / repeats


def _host_ms(fn, calls: int = 200) -> float:
    """Host milliseconds per call (launches queue behind each other)."""
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(calls):
        fn()
    host = (time.perf_counter() - start) / calls * 1e3
    torch.cuda.synchronize()
    return host


class Layer:
    """One KDA layer's cache (dim-first conv state, as served) and parameters."""

    def __init__(self, batch: int, heads: int, window: int, spec_slots: int):
        slots = batch * spec_slots + 1
        dim = 3 * heads * D
        self.conv = torch.randn(slots, dim, window + CONV_WIDTH - 2, dtype=DTYPE)
        self.state = torch.randn(slots, heads, D, D, dtype=DTYPE) * 0.05
        self.records = KDARecoverSSMRecords.allocate(
            batch, heads, window, D, DTYPE, torch.device("cuda")
        )
        self.A_log = torch.randn(heads) * 0.5
        self.dt_bias = torch.randn(heads * D) * 0.5
        self.conv_weight = torch.randn(dim, CONV_WIDTH) * 0.5  # FP32, as served
        self.kv_cache = (self.conv, self.state)
        self.recoverssm_records = self.records
        self.local_num_heads = heads
        self.head_dim = D
        self.gate_lower_bound = LOWER_BOUND


def _inputs(batch: int, heads: int, window: int):
    tokens = batch * window
    # The in-projection row: q / k / v, output gate, f_a, beta, alignment pad.
    width = 4 * heads * D + D + heads
    width += -width % 16
    projected = torch.randn(tokens, width, dtype=DTYPE)
    return SimpleNamespace(
        tokens=tokens,
        mixed_qkv=projected[:, : 3 * heads * D],
        raw_beta=projected[:, 4 * heads * D + D : 4 * heads * D + D + heads],
        raw_gate=torch.randn(tokens, heads * D, dtype=DTYPE),
        query_start=torch.arange(0, tokens + 1, window, dtype=torch.int32),
        slots=1 + torch.arange(batch, dtype=torch.int32),
        spec_slots=1 + torch.arange(batch * window, dtype=torch.int32).view(-1, window),
        accepted=torch.ones(batch, dtype=torch.int32),
        out=torch.empty(tokens, heads * D, dtype=DTYPE),
    )


def bench_verify(batch: int, heads: int, window: int, layers: int) -> dict:
    inp = _inputs(batch, heads, window)
    rss = Layer(batch, heads, window, 1)
    spec = Layer(batch, heads, window, window)
    conv_out = torch.empty_like(inp.mixed_qkv)

    def qkv():
        return (
            x.unflatten(-1, (heads, D)).unsqueeze(0)
            for x in conv_out.split(heads * D, -1)
        )

    def stock_spec():
        causal_conv1d_update(
            inp.mixed_qkv, spec.conv, spec.conv_weight, None, activation="silu",
            conv_state_indices=inp.spec_slots[:, 0], num_accepted_tokens=inp.accepted,
            query_start_loc=inp.query_start, max_query_len=window,
            validate_data=False, out=conv_out,
        )  # fmt: skip
        q, k, v = qkv()
        fused_recurrent_kda(
            q=q, k=k, v=v, raw_g=inp.raw_gate.view(1, -1, heads, D),
            raw_beta=inp.raw_beta.unsqueeze(0), A_log=spec.A_log,
            dt_bias=spec.dt_bias, lower_bound=LOWER_BOUND, initial_state=spec.state,
            cu_seqlens=inp.query_start, ssm_state_indices=inp.spec_slots,
            num_accepted_tokens=inp.accepted, out=inp.out.view(1, -1, heads, D),
        )  # fmt: skip

    def triton_verify():
        causal_conv1d_update(
            inp.mixed_qkv, rss.conv, rss.conv_weight, None, activation="silu",
            conv_state_indices=inp.slots, num_accepted_tokens=inp.accepted,
            query_start_loc=inp.query_start, max_query_len=window,
            validate_data=False, out=conv_out,
        )  # fmt: skip
        q, k, v = qkv()
        kda_recoverssm_verify(
            q=q, k=k, v=v, raw_g=inp.raw_gate.view(1, -1, heads, D),
            raw_beta=inp.raw_beta.unsqueeze(0), A_log=rss.A_log,
            dt_bias=rss.dt_bias, lower_bound=LOWER_BOUND, checkpoint_state=rss.state,
            records=rss.records, query_start_loc=inp.query_start,
            state_indices=inp.slots, spec_query_len=window,
            out=inp.out.view(1, -1, heads, D),
        )  # fmt: skip

    verify = CuteRecoverSSMVerify(
        num_heads=heads,
        window=window,
        conv_dim_first=True,
        conv_weight_dtype=torch.float32,
        device=torch.device("cuda"),
    )
    verify.compile()

    def cute_verify():
        verify(
            mixed_qkv=inp.mixed_qkv, raw_gate=inp.raw_gate, raw_beta=inp.raw_beta,
            A_log=rss.A_log, dt_bias=rss.dt_bias, lower_bound=LOWER_BOUND,
            state=rss.state, records=rss.records, query_start_loc=inp.query_start,
            state_indices=inp.slots, num_accepted_tokens=inp.accepted,
            conv_state=rss.conv, conv_weight=rss.conv_weight, out=inp.out,
        )  # fmt: skip

    return {
        name: _bench_graph(fn, layers) * 1e3
        for name, fn in (
            ("spec", stock_spec),
            ("triton", triton_verify),
            ("cute", cute_verify),
        )
    }


def bench_commit(batch: int, heads: int, window: int, layers: int, accept: int):
    stack = [Layer(batch, heads, window, 1) for _ in range(layers)]
    for layer in stack:
        layer.records.correction.normal_()
        layer.records.key_gate.normal_()
    compile_state_commit(heads, window, torch.device("cuda"), align=False)
    query_start = torch.arange(0, batch * window + 1, window, dtype=torch.int32)
    slots = 1 + torch.arange(batch, dtype=torch.int32)
    accepted = torch.full((batch,), accept, dtype=torch.int32)
    times = {}
    for name, cute in (("triton", False), ("cute", True)):
        context = KDARecoverSSMCommitContext.create(
            stack, spec_query_len=window, max_num_reqs=batch, cute_state_commit=cute
        )
        assert (context.state_commit is not None) == cute

        def commit(context=context):
            context.commit(accepted, slots, query_start)

        times[name] = _bench_graph(commit, 4) * 1e3
        times[f"{name} host"] = _host_ms(commit) * 1e3
    return times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64, 128]
    )
    parser.add_argument(
        "--window",
        type=int,
        nargs="+",
        default=[2, 4, 5, 8],
        help="verified tokens per request (1 + num_speculative_tokens)",
    )
    parser.add_argument("--heads", type=int, nargs="+", default=[96, 48, 24, 12])
    parser.add_argument("--layers", type=int, default=69, help="KDA layers")
    parser.add_argument(
        "--accept",
        type=int,
        default=None,
        help="accepted tokens per request in the commit (default: the window)",
    )
    args = parser.parse_args()
    os.environ["VLLM_SSM_CONV_STATE_LAYOUT"] = "DS"
    torch.set_default_device("cuda")
    torch.manual_seed(0)

    print(
        "verify: GPU us per layer; commit: GPU us per step for "
        f"{args.layers} layers (host us per call in parentheses); step: GPU us "
        f"of {args.layers} verifies + commit"
    )
    header = (
        "heads", "window", "batch", "verify spec", "verify triton", "verify cute",
        "vs spec", "vs triton", "commit triton", "commit cute", "speedup",
        "step triton", "step cute", "speedup",
    )  # fmt: skip
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for heads, window, batch in itertools.product(args.heads, args.window, args.batch):
        v = bench_verify(batch, heads, window, args.layers)
        c = bench_commit(batch, heads, window, args.layers, args.accept or window)
        step_triton = args.layers * v["triton"] + c["triton"]
        step_cute = args.layers * v["cute"] + c["cute"]
        row = (
            heads, window, batch, f"{v['spec']:.1f}", f"{v['triton']:.1f}",
            f"{v['cute']:.1f}", f"{v['spec'] / v['cute']:.2f}x",
            f"{v['triton'] / v['cute']:.2f}x",
            f"{c['triton']:.0f} ({c['triton host']:.0f})",
            f"{c['cute']:.0f} ({c['cute host']:.0f})",
            f"{c['triton'] / c['cute']:.2f}x",
            f"{step_triton:.0f}", f"{step_cute:.0f}",
            f"{step_triton / step_cute:.2f}x",
        )  # fmt: skip
        print("| " + " | ".join(map(str, row)) + " |", flush=True)


if __name__ == "__main__":
    main()
