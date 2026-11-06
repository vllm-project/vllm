# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the deferred KDA RecoverSSM verify against the model's RecoverSSM path.

The baseline is the RecoverSSM path of an FP32-state KDA layer per speculative
step: per layer the short convolution (``causal_conv1d_update``), the F_B gate
projection, the Triton RecoverSSM verify (``kda_recoverssm_verify``) and the gated
RMSNorm, then after sampling the commit that folds the accepted tokens into every
layer's checkpoint (``KDARecoverSSMCommitContext.commit``). The deferred path runs one
kernel per layer (``KdaDeferredVerify``: the last step's accepted tokens replayed
into the checkpoint, then the verify) and after sampling only records the accepted
counts. The floor is a copy of every layer's state pages: the one read and one
write of the state that a deferred step needs.

Each figure is a CUDA graph replay over all KDA layers, each with its own caches,
so every layer reads a cold state as in decode. The conv state layout follows
``VLLM_SSM_CONV_STATE_LAYOUT`` (default SD). Run the GLM-5.3-Flash shapes per rank
with:

    python benchmarks/kernels/benchmark_kda_recoverssm.py --model glm5next
"""

import argparse
import json
import statistics
from collections.abc import Callable
from types import SimpleNamespace

import torch

from vllm.model_executor.kernels.kda import deferred_recoverssm
from vllm.model_executor.kernels.kda.deferred_recoverssm import (
    KdaDeferredVerify,
    decode_conv_weight,
    record_pending,
)
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.models.kimi_k3.nvidia.ops.recoverssm import (
    KDARecoverSSMCommitContext,
    KDARecoverSSMRecords,
    _prepare_commit_plan_kernel,
    kda_recoverssm_verify,
)
from vllm.third_party.flash_linear_attention.ops.kda import rms_norm_gated
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

K = 128
LOWER_BOUND = -5.0
NORM_EPS = 1e-5
CONV_WIDTH = 4
# Per model: KDA heads per rank, KDA layers, verify window (1 + draft tokens).
_MODELS = {
    "glm5next": (64, 34, 6),
    "kimi_k3": (96, 69, 5),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(_MODELS), default="glm5next")
    parser.add_argument(
        "--batch",
        type=int,
        nargs="+",
        default=[8, 16, 32, 48, 64, 96, 128, 192],
        help="Speculative requests per rank.",
    )
    parser.add_argument(
        "--accepted", type=int, default=4, help="Tokens each request accepts."
    )
    parser.add_argument(
        "--kernel",
        choices=("auto", "ws", "fused"),
        default="auto",
        help="Force the deferred kernel (KdaFusedWS or KdaFused).",
    )
    parser.add_argument("--warmup-replays", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--json", type=str, help="Also write the results here.")
    return parser.parse_args()


def make_layer(heads: int, batch: int, window: int, gen: torch.Generator):
    dim, slots = heads * K, batch + 1

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device="cuda", generator=gen)

    conv_shape = (slots, 3 * dim, window + CONV_WIDTH - 2)
    if not is_conv_state_dim_first():
        conv_shape = (slots, window + CONV_WIDTH - 2, 3 * dim)
    records = KDARecoverSSMRecords.allocate(
        batch, heads, window, K, torch.bfloat16, torch.device("cuda")
    )
    records.correction.copy_(randn(*records.correction.shape))
    records.key_gate.copy_(randn(*records.key_gate.shape))
    conv_weight = (randn(3 * dim, 1, CONV_WIDTH) * 0.5).bfloat16()
    norm_weight = (torch.rand(K, device="cuda", generator=gen) + 0.5).bfloat16()
    return SimpleNamespace(
        kv_cache=(randn(*conv_shape).bfloat16(), randn(slots, heads, K, K) * 0.05),
        recoverssm_records=records,
        A_log=randn(heads) * 0.5,
        dt_bias=randn(dim) * 0.5,
        conv_weight=conv_weight,
        decoded_conv_weight=decode_conv_weight(conv_weight),
        # Integer multiples of 1/64 (and integer F_A): an exact F_B GEMM.
        f_b_weight=((randn(dim, K) * 3).round().clamp(-8, 8) / 64).bfloat16(),
        norm_weight=norm_weight,
        norm_weight_f32=norm_weight.float(),
        local_num_heads=heads,
        head_dim=K,
        gate_lower_bound=LOWER_BOUND,
    )


def clone_layer(layer: SimpleNamespace) -> SimpleNamespace:
    clone = SimpleNamespace(**vars(layer))
    clone.kv_cache = tuple(t.clone() for t in layer.kv_cache)
    records = layer.recoverssm_records
    clone.recoverssm_records = KDARecoverSSMRecords(
        records.correction.clone(), records.key_gate.clone()
    )
    return clone


def conv_view(layer: SimpleNamespace) -> torch.Tensor:
    """The [slots, 3 H K, window] view the kernels take."""
    conv = layer.kv_cache[0]
    return conv if is_conv_state_dim_first() else conv.transpose(-1, -2)


def make_inputs(heads: int, tokens: int, gen: torch.Generator):
    """(qkv, beta, F_A) views of a GLM-5.3 [q k v | b | f_a | g_a] projection, and
    the output gate (g_b_proj's output, which both paths take)."""
    dim = heads * K
    projected = torch.randn(
        tokens, 3 * dim + heads + 2 * K, device="cuda", generator=gen
    ).bfloat16()
    f_a = projected[:, 3 * dim + heads : 3 * dim + heads + K]
    f_a.copy_(
        (torch.randn(f_a.shape, device="cuda", generator=gen) * 1.5)
        .round()
        .clamp(-3, 3)
    )
    gate = torch.randn(tokens, dim, device="cuda", generator=gen).bfloat16()
    return projected[:, : 3 * dim], projected[:, 3 * dim : 3 * dim + heads], f_a, gate


def vllm_layer(layer, x, beta, f_a, gate, slots, starts, ones, window):
    tokens, heads = x.shape[0], layer.local_num_heads
    dim = heads * K
    g1 = (f_a @ layer.f_b_weight.t()).view(1, tokens, heads, K)
    mixed = causal_conv1d_update(
        x, conv_view(layer), layer.conv_weight.view(3 * dim, CONV_WIDTH), None,
        activation="silu", conv_state_indices=slots, num_accepted_tokens=ones,
        query_start_loc=starts, max_query_len=window, validate_data=False,
        out=torch.empty_like(x),
    )  # fmt: skip
    q, k, v = (t.reshape(1, tokens, heads, K) for t in mixed.split(dim, dim=-1))
    core = kda_recoverssm_verify(
        q, k, v, g1, beta.unsqueeze(0), layer.A_log, layer.dt_bias, LOWER_BOUND,
        layer.kv_cache[1], layer.recoverssm_records, starts, slots, window,
    )  # fmt: skip
    out = rms_norm_gated(
        core, gate.view(1, tokens, heads, K), layer.norm_weight, None, "sigmoid",
        eps=NORM_EPS,
    )  # fmt: skip
    return out.view(tokens, dim)


def deferred_layer(layer, verify, x, beta, f_a, gate, slots, ones, rows, pending, out):
    records = layer.recoverssm_records
    return verify(
        qkv=x, gate=gate, f_a=f_a, beta=beta, conv_weight=layer.decoded_conv_weight,
        conv_state=conv_view(layer), A_log=layer.A_log, dt_bias=layer.dt_bias,
        f_b_weight=layer.f_b_weight, norm_weight=layer.norm_weight_f32,
        norm_eps=NORM_EPS, lower_bound=LOWER_BOUND, state=layer.kv_cache[1],
        state_indices=slots.view(-1, 1), num_accepted_tokens=ones,
        correction=records.correction, key_gate=records.key_gate,
        record_rows=rows, pending=pending, out=out,
    )  # fmt: skip


def deferred_commit(ctx, accepted, slots, starts, rows, pending, window):
    """The commit plan, then each request's commit length becomes its pending
    replay."""
    _prepare_commit_plan_kernel[(slots.shape[0],)](
        accepted, None, slots, starts, None, None, ctx.commit_lens,
        ctx.final_state_indices, ctx.boundary_state_indices,
        ctx.boundary_recovery_lens, NULL_BLOCK_ID, 1, 1, accepted.stride(0), 0,
        slots.stride(0), starts.stride(0), 0, 0, 0, SPEC_QUERY_LEN=window,
        num_warps=1,
    )  # fmt: skip
    record_pending(slots, ctx.commit_lens, rows, pending, NULL_BLOCK_ID)


def check(layer, verify, inputs, slots, starts, ones, accepted, window):
    """One deferred step of a layer against the commit then the verify of the
    model's path, from the same checkpoint and records."""
    x, beta, f_a, gate = inputs
    batch = slots.shape[0]
    expected_layer = clone_layer(layer)
    ctx = KDARecoverSSMCommitContext.create(
        [expected_layer], spec_query_len=window, max_num_reqs=batch
    )
    ctx.commit(accepted, slots, starts)
    expected = vllm_layer(expected_layer, *inputs, slots, starts, ones, window)

    actual_layer = clone_layer(layer)
    rows = torch.randperm(batch, device="cuda").int()
    for dst, src in zip(
        vars(actual_layer.recoverssm_records).values(),
        vars(layer.recoverssm_records).values(),
    ):
        dst[rows.long()] = src
    pending = torch.zeros(batch, dtype=torch.int32, device="cuda")
    pending[rows.long()] = accepted
    actual = torch.empty_like(expected)
    deferred_layer(actual_layer, verify, x, beta, f_a, gate, slots, ones, rows,
                   pending, actual)  # fmt: skip

    written = slots.long()
    state, expected_state = (
        t.kv_cache[1][written] for t in (actual_layer, expected_layer)
    )
    state_error = (state - expected_state).abs().max() / expected_state.abs().max()
    assert state_error < 1e-5, state_error
    rms = expected.float().pow(2).mean(-1, keepdim=True).sqrt()
    out_error = ((actual.float() - expected.float()).abs() / rms).max()
    assert out_error < 6e-2, out_error


def capture_graph(op: Callable[[], object]) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            op()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        op()
    torch.cuda.current_stream().wait_stream(stream)
    return graph


def benchmark_graphs(
    graphs: dict[str, torch.cuda.CUDAGraph], warmup_replays: int, samples: int
) -> dict[str, float]:
    """Median replay time per graph in us, the graphs interleaved."""
    names = list(graphs)
    for _ in range(warmup_replays):
        for name in names:
            graphs[name].replay()
    torch.accelerator.synchronize()
    timings: dict[str, list[float]] = {name: [] for name in names}
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for sample in range(samples):
        for offset in range(len(names)):
            name = names[(sample + offset) % len(names)]
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            timings[name].append(start.elapsed_time(end) * 1000)
    return {name: statistics.median(values) for name, values in timings.items()}


def benchmark_batch(args, heads, num_layers, window, batch) -> dict:
    gen = torch.Generator(device="cuda").manual_seed(batch)
    layers = [make_layer(heads, batch, window, gen) for _ in range(num_layers)]
    inputs = make_inputs(heads, batch * window, gen)
    slots = (torch.randperm(batch, device="cuda", generator=gen) + 1).int()
    starts = torch.arange(
        0, batch * window + 1, window, dtype=torch.int32, device="cuda"
    )
    ones = torch.ones(batch, dtype=torch.int32, device="cuda")
    accepted = torch.full((batch,), args.accepted, dtype=torch.int32, device="cuda")
    rows = torch.randperm(batch, device="cuda", generator=gen).int()
    pending = torch.full((batch,), args.accepted, dtype=torch.int32, device="cuda")
    out = torch.empty(batch * window, heads * K, dtype=torch.bfloat16, device="cuda")
    verify = KdaDeferredVerify(heads, window)
    check(layers[0], verify, inputs, slots, starts, ones, accepted, window)
    kernel = deferred_recoverssm._kernel(
        heads, batch, window, conv_view(layers[0]), inputs[0]
    )

    ctx = KDARecoverSSMCommitContext.create(
        layers, spec_query_len=window, max_num_reqs=batch
    )
    scratch = torch.empty_like(layers[0].kv_cache[1])
    candidates = {
        "floor": lambda: [scratch.copy_(layer.kv_cache[1]) for layer in layers],
        "vllm_layers": lambda: [
            vllm_layer(layer, *inputs, slots, starts, ones, window) for layer in layers
        ],
        "vllm_commit": lambda: ctx.commit(accepted, slots, starts),
        "deferred_layers": lambda: [
            deferred_layer(layer, verify, *inputs, slots, ones, rows, pending, out)
            for layer in layers
        ],
        "deferred_commit": lambda: deferred_commit(
            ctx, accepted, slots, starts, rows, pending, window
        ),
    }
    graphs = {name: capture_graph(op) for name, op in candidates.items()}
    times = benchmark_graphs(graphs, args.warmup_replays, args.samples)

    vllm = times["vllm_layers"] + times["vllm_commit"]
    deferred = times["deferred_layers"] + times["deferred_commit"]
    return {
        "batch": batch,
        "kernel": type(kernel).__name__,
        "floor_ms": times["floor"] / 1e3,
        "vllm_layers_ms": times["vllm_layers"] / 1e3,
        "vllm_commit_ms": times["vllm_commit"] / 1e3,
        "vllm_ms": vllm / 1e3,
        "deferred_ms": deferred / 1e3,
        "deferred_commit_us": times["deferred_commit"],
        "latency_change": (deferred - vllm) / vllm,
        "floor_gap": (deferred - times["floor"]) / times["floor"],
    }


def print_results(args, heads, num_layers, window, results) -> None:
    layout = "DS" if is_conv_state_dim_first() else "SD"
    print(
        f"### Deferred KDA RecoverSSM vs the Triton RecoverSSM path "
        f"({args.model}: {heads} heads x {K} per rank, {num_layers} KDA layers, "
        f"q {window}, {args.accepted} accepted, conv state {layout}), ms per step"
    )
    print(
        "| batch | deferred kernel | floor | vLLM layers | vLLM commit | vLLM | "
        "deferred | vs vLLM | vs floor |"
    )
    print("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        print(
            f"| {r['batch']} | {r['kernel']} | {r['floor_ms']:.2f} | "
            f"{r['vllm_layers_ms']:.2f} | {r['vllm_commit_ms']:.2f} | "
            f"{r['vllm_ms']:.2f} | {r['deferred_ms']:.2f} | "
            f"{r['latency_change']:+.1%} | {r['floor_gap']:+.1%} |"
        )


def main() -> None:
    args = parse_args()
    heads, num_layers, window = _MODELS[args.model]
    if args.kernel != "auto":
        deferred_recoverssm.WS_MIN_ITEMS = 0 if args.kernel == "ws" else 1 << 62
    assert min(args.batch) > 0 and 1 <= args.accepted <= window
    results = []
    for batch in args.batch:
        results.append(benchmark_batch(args, heads, num_layers, window, batch))
        torch.accelerator.empty_cache()
    print_results(args, heads, num_layers, window, results)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
