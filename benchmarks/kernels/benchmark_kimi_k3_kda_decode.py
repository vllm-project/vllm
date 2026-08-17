# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark for the fused Kimi-K3 KDA decode kernel on ROCm.

Compares the single fused launch against the three-kernel Triton chain it
replaces (packed causal conv1d update -> recurrent delta-rule decode -> gated
output RMSNorm), at Kimi-K3 KDA shapes: 96 heads x 128, conv width 4,
gate_lower_bound -5.0. Per-step figures scale the per-layer number by the 69
KDA layers in the model.

The recurrent state dominates the traffic (num_heads x 128 x 128 fp32 read and
written per token), so the reported bandwidth is the useful metric: the fusion
removes launches and the intermediate QKV / core-output round trips, not the
state traffic itself.

Example:
    python benchmarks/kernels/benchmark_kimi_k3_kda_decode.py \
        --tokens 1 8 32 64 128 --heads 12

"""

import argparse
import functools

import torch

from vllm.triton_utils import triton

HEAD_DIM = 128
CONV_WIDTH = 4
GATE_LOWER_BOUND = -5.0
NORM_EPS = 1e-5
NUM_KDA_LAYERS = 69
DTYPE = torch.bfloat16


def _bench(fn) -> float:
    return triton.testing.do_bench(fn, warmup=50, rep=300, return_mode="median")


def _bench_graph_layers(calls: list) -> float:
    """Per-layer milliseconds for a graph holding one call per KDA layer."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for call in calls[:3]:
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for call in calls:
            call()
    total = triton.testing.do_bench(
        graph.replay, warmup=50, rep=300, return_mode="median"
    )
    return total / len(calls)


def _bench_graph(fn, repeats: int = NUM_KDA_LAYERS) -> float:
    """Per-call milliseconds under CUDA-graph replay, as decode actually runs.

    Eager timings credit a fusion for every Python dispatch it removes; inside a
    captured graph those are gone, so this is the number that decides whether
    the kernel is worth it in the server. The graph holds `repeats` calls (one
    per KDA layer) because a one-call graph is swamped by the ~19 us HIP
    graph-launch overhead, which a real 93-layer graph amortises away.
    """
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
        graph.replay, warmup=50, rep=300, return_mode="median"
    )
    return total / repeats


class Inputs:
    def __init__(self, num_tokens: int, num_heads: int) -> None:
        torch.manual_seed(0)
        device = "cuda"
        dim = num_heads * HEAD_DIM
        num_slots = num_tokens + 8
        self.num_heads = num_heads
        self.num_tokens = num_tokens
        self.mixed_qkv = torch.randn(num_tokens, 3 * dim, device=device, dtype=DTYPE)
        self.conv_weights = torch.randn(
            3 * dim, CONV_WIDTH, device=device, dtype=torch.float32
        )
        self.decode_conv1d_weight = torch.stack(
            [
                self.conv_weights[i * dim : (i + 1) * dim].transpose(0, 1).contiguous()
                for i in range(3)
            ]
        )
        self.conv_state = torch.randn(
            num_slots, CONV_WIDTH - 1, 3 * dim, device=device, dtype=DTYPE
        )
        self.recurrent_state = torch.randn(
            num_slots, num_heads, HEAD_DIM, HEAD_DIM, device=device, dtype=torch.float32
        )
        self.g1 = torch.randn(
            1, num_tokens, num_heads, HEAD_DIM, device=device, dtype=DTYPE
        )
        self.g2 = torch.randn(
            num_tokens, num_heads, HEAD_DIM, device=device, dtype=DTYPE
        )
        self.beta = torch.randn(1, num_tokens, num_heads, device=device, dtype=DTYPE)
        self.A_log = torch.randn(num_heads, device=device, dtype=torch.float32)
        self.dt_bias = torch.randn(dim, device=device, dtype=torch.float32)
        self.norm_weight_bf16 = torch.ones(HEAD_DIM, device=device, dtype=DTYPE)
        self.decode_norm_weight = self.norm_weight_bf16.float()
        # Slots start at 1: slot 0 is NULL_BLOCK_ID, which the fused kernel
        # treats as a padded row and skips, so timing it measures nothing.
        self.state_indices = torch.arange(
            1, num_tokens + 1, device=device, dtype=torch.int32
        )
        self.conv_state_t = self.conv_state.transpose(-1, -2)
        self.out = torch.empty(
            1, num_tokens, num_heads, HEAD_DIM, device=device, dtype=DTYPE
        )
        self.conv_out = torch.empty_like(self.mixed_qkv)

    def state_bytes(self) -> int:
        """Recurrent-state traffic, identical for both implementations."""
        return self.num_tokens * self.num_heads * HEAD_DIM * HEAD_DIM * 4 * 2


def _gated_rmsnorm(
    x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    x_float = x.float()
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    normed = x_float * torch.rsqrt(variance + eps) * weight.float()
    return (normed * torch.sigmoid(gate.float())).to(x.dtype)


def _triton_gated_norm(inp: Inputs, core_attn_out: torch.Tensor) -> torch.Tensor:
    from vllm.third_party.flash_linear_attention.ops.kda import rms_norm_gated

    return rms_norm_gated(
        core_attn_out,
        inp.g2,
        inp.norm_weight_bf16,
        None,
        activation="sigmoid",
        eps=NORM_EPS,
    )


def triton_chain(inp: Inputs, fused_norm: bool = False) -> None:
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )
    from vllm.models.kimi_k3.amd.ops.third_party.kda import (
        fused_recurrent_kda_packed_decode,
    )

    causal_conv1d_update(
        inp.mixed_qkv,
        inp.conv_state_t,
        inp.conv_weights,
        None,
        activation="silu",
        conv_state_indices=inp.state_indices,
        validate_data=False,
        out=inp.conv_out,
    )
    core_attn_out, _ = fused_recurrent_kda_packed_decode(
        mixed_qkv=inp.conv_out,
        raw_g=inp.g1,
        raw_beta=inp.beta,
        A_log=inp.A_log,
        dt_bias=inp.dt_bias,
        lower_bound=GATE_LOWER_BOUND,
        initial_state=inp.recurrent_state,
        state_indices=inp.state_indices,
    )
    if fused_norm:
        inp.out.copy_(_triton_gated_norm(inp, core_attn_out))
    else:
        inp.out.copy_(
            _gated_rmsnorm(core_attn_out, inp.g2, inp.norm_weight_bf16, NORM_EPS)
        )


def fused(inp: Inputs) -> None:
    from vllm import _custom_ops as ops

    ops.fused_kda_decode(
        x=inp.mixed_qkv,
        weight=inp.decode_conv1d_weight,
        bias=None,
        conv_state=inp.conv_state_t,
        raw_g=inp.g1,
        raw_beta=inp.beta,
        A_log=inp.A_log,
        dt_bias=inp.dt_bias,
        state_indices=inp.state_indices,
        state=inp.recurrent_state,
        out=inp.out,
        lower_bound=GATE_LOWER_BOUND,
        output_gate=inp.g2,
        norm_weight=inp.decode_norm_weight,
        norm_eps=NORM_EPS,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 32, 64, 128])
    parser.add_argument(
        "--graph",
        action="store_true",
        help="time under CUDA-graph replay instead of eager dispatch",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=1,
        help=(
            "with --graph, give each of N layers its own state buffers so the "
            "recurrent state cannot stay resident in Infinity Cache. "
            f"Use {NUM_KDA_LAYERS} for a production-shaped step (needs "
            "~0.9 GB per layer at 128 tokens / 12 heads)."
        ),
    )
    parser.add_argument(
        "--heads",
        type=int,
        nargs="+",
        default=[12],
        help="KDA heads per rank (96 total / TP size)",
    )
    args = parser.parse_args()

    if not hasattr(torch.ops._C, "fused_kda_decode"):
        raise SystemExit("vLLM was built without the fused KDA decode kernel")

    props = torch.cuda.get_device_properties(0)
    bench = _bench_graph if args.graph else _bench
    mode = "cuda-graph replay" if args.graph else "eager dispatch"
    print(f"device: {props.name} ({props.gcnArchName})  timing: {mode}")
    print(
        f"{'heads':>6} {'tokens':>7} {'eager-norm':>11} {'triton-norm':>12} "
        f"{'fused us':>9} {'speedup':>8} {'state TB/s':>11} {'saved ms/step':>14}"
    )
    for num_heads in args.heads:
        for num_tokens in args.tokens:
            if args.graph and args.layers > 1:
                layers = [Inputs(num_tokens, num_heads) for _ in range(args.layers)]
                eager_ms = _bench_graph_layers(
                    [functools.partial(triton_chain, i, False) for i in layers]
                )
                triton_ms = _bench_graph_layers(
                    [functools.partial(triton_chain, i, True) for i in layers]
                )
                fused_ms = _bench_graph_layers(
                    [functools.partial(fused, i) for i in layers]
                )
                inp = layers[0]
                del layers
                torch.accelerator.empty_cache()
            else:
                inp = Inputs(num_tokens, num_heads)
                eager_ms = bench(functools.partial(triton_chain, inp, False))
                triton_ms = bench(functools.partial(triton_chain, inp, True))
                fused_ms = bench(functools.partial(fused, inp))
            bandwidth = inp.state_bytes() * 1e3 / fused_ms / 1e12
            saved = (triton_ms - fused_ms) * NUM_KDA_LAYERS
            print(
                f"{num_heads:>6} {num_tokens:>7} {eager_ms * 1e3:>11.2f} "
                f"{triton_ms * 1e3:>12.2f} {fused_ms * 1e3:>9.2f} "
                f"{triton_ms / fused_ms:>7.2f}x {bandwidth:>10.2f} {saved:>14.3f}"
            )
    print(
        f"\n'eager-norm' is the chain as it runs today (FusedRMSNormGated falls "
        f"back to ~10 eager ops when custom_ops are off);\n'triton-norm' uses the "
        f"Triton rms_norm_gated kernel, which is the honest baseline for this "
        f"fusion.\nspeedup and saved ms/step are against 'triton-norm', over "
        f"{NUM_KDA_LAYERS} KDA layers per forward pass."
    )


if __name__ == "__main__":
    main()
