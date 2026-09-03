# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile Qwen3.5 GDN decode at production shapes.

This benchmark measures the Triton fallback used by Qwen3.5 speculative
decode and the packed recurrent fast path used by ordinary decode.  The
default dimensions match Qwen3.8-27B: 16 key heads, 48 value heads, 128-wide
keys/values, BF16 activations and convolution state, and FP32 recurrent state.

Example:
    python benchmarks/kernels/benchmark_qwen3_5_gdn_decode.py \
        --batches 1 8 24 --mtp 0 3 --layers 48
"""

import argparse
import functools

import torch

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_update,
)
from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
)
from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (
    layer_norm_fwd,
)
from vllm.triton_utils import triton

H = 16
HV = 48
K = 128
V = 128
CONV_WIDTH = 4
QKV_DIM = 2 * H * K + HV * V
NORM_EPS = 1e-6
NUM_GDN_LAYERS = 48
DTYPE = torch.bfloat16


class Inputs:
    def __init__(self, batch_size: int, num_spec_tokens: int) -> None:
        device = "cuda"
        self.batch_size = batch_size
        self.num_spec_tokens = num_spec_tokens
        self.query_len = num_spec_tokens + 1 if num_spec_tokens else 1
        self.num_tokens = batch_size * self.query_len
        # Slot zero is the null block and must not be timed as real work.
        num_slots = self.num_tokens + 1

        self.mixed_qkv = torch.randn(
            self.num_tokens, QKV_DIM, dtype=DTYPE, device=device
        )
        self.conv_out = torch.empty_like(self.mixed_qkv)
        self.conv_state = torch.empty(
            num_slots, QKV_DIM, CONV_WIDTH - 1, dtype=DTYPE, device=device
        )
        self.conv_weight = torch.randn(
            QKV_DIM, CONV_WIDTH, dtype=DTYPE, device=device
        )
        self.state = torch.empty(
            num_slots, HV, V, K, dtype=torch.float32, device=device
        )
        self.ba = torch.randn(
            self.num_tokens, 2 * HV, dtype=DTYPE, device=device
        )
        # Match the non-contiguous views returned by Qwen3.5's packed BA split.
        self.b, self.a = self.ba.chunk(2, dim=-1)
        self.A_log = torch.randn(HV, dtype=torch.float32, device=device)
        self.dt_bias = torch.randn(HV, dtype=torch.float32, device=device)
        self.output_gate = torch.randn(
            self.num_tokens, HV, V, dtype=DTYPE, device=device
        )
        self.norm_weight = torch.randn(V, dtype=DTYPE, device=device)
        self.out = torch.empty_like(self.output_gate)
        self.state_indices = torch.arange(
            1, self.num_tokens + 1, dtype=torch.int32, device=device
        ).view(batch_size, self.query_len)
        self.cu_seqlens = torch.arange(
            0,
            self.num_tokens + 1,
            self.query_len,
            dtype=torch.int32,
            device=device,
        )
        self.num_accepted_tokens = torch.ones(
            batch_size, dtype=torch.int32, device=device
        )

    def state_bytes(self) -> int:
        """Minimum recurrent-state traffic for the scheduled tokens."""
        return self.num_tokens * HV * V * K * 4 * 2


def _conv(inp: Inputs) -> torch.Tensor:
    if inp.num_spec_tokens:
        state_indices = inp.state_indices[:, 0]
        return causal_conv1d_update(
            inp.mixed_qkv,
            inp.conv_state,
            inp.conv_weight,
            None,
            "silu",
            conv_state_indices=state_indices,
            num_accepted_tokens=inp.num_accepted_tokens,
            query_start_loc=inp.cu_seqlens,
            max_query_len=inp.query_len,
            validate_data=False,
            out=inp.conv_out,
        )
    return causal_conv1d_update(
        inp.mixed_qkv,
        inp.conv_state,
        inp.conv_weight,
        None,
        "silu",
        conv_state_indices=inp.state_indices[:, 0],
        validate_data=False,
        out=inp.conv_out,
    )


def _rearrange_mixed_qkv(
    mixed_qkv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mirror QwenGatedDeltaNetAttention.rearrange_mixed_qkv."""
    query, key, value = torch.split(
        mixed_qkv, [H * K, H * K, HV * V], dim=-1
    )
    fused = torch.cat(
        [query.reshape(-1), key.reshape(-1), value.reshape(-1)], dim=0
    )
    q_size = mixed_qkv.shape[0] * H * K
    k_size = q_size
    query = fused[:q_size].view(1, mixed_qkv.shape[0], H, K)
    key = fused[q_size : q_size + k_size].view(1, mixed_qkv.shape[0], H, K)
    value = fused[q_size + k_size :].view(1, mixed_qkv.shape[0], HV, V)
    return query, key, value


def _norm(inp: Inputs, core_attn_out: torch.Tensor) -> None:
    layer_norm_fwd(
        core_attn_out.reshape(-1, V),
        inp.norm_weight,
        None,
        NORM_EPS,
        z=inp.output_gate.reshape(-1, V),
        out=inp.out.reshape(-1, V),
        group_size=V,
        norm_before_gate=True,
        is_rms_norm=True,
        activation="silu",
    )


def regular_chain(inp: Inputs) -> None:
    conv_out = _conv(inp)
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=conv_out,
        a=inp.a.contiguous(),
        b=inp.b.contiguous(),
        A_log=inp.A_log,
        dt_bias=inp.dt_bias,
        scale=K**-0.5,
        initial_state=inp.state,
        out=inp.out.unsqueeze(1),
        ssm_state_indices=inp.state_indices[:, 0],
        use_qk_l2norm_in_kernel=True,
    )
    _norm(inp, inp.out)


def mtp_chain(inp: Inputs) -> None:
    conv_out = _conv(inp)
    query, key, value = _rearrange_mixed_qkv(conv_out)
    core_attn_out, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=inp.A_log,
        a=inp.a.contiguous(),
        b=inp.b.contiguous(),
        dt_bias=inp.dt_bias,
        q=query,
        k=key,
        v=value,
        initial_state=inp.state,
        inplace_final_state=True,
        cu_seqlens=inp.cu_seqlens,
        ssm_state_indices=inp.state_indices,
        num_accepted_tokens=inp.num_accepted_tokens,
        use_qk_l2norm_in_kernel=True,
    )
    _norm(inp, core_attn_out.squeeze(0))


def fused_mtp_chain(inp: Inputs) -> None:
    from vllm import _custom_ops as ops

    conv_out = _conv(inp)
    ops.fused_gdn_decode_post_conv_mtp(
        mixed_qkv=conv_out,
        a=inp.a,
        b=inp.b,
        A_log=inp.A_log,
        dt_bias=inp.dt_bias,
        state_indices=inp.state_indices,
        cu_seqlens=inp.cu_seqlens,
        num_accepted_tokens=inp.num_accepted_tokens,
        state=inp.state,
        output_gate=inp.output_gate,
        norm_weight=inp.norm_weight,
        out=inp.out,
        scale=K**-0.5,
        norm_eps=NORM_EPS,
        output_gate_activation="silu",
    )


def _bench_graph_layers(calls: list[functools.partial]) -> float:
    """Return per-layer milliseconds with graph-launch cost amortized."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for call in calls:
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for call in calls:
            call()
    total_ms = triton.testing.do_bench(
        graph.replay, warmup=25, rep=100, return_mode="median"
    )
    return total_ms / len(calls)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 8, 24])
    parser.add_argument(
        "--mtp",
        nargs="+",
        type=int,
        default=[0, 3],
        help="numbers of speculative tokens; zero selects ordinary decode",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=NUM_GDN_LAYERS,
        help="distinct layer buffers captured in the graph",
    )
    parser.add_argument(
        "--trace-once",
        action="store_true",
        help="run one synchronized chain after compilation for rocprofv3",
    )
    args = parser.parse_args()

    props = torch.cuda.get_device_properties(0)
    print(f"device: {props.name} ({props.gcnArchName})")
    print(
        f"{'batch':>5} {'mtp':>4} {'tokens':>6} {'triton us':>10} "
        f"{'fused us':>10} {'speedup':>8}"
    )
    for batch_size in args.batches:
        for num_spec_tokens in args.mtp:
            if num_spec_tokens == 0:
                inputs = [Inputs(batch_size, 0) for _ in range(args.layers)]
                chain = regular_chain
            else:
                inputs = [
                    Inputs(batch_size, num_spec_tokens) for _ in range(args.layers)
                ]
                chain = mtp_chain
            calls = [functools.partial(chain, inp) for inp in inputs]
            if args.trace_once:
                for call in calls:
                    call()
                torch.cuda.synchronize()
                elapsed_ms = float("nan")
                fused_ms = float("nan")
            else:
                elapsed_ms = _bench_graph_layers(calls)
                if num_spec_tokens:
                    fused_inputs = [
                        Inputs(batch_size, num_spec_tokens)
                        for _ in range(args.layers)
                    ]
                    fused_ms = _bench_graph_layers(
                        [
                            functools.partial(fused_mtp_chain, inp)
                            for inp in fused_inputs
                        ]
                    )
                else:
                    fused_ms = float("nan")
            speedup = elapsed_ms / fused_ms
            print(
                f"{batch_size:5d} {num_spec_tokens:4d} "
                f"{inputs[0].num_tokens:6d} {elapsed_ms * 1e3:10.2f} "
                f"{fused_ms * 1e3:10.2f} {speedup:8.3f}"
            )


if __name__ == "__main__":
    main()
