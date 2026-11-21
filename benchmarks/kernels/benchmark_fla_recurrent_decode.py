# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode microbenchmark for the shared FLA recurrent delta-rule launchers.

Times `fused_recurrent_gated_delta_rule_fwd`,
`fused_sigmoid_gating_delta_rule_update` and `fused_recurrent_kda_fwd` at
decode shapes (varlen layout, as the GDN / KDA layers call them), with and
without spec decode. Each measurement is a CUDA graph holding one call per
layer, every layer with its own recurrent state, so the state is read from
DRAM rather than L2.

To compare two revisions of the kernels, run once per revision with `--save`
and then `--compare`:

    python benchmarks/kernels/benchmark_fla_recurrent_decode.py --save base.json
    # switch revision
    python benchmarks/kernels/benchmark_fla_recurrent_decode.py --save new.json
    python benchmarks/kernels/benchmark_fla_recurrent_decode.py \
        --compare base.json new.json
"""

import argparse
import json
import math

import torch

from vllm.triton_utils import triton

DEVICE = "cuda"
DTYPE = torch.bfloat16
LAUNCHERS = ("recurrent", "sigmoid_gating", "kda")


class Inputs:
    def __init__(
        self,
        launcher: str,
        batch: int,
        num_v_heads: int,
        head_dim: int,
        spec_len: int,
        state_dtype: torch.dtype,
    ) -> None:
        # KDA allocates its output like k, so it needs H == HV.
        H = num_v_heads if launcher == "kda" else max(num_v_heads // 2, 1)
        HV, K, V, T = num_v_heads, head_dim, head_dim, spec_len
        tokens = batch * T
        self.launcher = launcher
        self.scale = K**-0.5
        self.q = torch.randn(1, tokens, H, K, device=DEVICE, dtype=DTYPE)
        self.k = torch.randn(1, tokens, H, K, device=DEVICE, dtype=DTYPE)
        self.v = torch.randn(1, tokens, HV, V, device=DEVICE, dtype=DTYPE)
        self.a = torch.randn(tokens, HV, device=DEVICE, dtype=DTYPE)
        self.b = torch.randn(tokens, HV, device=DEVICE, dtype=DTYPE)
        self.A_log = torch.randn(HV, device=DEVICE, dtype=torch.float32)
        self.dt_bias = torch.randn(HV, device=DEVICE, dtype=torch.float32)
        g_shape = (1, tokens, HV, K) if launcher == "kda" else (1, tokens, HV)
        self.g = -torch.rand(g_shape, device=DEVICE, dtype=torch.float32)
        self.beta = torch.rand(1, tokens, HV, device=DEVICE, dtype=DTYPE)
        self.cu_seqlens = torch.arange(
            0, tokens + 1, T, device=DEVICE, dtype=torch.int32
        )
        # Slot 0 is NULL_BLOCK_ID, which the kernel skips.
        slots = torch.randperm(tokens, device=DEVICE, dtype=torch.int32) + 1
        if T == 1:
            self.state_indices = slots
            self.num_accepted_tokens = None
        else:
            self.state_indices = slots.view(batch, T)
            self.num_accepted_tokens = torch.randint(
                1, T + 1, (batch,), device=DEVICE, dtype=torch.int32
            )
        self.state = torch.randn(tokens + 1, HV, V, K, device=DEVICE, dtype=state_dtype)

    def __call__(self) -> None:
        common = dict(
            q=self.q,
            k=self.k,
            v=self.v,
            scale=self.scale,
            initial_state=self.state,
            inplace_final_state=True,
            cu_seqlens=self.cu_seqlens,
            ssm_state_indices=self.state_indices,
            num_accepted_tokens=self.num_accepted_tokens,
            use_qk_l2norm_in_kernel=True,
        )
        if self.launcher == "recurrent":
            from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
                fused_recurrent_gated_delta_rule_fwd,
            )

            fused_recurrent_gated_delta_rule_fwd(g=self.g, beta=self.beta, **common)
        elif self.launcher == "sigmoid_gating":
            from vllm.third_party.flash_linear_attention.ops import (
                fused_sigmoid_gating_delta_rule_update,
            )

            fused_sigmoid_gating_delta_rule_update(
                A_log=self.A_log, a=self.a, b=self.b, dt_bias=self.dt_bias, **common
            )
        else:
            from vllm.third_party.flash_linear_attention.ops.kda import (
                fused_recurrent_kda_fwd,
            )

            fused_recurrent_kda_fwd(g=self.g, beta=self.beta, **common)


def _bench_graph_layers(calls: list) -> float:
    """Per-call microseconds for a CUDA graph holding every call once."""
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
    total_ms = triton.testing.do_bench(
        graph.replay, warmup=50, rep=300, return_mode="median"
    )
    del graph
    return total_ms * 1e3 / len(calls)


def run(args: argparse.Namespace) -> list[dict]:
    state_dtype = getattr(torch, args.state_dtype)
    elem = torch.finfo(state_dtype).bits // 8
    budget = int(args.state_budget_gb * 2**30)
    print(f"device: {torch.cuda.get_device_name()}  state: {args.state_dtype}")
    print(
        f"{'launcher':>15} {'T':>2} {'HV':>3} {'B':>5} {'layers':>6} "
        f"{'us':>9} {'state GB/s':>11}"
    )
    rows = []
    for spec_len in args.spec_len:
        for heads in args.heads:
            for batch in args.batch:
                layer_bytes = (batch * spec_len + 1) * heads * args.head_dim**2 * elem
                if layer_bytes > budget:
                    continue
                layers = max(1, min(args.layers, budget // layer_bytes))
                for launcher in args.launchers:
                    torch.manual_seed(0)
                    calls = [
                        Inputs(
                            launcher,
                            batch,
                            heads,
                            args.head_dim,
                            spec_len,
                            state_dtype,
                        )
                        for _ in range(layers)
                    ]
                    us = _bench_graph_layers(calls)
                    del calls
                    torch.accelerator.empty_cache()
                    # One state read per sequence; spec decode writes T states.
                    traffic = batch * (1 + spec_len) * heads * args.head_dim**2
                    gbps = traffic * elem / (us * 1e-6) / 1e9
                    print(
                        f"{launcher:>15} {spec_len:>2} {heads:>3} {batch:>5} "
                        f"{layers:>6} {us:>9.2f} {gbps:>11.1f}"
                    )
                    rows.append(
                        dict(
                            launcher=launcher,
                            spec_len=spec_len,
                            heads=heads,
                            batch=batch,
                            us=us,
                        )
                    )
    return rows


def compare(base_path: str, new_path: str) -> None:
    def load(path: str) -> dict:
        with open(path) as f:
            data = json.load(f)
        rows = {
            (r["launcher"], r["spec_len"], r["heads"], r["batch"]): r["us"]
            for r in data["rows"]
        }
        return data["meta"], rows

    base_meta, base = load(base_path)
    new_meta, new = load(new_path)
    print(f"base: {base_meta}\nnew:  {new_meta}")
    print(
        f"{'launcher':>15} {'T':>2} {'HV':>3} {'B':>5} "
        f"{'base us':>9} {'new us':>9} {'new/base':>9}"
    )
    ratios = []
    for key in base:
        if key not in new:
            continue
        ratio = new[key] / base[key]
        ratios.append(ratio)
        print(
            f"{key[0]:>15} {key[1]:>2} {key[2]:>3} {key[3]:>5} "
            f"{base[key]:>9.2f} {new[key]:>9.2f} {ratio:>9.3f}"
        )
    geomean = math.exp(sum(map(math.log, ratios)) / len(ratios))
    print(
        f"\ngeomean new/base: {geomean:.4f}  "
        f"min: {min(ratios):.3f}  max: {max(ratios):.3f}  n={len(ratios)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--launchers", nargs="+", choices=LAUNCHERS, default=list(LAUNCHERS)
    )
    parser.add_argument(
        "--batch", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256]
    )
    parser.add_argument(
        "--heads", type=int, nargs="+", default=[8, 16, 32], help="value heads"
    )
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--spec-len",
        type=int,
        nargs="+",
        default=[1, 4],
        help="tokens per sequence; >1 runs the spec-decode path",
    )
    parser.add_argument(
        "--state-dtype", choices=("bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=36,
        help="calls per graph, each with its own state (capped by budget)",
    )
    parser.add_argument(
        "--state-budget-gb",
        type=float,
        default=1.5,
        help="max total recurrent-state memory per measurement",
    )
    parser.add_argument("--label", default="")
    parser.add_argument("--save", help="write results to this JSON file")
    parser.add_argument("--compare", nargs=2, metavar=("BASE", "NEW"))
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    rows = run(args)
    if args.save:
        meta = dict(
            label=args.label,
            device=torch.cuda.get_device_name(),
            state_dtype=args.state_dtype,
            torch=torch.__version__,
            triton=triton.__version__,
        )
        with open(args.save, "w") as f:
            json.dump(dict(meta=meta, rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
