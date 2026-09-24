# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark Helion and Triton packed KDA decode kernels.

The default matrix covers the Kimi-K3 per-rank head count (12), Kimi Linear
TP=2 (16), and Kimi Linear TP=1 (32), with every state dtype accepted by the
Helion kernel. CUDA-graph timings use independent state pools for each layer so
the recurrent state cannot remain artificially resident in cache.

Example:
    python benchmarks/kernels/benchmark_helion_kda_decode.py \
        --mode graph --layers 20 --output-json /tmp/kda-decode.json
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from vllm.kernels.helion.ops.kda.kda_decode import (
    helion_fused_recurrent_kda_packed_decode,
)
from vllm.models.kimi_k3.nvidia.ops.third_party.kda import (
    fused_recurrent_kda_packed_decode,
)
from vllm.triton_utils import triton

HEAD_DIM = 128
STATE_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}
GATE_LOWER_BOUNDS = {
    "bounded": -3.0,
    "unbounded": None,
}


class Inputs:
    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        state_dtype: torch.dtype,
        lower_bound: float | None,
        seed: int,
    ) -> None:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.lower_bound = lower_bound
        self.mixed_qkv = torch.randn(
            batch_size,
            3 * num_heads * HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        self.raw_g = torch.randn(
            1,
            batch_size,
            num_heads,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        self.raw_beta = torch.randn(
            1,
            batch_size,
            num_heads,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        self.a_log = 0.2 * torch.randn(
            num_heads,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        self.dt_bias = 0.1 * torch.randn(
            num_heads * HEAD_DIM,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        self.state_indices = torch.arange(
            1,
            batch_size + 1,
            dtype=torch.int32,
            device="cuda",
        )
        state = 0.01 * torch.randn(
            batch_size + 1,
            num_heads,
            HEAD_DIM,
            HEAD_DIM,
            dtype=state_dtype,
            device="cuda",
            generator=generator,
        )
        self.helion_state = state.clone()
        self.triton_state = state.clone()
        self.helion_out = torch.empty(
            batch_size,
            1,
            num_heads,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )

    def run_helion(self) -> tuple[torch.Tensor, torch.Tensor]:
        return helion_fused_recurrent_kda_packed_decode(
            mixed_qkv=self.mixed_qkv,
            a=self.raw_g.view(self.batch_size, -1),
            b=self.raw_beta.view(self.batch_size, -1),
            A_log=self.a_log,
            dt_bias=self.dt_bias,
            scale=HEAD_DIM**-0.5,
            initial_state=self.helion_state,
            out=self.helion_out,
            ssm_state_indices=self.state_indices,
            use_qk_l2norm_in_kernel=True,
            lower_bound=self.lower_bound,
        )

    def run_triton(self) -> tuple[torch.Tensor, torch.Tensor]:
        return fused_recurrent_kda_packed_decode(
            mixed_qkv=self.mixed_qkv,
            raw_g=self.raw_g,
            raw_beta=self.raw_beta,
            A_log=self.a_log,
            dt_bias=self.dt_bias.view(self.num_heads, HEAD_DIM),
            lower_bound=self.lower_bound,
            initial_state=self.triton_state,
            state_indices=self.state_indices,
            scale=HEAD_DIM**-0.5,
        )


def _bench(
    calls: list[Callable[[], object]],
    use_cudagraph: bool,
    warmup: int,
    rep: int,
) -> float:
    def run_all() -> None:
        for call in calls:
            call()

    run_all()
    torch.cuda.synchronize()
    if use_cudagraph:
        total_ms = triton.testing.do_bench_cudagraph(
            run_all,
            rep=rep,
            quantiles=[0.5],
        )
    else:
        total_ms = triton.testing.do_bench(
            run_all,
            warmup=warmup,
            rep=rep,
            quantiles=[0.5],
        )
    return total_ms * 1000 / len(calls)


def _benchmark_case(
    batch_size: int,
    num_heads: int,
    state_dtype: torch.dtype,
    lower_bound: float | None,
    layers: int,
    use_cudagraph: bool,
    warmup: int,
    rep: int,
) -> dict[str, float]:
    correctness_inputs = Inputs(
        batch_size,
        num_heads,
        state_dtype,
        lower_bound,
        seed=batch_size * 1000 + num_heads,
    )
    helion_out, _ = correctness_inputs.run_helion()
    triton_out, _ = correctness_inputs.run_triton()
    torch.cuda.synchronize()
    output_max_abs = (
        (helion_out.float() - triton_out.transpose(0, 1).float()).abs().max().item()
    )
    state_max_abs = (
        (
            correctness_inputs.helion_state.float()
            - correctness_inputs.triton_state.float()
        )
        .abs()
        .max()
        .item()
    )

    inputs = [
        Inputs(
            batch_size,
            num_heads,
            state_dtype,
            lower_bound,
            seed=batch_size * 1000 + num_heads + layer + 1,
        )
        for layer in range(layers)
    ]
    helion_us = _bench(
        [item.run_helion for item in inputs],
        use_cudagraph,
        warmup,
        rep,
    )
    triton_us = _bench(
        [item.run_triton for item in inputs],
        use_cudagraph,
        warmup,
        rep,
    )
    return {
        "helion_us": helion_us,
        "triton_us": triton_us,
        "triton_over_helion": triton_us / helion_us,
        "output_max_abs": output_max_abs,
        "state_max_abs": state_max_abs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", type=int, nargs="+", default=[12, 16, 32])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument(
        "--state-dtypes",
        nargs="+",
        choices=STATE_DTYPES,
        default=list(STATE_DTYPES),
    )
    parser.add_argument(
        "--gate-modes",
        nargs="+",
        choices=GATE_LOWER_BOUNDS,
        default=list(GATE_LOWER_BOUNDS),
    )
    parser.add_argument("--layers", type=int, default=20)
    parser.add_argument("--mode", choices=("eager", "graph"), default="graph")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rep", type=int, default=200)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.layers <= 0:
        parser.error("--layers must be positive")

    use_cudagraph = args.mode == "graph"
    print(
        f"device={torch.cuda.get_device_name()} mode={args.mode} layers={args.layers}"
    )
    print(
        "gate,heads,state_dtype,batch,helion_us,triton_us,triton/helion,"
        "output_max_abs,state_max_abs"
    )
    results: list[dict[str, Any]] = []
    for gate_mode in args.gate_modes:
        lower_bound = GATE_LOWER_BOUNDS[gate_mode]
        for num_heads in args.heads:
            for state_dtype_name in args.state_dtypes:
                state_dtype = STATE_DTYPES[state_dtype_name]
                for batch_size in args.batch_sizes:
                    timings = _benchmark_case(
                        batch_size,
                        num_heads,
                        state_dtype,
                        lower_bound,
                        args.layers,
                        use_cudagraph,
                        args.warmup,
                        args.rep,
                    )
                    result = {
                        "gate": gate_mode,
                        "lower_bound": lower_bound,
                        "heads": num_heads,
                        "state_dtype": state_dtype_name,
                        "batch_size": batch_size,
                        **timings,
                    }
                    results.append(result)
                    print(
                        f"{gate_mode},{num_heads},{state_dtype_name},{batch_size},"
                        f"{timings['helion_us']:.2f},{timings['triton_us']:.2f},"
                        f"{timings['triton_over_helion']:.3f},"
                        f"{timings['output_max_abs']:.3e},"
                        f"{timings['state_max_abs']:.3e}"
                    )
                    del timings
                    torch.cuda.empty_cache()

    if args.output_json is not None:
        payload = {
            "device": torch.cuda.get_device_name(),
            "mode": args.mode,
            "head_dim": HEAD_DIM,
            "layers": args.layers,
            "warmup": args.warmup,
            "rep": args.rep,
            "results": results,
        }
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
