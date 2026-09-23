# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the indexed-state Helion and Triton KDA prefill paths.

This exercises the unbounded gate and state-pool contract used by Kimi Linear.
Packed cases pass scheduler-precomputed chunk metadata to Helion, matching the
runtime integration. FlashKDA is not included because it requires the bounded
gate used by Kimi-K3; use ``benchmark_kimi_k3_kda_prefill.py`` for that
three-way comparison.

Example:
    python benchmarks/kernels/benchmark_helion_kda_prefill.py \
        --mode graph --layers 20 --output-json /tmp/kda-prefill.json
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from vllm.kernels.helion.ops.kda.kda_prefill import chunk_kda as helion_chunk_kda
from vllm.model_executor.layers.mamba.ops.gather_initial_states import (
    gather_initial_states,
)
from vllm.models.kimi_k3.nvidia.ops.third_party.kda import (
    chunk_kda_with_fused_gate,
)
from vllm.third_party.flash_linear_attention.ops.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from vllm.triton_utils import triton

HEAD_DIM = 128
CHUNK_SIZE = 64
STATE_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def _parse_case(value: str) -> list[int]:
    lengths = [int(item) for item in value.split(",")]
    if not lengths or any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError("sequence lengths must be positive")
    return lengths


class Inputs:
    def __init__(
        self,
        lengths: list[int],
        num_heads: int,
        state_dtype: torch.dtype,
        seed: int,
    ) -> None:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        num_tokens = sum(lengths)
        num_sequences = len(lengths)
        shape = (1, num_tokens, num_heads, HEAD_DIM)
        self.q = torch.randn(
            shape,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        self.k = torch.randn_like(self.q)
        self.helion_v = torch.randn_like(self.q)
        self.triton_v = self.helion_v.clone()
        self.raw_g = torch.randn_like(self.q)
        self.raw_beta = torch.randn(
            (1, num_tokens, num_heads),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        self.a_log = 0.2 * torch.randn(
            num_heads,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        self.dt_bias = 0.1 * torch.randn(
            num_heads * HEAD_DIM,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        self.cu_seqlens = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            device="cuda",
            dtype=torch.int32,
        )
        self.chunk_indices = prepare_chunk_indices(self.cu_seqlens, CHUNK_SIZE)
        self.chunk_offsets = prepare_chunk_offsets(self.cu_seqlens, CHUNK_SIZE)
        self.state_indices = torch.arange(
            1,
            num_sequences + 1,
            device="cuda",
            dtype=torch.int32,
        )
        self.has_initial_state = torch.zeros(
            num_sequences,
            device="cuda",
            dtype=torch.bool,
        )
        state = 0.01 * torch.randn(
            num_sequences + 1,
            num_heads,
            HEAD_DIM,
            HEAD_DIM,
            device="cuda",
            dtype=state_dtype,
            generator=generator,
        )
        self.helion_state = state.clone()
        self.triton_state = state.clone()

    def run_helion(self) -> torch.Tensor:
        is_varlen = self.state_indices.numel() > 1
        return helion_chunk_kda(
            q=self.q,
            k=self.k,
            v=self.helion_v,
            g=self.raw_g,
            beta=self.raw_beta,
            scale=HEAD_DIM**-0.5,
            initial_state=self.helion_state,
            initial_state_indices=self.state_indices,
            has_initial_state=self.has_initial_state,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=self.cu_seqlens if is_varlen else None,
            chunk_indices=self.chunk_indices if is_varlen else None,
            chunk_offsets=self.chunk_offsets if is_varlen else None,
            A_log=self.a_log,
            dt_bias=self.dt_bias,
            lower_bound=None,
            beta_is_logit=True,
        )

    def run_triton(self) -> torch.Tensor:
        initial_state = gather_initial_states(
            self.triton_state,
            self.state_indices,
            self.has_initial_state,
        )
        output, final_state = chunk_kda_with_fused_gate(
            q=self.q,
            k=self.k,
            v=self.triton_v,
            raw_g=self.raw_g,
            raw_beta=self.raw_beta,
            A_log=self.a_log,
            g_bias=self.dt_bias,
            lower_bound=None,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=self.cu_seqlens,
        )
        self.triton_state[self.state_indices] = final_state.to(self.triton_state.dtype)
        return output


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
    lengths: list[int],
    num_heads: int,
    state_dtype: torch.dtype,
    layers: int,
    use_cudagraph: bool,
    warmup: int,
    rep: int,
) -> dict[str, float]:
    seed = sum(lengths) * 100 + num_heads
    correctness_inputs = Inputs(lengths, num_heads, state_dtype, seed)
    helion_out = correctness_inputs.run_helion()
    triton_out = correctness_inputs.run_triton()
    torch.cuda.synchronize()
    output_max_abs = (helion_out.float() - triton_out.float()).abs().max().item()
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
        Inputs(lengths, num_heads, state_dtype, seed + layer + 1)
        for layer in range(layers)
    ]
    # Scheduler metadata is shared by every layer in an actual model forward.
    # Sharing the tensor objects also exercises FLA's metadata cache correctly.
    metadata_source = inputs[0]
    for item in inputs[1:]:
        item.cu_seqlens = metadata_source.cu_seqlens
        item.chunk_indices = metadata_source.chunk_indices
        item.chunk_offsets = metadata_source.chunk_offsets
        item.state_indices = metadata_source.state_indices
        item.has_initial_state = metadata_source.has_initial_state
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
    parser.add_argument(
        "--state-dtypes",
        nargs="+",
        choices=STATE_DTYPES,
        default=list(STATE_DTYPES),
    )
    parser.add_argument(
        "--case",
        action="append",
        type=_parse_case,
        dest="cases",
        help="comma-separated packed sequence lengths; may be repeated",
    )
    parser.add_argument("--layers", type=int, default=20)
    parser.add_argument("--mode", choices=("eager", "graph"), default="graph")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rep", type=int, default=200)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.layers <= 0:
        parser.error("--layers must be positive")
    cases = args.cases or [
        [128],
        [64, 64],
        [17, 111],
        [17, 31, 47, 63, 79, 95, 111, 127],
    ]
    use_cudagraph = args.mode == "graph"

    print(
        f"device={torch.cuda.get_device_name()} mode={args.mode} layers={args.layers}"
    )
    print(
        "heads,state_dtype,lengths,helion_us,triton_us,triton/helion,"
        "output_max_abs,state_max_abs"
    )
    results: list[dict[str, Any]] = []
    for num_heads in args.heads:
        for state_dtype_name in args.state_dtypes:
            state_dtype = STATE_DTYPES[state_dtype_name]
            for lengths in cases:
                timings = _benchmark_case(
                    lengths,
                    num_heads,
                    state_dtype,
                    args.layers,
                    use_cudagraph,
                    args.warmup,
                    args.rep,
                )
                result = {
                    "heads": num_heads,
                    "state_dtype": state_dtype_name,
                    "lengths": lengths,
                    **timings,
                }
                results.append(result)
                print(
                    f"{num_heads},{state_dtype_name},{lengths},"
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
            "lower_bound": None,
            "layers": args.layers,
            "warmup": args.warmup,
            "rep": args.rep,
            "results": results,
        }
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
