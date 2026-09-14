# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare Kimi-K3 TP8 KDA sequential and multistream projections on B300."""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable

import torch

from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    shape_dynamic_skinny_gemm,
)
from vllm.models.kimi_k3.nvidia.low_latency_gemm import (
    KDA_PROJECTION_OVERLAP_MAX_TOKENS,
    run_kda_projection_overlap,
    try_low_latency_gemm,
)

ProjectionOutput = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _capture(
    fn: Callable[[], list[ProjectionOutput]],
) -> tuple[torch.cuda.CUDAGraph, list[ProjectionOutput]]:
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        fn()
        fn()
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        outputs = fn()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.accelerator.synchronize()
    return graph, outputs


def _benchmark_graph(
    graph: torch.cuda.CUDAGraph,
    num_weight_sets: int,
    repeats: int,
) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / repeats / num_weight_sets


def _validate(
    output: ProjectionOutput,
    hidden_states: torch.Tensor,
    packed_weight: torch.Tensor,
    f_b_weight: torch.Tensor,
) -> None:
    expected_qkvg = torch.nn.functional.linear(
        hidden_states.float(), packed_weight[:6144].float()
    )
    expected_fab = torch.nn.functional.linear(
        hidden_states.float(), packed_weight[6144:6284].float()
    )
    expected_g1 = torch.nn.functional.linear(
        expected_fab[:, :128].to(torch.bfloat16).float(),
        f_b_weight.float(),
    )
    for actual, expected in zip(
        output,
        (expected_qkvg, expected_g1, expected_fab[:, 128:]),
    ):
        cosine = torch.nn.functional.cosine_similarity(
            actual.float().flatten(), expected.flatten(), dim=0
        ).item()
        if cosine <= 0.999:
            raise AssertionError(f"projection cosine similarity is {cosine}")


def _sequential_projection(
    hidden_states: torch.Tensor,
    packed_weight: torch.Tensor,
    f_b_weight: torch.Tensor,
) -> ProjectionOutput:
    projected = try_low_latency_gemm(hidden_states, packed_weight)
    if projected is None:
        projected = torch.mm(hidden_states, packed_weight.t())
    f_a = projected[:, 6144:6272]
    g1 = torch.mm(f_a, f_b_weight.t())
    return projected[:, :6144], g1, projected[:, 6272:6284]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=range(1, 17))
    parser.add_argument("--weight-sets", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    if torch.cuda.get_device_capability() != (10, 3):
        raise SystemExit("This benchmark requires an SM103 GPU")
    if not shape_dynamic_skinny_gemm.is_available():
        raise SystemExit("This benchmark requires CuTe DSL")
    required_ops = ("dsv3_fused_a_gemm",)
    if not all(hasattr(torch.ops._C, name) for name in required_ops):
        raise SystemExit("This benchmark requires the KDA projection GEMMs")
    if not all(1 <= num_tokens <= 16 for num_tokens in args.tokens):
        raise SystemExit("--tokens must be in [1, 16]")

    torch.manual_seed(42)
    packed_weights = [
        torch.randn(6288, 7168, dtype=torch.bfloat16, device="cuda")
        for _ in range(args.weight_sets)
    ]
    f_b_weights = [
        torch.randn(1536, 128, dtype=torch.bfloat16, device="cuda")
        for _ in range(args.weight_sets)
    ]

    from flashinfer.autotuner import autotune
    from flashinfer.gemm import mm_bf16

    with torch.inference_mode(), autotune(tune_mode=True):
        mm_bf16(
            torch.empty(
                (KDA_PROJECTION_OVERLAP_MAX_TOKENS, 7168),
                dtype=torch.bfloat16,
                device="cuda",
            ),
            packed_weights[0][:6144].t(),
            pdl=True,
            backend="cute-dsl",
        )

    print("M  sequential (us)  multistream (us)  fastest")
    for num_tokens in args.tokens:
        hidden_states = [
            torch.randn(
                num_tokens,
                7168,
                dtype=torch.bfloat16,
                device="cuda",
            )
            for _ in range(args.weight_sets)
        ]

        def sequential(
            hidden_states: list[torch.Tensor] = hidden_states,
        ) -> list[ProjectionOutput]:
            return [
                _sequential_projection(x, weight, f_b_weight)
                for x, weight, f_b_weight in zip(
                    hidden_states, packed_weights, f_b_weights
                )
            ]

        candidates: list[tuple[str, Callable[[], list[ProjectionOutput]]]] = [
            ("sequential", sequential)
        ]
        if num_tokens <= KDA_PROJECTION_OVERLAP_MAX_TOKENS:
            aux_stream = torch.cuda.Stream()
            events = [
                (torch.cuda.Event(), torch.cuda.Event())
                for _ in range(args.weight_sets)
            ]

            def overlap(
                hidden_states: list[torch.Tensor] = hidden_states,
                aux_stream: torch.cuda.Stream = aux_stream,
                events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = events,
            ) -> list[ProjectionOutput]:
                return [
                    run_kda_projection_overlap(x, weight, f_b_weight, aux_stream, pair)
                    for x, weight, f_b_weight, pair in zip(
                        hidden_states, packed_weights, f_b_weights, events
                    )
                ]

            candidates.append(("multistream", overlap))

        graphs = {}
        for name, candidate in candidates:
            graph, outputs = _capture(candidate)
            graph.replay()
            torch.accelerator.synchronize()
            _validate(outputs[0], hidden_states[0], packed_weights[0], f_b_weights[0])
            graphs[name] = graph

        samples = {name: [] for name in graphs}
        graph_items = list(graphs.items())
        for trial in range(args.trials):
            ordered_graphs = graph_items if trial % 2 == 0 else graph_items[::-1]
            for name, graph in ordered_graphs:
                samples[name].append(
                    _benchmark_graph(graph, args.weight_sets, args.repeats)
                )
        results = {name: statistics.median(values) for name, values in samples.items()}

        sequential_us = results["sequential"]
        multistream_us = results.get("multistream")
        multistream_text = (
            f"{multistream_us:18.3f}"
            if multistream_us is not None
            else "                 -"
        )
        fastest = (
            "multistream"
            if multistream_us is not None and multistream_us < sequential_us
            else "sequential"
        )
        print(f"{num_tokens:2d} {sequential_us:15.3f}{multistream_text}  {fastest}")


if __name__ == "__main__":
    main()
