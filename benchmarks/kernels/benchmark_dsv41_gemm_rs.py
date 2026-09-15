# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP4 MXFP8 native GEMM-RS correctness and latency smoke."""

import argparse
import copy
import json
import os
import statistics
from collections.abc import Callable
from pathlib import Path

import torch
import torch.distributed as dist

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.parallel_state import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.models.common.ops.sequence_parallel import sp_reduce_scatter
from vllm.models.deepseek_v41.nvidia.ops.gemm_rs import Mxfp8GemmRS, WoBGemmRS
from vllm.models.deepseek_v41.quant_config import DeepseekV4FP8Config


def capture_graph(
    op: Callable[[], torch.Tensor],
    stream: torch.cuda.Stream,
    cpu_group: dist.ProcessGroup,
) -> tuple[torch.cuda.CUDAGraph, list[torch.Tensor | None]]:
    result: list[torch.Tensor | None] = [None]
    stream.wait_stream(torch.cuda.current_stream())
    dist.barrier(group=cpu_group)
    with torch.cuda.stream(stream):
        for _ in range(3):
            result[0] = op()
    stream.synchronize()
    dist.barrier(group=cpu_group)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result[0] = op()
    torch.cuda.current_stream().wait_stream(stream)
    dist.barrier(group=cpu_group)
    return graph, result


def benchmark_graphs(
    candidate_graphs: dict[str, list[torch.cuda.CUDAGraph]],
    warmup_replays: int,
    samples: int,
    device_group: dist.ProcessGroup,
    device_barrier: Callable[[], None],
) -> dict[str, float]:
    candidate_names = list(candidate_graphs)
    for round_index in range(warmup_replays):
        for candidate_index in range(len(candidate_names)):
            candidate_id = (round_index + candidate_index) % len(candidate_names)
            name = candidate_names[candidate_id]
            graphs = candidate_graphs[name]
            device_barrier()
            graphs[round_index % len(graphs)].replay()
    torch.accelerator.synchronize()

    timings: dict[str, list[float]] = {name: [] for name in candidate_names}
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for sample_index in range(samples):
        for candidate_index in range(len(candidate_names)):
            candidate_id = (sample_index + candidate_index) % len(candidate_names)
            name = candidate_names[candidate_id]
            graphs = candidate_graphs[name]
            device_barrier()
            start.record()
            graphs[sample_index % len(graphs)].replay()
            end.record()
            end.synchronize()

            elapsed = torch.tensor(
                start.elapsed_time(end) * 1000,
                dtype=torch.float64,
                device=torch.accelerator.current_device_index(),
            )
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=device_group)
            timings[name].append(elapsed.item())
    return {name: statistics.median(values) for name, values in timings.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", nargs="+", type=int, default=[128, 129, 1024, 4096])
    parser.add_argument("--workspaces", type=int, default=10)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    torch.manual_seed(2026 + rank)
    init_distributed_environment()
    with set_current_vllm_config(VllmConfig()):
        initialize_model_parallel(tensor_model_parallel_size=4)
        with torch.device("cuda"):
            linear = RowParallelLinear(
                8192,
                5120,
                bias=False,
                input_is_parallel=True,
                reduce_results=False,
                return_bias=False,
                params_dtype=torch.bfloat16,
                quant_config=DeepseekV4FP8Config(
                    is_checkpoint_fp8_serialized=True, weight_block_size=[32, 32]
                ),
                prefix="model.layers.0.attn.wo_b",
            )
        with torch.no_grad():
            linear.weight.copy_(torch.randn_like(linear.weight, dtype=torch.float32))
            linear.weight_scale.view(torch.uint8).random_(119, 124)
        linear.quant_method.process_weights_after_loading(linear)
        b = linear.weight.t()
        assert b.is_contiguous() and b.shape == (5120, 2048)
        fused = Mxfp8GemmRS(max(args.tokens), 5120, 2048)
        linears = [linear] + [copy.deepcopy(linear) for _ in range(args.workspaces - 1)]
        tp = get_tp_group()
        stream = torch.cuda.Stream()
        sync_input = torch.zeros(1, device="cuda")
        sync_output = torch.empty_like(sync_input)

        def device_barrier():
            tp.device_communicator.pynccl_comm.all_reduce(sync_input, sync_output)

        results = []
        for m in args.tokens:
            x = torch.randn(m, 2048, device="cuda", dtype=torch.bfloat16)
            partial = linear(x)
            ref = sp_reduce_scatter(partial).clone()
            bound = partial.float().abs() * torch.finfo(torch.bfloat16).eps * 2
            dist.all_reduce(bound)
            local_m = (m + 3) // 4
            bound = torch.nn.functional.pad(bound, (0, 0, 0, local_m * 4 - m))
            bound = bound[rank * local_m : (rank + 1) * local_m]
            bound += ref.float().abs() * torch.finfo(torch.bfloat16).eps

            def assert_valid(actual, expected, bound=bound):
                error = (actual.float() - expected.float()).abs()
                assert torch.all(error <= bound)
                assert (
                    error.square().mean() < expected.float().square().mean() * 0.005**2
                )

            actual = fused(x, b, linear.weight_scale)
            torch.accelerator.synchronize()
            delta = actual.float() - ref.float()
            stats = torch.stack(
                (
                    delta.abs().max(),
                    delta.square().mean().sqrt(),
                    ref.float().square().mean().sqrt(),
                )
            )
            dist.all_reduce(stats, op=dist.ReduceOp.MAX)
            valid = (delta.abs() <= bound).all().int()
            dist.all_reduce(valid, op=dist.ReduceOp.MIN)
            result = dict(tokens=m, valid=bool(valid.item()), stats=stats.tolist())
            assert result["valid"], result
            assert_valid(WoBGemmRS(fused, linear, True)(x), ref)
            # Changed inputs and different grid sizes must not consume stale tiles.
            for _ in range(3):
                x.neg_()
                expected = sp_reduce_scatter(linear(x)).clone()
                actual = fused(x, b, linear.weight_scale)
                assert_valid(actual, expected)
            graphs = {"baseline_us": [], "fused_us": []}
            inputs = [x.clone() for _ in linears]
            for current_linear, current_x in zip(linears, inputs):
                current_b = current_linear.weight.t()
                current_scale = current_linear.weight_scale
                for name, op in (
                    (
                        "baseline_us",
                        lambda x=current_x, linear=current_linear: sp_reduce_scatter(
                            linear(x)
                        ),
                    ),
                    (
                        "fused_us",
                        lambda x=current_x, b=current_b, scale=current_scale: fused(
                            x, b, scale
                        ),
                    ),
                ):
                    graph, outputs = capture_graph(op, stream, tp.cpu_group)
                    for _ in range(3):
                        current_x.neg_()
                        expected = sp_reduce_scatter(current_linear(current_x)).clone()
                        graph.replay()
                        torch.accelerator.synchronize()
                        assert_valid(outputs[0], expected)
                    graphs[name].append(graph)
            result.update(
                benchmark_graphs(
                    graphs, 5, args.samples, tp.device_group, device_barrier
                )
            )
            result["workspaces"] = args.workspaces
            if args.profile and m == max(args.tokens):
                dist.barrier(group=tp.cpu_group)
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as prof:
                    for name, group_graphs in graphs.items():
                        with torch.profiler.record_function(name):
                            group_graphs[0].replay()
                            torch.accelerator.synchronize()
                if rank == 0:
                    prof.export_chrome_trace(
                        str(args.output.with_suffix(".trace.json"))
                    )
            result["speedup"] = result["baseline_us"] / result["fused_us"]
            results.append(result)
            if rank == 0:
                print(json.dumps(result), flush=True)
                args.output.write_text(json.dumps(results, indent=2) + "\n")
    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
