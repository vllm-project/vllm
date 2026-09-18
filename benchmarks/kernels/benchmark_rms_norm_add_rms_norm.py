# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the fused post-norm -> residual add -> pre-norm boundary.

Compares, per boundary, the two-op sequence (rms_norm + fused_add_rms_norm)
against the single rms_norm_add_rms_norm op for each registered provider.
Timings are taken from CUDA-graph replays so that launch overhead, which is
what the fusion removes at decode batch sizes, is included.
"""

import torch

import vllm.kernels  # noqa: F401  registers op implementations
from vllm import ir
from vllm.benchmarks.lib.utils import default_vllm_config
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, set_random_seed

BOUNDARIES_PER_GRAPH = 32


def _graph_time_us(fn, num_iters: int) -> float:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / num_iters / BOUNDARIES_PER_GRAPH


@torch.inference_mode()
@default_vllm_config()
def main(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    seed: int = 0,
    num_iters: int = 200,
) -> None:
    set_random_seed(seed)
    torch.set_default_device("cuda")

    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    residual = torch.randn(num_tokens, hidden_size, dtype=dtype) * 4
    weight = torch.randn(hidden_size, dtype=dtype) * 0.1 + 1.0
    weight_residual = torch.randn(hidden_size, dtype=dtype) * 0.1 + 1.0
    epsilon = 1e-6
    xs = [x.clone() for _ in range(BOUNDARIES_PER_GRAPH)]
    rs = [residual.clone() for _ in range(BOUNDARIES_PER_GRAPH)]

    def two_ops(providers: list[str]):
        def run():
            with (
                ir.ops.rms_norm.set_priority(providers),
                ir.ops.fused_add_rms_norm.set_priority(providers),
            ):
                for i in range(BOUNDARIES_PER_GRAPH):
                    h = ir.ops.rms_norm(xs[i], weight, epsilon)
                    ir.ops.fused_add_rms_norm(h, rs[i], weight_residual, epsilon)

        return run

    def fused(providers: list[str]):
        def run():
            with ir.ops.rms_norm_add_rms_norm.set_priority(providers):
                for i in range(BOUNDARIES_PER_GRAPH):
                    ir.ops.rms_norm_add_rms_norm(
                        xs[i], rs[i], weight, weight_residual, epsilon
                    )

        return run

    def two_ops_compiled():
        # What the compiled model runs when "native" wins the priority list:
        # Inductor codegen for the two native ops.
        rms_native = ir.ops.rms_norm.impls["native"].impl_fn
        add_native = ir.ops.fused_add_rms_norm.impls["native"].impl_fn

        def boundary(x_i, r_i):
            return add_native(
                rms_native(x_i, weight, epsilon), r_i, weight_residual, epsilon
            )

        boundary_c = torch.compile(boundary, dynamic=False)
        boundary_c(xs[0], rs[0])

        def run():
            for i in range(BOUNDARIES_PER_GRAPH):
                boundary_c(xs[i], rs[i])

        return run

    rows = [
        ("rms_norm + fused_add_rms_norm [native, torch.compile]", two_ops_compiled())
    ]
    for provider, impl in ir.ops.rms_norm.impls.items():
        if impl.supported and impl.supports_args(x, weight, epsilon):
            rows.append(
                (
                    f"rms_norm + fused_add_rms_norm [{provider}, eager]",
                    two_ops([provider, "native"]),
                )
            )
    for provider, impl in ir.ops.rms_norm_add_rms_norm.impls.items():
        if impl.supported and impl.supports_args(
            x, residual, weight, weight_residual, epsilon
        ):
            rows.append(
                (
                    f"rms_norm_add_rms_norm [{provider}, eager]",
                    fused([provider, "native"]),
                )
            )

    print(f"num_tokens={num_tokens} hidden_size={hidden_size} dtype={dtype}")
    for name, fn in rows:
        print(f"  {name:58s} {_graph_time_us(fn, num_iters):8.3f} us / boundary")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the fused rms_norm_add_rms_norm op."
    )
    parser.add_argument("--num-tokens", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=5376)
    parser.add_argument(
        "--dtype", type=str, choices=["half", "bfloat16", "float"], default="bfloat16"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-iters", type=int, default=200)
    args = parser.parse_args()
    print(args)

    main(
        num_tokens=args.num_tokens,
        hidden_size=args.hidden_size,
        dtype=STR_DTYPE_TO_TORCH_DTYPE[args.dtype],
        seed=args.seed,
        num_iters=args.num_iters,
    )
