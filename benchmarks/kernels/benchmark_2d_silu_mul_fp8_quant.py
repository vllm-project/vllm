# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import Any

import torch
import torch.utils.benchmark as TBenchmark
from torch.utils.benchmark import Measurement as TMeasurement

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
    silu_mul_per_token_group_quant_fp8_colmajor,
)

from .utils import ArgPool, Bench, CudaGraphBenchParams

GROUP_SIZE = 128
FLOAT8_T = torch.float8_e4m3fn


def print_timers(timers: list[TMeasurement], cuda_graph_nops: int):
    print(
        f"Note : The timings reported above is for {cuda_graph_nops} "
        "consecutive invocations of the benchmarking functions. "
        f"Please divide by {cuda_graph_nops} for single invocation "
        "timings."
    )
    compare = TBenchmark.Compare(timers)
    compare.print()


class ImplType(Enum):
    SILU_MUL_PER_TOKEN_GROUP_QUANT_FP8_COLMAJOR = 1
    REFERENCE = 2

    def get_impl(self):
        if self == ImplType.SILU_MUL_PER_TOKEN_GROUP_QUANT_FP8_COLMAJOR:
            return silu_mul_per_token_group_quant_fp8_colmajor
        elif self == ImplType.REFERENCE:
            return reference
        raise ValueError(f"Unrecognized ImplType {self}")


@dataclass
class BenchmarkTensors:
    input: torch.Tensor
    output: torch.Tensor

    # Reference act output tensor
    ref_act_out: torch.Tensor
    ref_quant_out: torch.Tensor

    @staticmethod
    def make(T: int, N: int) -> "BenchmarkTensors":
        assert T % GROUP_SIZE == 0
        assert N % (GROUP_SIZE * 2) == 0

        input = torch.rand((T, N), dtype=torch.bfloat16, device="cuda")

        # silu_mul_per_token_group_quant_fp8_colmajor output.
        output = torch.rand((T, N // 2), dtype=torch.bfloat16, device="cuda").to(
            FLOAT8_T
        )

        # reference output.
        ref_act_out = torch.empty((T, N // 2), dtype=torch.bfloat16, device="cuda")
        ref_quant_out = torch.empty(
            (T, N // 2), dtype=torch.bfloat16, device="cuda"
        ).to(FLOAT8_T)

        return BenchmarkTensors(
            input=input,
            output=output,
            ref_act_out=ref_act_out,
            ref_quant_out=ref_quant_out,
        )

    @property
    def T(self):
        return self.input.size(0)

    @property
    def N(self):
        return self.input.size(1)

    def make_impl_kwargs(self, impl_type: ImplType) -> dict[str, Any]:
        if impl_type == ImplType.SILU_MUL_PER_TOKEN_GROUP_QUANT_FP8_COLMAJOR:
            return {"input": self.input, "output": self.output}
        elif impl_type == ImplType.REFERENCE:
            return {
                "input": self.input,
                "act_out": self.ref_act_out,
                "quant_out": self.ref_quant_out,
            }
        raise ValueError(f"Unrecognized impl_type {impl_type}")


def reference(
    input: torch.Tensor, act_out: torch.Tensor, quant_out: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.ops._C.silu_and_mul(act_out, input)
    ref_output, ref_output_scales = per_token_group_quant_fp8(
        act_out,
        GROUP_SIZE,
        column_major_scales=True,
        out_q=quant_out,
    )
    return ref_output, ref_output_scales


def bench_impl(
    bench_tensors: list[BenchmarkTensors], impl_type: ImplType
) -> TMeasurement:
    T = bench_tensors[0].T
    N = bench_tensors[0].N

    arg_pool_size = len(bench_tensors)
    kwargs_list = [bt.make_impl_kwargs(impl_type) for bt in bench_tensors]

    # warmup
    for kwargs in kwargs_list:
        impl_type.get_impl()(**kwargs)
    torch.cuda.synchronize()

    # Merge into a single kwargs and qualify arguments as ArgPool
    kwargs = {k: ArgPool([]) for k in kwargs_list[0]}
    for _kwargs in kwargs_list:
        for k, v in _kwargs.items():
            kwargs[k].values.append(v)

    cuda_graph_params = None
    cuda_graph_params = CudaGraphBenchParams(arg_pool_size)
    timer = None
    with Bench(
        cuda_graph_params,
        "silu-mul-quant",
        f"num_tokens={T}, N={N}",
        impl_type.name,
        impl_type.get_impl(),
        **kwargs,
    ) as bench:
        timer = bench.run()
    return timer


def test_correctness(T: int, N: int):
    print(f"Testing num_tokens={T}, N={N} ...")

    bench_tensor = BenchmarkTensors.make(T, N)

    def output_from_impl(impl: ImplType) -> tuple[torch.Tensor, torch.Tensor]:
        return impl.get_impl()(**bench_tensor.make_impl_kwargs(impl))

    # reference output
    ref_out_q, ref_out_s = output_from_impl(ImplType.REFERENCE)

    # test ouptut
    out_q, out_s = output_from_impl(
        ImplType.SILU_MUL_PER_TOKEN_GROUP_QUANT_FP8_COLMAJOR
    )

    torch.testing.assert_close(
        ref_out_q.to(torch.float32), out_q.to(torch.float32), atol=32, rtol=1e-3
    )
    torch.testing.assert_close(ref_out_s, out_s)


def run(Ts: list[int], Ns: list[int], arg_pool_size: int) -> list[TMeasurement]:
    timers = []
    for N, T in product(Ns, Ts):
        test_correctness(T, N)

        bench_tensors: list[BenchmarkTensors] = [
            BenchmarkTensors.make(T, N) for _ in range(arg_pool_size)
        ]

        silu_mul_quant_timer = bench_impl(
            bench_tensors, ImplType.SILU_MUL_PER_TOKEN_GROUP_QUANT_FP8_COLMAJOR
        )
        timers.append(silu_mul_quant_timer)
        reference_timer = bench_impl(bench_tensors, ImplType.REFERENCE)
        timers.append(reference_timer)

        print_timers(
            [silu_mul_quant_timer, reference_timer], cuda_graph_nops=arg_pool_size
        )

    print_timers(timers, cuda_graph_nops=arg_pool_size)

    return timers


if __name__ == "__main__":
    T = [128 * i for i in range(1, 16)] + [2048 * i for i in range(1, 65)]
    N = [2048, 4096, 8192]

    print(f"T = {T}, N = {N}")
    run(T, N, arg_pool_size=8)
