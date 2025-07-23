# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pickle as pkl
import time
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product
from typing import Callable, Optional

import torch
import torch.utils.benchmark as TBenchmark
from torch.utils.benchmark import Measurement as TMeasurement
from tqdm import tqdm

import vllm._custom_ops as ops
from vllm.model_executor.layers.layernorm import RMSNorm


@dataclass
class bench_params_t:
    num_tokens: int
    hidden_size: int
    add_residual: bool
    dtype: torch.dtype

    def description(self):
        return (
            f"N {self.num_tokens} "
            f"x D {self.hidden_size} "
            f"x R {self.add_residual} "
            f"x DT {self.dtype}"
        )


def get_bench_params() -> list[bench_params_t]:
    ## Test Fixtures
    NUM_TOKENS = [2**x for x in range(11)]
    HIDDEN_SIZES = list(range(1024, 8129, 1024))
    ADD_RESIDUAL = [True, False]
    DTYPES = [torch.bfloat16, torch.float]

    combinations = product(NUM_TOKENS, HIDDEN_SIZES, ADD_RESIDUAL, DTYPES)
    bench_params = list(
        map(lambda x: bench_params_t(x[0], x[1], x[2], x[3]), combinations)
    )
    return bench_params


# Reference impls
def unfused_int8_impl(
    rms_norm_layer: RMSNorm,
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    quant_dtype: torch.dtype,
):
    # Norm
    torch_out = None
    if residual is None:
        torch_out = rms_norm_layer.forward_cuda(x, residual)
    else:
        torch_out, _ = rms_norm_layer.forward_cuda(x, residual)

    # Quant
    torch_out, _, _ = ops.scaled_int8_quant(torch_out)


def unfused_fp8_impl(
    rms_norm_layer: RMSNorm,
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    quant_dtype: torch.dtype,
):
    # Norm
    torch_out = None
    if residual is None:
        torch_out = rms_norm_layer.forward_cuda(x, residual)
    else:
        torch_out, _ = rms_norm_layer.forward_cuda(x, residual)

    # Quant
    torch_out, _ = ops.scaled_fp8_quant(torch_out)


def fused_impl(
    rms_norm_layer: RMSNorm,  # this stores the weights
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    quant_dtype: torch.dtype,
):
    out, _ = ops.rms_norm_dynamic_per_token_quant(
        x, rms_norm_layer.weight, 1e-6, quant_dtype, residual=residual
    )


# Bench functions
def bench_fn(
    rms_norm_layer: RMSNorm,
    x: torch.Tensor,
    residual: torch.Tensor,
    quant_dtype: torch.dtype,
    label: str,
    sub_label: str,
    fn: Callable,
    description: str,
) -> TMeasurement:
    min_run_time = 1

    globals = {
        "rms_norm_layer": rms_norm_layer,
        "x": x,
        "residual": residual,
        "quant_dtype": quant_dtype,
        "fn": fn,
    }
    return TBenchmark.Timer(
        stmt="fn(rms_norm_layer, x, residual, quant_dtype)",
        globals=globals,
        label=label,
        sub_label=sub_label,
        description=description,
    ).blocked_autorange(min_run_time=min_run_time)


def bench(params: bench_params_t, label: str, sub_label: str) -> Iterable[TMeasurement]:
    # Make inputs
    layer = RMSNorm(params.hidden_size, 1e-6).to(dtype=params.dtype)
    # Make weights
    layer.weight.data.normal_(mean=1.0, std=0.1)
    # Make inputs
    scale = 1 / params.hidden_size
    x = (
        torch.randn(
            params.num_tokens, params.hidden_size, dtype=params.dtype, device="cuda"
        )
        * scale
    )
    residual = (
        (torch.randn_like(x) * scale).to(device="cuda") if params.add_residual else None
    )

    timers = []

    # unfused int8 impl.
    timers.append(
        bench_fn(
            layer,
            x,
            residual,
            torch.int8,
            label,
            sub_label,
            unfused_int8_impl,
            "unfused_int8_impl",
        )
    )

    # unfused fp8 impl.
    timers.append(
        bench_fn(
            layer,
            x,
            residual,
            torch.float8_e4m3fn,
            label,
            sub_label,
            unfused_fp8_impl,
            "unfused_fp8_impl",
        )
    )

    # fused int8 impl.
    timers.append(
        bench_fn(
            layer,
            x,
            residual,
            torch.int8,
            label,
            sub_label,
            fused_impl,
            "fused_int8_impl",
        )
    )

    # fused fp8 impl.
    timers.append(
        bench_fn(
            layer,
            x,
            residual,
            torch.float8_e4m3fn,
            label,
            sub_label,
            fused_impl,
            "fused_fp8_impl",
        )
    )

    print_timers(timers)

    return timers


# launch bench
# runner
def print_timers(timers: Iterable[TMeasurement]):
    compare = TBenchmark.Compare(timers)
    compare.print()


def main():
    torch.set_default_device("cuda")
    bench_params = get_bench_params()

    timers = []
    for bp in tqdm(bench_params):
        timers.extend(bench(bp, "rms-norm-dynamic-per-token-quant", bp.description()))
    print_timers(timers)

    # pickle all the results
    timestamp = int(time.time())
    with open(f"rms_norm_dpt_quant-{timestamp}.pkl", "wb") as f:
        pkl.dump(timers, f)


if __name__ == "__main__":
    main()
