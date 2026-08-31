# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark Qwen4Exp PLE packed-varlen convolution against dense packing."""

import argparse
import gc
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import pandas as pd
import torch
import torch.nn.functional as F
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.models.qwen4_exp.nvidia.ops.ple_conv import varlen_dilated_conv1d


@dataclass(frozen=True)
class Case:
    name: str
    lengths: tuple[int, ...]

    @property
    def num_tokens(self) -> int:
        return sum(self.lengths)


def balanced_lengths(num_tokens: int, batch_size: int) -> tuple[int, ...]:
    base, remainder = divmod(num_tokens, batch_size)
    return tuple(base + (sequence < remainder) for sequence in range(batch_size))


INCIDENT_LENGTHS = (9584, *(966 for _ in range(22)), 965, 965)
LATENCY_CASES = (
    Case("single-long", (32768,)),
    Case("uniform-25", balanced_lengths(32766, 25)),
    Case("incident-skew-25", INCIDENT_LENGTHS),
    Case("uniform-512", (64,) * 512),
)
MEMORY_CASES = (
    Case("uniform-8k", balanced_lengths(8192, 25)),
    Case("uniform-16k", balanced_lengths(16384, 25)),
    LATENCY_CASES[1],
    LATENCY_CASES[2],
)


def query_start_loc(lengths: tuple[int, ...], device: torch.device) -> torch.Tensor:
    starts = torch.zeros(len(lengths) + 1, dtype=torch.int64, device=device)
    torch.cumsum(
        torch.tensor(lengths, dtype=torch.int64, device=device),
        dim=0,
        out=starts[1:],
    )
    return starts


def dense_dilated_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    starts: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    dilation: int,
    max_len: int,
) -> torch.Tensor:
    """Previous PLE prefill implementation used as the benchmark baseline."""
    num_tokens, channels = x.shape
    num_sequences = starts.numel() - 1
    state_len = (weight.shape[1] - 1) * dilation
    lengths = starts[1:] - starts[:-1]
    positions = torch.arange(num_tokens, device=x.device, dtype=torch.int64)
    sequence_indices = torch.searchsorted(starts[1:], positions, right=True)
    column_indices = positions - starts[sequence_indices]

    packed_tokens = x.new_zeros((num_sequences, max_len, channels))
    packed_tokens[sequence_indices, column_indices] = x
    packed_tokens = packed_tokens.transpose(1, 2).contiguous()

    state = conv_state.index_select(0, state_indices)[..., :state_len].to(x.dtype)
    initial_state = torch.where(
        has_initial_state.view(num_sequences, 1, 1),
        state,
        torch.zeros_like(state),
    )
    history = torch.cat((initial_state, packed_tokens), dim=-1)
    conv_output = F.conv1d(
        history,
        weight.unsqueeze(1).contiguous(),
        groups=channels,
        dilation=dilation,
    )
    conv_output = F.silu(conv_output).transpose(1, 2).contiguous()

    output = torch.empty_like(x)
    output.copy_(conv_output[sequence_indices, column_indices])
    state_starts = lengths.view(num_sequences, 1, 1)
    state_offsets = torch.arange(state_len, device=x.device, dtype=torch.int64).view(
        1, 1, state_len
    )
    next_state = history.gather(
        dim=2,
        index=(state_starts + state_offsets).expand(-1, channels, -1),
    )
    existing_state = conv_state.index_select(0, state_indices)
    existing_state[..., :state_len] = next_state.to(conv_state.dtype)
    conv_state.index_copy_(0, state_indices, existing_state)
    return output


def make_case_tensors(
    case: Case,
    channels: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    x = torch.randn(case.num_tokens, channels, device=device, dtype=torch.bfloat16)
    weight = torch.randn(channels, 4, device=device, dtype=torch.bfloat16)
    state = torch.randn(
        len(case.lengths) + 1,
        channels,
        9,
        device=device,
        dtype=torch.float32,
    )
    starts = query_start_loc(case.lengths, device)
    state_indices = torch.arange(
        1, len(case.lengths) + 1, device=device, dtype=torch.int64
    )
    has_initial_state = torch.ones(len(case.lengths), device=device, dtype=torch.bool)
    return x, weight, state, starts, state_indices, has_initial_state


def check_correctness(device: torch.device) -> None:
    case = Case("correctness", (2, 5, 19, 3))
    x, weight, state, starts, state_indices, has_initial_state = make_case_tensors(
        case, 37, device
    )
    dense_state = state.clone()
    varlen_state = state.clone()
    expected = dense_dilated_conv1d(
        x,
        weight,
        dense_state,
        starts,
        state_indices,
        has_initial_state,
        dilation=3,
        max_len=max(case.lengths),
    )
    actual = varlen_dilated_conv1d(
        x,
        weight,
        varlen_state,
        starts,
        state_indices,
        has_initial_state,
        dilation=3,
    )
    torch.accelerator.synchronize()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(varlen_state, dense_state, rtol=0, atol=0)


def measure_peak_gib(run: Callable[[], torch.Tensor]) -> float:
    warmup_output = run()
    torch.accelerator.synchronize()
    del warmup_output
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    output = run()
    torch.accelerator.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline
    del output
    return peak / 2**30


def benchmark_us(
    run: Callable[[], torch.Tensor],
    dry_run_time_ms: int,
    repeat_time_ms: int,
) -> float:
    for _ in range(3):
        run()
    torch.accelerator.synchronize()
    samples = bench_gpu_time_with_cupti(
        run,
        dry_run_time_ms=dry_run_time_ms,
        repeat_time_ms=repeat_time_ms,
        use_cuda_graph=False,
        cold_l2_cache=True,
    )
    return statistics.median(samples) * 1e3


def run_memory_cases(channels: int, device: torch.device) -> pd.DataFrame:
    rows = []
    for case in MEMORY_CASES:
        tensors = make_case_tensors(case, channels, device)
        x, weight, state, starts, state_indices, has_initial_state = tensors

        run_varlen = partial(
            varlen_dilated_conv1d,
            x,
            weight,
            state,
            starts,
            state_indices,
            has_initial_state,
            dilation=3,
        )

        varlen_peak = measure_peak_gib(run_varlen)
        dense_peak = None
        run_dense = None
        dense_state = None
        if case.name in {"uniform-25", "incident-skew-25"}:
            dense_state = state.clone()
            run_dense = partial(
                dense_dilated_conv1d,
                x,
                weight,
                dense_state,
                starts,
                state_indices,
                has_initial_state,
                dilation=3,
                max_len=max(case.lengths),
            )
            dense_peak = measure_peak_gib(run_dense)
        rows.append(
            {
                "case": case.name,
                "batch": len(case.lengths),
                "tokens": case.num_tokens,
                "max_len": max(case.lengths),
                "dense_rows/token": (
                    len(case.lengths) * max(case.lengths) / case.num_tokens
                ),
                "varlen_peak_gib": varlen_peak,
                "dense_peak_gib": dense_peak,
            }
        )
        del (
            tensors,
            x,
            weight,
            state,
            starts,
            state_indices,
            has_initial_state,
            run_varlen,
            run_dense,
            dense_state,
        )
        gc.collect()
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def run_latency_cases(
    channels: int,
    device: torch.device,
    dry_run_time_ms: int,
    repeat_time_ms: int,
) -> pd.DataFrame:
    rows = []
    element_size = torch.tensor([], dtype=torch.bfloat16).element_size()
    for case in LATENCY_CASES:
        tensors = make_case_tensors(case, channels, device)
        x, weight, state, starts, state_indices, has_initial_state = tensors
        dense_state = state.clone()

        run_varlen = partial(
            varlen_dilated_conv1d,
            x,
            weight,
            state,
            starts,
            state_indices,
            has_initial_state,
            dilation=3,
        )
        run_dense = partial(
            dense_dilated_conv1d,
            x,
            weight,
            dense_state,
            starts,
            state_indices,
            has_initial_state,
            dilation=3,
            max_len=max(case.lengths),
        )

        varlen_us = benchmark_us(run_varlen, dry_run_time_ms, repeat_time_ms)
        dense_us = benchmark_us(run_dense, dry_run_time_ms, repeat_time_ms)
        ideal_bytes = (
            2 * case.num_tokens * channels + channels * weight.shape[1]
        ) * element_size
        rows.append(
            {
                "case": case.name,
                "batch": len(case.lengths),
                "tokens": case.num_tokens,
                "max_len": max(case.lengths),
                "varlen_us": varlen_us,
                "dense_us": dense_us,
                "speedup": dense_us / varlen_us,
                "varlen_ideal_gbps": ideal_bytes / (varlen_us * 1e3),
                "dense_ideal_gbps": ideal_bytes / (dense_us * 1e3),
            }
        )
        del (
            tensors,
            x,
            weight,
            state,
            starts,
            state_indices,
            has_initial_state,
            dense_state,
            run_varlen,
            run_dense,
        )
        gc.collect()
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("memory", "latency", "all"), default="all")
    parser.add_argument("--channels", type=int, default=10240)
    parser.add_argument("--dry-run-time-ms", type=int, default=25)
    parser.add_argument("--repeat-time-ms", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    check_correctness(device)
    print(
        pd.Series(
            {
                "gpu": torch.cuda.get_device_name(device),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "dtype": "bfloat16",
                "channels": args.channels,
                "cupti_cold_l2": True,
                "cuda_graph": False,
            },
            name="value",
        ).to_string()
    )
    if args.mode in ("memory", "all"):
        memory = run_memory_cases(args.channels, device)
        print("\nPeak incremental allocation")
        print(memory.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    if args.mode in ("latency", "all"):
        latency = run_latency_cases(
            args.channels,
            device,
            args.dry_run_time_ms,
            args.repeat_time_ms,
        )
        print("\nCUPTI latency (eager production operator, cold L2)")
        print(latency.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
