# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
from collections.abc import Callable
from typing import Literal

import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.worker.gpu.sample.thinking_budget import ThinkingBudgetState
from vllm.v1.worker.gpu.states import RequestState

START_TOKEN_ID = 90
END_TOKEN_ID = 91
TOKEN_ID = 10
VOCAB_SIZE = 128
QUANTILES = [0.5, 0.2, 0.8]


class ReasoningConfig:
    reasoning_start_token_ids = [START_TOKEN_ID]
    reasoning_end_token_ids = [END_TOKEN_ID]
    natural_reasoning_end_token_ids = [END_TOKEN_ID]


@dataclasses.dataclass
class BenchmarkCase:
    req_states: RequestState
    state: ThinkingBudgetState
    run: Callable[[], None]


def create_case(
    history_len: int,
    device: torch.device,
    extra_tokens: int = 0,
    history_type: Literal["reasoning", "prefill"] = "reasoning",
    batch_size: int = 1,
    budget_type: Literal["active", "forced", "mixed"] = "active",
) -> BenchmarkCase:
    assert history_len >= 2
    if history_type == "prefill":
        tokens = [*([TOKEN_ID] * (history_len - 1)), START_TOKEN_ID]
    else:
        tokens = [TOKEN_ID, START_TOKEN_ID, *([TOKEN_ID] * (history_len - 2))]
    req_states = RequestState(
        max_num_reqs=batch_size,
        max_model_len=history_len + extra_tokens + 1,
        max_num_batched_tokens=batch_size,
        num_speculative_steps=1,
        vocab_size=VOCAB_SIZE,
        device=device,
    )
    req_indices = []
    for req_num in range(batch_size):
        req_id = f"benchmark-{req_num}"
        req_states.add_request(
            req_id=req_id,
            prompt_len=1,
            all_token_ids=tokens,
            num_computed_tokens=history_len,
            max_tokens=extra_tokens + 1,
        )
        req_indices.append(req_states.req_id_to_index[req_id])
    req_states.apply_staged_writes()

    state = ThinkingBudgetState(req_states, ReasoningConfig())
    active_budget = history_len + extra_tokens + 1
    forced_budget = max(1, history_len - 2)
    for req_num, req_idx in enumerate(req_indices):
        if budget_type == "forced":
            budget = forced_budget
        elif budget_type == "mixed" and req_num % 2 == 1:
            budget = None
        else:
            budget = active_budget
        state.add_request(
            req_idx,
            SamplingParams(thinking_token_budget=budget),
        )
    state.apply_staged_writes()

    idx_mapping = torch.tensor(req_indices, dtype=torch.int32, device=device)
    logits = torch.zeros((batch_size, VOCAB_SIZE), device=device)
    idx_mapping_np = idx_mapping.cpu().numpy()
    input_ids = torch.full((batch_size,), TOKEN_ID, dtype=torch.int32, device=device)
    local_pos = torch.zeros(batch_size, dtype=torch.int32, device=device)

    def run() -> None:
        state.apply(
            logits,
            idx_mapping,
            idx_mapping,
            idx_mapping_np,
            input_ids,
            local_pos,
        )

    run()
    torch.accelerator.synchronize(device)
    active_req_indices = [
        req_idx
        for req_num, req_idx in enumerate(req_indices)
        if budget_type != "mixed" or req_num % 2 == 0
    ]
    assert torch.all(state.cached_scan_pos[active_req_indices] == history_len).item()
    if budget_type == "forced":
        assert torch.all(logits[:, END_TOKEN_ID] == 1.0e9).item()
    return BenchmarkCase(req_states, state, run)


def benchmark_cached(
    case: BenchmarkCase, warmup_ms: int, rep_ms: int
) -> tuple[float, float, float]:
    median_ms, min_ms, max_ms = triton.testing.do_bench(
        case.run,
        warmup=warmup_ms,
        rep=rep_ms,
        quantiles=QUANTILES,
    )
    return median_ms, min_ms, max_ms


def summarize(timings_ms: list[float]) -> tuple[float, float, float]:
    median_ms, min_ms, max_ms = np.quantile(timings_ms, QUANTILES)
    return float(median_ms), float(min_ms), float(max_ms)


def time_cuda_call(run: Callable[[], None]) -> float:
    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)
    start.record()
    run()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def benchmark_incremental_decode(
    history_len: int,
    device: torch.device,
    warmup_steps: int,
    iterations: int,
) -> tuple[float, float, float]:
    case = create_case(history_len, device, warmup_steps + iterations)
    timings_ms: list[float] = []
    total_len = history_len
    for step in range(warmup_steps + iterations):
        case.req_states.all_token_ids.stage_write(0, total_len, [TOKEN_ID])
        total_len += 1
        case.req_states.total_len.stage_write_elem(0, total_len)
        case.req_states.apply_staged_writes()
        torch.accelerator.synchronize(device)

        elapsed_ms = time_cuda_call(case.run)
        if step >= warmup_steps:
            timings_ms.append(elapsed_ms)

    assert case.state.cached_scan_pos[0].item() == total_len
    return summarize(timings_ms)


def benchmark_cold_scan(
    history_len: int,
    device: torch.device,
    warmup_steps: int,
    iterations: int,
    history_type: Literal["reasoning", "prefill"],
) -> tuple[float, float, float]:
    case = create_case(history_len, device, history_type=history_type)
    timings_ms: list[float] = []
    for step in range(warmup_steps + iterations):
        case.state.cached_last_start.fill_(-1)
        case.state.cached_last_end.fill_(-1)
        case.state.cached_scan_pos.zero_()
        torch.accelerator.synchronize(device)

        elapsed_ms = time_cuda_call(case.run)
        if step >= warmup_steps:
            timings_ms.append(elapsed_ms)

    return summarize(timings_ms)


def main() -> None:
    parser = FlexibleArgumentParser(
        description=(
            "Benchmark thinking-budget scan, forced-end, and batched request "
            "overhead across token history lengths, including worst-case cold "
            "resume reconstruction."
        )
    )
    parser.add_argument(
        "--history-lengths",
        type=int,
        nargs="+",
        default=[128, 16384, 32768],
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=[
            "cached",
            "incremental-decode",
            "cold-prefill",
            "cold-resume-worst-case",
            "forced-end",
            "batched-budgeted",
            "batched-mixed",
        ],
        default=[
            "cached",
            "incremental-decode",
            "cold-prefill",
            "cold-resume-worst-case",
            "forced-end",
            "batched-budgeted",
            "batched-mixed",
        ],
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8, 32],
        help="Batch sizes used by batched-budgeted and batched-mixed modes.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--warmup-ms",
        type=int,
        default=100,
        help="Warmup duration passed to triton.testing.do_bench.",
    )
    parser.add_argument(
        "--rep-ms",
        type=int,
        default=500,
        help="Measurement duration passed to triton.testing.do_bench.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help="Warmup steps for incremental-decode and cold modes.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Measured steps for incremental-decode and cold modes.",
    )
    parser.add_argument(
        "--max-slowdown",
        type=float,
        default=None,
        help=(
            "Fail if cached or incremental-decode median latency exceeds this "
            "multiple of its shortest-history latency. Cold modes are excluded "
            "because worst-case cold-resume cost scales with scanned history."
        ),
    )
    args = parser.parse_args()

    history_lengths = sorted(set(args.history_lengths))
    if not history_lengths or history_lengths[0] < 2:
        parser.error("history lengths must be at least 2")
    if args.warmup_steps < 0 or args.iterations <= 0:
        parser.error("warmup steps must be non-negative and iterations positive")
    batch_sizes = sorted(set(args.batch_sizes))
    if not batch_sizes or batch_sizes[0] <= 0:
        parser.error("batch sizes must be positive")

    device = torch.device(args.device)
    results: dict[str, dict[int, tuple[float, float, float]]] = {}
    for mode in args.modes:
        mode_batch_sizes = batch_sizes if mode.startswith("batched-") else [1]
        if mode == "batched-mixed":
            mode_batch_sizes = [size for size in mode_batch_sizes if size >= 2]
            if not mode_batch_sizes:
                parser.error("batched-mixed requires a batch size of at least 2")
        for batch_size in mode_batch_sizes:
            label = f"{mode}-b{batch_size}" if mode.startswith("batched-") else mode
            mode_results: dict[int, tuple[float, float, float]] = {}
            for history_len in history_lengths:
                if mode == "cached":
                    case = create_case(history_len, device)
                    result = benchmark_cached(case, args.warmup_ms, args.rep_ms)
                elif mode == "incremental-decode":
                    result = benchmark_incremental_decode(
                        history_len,
                        device,
                        args.warmup_steps,
                        args.iterations,
                    )
                elif mode == "cold-prefill":
                    result = benchmark_cold_scan(
                        history_len,
                        device,
                        args.warmup_steps,
                        args.iterations,
                        "prefill",
                    )
                elif mode == "cold-resume-worst-case":
                    result = benchmark_cold_scan(
                        history_len,
                        device,
                        args.warmup_steps,
                        args.iterations,
                        "reasoning",
                    )
                else:
                    budget_type = (
                        "forced"
                        if mode == "forced-end"
                        else "mixed"
                        if mode == "batched-mixed"
                        else "active"
                    )
                    case = create_case(
                        history_len,
                        device,
                        batch_size=batch_size,
                        budget_type=budget_type,
                    )
                    result = benchmark_cached(case, args.warmup_ms, args.rep_ms)
                mode_results[history_len] = result
            results[label] = mode_results

    print("mode                       history_len  median_us  p20_us  p80_us  slowdown")
    for label, mode_results in results.items():
        baseline_ms = mode_results[history_lengths[0]][0]
        for history_len in history_lengths:
            median_ms, min_ms, max_ms = mode_results[history_len]
            slowdown = median_ms / baseline_ms
            print(
                f"{label:<26} {history_len:>11}  {median_ms * 1000:>9.3f}  "
                f"{min_ms * 1000:>6.3f}  {max_ms * 1000:>6.3f}  "
                f"{slowdown:>7.3f}x"
            )

    if args.max_slowdown is not None:
        for mode in ("cached", "incremental-decode"):
            if mode not in results:
                continue
            mode_results = results[mode]
            baseline_ms = mode_results[history_lengths[0]][0]
            worst_history_len = max(
                mode_results, key=lambda length: mode_results[length][0]
            )
            worst_slowdown = mode_results[worst_history_len][0] / baseline_ms
            if worst_slowdown > args.max_slowdown:
                raise SystemExit(
                    f"thinking-budget {mode} slowdown {worst_slowdown:.3f}x "
                    f"at history length {worst_history_len} exceeds "
                    f"{args.max_slowdown:.3f}x"
                )


if __name__ == "__main__":
    main()
