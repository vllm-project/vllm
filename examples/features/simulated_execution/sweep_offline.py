# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sweep a scheduler setting in simulated time.

Runs a Poisson workload through the real scheduler and KV-cache manager with
the forward pass replaced by a timing model, once per `max_num_seqs` value,
and prints the latency of each configuration. Every run takes only as long as
the engine's host work, not as long as the workload.

With a recording of the target deployment (`record_step_timing`, see
docs/features/simulated_execution.md) pass `--replay /path/to/steps.jsonl`;
otherwise a linear timing model stands in for the hardware.

    python examples/features/simulated_execution/sweep_offline.py \
        --model Qwen/Qwen3-0.6B --rate 20 --num-requests 200
"""

import argparse

import numpy as np

from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.simulated.offline import TimedRequest, simulate


def workload(args: argparse.Namespace) -> list[TimedRequest]:
    rng = np.random.default_rng(args.seed)
    arrivals = np.cumsum(rng.exponential(1.0 / args.rate, args.num_requests))
    params = SamplingParams(max_tokens=args.output_len, ignore_eos=True)
    return [
        TimedRequest(
            prompt_token_ids=rng.integers(1000, 30000, args.input_len).tolist(),
            sampling_params=params,
            arrival_time=float(t),
        )
        for t in arrivals
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--num-requests", type=int, default=200)
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--max-num-seqs", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=8 * 1024**3)
    parser.add_argument("--replay", help="step log recorded on the target")
    parser.add_argument("--clock", default="fast_forward")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.replay:
        timing = {"timing": "replay", "timing_args": {"path": args.replay}}
    else:
        timing = {
            "timing": "linear",
            "timing_args": {
                "fixed_us": 1500,
                "prefill_us_per_token": 10,
                "decode_us_per_req": 15,
            },
        }
    requests = workload(args)

    print(
        f"{'max_num_seqs':>12} {'TTFT p50':>9} {'TTFT p99':>9} {'TPOT p50':>9} "
        f"{'tok/s':>8} {'sim s':>7} {'wall s':>7}"
    )
    for max_num_seqs in args.max_num_seqs:
        engine_args = EngineArgs(
            model=args.model,
            max_model_len=args.input_len + args.output_len,
            max_num_seqs=max_num_seqs,
            kv_cache_memory_bytes=args.kv_cache_memory_bytes,
            simulation_config={"simulate_forward": True, "clock": args.clock, **timing},
        )
        s = simulate(engine_args, requests).summary()
        print(
            f"{max_num_seqs:>12} {s['median_ttft_ms']:>9.2f} "
            f"{s['p99_ttft_ms']:>9.2f} {s['median_tpot_ms']:>9.2f} "
            f"{s['output_throughput']:>8.0f} {s['duration_s']:>7.2f} "
            f"{s['wall_time_s']:>7.2f}"
        )


if __name__ == "__main__":
    main()
