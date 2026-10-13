# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math

import pytest

from vllm.benchmarks.sweep.param_sweep import ParameterSweepItem
from vllm.benchmarks.sweep.simulate import (
    max_rate_meeting_slo,
    simulated_serve_cmd,
    split_roles,
)
from vllm.entrypoints.launchers.cli_args import make_arg_parser
from vllm.utils.argparse_utils import FlexibleArgumentParser


def _queue(capacity: float):
    """Runs of a queue whose TTFT grows with the rate and diverges at
    `capacity`, the throughput when all requests arrive at once."""
    rates: list[float] = []

    def evaluate(rate: float) -> dict[str, object]:
        rates.append(rate)
        ttft = 10 / (1 - rate / capacity) if rate < capacity else 1e6
        return {"p99_ttft_ms": ttft, "request_throughput": min(rate, capacity)}

    return evaluate, rates


def test_max_rate_meeting_slo():
    evaluate, rates = _queue(capacity=20)
    # TTFT <= 40 ms up to 3/4 of the capacity.
    rate = max_rate_meeting_slo(evaluate, {"p99_ttft_ms": 40}, search_iters=6)
    assert 15 * 0.98 <= rate <= 15
    assert rates[0] == math.inf
    assert len(rates) <= 1 + 2 + 6


@pytest.mark.parametrize(("bound", "expected"), [(1e9, math.inf), (1, 0.0)])
def test_max_rate_meeting_slo_unbounded_or_infeasible(bound, expected):
    evaluate, _ = _queue(capacity=20)
    assert max_rate_meeting_slo(evaluate, {"p99_ttft_ms": bound}, 4) == expected


def test_max_rate_meeting_slo_unknown_metric():
    evaluate, _ = _queue(capacity=20)
    with pytest.raises(ValueError, match="not reported"):
        max_rate_meeting_slo(evaluate, {"p99_tpot_ms": 10}, 4)


def test_simulated_serve_cmd():
    cmd = simulated_serve_cmd(
        ["vllm", "serve", "Qwen/Qwen3-0.6B", "--max-num-seqs", "16"],
        {"timing": "replay", "timing_args": {"path": "a.jsonl", "k": 2}},
        ParameterSweepItem(
            {"max_num_seqs": 8, "simulation_config.timing_args.path": "b.jsonl"}
        ),
    )
    args = make_arg_parser(FlexibleArgumentParser()).parse_args(cmd[2:])
    assert args.max_num_seqs == 8
    sim = args.simulation_config
    assert sim.simulate_forward and sim.clock == "virtual"
    assert sim.timing == "replay"
    assert sim.timing_args == {"path": "b.jsonl", "k": 2}


def test_split_roles():
    common, roles, counts = split_roles(
        ParameterSweepItem({"_benchmark_name": "a", "max_model_len": 2048})
    )
    assert common == {"_benchmark_name": "a", "max_model_len": 2048}
    assert counts == {}

    common, roles, counts = split_roles(
        ParameterSweepItem(
            {"_num_prefill": 2, "max_model_len": 2048, "_decode.max_num_seqs": 64}
        )
    )
    assert common == {"max_model_len": 2048}
    assert roles == {"prefill": {}, "decode": {"max_num_seqs": 64}}
    assert counts == {"prefill": 2, "decode": 1}
