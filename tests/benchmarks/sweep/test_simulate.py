# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import math

import pytest

from vllm.benchmarks.sweep.param_sweep import ParameterSweepItem
from vllm.benchmarks.sweep.simulate import (
    SweepSimulateArgs,
    max_rate_meeting_slo,
    run_main,
    simulated_serve_cmd,
)
from vllm.entrypoints.launchers.cli_args import make_arg_parser
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.utils.argparse_utils import FlexibleArgumentParser

needs_simulation = pytest.mark.skipif(
    not HAS_TRITON
    or not (current_platform.is_cpu() or current_platform.is_cuda_alike()),
    reason="Simulated forward needs Triton, and the CPU or a CUDA-alike platform.",
)


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


def _sweep(tmp_path, serve_params, output_len, simulation_config, *options):
    """Run `vllm bench sweep simulate` of Qwen3-0.6B on 40 random requests
    into experiment `tmp_path/exp`."""
    serve_params_path = tmp_path / "serve_params.json"
    serve_params_path.write_text(json.dumps(serve_params))
    parser = SweepSimulateArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(
        [
            "--serve-cmd",
            "vllm serve Qwen/Qwen3-0.6B --max-model-len 1024 "
            "--kv-cache-memory-bytes 1073741824",
            "--bench-cmd",
            "vllm bench serve --dataset-name random --random-input-len 128 "
            f"--random-output-len {output_len} --num-prompts 40",
            "--serve-params",
            str(serve_params_path),
            "--simulation-config",
            json.dumps(simulation_config),
            "--search-iters",
            "3",
            "-o",
            str(tmp_path),
            "-e",
            "exp",
            *options,
        ]
    )
    run_main(SweepSimulateArgs.from_cli_args(args))
    return tmp_path / "exp"


@needs_simulation
def test_sweep_simulate_slo(tmp_path) -> None:
    """A larger batch meets the TTFT SLO at a higher request rate, and the
    results are written where the sweep plots read them."""
    exp = _sweep(
        tmp_path,
        [{"max_num_seqs": 2}, {"max_num_seqs": 16}],
        16,
        {
            "timing": "linear",
            "timing_args": {"fixed_us": 2000, "prefill_us_per_token": 10},
        },
        "--slo",
        "p99_ttft_ms=50",
        "-j",
        "2",
    )
    max_rates = {}
    for max_num_seqs in (2, 16):
        comb_dir = exp / f"SERVE--max_num_seqs={max_num_seqs}"
        result = json.loads((comb_dir / "result.json").read_text())
        assert result["p99_ttft_ms"] <= 50
        max_rates[max_num_seqs] = result["max_request_rate"]
        runs = [json.loads(p.read_text()) for p in comb_dir.rglob("summary.json")]
        assert len(runs) > 3
        assert all(run["completed"] == 40 for (run,) in runs)
    assert 0 < max_rates[2] < max_rates[16]
    assert (exp / "summary.csv").exists() and (exp / "slo_summary.csv").exists()


@needs_simulation
def test_sweep_simulate_disaggregated(tmp_path) -> None:
    """A second decode instance raises the SLO-meeting rate of a decode-bound
    workload, with filled KV caches and a read posting cost."""
    exp = _sweep(
        tmp_path,
        [
            {"_benchmark_name": "1p1d", "_num_prefill": 1, "_num_decode": 1},
            {"_benchmark_name": "1p2d", "_num_prefill": 1, "_num_decode": 2},
        ],
        32,
        {
            "timing": "linear",
            "timing_args": {"fixed_us": 1000, "decode_us_per_req": 300},
            "kv_transfer_timing": {
                "load_gbps": 10,
                "post_us_per_read": 100,
                "post_us_per_run": 50,
            },
        },
        "--slo",
        "mean_tpot_ms=4",
        "--fill-kv-caches",
        "--handoff-latency-ms",
        "1",
    )
    results = {
        name: json.loads((exp / f"SERVE--{name}" / "result.json").read_text())
        for name in ("1p1d", "1p2d")
    }
    assert [r["gpu_count"] for r in results.values()] == [2, 3]
    assert all(r["kv_caches_filled"] for r in results.values())
    assert all(
        r["mean_tpot_ms"] <= 4 and r["completed"] == 40 for r in results.values()
    )
    assert 0 < results["1p1d"]["max_request_rate"] < results["1p2d"]["max_request_rate"]
