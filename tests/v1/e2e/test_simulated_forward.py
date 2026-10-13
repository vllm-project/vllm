# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from tests.conftest import VllmRunner
from tests.models.registry import HF_EXAMPLE_MODELS
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.triton_utils import HAS_TRITON
from vllm.v1.metrics.reader import Counter

pytestmark = [
    pytest.mark.cpu_model,
    pytest.mark.skipif(not HAS_TRITON, reason="Simulated forward requires Triton."),
]

SIMULATED_FORWARD_MODELS = [
    pytest.param("Qwen/Qwen3-235B-A22B", 1, id="qwen3-235b-a22b"),
    pytest.param("allenai/Olmo-3-7B-Think", 2, id="olmo3-hybrid-attention"),
    pytest.param("openai/gpt-oss-120b", 2, id="gpt-oss-120b"),
]


def _check_model_available(model: str) -> None:
    try:
        model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    except ValueError:
        return
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")


def _get_kv_cache_group_count(vllm_model: VllmRunner) -> int | None:
    engine_core_client = vllm_model.llm.llm_engine.engine_core
    engine_core = getattr(engine_core_client, "engine_core", None)
    if engine_core is None:
        return None

    kv_cache_config = getattr(engine_core.scheduler, "kv_cache_config", None)
    if kv_cache_config is None:
        return None
    return len(kv_cache_config.kv_cache_groups)


def _get_counter_value(vllm_model: VllmRunner, metric_name: str) -> int:
    vllm_model.llm.llm_engine.do_log_stats()
    value = 0
    found = False
    for metric in vllm_model.llm.get_metrics():
        if isinstance(metric, Counter) and metric.name == metric_name:
            found = True
            value += metric.value
    assert found, f"Metric not found: {metric_name}"
    return value


@pytest.mark.parametrize(("model", "min_kv_cache_groups"), SIMULATED_FORWARD_MODELS)
def test_simulate_forward_model_matrix(
    vllm_runner: type[VllmRunner],
    model: str,
    min_kv_cache_groups: int,
) -> None:
    if not (current_platform.is_cpu() or current_platform.is_cuda_alike()):
        pytest.skip("Simulated forward needs the CPU or a CUDA-alike platform.")

    _check_model_available(model)

    # No max_tokens: generation must stop after the trace instead of padding
    # to max_model_len.
    sampling_params = SamplingParams(
        temperature=0.0,
        detokenize=False,
        trace_decode_token_ids=[501, 502],
    )
    prompt = [1000, 1001, 1002, 1003]

    with vllm_runner(
        model,
        simulate_forward=True,
        disable_hybrid_kv_cache_manager=False,
        kv_cache_memory_bytes=16 * 1024**3,
        max_model_len=1024,
    ) as vllm_model:
        outputs = vllm_model.generate([prompt], sampling_params=sampling_params)
        assert outputs[0][0][0] == prompt + [501, 502]

        group_count = _get_kv_cache_group_count(vllm_model)
        if group_count is not None:
            assert group_count >= min_kv_cache_groups


def test_simulate_forward_prefix_cache_hybrid_retention_zero(
    monkeypatch: pytest.MonkeyPatch,
    vllm_runner: type[VllmRunner],
) -> None:
    if not (current_platform.is_cpu() or current_platform.is_cuda_alike()):
        pytest.skip("Simulated forward needs the CPU or a CUDA-alike platform.")

    model = "openai/gpt-oss-120b"
    _check_model_available(model)

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_PREFIX_CACHE_RETENTION_INTERVAL", "0")

    sampling_params = SamplingParams(
        temperature=0.0,
        detokenize=False,
        trace_decode_token_ids=[501, 502],
    )
    prompt = [1000 + (i % 127) for i in range(512)]

    with vllm_runner(
        model,
        simulate_forward=True,
        disable_hybrid_kv_cache_manager=False,
        enable_prefix_caching=True,
        disable_log_stats=False,
        kv_cache_memory_bytes=16 * 1024**3,
        max_model_len=1024,
    ) as vllm_model:
        group_count = _get_kv_cache_group_count(vllm_model)
        if group_count is not None:
            assert group_count >= 2

        outputs = vllm_model.generate([prompt], sampling_params=sampling_params)
        assert outputs[0][0][0][-2:] == [501, 502]
        hits_after_first = _get_counter_value(vllm_model, "vllm:prefix_cache_hits")

        second_sampling_params = SamplingParams(
            temperature=0.0,
            detokenize=False,
            trace_decode_token_ids=[503, 504],
        )
        outputs = vllm_model.generate([prompt], sampling_params=second_sampling_params)
        assert outputs[0][0][0][-2:] == [503, 504]
        hits_after_second = _get_counter_value(vllm_model, "vllm:prefix_cache_hits")

        assert hits_after_second > hits_after_first


def test_simulate_offline_virtual_clock() -> None:
    """Under the virtual clock, request timestamps are exactly the constant
    timing model's step times, idle gaps are skipped, and arrivals are
    honored."""
    if not (current_platform.is_cpu() or current_platform.is_cuda_alike()):
        pytest.skip("Simulated forward needs the CPU or a CUDA-alike platform.")
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.worker.simulated.offline import TimedRequest, simulate

    step_s = 1e-3
    args = EngineArgs(
        model="Qwen/Qwen3-0.6B",
        max_model_len=1024,
        kv_cache_memory_bytes=1024**3,
        simulation_config={
            "simulate_forward": True,
            "clock": "virtual",
            "timing": "constant",
            "timing_args": {"fixed_us": step_s * 1e6},
        },
    )
    params = SamplingParams(max_tokens=8, ignore_eos=True)
    requests = [TimedRequest(list(range(16)), params, 0.0) for _ in range(4)]
    requests.append(TimedRequest(list(range(16)), params, 10.0))

    result = simulate(args, requests)

    for timing in result.requests:
        # One prefill step yields the first token, then one token per step.
        expected = [timing.arrival_time + k * step_s for k in range(1, 9)]
        assert timing.num_output_tokens == 8
        assert timing.token_times == pytest.approx(expected)
        assert timing.finish_time == pytest.approx(expected[-1])
    assert result.duration == pytest.approx(10.0 + 8 * step_s)
    assert result.summary()["mean_tpot_ms"] == pytest.approx(step_s * 1e3)

    # A frontend latency delays each request's admission, so it adds to TTFT
    # and end-to-end latency but not to TPOT.
    latency_s = 5e-3
    shifted = simulate(args, requests, frontend_latency_s=latency_s)
    for before, after in zip(result.requests, shifted.requests):
        assert after.arrival_time == before.arrival_time
        assert after.ttft == pytest.approx(before.ttft + latency_s)
        assert after.e2el == pytest.approx(before.e2el + latency_s)
        assert after.tpot == pytest.approx(before.tpot)


def test_sweep_simulate_slo(tmp_path) -> None:
    """`vllm bench sweep simulate` finds a higher SLO-meeting request rate
    for a larger batch, and writes results the sweep plots read."""
    if not (current_platform.is_cpu() or current_platform.is_cuda_alike()):
        pytest.skip("Simulated forward needs the CPU or a CUDA-alike platform.")
    from vllm.benchmarks.sweep.simulate import SweepSimulateArgs, run_main
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    serve_params = tmp_path / "serve_params.json"
    serve_params.write_text(json.dumps([{"max_num_seqs": 2}, {"max_num_seqs": 16}]))
    parser = SweepSimulateArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(
        [
            "--serve-cmd",
            "vllm serve Qwen/Qwen3-0.6B --max-model-len 1024 "
            "--kv-cache-memory-bytes 1073741824",
            "--bench-cmd",
            "vllm bench serve --dataset-name random --random-input-len 128 "
            "--random-output-len 16 --num-prompts 40",
            "--serve-params",
            str(serve_params),
            "--simulation-config",
            json.dumps(
                {
                    "timing": "constant",
                    "timing_args": {"fixed_us": 2000, "prefill_us_per_token": 10},
                }
            ),
            "--slo",
            "p99_ttft_ms=50",
            "--search-iters",
            "3",
            "-j",
            "2",
            "-o",
            str(tmp_path),
            "-e",
            "exp",
        ]
    )
    run_main(SweepSimulateArgs.from_cli_args(args))

    exp = tmp_path / "exp"
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


def test_sweep_simulate_disaggregated(tmp_path) -> None:
    """A disaggregated combination runs its prefill and decode instances, and a
    second decode instance raises the SLO-meeting rate of a decode-bound
    workload, also with filled KV caches and a read posting cost; runs record
    their GPU count."""
    if not (current_platform.is_cpu() or current_platform.is_cuda_alike()):
        pytest.skip("Simulated forward needs the CPU or a CUDA-alike platform.")
    from vllm.benchmarks.sweep.simulate import SweepSimulateArgs, run_main
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    serve_params = tmp_path / "serve_params.json"
    serve_params.write_text(
        json.dumps(
            [
                {"_benchmark_name": "1p1d", "_num_prefill": 1, "_num_decode": 1},
                {"_benchmark_name": "1p2d", "_num_prefill": 1, "_num_decode": 2},
            ]
        )
    )
    parser = SweepSimulateArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(
        [
            "--serve-cmd",
            "vllm serve Qwen/Qwen3-0.6B --max-model-len 1024 "
            "--kv-cache-memory-bytes 1073741824",
            "--bench-cmd",
            "vllm bench serve --dataset-name random --random-input-len 128 "
            "--random-output-len 32 --num-prompts 40",
            "--serve-params",
            str(serve_params),
            "--simulation-config",
            json.dumps(
                {
                    "timing": "constant",
                    "timing_args": {"fixed_us": 1000, "decode_us_per_req": 300},
                    "kv_transfer_timing": {
                        "load_gbps": 10,
                        "post_us_per_read": 100,
                        "post_us_per_run": 50,
                    },
                }
            ),
            "--slo",
            "mean_tpot_ms=4",
            "--fill-kv-caches",
            "--search-iters",
            "3",
            "--handoff-latency-ms",
            "1",
            "-o",
            str(tmp_path),
            "-e",
            "exp",
        ]
    )
    run_main(SweepSimulateArgs.from_cli_args(args))

    results = {
        name: json.loads(
            (tmp_path / "exp" / f"SERVE--{name}" / "result.json").read_text()
        )
        for name in ("1p1d", "1p2d")
    }
    assert [r["gpu_count"] for r in results.values()] == [2, 3]
    assert all(r["kv_caches_filled"] for r in results.values())
    assert all(
        r["mean_tpot_ms"] <= 4 and r["completed"] == 40 for r in results.values()
    )
    assert 0 < results["1p1d"]["max_request_rate"] < results["1p2d"]["max_request_rate"]


def test_simulate_offline_host_time_holds_engine_thread() -> None:
    """A step whose host time exceeds its device time paces the run by the
    host time: the next step cannot start before the engine thread is free."""
    if not (current_platform.is_cpu() or current_platform.is_cuda_alike()):
        pytest.skip("Simulated forward needs the CPU or a CUDA-alike platform.")
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.worker.simulated.offline import TimedRequest, simulate

    args = EngineArgs(
        model="Qwen/Qwen3-0.6B",
        max_model_len=1024,
        kv_cache_memory_bytes=1024**3,
        simulation_config={
            "simulate_forward": True,
            "clock": "virtual",
            "timing": "constant",
            "timing_args": {"fixed_us": 1000, "host_us": 3000},
        },
    )
    params = SamplingParams(max_tokens=8, ignore_eos=True)
    result = simulate(args, [TimedRequest(list(range(16)), params)])
    # Under async scheduling step k+1's host work runs before step k's output
    # is consumed. Nothing follows the last step, so its device time ends with
    # that host work and its token arrives together with the previous one.
    assert result.requests[0].itls == pytest.approx([3e-3] * 6 + [0.0])


def test_simulate_offline_sessions() -> None:
    """A session's next turn arrives a think time after the previous one
    finished, extends its prompt and output, and reuses their KV cache."""
    if not (current_platform.is_cpu() or current_platform.is_cuda_alike()):
        pytest.skip("Simulated forward needs the CPU or a CUDA-alike platform.")
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.worker.simulated.offline import TimedSession, Turn, simulate

    args = EngineArgs(
        model="Qwen/Qwen3-0.6B",
        max_model_len=1024,
        kv_cache_memory_bytes=1024**3,
        simulation_config={
            "simulate_forward": True,
            "clock": "virtual",
            "timing": "constant",
            "timing_args": {"fixed_us": 1000, "prefill_us_per_token": 10},
        },
    )
    params = SamplingParams(max_tokens=12, ignore_eos=True)
    session = TimedSession(
        [Turn(list(range(56)), params), Turn(list(range(100, 116)), params, 0.5)],
        arrival_time=1.0,
    )
    first, second = simulate(args, [session]).requests

    assert first.ttft == pytest.approx(1000e-6 + 56 * 10e-6)
    assert second.arrival_time == pytest.approx(first.finish_time + 0.5)
    assert second.num_prompt_tokens == 56 + 12 + 16
    # The KV of the prompt and the first 8 output tokens (64 tokens, whole
    # blocks) is cached only if the output follows the prompt; 20 tokens are
    # prefilled.
    assert second.ttft == pytest.approx(1000e-6 + 20 * 10e-6)


def test_simulate_offline_kv_offloading(tmp_path) -> None:
    """A prompt evicted from the GPU KV cache is loaded back from the CPU
    cache: its request waits exactly for the modeled load, then prefills only
    its last token."""
    if not (current_platform.is_cpu() or current_platform.is_cuda_alike()):
        pytest.skip("Simulated forward needs the CPU or a CUDA-alike platform.")
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.worker.simulated.offline import TimedRequest, simulate

    step_s, load_gbps = 1e-3, 100
    log = tmp_path / "steps.jsonl"
    args = EngineArgs(
        model="Qwen/Qwen3-0.6B",
        max_model_len=1024,
        # 1056 tokens of Qwen3-0.6B KV cache.
        kv_cache_memory_bytes=1056 * 114688,
        kv_transfer_config={
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"cpu_bytes_to_use": 1 << 30},
        },
        simulation_config={
            "simulate_forward": True,
            "clock": "virtual",
            "timing": "constant",
            "timing_args": {"fixed_us": step_s * 1e6},
            "kv_transfer_timing": {"load_gbps": load_gbps, "store_gbps": load_gbps},
            "log_path": str(log),
        },
    )
    params = SamplingParams(max_tokens=4, ignore_eos=True)
    a, b = list(range(1000, 1640)), list(range(5000, 5960))
    # b evicts most of a from the GPU cache, but not from the CPU cache.
    result = simulate(
        args,
        [TimedRequest(a, params, 0.0), TimedRequest(b, params, 1.0)]
        + [TimedRequest(a, params, 2.0)],
    )

    records = [json.loads(line) for line in log.read_text().splitlines()]
    [load] = [
        r["kv_transfers"]["load"]
        for r in records
        if "load" in r.get("kv_transfers", {})
    ]
    # The third request's first step runs all of its prefill. Clock times are
    # absolute, and the first step starts when the first request arrives.
    steps = [r for r in records if "step" in r]
    step = next(r for r in steps if r["t_start"] >= steps[0]["t_start"] + 2.0)
    assert [
        (r["num_scheduled_tokens"], r["num_computed_tokens"])
        for r in step["step"]["requests"]
    ] == [(1, 639)]
    assert result.requests[2].ttft == pytest.approx(
        load["bytes"] / (load_gbps * 1e9) + step_s
    )


def test_simulate_disaggregated_prefill() -> None:
    """A request is prefilled on the prefill instance, its KV read by the
    decode instance in the modeled time, and decoded there; the prefill
    instance frees its blocks once they are read."""
    if not (current_platform.is_cpu() or current_platform.is_cuda_alike()):
        pytest.skip("Simulated forward needs the CPU or a CUDA-alike platform.")
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.worker.simulated.disaggregated import simulate_disaggregated
    from vllm.v1.worker.simulated.offline import TimedRequest

    load_gbps = 10
    args = EngineArgs(
        model="Qwen/Qwen3-0.6B",
        max_model_len=2048,
        kv_cache_memory_bytes=1024**3,
        simulation_config={
            "simulate_forward": True,
            "timing": "constant",
            "timing_args": {"fixed_us": 1000, "prefill_us_per_token": 10},
            "kv_transfer_timing": {"load_gbps": load_gbps},
        },
    )
    params = SamplingParams(max_tokens=4, ignore_eos=True)
    requests = [
        TimedRequest(list(range(1000, 1640)), params, 0.0),
        TimedRequest(list(range(3000, 3320)), params, 0.5),
    ]
    result = simulate_disaggregated(args, args, requests)

    for request in result.requests:
        prompt = request.num_prompt_tokens
        prefill_s = (1000 + 10 * prompt) * 1e-6
        # Qwen3-0.6B's KV is 114688 bytes per token.
        read_s = prompt * 114688 / (load_gbps * 1e9)
        # The decode instance recomputes the last prompt token.
        assert request.ttft == pytest.approx(prefill_s + read_s + 1010e-6)
        assert request.num_output_tokens == 4


class AcceptAllDrafts:
    """Timing provider that accepts every draft, overriding the rates."""

    def __init__(self, vllm_config, **kwargs):
        pass

    def estimate(self, step):
        from vllm.v1.worker.simulated.timing import StepCost

        return StepCost(
            device_s=1e-3,
            num_accepted=tuple(r.num_spec_tokens for r in step.requests),
        )


@pytest.mark.parametrize(
    ("rates", "timing", "chunks"),
    [
        ([1.0], "constant", 5),
        ([0.0], "constant", 16),
        ([0.0], f"{__name__}.AcceptAllDrafts", 5),
    ],
)
def test_simulate_offline_speculative_decoding(
    rates: list[float], timing: str, chunks: int
) -> None:
    """Accepted drafts follow `spec_acceptance_rates` unless the timing
    provider sets them: with every draft accepted a step yields 1 + 3 tokens,
    with none it yields one."""
    if not (current_platform.is_cpu() or current_platform.is_cuda_alike()):
        pytest.skip("Simulated forward needs the CPU or a CUDA-alike platform.")
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.worker.simulated.offline import TimedRequest, simulate

    args = EngineArgs(
        model="Qwen/Qwen3-0.6B",
        max_model_len=1024,
        kv_cache_memory_bytes=1024**3,
        speculative_config={
            "method": "ngram",
            "num_speculative_tokens": 3,
            "prompt_lookup_max": 3,
            "prompt_lookup_min": 2,
        },
        simulation_config={
            "simulate_forward": True,
            "clock": "virtual",
            "timing": timing,
            "timing_args": {"fixed_us": 1000} if timing == "constant" else {},
            "spec_acceptance_rates": rates,
        },
    )
    params = SamplingParams(max_tokens=16, ignore_eos=True)
    result = simulate(args, [TimedRequest(list(range(16)), params) for _ in range(4)])
    for request in result.requests:
        assert request.num_output_tokens == 16
        assert len(request.token_times) == chunks


@pytest.mark.parametrize(("tp", "pp"), [(2, 1), (1, 2)])
def test_simulate_forward_multi_rank(
    vllm_runner: type[VllmRunner], tp: int, pp: int
) -> None:
    """Every rank runs the bookkeeping; the output rank's trace tokens come
    back as on a single rank."""
    # CPU ranks need no devices; CUDA ranks need one each.
    if not current_platform.is_cpu() and (
        not current_platform.is_cuda_alike()
        or current_platform.device_count() < tp * pp
    ):
        pytest.skip(f"Needs the CPU platform or {tp * pp} CUDA devices.")
    prompt = list(range(1000, 1064))
    sampling_params = SamplingParams(trace_decode_token_ids=[501, 502, 503])
    with vllm_runner(
        "Qwen/Qwen3-0.6B",
        simulate_forward=True,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        kv_cache_memory_bytes=1024**3,
        max_model_len=1024,
    ) as vllm_model:
        outputs = vllm_model.generate([prompt] * 4, sampling_params=sampling_params)
    for output in outputs:
        assert output[0][0][-3:] == [501, 502, 503]
