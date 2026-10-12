# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
