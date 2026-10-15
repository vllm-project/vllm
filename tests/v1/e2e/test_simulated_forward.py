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
    pytest.mark.skipif(
        not (current_platform.is_cpu() or current_platform.is_cuda_alike()),
        reason="Simulated forward needs the CPU or a CUDA-alike platform.",
    ),
]

SIMULATED_FORWARD_MODELS = [
    pytest.param("Qwen/Qwen3-235B-A22B", 1, id="qwen3-235b-a22b"),
    pytest.param("allenai/Olmo-3-7B-Think", 2, id="olmo3-hybrid-attention"),
    pytest.param("openai/gpt-oss-120b", 2, id="gpt-oss-120b"),
]


@pytest.mark.parametrize(
    "kwargs, error",
    [
        (
            {
                "kv_transfer_config": {
                    "kv_connector": "ExampleConnector",
                    "kv_role": "kv_both",
                }
            },
            "ExampleConnector connector",
        ),
        (
            {"simulation_config": {"clock": "virtual"}, "tensor_parallel_size": 2},
            "supports one rank",
        ),
        ({"enable_lora": True}, "LoRA"),
    ],
)
def test_simulated_config_rejections(kwargs, error) -> None:
    from vllm.engine.arg_utils import EngineArgs

    with pytest.raises(ValueError, match=error):
        EngineArgs(
            model="Qwen/Qwen3-0.6B", max_model_len=1024, simulate_forward=True, **kwargs
        ).create_engine_config()


def test_simulate_forward_quantized_checkpoint(
    tmp_path, monkeypatch: pytest.MonkeyPatch, vllm_runner: type[VllmRunner]
) -> None:
    """A quantized checkpoint is simulated as deployed: its layers take the
    checkpoint's quantization, and its KV cache the dtype it declares."""
    from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

    # apply_model() ships a function to the workers.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    config = get_hf_file_to_dict("config.json", "Qwen/Qwen3-0.6B")
    config["quantization_config"] = {
        "quant_method": "modelopt",
        "quant_algo": "FP8",
        "kv_cache_quant_algo": "FP8",
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    with vllm_runner(
        str(tmp_path),
        tokenizer_name="Qwen/Qwen3-0.6B",
        simulate_forward=True,
        kv_cache_memory_bytes=1024**3,
        max_model_len=1024,
    ) as vllm_model:
        vllm_config = vllm_model.llm.llm_engine.vllm_config
        assert vllm_config.cache_config.cache_dtype == "fp8_e4m3"
        weight_dtypes = vllm_model.apply_model(
            lambda model: str(model.model.layers[0].self_attn.qkv_proj.weight.dtype)
        )
        outputs = vllm_model.generate_greedy([[1000, 1001, 1002]], max_tokens=4)
    assert weight_dtypes == ["torch.float8_e4m3fn"]
    assert len(outputs[0][0]) == 3 + 4


@pytest.mark.skipif(
    current_platform.is_cpu(), reason="The CPU build has no FP8 quantization kernels."
)
def test_simulate_forward_online_quantization(
    monkeypatch: pytest.MonkeyPatch, vllm_runner: type[VllmRunner]
) -> None:
    """Online quantization quantizes the meta weights as they load."""
    # apply_model() ships a function to the workers.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    with vllm_runner(
        "Qwen/Qwen3-0.6B",
        quantization="fp8",
        simulate_forward=True,
        kv_cache_memory_bytes=1024**3,
        max_model_len=1024,
    ) as vllm_model:
        weight_dtypes = vllm_model.apply_model(
            lambda model: str(model.model.layers[0].self_attn.qkv_proj.weight.dtype)
        )
        outputs = vllm_model.generate_greedy([[1000, 1001, 1002]], max_tokens=4)
    assert weight_dtypes == ["torch.float8_e4m3fn"]
    assert len(outputs[0][0]) == 3 + 4


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
            "timing": "linear",
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


def test_simulate_offline_host_time_holds_engine_thread() -> None:
    """A step whose host time exceeds its device time paces the run by the
    host time: the next step cannot start before the engine thread is free."""
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.worker.simulated.offline import TimedRequest, simulate

    args = EngineArgs(
        model="Qwen/Qwen3-0.6B",
        max_model_len=1024,
        kv_cache_memory_bytes=1024**3,
        simulation_config={
            "simulate_forward": True,
            "clock": "virtual",
            "timing": "linear",
            "timing_args": {"fixed_us": 1000, "host_us": 3000},
        },
    )
    params = SamplingParams(max_tokens=8, ignore_eos=True)
    result = simulate(args, [TimedRequest(list(range(16)), params)])
    # The 3 ms host time, not the 1 ms device time, paces the steps.
    assert result.requests[0].itls[:6] == pytest.approx([3e-3] * 6)


def test_simulate_offline_sessions() -> None:
    """A session's next turn arrives a think time after the previous one
    finished, extends its prompt and output, and reuses their KV cache."""
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.worker.simulated.offline import TimedSession, Turn, simulate

    args = EngineArgs(
        model="Qwen/Qwen3-0.6B",
        max_model_len=1024,
        kv_cache_memory_bytes=1024**3,
        simulation_config={
            "simulate_forward": True,
            "clock": "virtual",
            "timing": "linear",
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
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.worker.simulated.offline import TimedRequest, simulate

    step_s, load_gbps = 1e-3, 100
    log = tmp_path / "steps.jsonl"
    args = EngineArgs(
        model="Qwen/Qwen3-0.6B",
        max_model_len=1024,
        # 1056 tokens of Qwen3-0.6B KV cache (28 layers x 8 KV heads x 128
        # dims x K and V x 2 bytes = 114688 bytes per token).
        kv_cache_memory_bytes=1056 * 114688,
        kv_transfer_config={
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"cpu_bytes_to_use": 1 << 30},
        },
        simulation_config={
            "simulate_forward": True,
            "clock": "virtual",
            "timing": "linear",
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
    decode instance in the modeled time, and decoded there."""
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
            "timing": "linear",
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
        ([1.0], "linear", 5),
        ([0.0], "linear", 16),
        ([0.0], f"{__name__}.AcceptAllDrafts", 5),
    ],
)
def test_simulate_offline_speculative_decoding(
    rates: list[float], timing: str, chunks: int
) -> None:
    """Accepted drafts follow `spec_acceptance_rates` unless the timing
    provider sets them: with every draft accepted a step yields 1 + 3 tokens,
    with none it yields one."""
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
            "timing_args": {"fixed_us": 1000} if timing == "linear" else {},
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


def test_simulate_forward_prefill_context_parallel(
    vllm_runner: type[VllmRunner],
) -> None:
    """PCP ranks run their shard of each prefill, and the request still
    samples once its whole prefill is done."""
    if not current_platform.is_cuda_alike() or current_platform.device_count() < 2:
        pytest.skip("Needs 2 CUDA devices.")
    sampling_params = SamplingParams(trace_decode_token_ids=[501, 502, 503])
    # PCP needs an MLA attention backend.
    with vllm_runner(
        "deepseek-ai/DeepSeek-V2-Lite",
        simulate_forward=True,
        prefill_context_parallel_size=2,
        hf_overrides={"num_hidden_layers": 1},
        trust_remote_code=True,
        kv_cache_memory_bytes=1024**3,
        max_model_len=1024,
    ) as vllm_model:
        outputs = vllm_model.generate(
            [list(range(1000, 1300))] * 4, sampling_params=sampling_params
        )
    for output in outputs:
        assert output[0][0][-3:] == [501, 502, 503]


@pytest.mark.skipif(not current_platform.is_cpu(), reason="Tests the CPU platform.")
def test_simulate_forward_cpu_hybrid_prefix_cache_multi_rank(
    vllm_runner: type[VllmRunner],
) -> None:
    """Workers raise the block size to fit a hybrid model's Mamba pages. The
    engine core reruns VllmConfig.__post_init__ after its handshake and must
    keep that size: Mamba prefill checkpoints land on its block boundaries,
    so a repeated prompt hits up to its last full block."""
    from vllm.inputs import TokensPrompt

    params = SamplingParams(max_tokens=1, detokenize=False)
    with vllm_runner(
        "Qwen/Qwen3.8-2.4T-A95B-FP8",
        hf_overrides={"num_hidden_layers": 4},
        simulate_forward=True,
        # Unset, as the platform resolves it (the runner sets one by default).
        block_size=None,
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        # The whole prompt in one chunk, which no block boundary splits.
        max_num_batched_tokens=16384,
        kv_cache_memory_bytes=4 * 1024**3,
        max_model_len=16384,
    ) as vllm_model:
        block_size = vllm_model.llm.llm_engine.vllm_config.cache_config.block_size
        # Ends mid-block: the prefill must be split at the last block boundary.
        num_tokens = 4 * block_size + block_size // 2
        prompt = TokensPrompt(
            prompt_token_ids=[1000 + i % 5000 for i in range(num_tokens)]
        )
        hits = [
            vllm_model.llm.generate(prompt, params)[0].num_cached_tokens
            for _ in range(2)
        ]
    assert hits == [0, 4 * block_size]
