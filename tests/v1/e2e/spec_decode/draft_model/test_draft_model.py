# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from typing import Any

import pytest

from tests.models.utils import check_logprobs_close
from tests.utils import multi_gpu_only, single_gpu_only
from vllm import SamplingParams
from vllm.config import CompilationConfig, VllmConfig, replace
from vllm.config.kernel import MoEBackend
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform

from ..utils import (
    Messages,
    compute_acceptance_len,
    compute_acceptance_rate,
    evaluate_llm_for_gsm8k,
    get_instruct_coder_messages,
    get_test_prompts,
    greedy_sampling,
    stochastic_sampling,
)


# Tomas R: TODO: Improve naming of this class and
# assert_draft_model_correctness and move to common SD test utils
@dataclass(slots=True)
class ArgsTest:
    target_model: str
    draft_model: str
    sampling_config: SamplingParams
    num_speculative_tokens: int
    expected_acceptance_rate: float
    expected_acceptance_len: float
    expected_gsm8k_accuracy: float = 0.0  # skip by default
    # Defaults
    method: str = "draft_model"
    enforce_eager: bool = True
    target_tensor_parallel_size: int = 1
    draft_tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    max_model_len: int = 2048
    gpu_memory_utilization: float | None = 0.5
    dataset: str = "test_prompts"
    # Doubling 100 to 200 reduces sampling standard error by about 29%.
    num_prompts: int = 200
    max_num_seqs: int = 100
    enable_prefix_caching: bool | None = None
    enable_adaptive_verification: bool = False
    chat_template_kwargs: dict | None = None
    reject_full_acceptance: bool = False
    engine_kwargs: dict[str, Any] = field(default_factory=dict)


def get_messages(dataset: str, n: int) -> list[Messages]:
    if dataset == "test_prompts":
        return get_test_prompts(mm_enabled=False, num_prompts=n)
    if dataset == "likaixin/InstructCoder":
        return get_instruct_coder_messages(n=n)
    raise NotImplementedError(f"Dataset '{dataset}' not implemented")


def some_high_acceptance_metrics() -> dict:
    return {
        "sampling_config": greedy_sampling(),
        "num_speculative_tokens": 3,
        "expected_acceptance_len": 3.4,  # ref: 3.75
        "expected_acceptance_rate": 0.8,  # ref: 0.9
    }


@single_gpu_only
@pytest.mark.parametrize("draft_temperature_scale", [0.0, 0.5])
@pytest.mark.parametrize(
    "method,model",
    [
        ("draft_model", "Qwen/Qwen3-0.6B"),
        ("mtp", "Qwen/Qwen3.5-0.8B-Base"),
    ],
)
def test_draft_temperature_scale_preserves_greedy_target(
    vllm_runner, monkeypatch, draft_temperature_scale, method, model
):
    """Scaling draft temperatures preserves greedy target tokens and logprobs."""
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1" if method == "draft_model" else "0")
    prompts = [
        "The capital of France is",
        "Write a short story about a robot:",
        "Explain why the sky is blue:",
        "The first ten prime numbers are",
    ]
    results = []
    logprob_results = []
    acceptance_rates = []
    for scale in (None, draft_temperature_scale):
        spec_config: dict[str, Any] = {
            "method": method,
            "model": model,
            "num_speculative_tokens": 3,
            "draft_sample_method": "greedy" if scale is None else "probabilistic",
        }
        if scale is not None:
            spec_config["draft_temperature_scale"] = scale
        with vllm_runner(
            model,
            block_size=None,
            enable_chunked_prefill=None,
            max_model_len=256,
            max_num_seqs=len(prompts),
            speculative_config=spec_config,
            disable_log_stats=False,
            limit_mm_per_prompt={"image": 0, "video": 0},
        ) as runner:
            outputs = runner.llm.generate(
                prompts,
                SamplingParams(
                    temperature=0,
                    seed=42,
                    max_tokens=32,
                    ignore_eos=True,
                    logprobs=5 if method == "mtp" else None,
                ),
            )
            results.append([list(out.outputs[0].token_ids) for out in outputs])
            if method == "mtp":
                logprob_results.append(
                    [
                        (
                            list(out.outputs[0].token_ids),
                            out.outputs[0].text,
                            out.outputs[0].logprobs,
                        )
                        for out in outputs
                    ]
                )
            acceptance_rates.append(compute_acceptance_rate(runner.llm.get_metrics()))

    assert all(rate > 0 for rate in acceptance_rates)
    if method == "draft_model":
        assert results[0] == results[1]
    else:
        check_logprobs_close(
            outputs_0_lst=logprob_results[0],
            outputs_1_lst=logprob_results[1],
            name_0="greedy_draft",
            name_1=f"draft_temperature_scale={draft_temperature_scale}",
        )


@single_gpu_only
def test_draft_temperature_scale_handles_mixed_target_temperatures(
    vllm_runner, monkeypatch
):
    """Identity scaling preserves seeded output; half scaling changes acceptance."""
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    model = "Qwen/Qwen3-0.6B"
    prompts = [
        "The capital of France is",
        "Write a short story about a robot:",
        "Explain why the sky is blue:",
        "The first ten prime numbers are",
    ]
    params = [
        SamplingParams(temperature=t, seed=42 + i, max_tokens=32, ignore_eos=True)
        for i, t in enumerate((0.0, 0.5, 1.0, 1.5))
    ]
    results = []
    acceptance_rates = []
    for scale in (None, 1.0, 0.5):
        options = {} if scale is None else {"draft_temperature_scale": scale}
        with vllm_runner(
            model,
            max_model_len=256,
            max_num_seqs=4,
            speculative_config={
                "method": "draft_model",
                "model": model,
                "num_speculative_tokens": 3,
                "draft_sample_method": "probabilistic",
                **options,
            },
            disable_log_stats=False,
        ) as runner:
            outputs = runner.llm.generate(prompts, params)
            results.append([list(out.outputs[0].token_ids) for out in outputs])
            acceptance_rates.append(compute_acceptance_rate(runner.llm.get_metrics()))
    assert results[0] == results[1]
    assert acceptance_rates[0] == acceptance_rates[1]
    assert 0 < acceptance_rates[2] < acceptance_rates[0]
    assert [p.temperature for p in params] == [0.0, 0.5, 1.0, 1.5]


cases = [
    # Same model for draft and target, greedy sampling.
    ArgsTest(
        target_model="Qwen/Qwen3-0.6B",
        draft_model="Qwen/Qwen3-0.6B",
        sampling_config=greedy_sampling(),
        num_speculative_tokens=3,  # K
        expected_acceptance_len=0.98 * (3 + 1),  # epsilon discount of K + 1
        expected_acceptance_rate=0.98,  # slight epsilon
        expected_gsm8k_accuracy=0.25,  # ref: 35-40%
    ),
    # Smaller draft model, stochastic sampling.
    ArgsTest(
        target_model="Qwen/Qwen3-1.7B",
        draft_model="Qwen/Qwen3-0.6B",
        sampling_config=stochastic_sampling(),
        num_speculative_tokens=3,
        expected_acceptance_len=3.4,  # ref: 3.7
        expected_acceptance_rate=0.80,  # ref: 0.90
        expected_gsm8k_accuracy=0.5,  # ref: 60%. Note gsm8k always runs greedy sampling
    ),
]


@pytest.mark.parametrize("args", cases)
@pytest.mark.parametrize("enforce_eager", [True, False])
@single_gpu_only
def test_draft_model_correctness(args: ArgsTest, enforce_eager: bool, vllm_runner):
    args.enforce_eager = enforce_eager
    assert_draft_model_correctness(args, vllm_runner)


@single_gpu_only
def test_draft_model_realistic_example(vllm_runner):
    args = ArgsTest(
        target_model="Qwen/Qwen3-1.7B",
        draft_model="Qwen/Qwen3-0.6B",
        dataset="likaixin/InstructCoder",
        num_speculative_tokens=3,
        sampling_config=greedy_sampling(),
        enforce_eager=False,
        expected_acceptance_len=2.6,  # ref: 2.86
        expected_acceptance_rate=0.5,  # ref: 0.62
    )
    assert_draft_model_correctness(args, vllm_runner)


@pytest.mark.parametrize(
    "models",
    [
        # target_model,         draft_model
        ("Qwen/Qwen3-1.7B-FP8", "Qwen/Qwen3-0.6B"),  # target quantized
        ("Qwen/Qwen3-1.7B", "Qwen/Qwen3-0.6B-FP8"),  # draft quantized
    ],
    ids=["target_quantized", "draft_quantized"],
)
@pytest.mark.parametrize("enforce_eager", [True, False])
@single_gpu_only
def test_draft_model_quantization(
    models: tuple[str, str], enforce_eager: bool, vllm_runner
):
    tgt_model, draft_model = models
    sd_case = ArgsTest(
        target_model=tgt_model,
        draft_model=draft_model,
        **some_high_acceptance_metrics(),
        enforce_eager=enforce_eager,
    )
    assert_draft_model_correctness(sd_case, vllm_runner)


@single_gpu_only
def test_draft_model_transformers_backend(vllm_runner):
    """Ensure that Attention prefixes don't collide between the target and
    draft models when both models use the Transformers modelling backend."""
    sd_case = ArgsTest(
        target_model="allenai/OLMo-2-0425-1B-Instruct",
        draft_model="allenai/OLMo-2-0425-1B-Instruct",
        sampling_config=greedy_sampling(),
        num_speculative_tokens=3,
        expected_acceptance_len=0.98 * (3 + 1),  # epsilon discount of K + 1
        expected_acceptance_rate=0.98,  # slight epsilon
    )
    assert_draft_model_correctness(sd_case, vllm_runner)


@multi_gpu_only(num_gpus=2)
def test_draft_model_tensor_parallelism(vllm_runner):
    """Ensure spec decode works when running with TP > 1."""
    sd_case = ArgsTest(
        target_model="Qwen/Qwen3-1.7B",
        target_tensor_parallel_size=2,
        draft_model="Qwen/Qwen3-0.6B",
        draft_tensor_parallel_size=2,
        **some_high_acceptance_metrics(),
        enforce_eager=False,
        expected_gsm8k_accuracy=0.5,
    )
    assert_draft_model_correctness(sd_case, vllm_runner)


@multi_gpu_only(num_gpus=2)
@pytest.mark.parametrize(
    "load_kwargs",
    [
        {"load_format": "auto"},
        {"load_format": "fastsafetensors"},
    ],
    ids=["auto", "fastsafetensors"],
)
def test_draft_model_pipeline_parallelism(vllm_runner, load_kwargs: dict):
    """Ensure the draft model runs whole on the last stage when PP > 1.

    fastsafetensors synchronizes across ranks, so it must load the draft model
    without the other stages.
    """
    sd_case = ArgsTest(
        target_model="Qwen/Qwen3-1.7B",
        draft_model="Qwen/Qwen3-0.6B",
        pipeline_parallel_size=2,
        **some_high_acceptance_metrics(),
        enforce_eager=False,
        engine_kwargs=load_kwargs,
    )
    assert_draft_model_correctness(sd_case, vllm_runner)


@multi_gpu_only(num_gpus=2)
def test_draft_model_engine_args_tensor_parallelism():
    """Ensure the vllm_config for the draft model is created correctly,
    and independently of the target model (quantization, TP, etc.)"""
    engine_args = EngineArgs(
        model="Qwen/Qwen3-1.7B-FP8",  # <<< tgt quantized
        tensor_parallel_size=2,
        speculative_config={
            "model": "Qwen/Qwen3-0.6B",  # <<< draft not quantized
            "method": "draft_model",
            "num_speculative_tokens": 3,
            "draft_tensor_parallel_size": 1,  # <<< valid arg name
        },
    )
    target_config: VllmConfig = engine_args.create_engine_config()
    assert target_config.parallel_config.tensor_parallel_size == 2
    assert target_config.quant_config is not None
    assert target_config.quant_config.get_name() == "fp8"

    speculative_config = target_config.speculative_config
    assert speculative_config is not None
    draft_config: VllmConfig = replace(
        target_config,
        quant_config=None,
        parallel_config=replace(
            speculative_config.draft_parallel_config,
            rank=target_config.parallel_config.rank,
        ),
        model_config=speculative_config.draft_model_config,
    )
    assert draft_config.parallel_config.tensor_parallel_size == 1
    assert draft_config.quant_config is None


def _apply_draft_moe_backend(vllm_config: VllmConfig) -> VllmConfig:
    """Replicate the draft moe_backend override so we can test it without
    instantiating a full speculator."""
    spec_cfg = vllm_config.speculative_config
    assert spec_cfg is not None
    if spec_cfg.moe_backend is not None:
        return replace(
            vllm_config,
            kernel_config=replace(
                vllm_config.kernel_config,
                moe_backend=spec_cfg.moe_backend,
            ),
        )
    return vllm_config


def _platform_moe_backend(monkeypatch: pytest.MonkeyPatch) -> MoEBackend:
    if current_platform.is_rocm():
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
        return "aiter"
    if current_platform.is_cuda():
        return "flashinfer_trtllm"
    return "auto"


def test_draft_model_moe_backend_override(monkeypatch: pytest.MonkeyPatch):
    """When moe_backend is set in speculative_config, the draft VllmConfig
    should use it while the target keeps its own setting."""
    target_moe_backend = _platform_moe_backend(monkeypatch)
    engine_args = EngineArgs(
        model="Qwen/Qwen3-1.7B",
        tensor_parallel_size=1,
        moe_backend=target_moe_backend,
        speculative_config={
            "model": "Qwen/Qwen3-0.6B",
            "method": "draft_model",
            "num_speculative_tokens": 3,
            "moe_backend": "triton",
        },
    )
    tgt_config: VllmConfig = engine_args.create_engine_config()
    assert tgt_config.kernel_config.moe_backend == target_moe_backend
    assert tgt_config.speculative_config is not None
    assert tgt_config.speculative_config.moe_backend == "triton"

    draft_config = _apply_draft_moe_backend(tgt_config)
    assert draft_config.kernel_config.moe_backend == "triton"
    # Target config must be unaffected.
    assert tgt_config.kernel_config.moe_backend == target_moe_backend


def test_draft_model_moe_backend_inherits_target(monkeypatch: pytest.MonkeyPatch):
    """When moe_backend is not set in speculative_config, the draft should
    inherit the target's moe_backend."""
    target_moe_backend = _platform_moe_backend(monkeypatch)
    engine_args = EngineArgs(
        model="Qwen/Qwen3-1.7B",
        tensor_parallel_size=1,
        moe_backend=target_moe_backend,
        speculative_config={
            "model": "Qwen/Qwen3-0.6B",
            "method": "draft_model",
            "num_speculative_tokens": 3,
        },
    )
    tgt_config: VllmConfig = engine_args.create_engine_config()
    assert tgt_config.kernel_config.moe_backend == target_moe_backend
    assert tgt_config.speculative_config is not None
    assert tgt_config.speculative_config.moe_backend is None

    draft_config = _apply_draft_moe_backend(tgt_config)
    assert draft_config.kernel_config.moe_backend == target_moe_backend
    assert draft_config is tgt_config


def test_draft_model_moe_backend_default_auto():
    """When neither target nor draft set moe_backend explicitly, both should
    default to 'auto'."""
    engine_args = EngineArgs(
        model="Qwen/Qwen3-1.7B",
        tensor_parallel_size=1,
        speculative_config={
            "model": "Qwen/Qwen3-0.6B",
            "method": "draft_model",
            "num_speculative_tokens": 3,
        },
    )
    tgt_config: VllmConfig = engine_args.create_engine_config()
    assert tgt_config.kernel_config.moe_backend == "auto"
    assert tgt_config.speculative_config is not None
    assert tgt_config.speculative_config.moe_backend is None

    draft_config = _apply_draft_moe_backend(tgt_config)
    assert draft_config.kernel_config.moe_backend == "auto"
    assert draft_config is tgt_config


def test_draft_model_engine_args_rejects_invalid_tp_argname():
    """The user should pass "draft_tensor_parallel_size" rather than
    "tensor_parallel_size". We enforce this with validation."""
    engine_args = EngineArgs(
        model="Qwen/Qwen3-1.7B",
        tensor_parallel_size=1,
        speculative_config={
            "model": "Qwen/Qwen3-0.6B",
            "method": "draft_model",
            "num_speculative_tokens": 3,
            "tensor_parallel_size": 1,  # <<< invalid arg name
        },
    )
    with pytest.raises(ValueError, match="draft_tensor_parallel_size"):
        engine_args.create_engine_config()


def assert_draft_model_correctness(args: ArgsTest, vllm_runner):
    """Check speculative decoding acceptance metrics and GSM8K accuracy."""
    test_prompts: list[Messages] = get_messages(
        dataset=args.dataset, n=args.num_prompts
    )

    kwargs = vllm_runner_kwargs(args)
    with vllm_runner(args.target_model, **kwargs) as spec_runner:
        spec_llm = spec_runner.llm

        # we don't check the outputs, only check the metrics
        spec_llm.chat(
            test_prompts,
            args.sampling_config,
            chat_template_kwargs=args.chat_template_kwargs,
        )
        metrics = spec_llm.get_metrics()
        acceptance_rate: float = compute_acceptance_rate(metrics)
        acceptance_len: float = compute_acceptance_len(metrics)

        # Need to evaluate after getting metrics to avoid polluting the AR
        evaluate_llm_for_gsm8k(
            spec_llm, expected_accuracy_threshold=args.expected_gsm8k_accuracy
        )

        print(
            f"spec-decode: target={args.target_model}, draft={args.draft_model}, "
            f"temperature={args.sampling_config.temperature:.2f}, "
            f"acceptance_rate={acceptance_rate:.2f}, "
            f"acceptance_len={acceptance_len:.2f}, "
        )

        context = (
            f"method={args.method} "
            f"target={args.target_model}, draft={args.draft_model}, "
            f"eager={args.enforce_eager}, "
            f"acceptance_rate={acceptance_rate:.3f}, "
            f"acceptance_len={acceptance_len:.3f}"
        )
        assert acceptance_rate >= args.expected_acceptance_rate, (
            f"{context}; expected acceptance_rate >= "
            f"{args.expected_acceptance_rate:.3f}"
        )
        assert acceptance_len >= args.expected_acceptance_len, (
            f"{context}; expected acceptance_len >= {args.expected_acceptance_len:.3f}"
        )
        if args.reject_full_acceptance:
            assert acceptance_len < args.num_speculative_tokens + 1, (
                f"{context}; acceptance_len is at its ceiling"
            )
        # Assert async scheduling is active by default.
        has_async = spec_llm.llm_engine.vllm_config.scheduler_config.async_scheduling

    assert has_async, f"Expected async_scheduling=True for {args.method} spec decode"


def vllm_runner_kwargs(args: ArgsTest) -> dict[str, Any]:
    kwargs = dict(
        block_size=None,
        enable_chunked_prefill=None,
        compilation_config=CompilationConfig(),
        speculative_config={
            "model": args.draft_model,
            "method": args.method,
            "num_speculative_tokens": args.num_speculative_tokens,
            "max_model_len": args.max_model_len,
            "enforce_eager": args.enforce_eager,
            "draft_tensor_parallel_size": args.draft_tensor_parallel_size,
            "enable_adaptive_verification": args.enable_adaptive_verification,
        },
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.target_tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        enforce_eager=args.enforce_eager,
        disable_log_stats=False,  # enables get_metrics()
        enable_prefix_caching=args.enable_prefix_caching,
    )
    kwargs.update(args.engine_kwargs)
    return kwargs
