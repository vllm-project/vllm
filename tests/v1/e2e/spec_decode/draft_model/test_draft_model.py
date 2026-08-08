# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import pytest
import torch

from tests.utils import multi_gpu_only, single_gpu_only
from vllm import LLM, SamplingParams
from vllm.config import VllmConfig, replace
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.engine.arg_utils import EngineArgs

from ..utils import (
    Messages,
    _skip_if_insufficient_gpus_for_tp,
    compute_acceptance_len,
    compute_acceptance_rate,
    evaluate_llm_for_gsm8k,
    get_instruct_coder_messages,
    get_test_prompts,
    greedy_sampling,
    stochastic_sampling,
)


@dataclass
class ArgsTest:
    target_model: str
    draft_model: str
    sampling_config: SamplingParams
    num_speculative_tokens: int
    expected_acceptance_rate: float
    expected_acceptance_len: float
    expected_gsm8k_accuracy: float = 0.0  # skip by default
    # Defaults
    enforce_eager: bool = True
    parallel_drafting: bool = False
    target_tensor_parallel_size: int = 1
    draft_tensor_parallel_size: int = 1
    max_model_len: int = 2048
    gpu_memory_utilization: float = 0.5
    dataset: str = "test_prompts"
    num_prompts: int = 100


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
    # Same Gemma3 model for draft and target. This exercises multi-group KV
    # draft model metadata handling.
    ArgsTest(
        target_model="google/gemma-3-270m-it",
        draft_model="google/gemma-3-270m-it",
        sampling_config=greedy_sampling(),
        num_speculative_tokens=3,
        expected_acceptance_len=0.98 * (3 + 1),  # epsilon discount of K + 1
        expected_acceptance_rate=0.98,
    ),
]


@pytest.mark.parametrize("args", cases)
@pytest.mark.parametrize("enforce_eager", [True, False])
@single_gpu_only
def test_draft_model_correctness(args: ArgsTest, enforce_eager: bool):
    args.enforce_eager = enforce_eager
    assert_draft_model_correctness(args)


@single_gpu_only
def test_draft_model_realistic_example():
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
    assert_draft_model_correctness(args)


@single_gpu_only
def test_draft_model_parallel_drafting():
    args = ArgsTest(
        target_model="Qwen/Qwen3-1.7B",
        draft_model="amd/PARD-Qwen3-0.6B",
        dataset="likaixin/InstructCoder",
        num_speculative_tokens=3,
        sampling_config=greedy_sampling(),
        parallel_drafting=True,
        enforce_eager=False,
        expected_acceptance_len=2.3,  # ref: 2.52
        expected_acceptance_rate=0.4,  # ref: 0.51
    )
    assert_draft_model_correctness(args)


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
def test_draft_model_quantization(models: tuple[str, str], enforce_eager: bool):
    tgt_model, draft_model = models
    sd_case = ArgsTest(
        target_model=tgt_model,
        draft_model=draft_model,
        **some_high_acceptance_metrics(),
        enforce_eager=enforce_eager,
    )
    assert_draft_model_correctness(sd_case)


@multi_gpu_only(num_gpus=2)
def test_draft_model_tensor_parallelism():
    """Ensure spec decode works when running with TP > 1."""
    _skip_if_insufficient_gpus_for_tp(2)
    sd_case = ArgsTest(
        target_model="Qwen/Qwen3-1.7B",
        target_tensor_parallel_size=2,
        draft_model="Qwen/Qwen3-0.6B",
        draft_tensor_parallel_size=2,
        **some_high_acceptance_metrics(),
        enforce_eager=False,
        expected_gsm8k_accuracy=0.5,
    )
    assert_draft_model_correctness(sd_case)


@multi_gpu_only(num_gpus=2)
def test_draft_model_engine_args_tensor_parallelism():
    """Ensure the vllm_config for the draft model is created correctly,
    and independently of the target model (quantization, TP, etc.)"""
    _skip_if_insufficient_gpus_for_tp(2)

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
    assert target_config.quant_config.get_name() == "fp8"

    speculative_config = target_config.speculative_config
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
    """Replicate SpecDecodeBaseProposer._create_draft_vllm_config logic
    so we can test it without instantiating a full proposer."""
    spec_cfg = vllm_config.speculative_config
    if spec_cfg.moe_backend is not None:
        return replace(
            vllm_config,
            kernel_config=replace(
                vllm_config.kernel_config,
                moe_backend=spec_cfg.moe_backend,
            ),
        )
    return vllm_config


def test_draft_model_moe_backend_override():
    """When moe_backend is set in speculative_config, the draft VllmConfig
    should use it while the target keeps its own setting."""
    engine_args = EngineArgs(
        model="Qwen/Qwen3-1.7B",
        tensor_parallel_size=1,
        moe_backend="flashinfer_trtllm",
        speculative_config={
            "model": "Qwen/Qwen3-0.6B",
            "method": "draft_model",
            "num_speculative_tokens": 3,
            "moe_backend": "triton",
        },
    )
    tgt_config: VllmConfig = engine_args.create_engine_config()
    assert tgt_config.kernel_config.moe_backend == "flashinfer_trtllm"
    assert tgt_config.speculative_config.moe_backend == "triton"

    draft_config = _apply_draft_moe_backend(tgt_config)
    assert draft_config.kernel_config.moe_backend == "triton"
    # Target config must be unaffected.
    assert tgt_config.kernel_config.moe_backend == "flashinfer_trtllm"


def test_draft_model_moe_backend_inherits_target():
    """When moe_backend is not set in speculative_config, the draft should
    inherit the target's moe_backend."""
    engine_args = EngineArgs(
        model="Qwen/Qwen3-1.7B",
        tensor_parallel_size=1,
        moe_backend="flashinfer_cutlass",
        speculative_config={
            "model": "Qwen/Qwen3-0.6B",
            "method": "draft_model",
            "num_speculative_tokens": 3,
        },
    )
    tgt_config: VllmConfig = engine_args.create_engine_config()
    assert tgt_config.kernel_config.moe_backend == "flashinfer_cutlass"
    assert tgt_config.speculative_config.moe_backend is None

    draft_config = _apply_draft_moe_backend(tgt_config)
    assert draft_config.kernel_config.moe_backend == "flashinfer_cutlass"
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
    with pytest.raises(ValueError):
        engine_args.create_engine_config()


def assert_draft_model_correctness(args: ArgsTest):
    """Compare the outputs using and not using speculative decoding.
    In the greedy decoding case, the outputs must match EXACTLY."""
    test_prompts: list[Messages] = get_messages(
        dataset=args.dataset, n=args.num_prompts
    )

    spec_llm = LLM(
        model=args.target_model,
        speculative_config={
            "model": args.draft_model,
            "method": "draft_model",
            "num_speculative_tokens": args.num_speculative_tokens,
            "max_model_len": args.max_model_len,
            "enforce_eager": args.enforce_eager,
            "draft_tensor_parallel_size": args.draft_tensor_parallel_size,
            "parallel_drafting": args.parallel_drafting,
        },
        max_num_seqs=100,  # limit cudagraph capture runtime
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.target_tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        disable_log_stats=False,  # enables get_metrics()
    )

    # we don't check the outputs, only check the metrics
    spec_llm.chat(test_prompts, args.sampling_config)
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

    assert acceptance_rate >= args.expected_acceptance_rate
    assert acceptance_len >= args.expected_acceptance_len
    has_async = spec_llm.llm_engine.vllm_config.scheduler_config.async_scheduling
    del spec_llm  # CLEANUP
    torch.accelerator.empty_cache()
    cleanup_dist_env_and_memory()
    assert has_async, "Expected async_scheduling=True for draft_model spec decode"
