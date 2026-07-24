# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Acceptance-length regression tests for EAGLE3 and MTP speculative decoding.

Each test runs MT-Bench inference and asserts the mean acceptance length stays
within tolerance of a calibrated baseline.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch

from tests.conftest import VllmRunner
from tests.evals.gsm8k.gsm8k_eval import evaluate_gsm8k_offline
from tests.utils import large_gpu_mark
from vllm import SamplingParams
from vllm.benchmarks.datasets import get_samples
from vllm.inputs import TokensPrompt
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import AttentionSelectorConfig
from vllm.v1.metrics.reader import Counter, Vector

# Default test parameters
DEFAULT_NUM_SPEC_TOKENS = 3
DEFAULT_NUM_PROMPTS = 80
DEFAULT_OUTPUT_LEN = 256
DEFAULT_MAX_MODEL_LEN = 16384
DEFAULT_RTOL = 0.05

# TP sizes to test
TP_SIZES = [1, 2, 4]

# Backends excluded from testing due to significantly different behavior
EXCLUDED_BACKENDS = {AttentionBackendEnum.FLEX_ATTENTION}


@dataclass
class SpecDecodeModelConfig:
    method: str
    model: str
    expected_acceptance_length: float
    # Draft model (EAGLE3 only; MTP heads ship with the verifier)
    drafter: str | None = None
    num_speculative_tokens: int = DEFAULT_NUM_SPEC_TOKENS
    expected_acceptance_lengths_per_pos: list[float] = field(default_factory=list)
    id: str = ""
    tp_size: int = 1
    gpu_memory_utilization: float = 0.7
    # Backends that are incompatible with this model (will be skipped)
    excluded_backends: set[AttentionBackendEnum] = field(default_factory=set)
    marks: list = field(default_factory=list)
    # Custom relative tolerance (defaults to DEFAULT_RTOL if None)
    rtol: float | None = None
    rocm_expected_acceptance_lengths_per_pos: list[float] = field(default_factory=list)
    # Extra VllmRunner kwargs (e.g. quantization, trust_remote_code)
    extra_kwargs: dict = field(default_factory=dict)
    # If set, also assert GSM8k accuracy stays above this floor
    expected_gsm8k_accuracy: float | None = None


# Baselines measured on the MT-Bench dataset with
# examples/features/speculative_decoding/spec_decode_offline.py
EAGLE3_MODEL_CONFIGS = [
    SpecDecodeModelConfig(
        method="eagle3",
        model="meta-llama/Llama-3.1-8B-Instruct",
        drafter="RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3",
        expected_acceptance_length=2.60,
        expected_acceptance_lengths_per_pos=[0.7296, 0.5208, 0.3545],
        id="llama3-8b-eagle3",
    ),
    SpecDecodeModelConfig(
        method="eagle3",
        model="Qwen/Qwen3-8B",
        drafter="RedHatAI/Qwen3-8B-speculator.eagle3",
        expected_acceptance_length=2.26,
        expected_acceptance_lengths_per_pos=[0.6541, 0.3993, 0.2020],
        id="qwen3-8b-eagle3",
    ),
    SpecDecodeModelConfig(
        method="eagle3",
        model="openai/gpt-oss-20b",
        drafter="RedHatAI/gpt-oss-20b-speculator.eagle3",
        expected_acceptance_length=2.56,
        expected_acceptance_lengths_per_pos=[0.7165, 0.5120, 0.3337],
        id="gpt-oss-20b-eagle3",
        # FLASHINFER incompatible: gpt-oss-20b uses sink attention which
        # FLASHINFER does not support ("sink setting not supported")
        excluded_backends={AttentionBackendEnum.FLASHINFER},
        rocm_expected_acceptance_lengths_per_pos=[0.7040, 0.4820, 0.3350],
    ),
    SpecDecodeModelConfig(
        method="eagle3",
        model="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
        drafter="nm-testing/Speculator-Qwen3-30B-MOE-VL-Eagle3",
        expected_acceptance_length=1.35,
        expected_acceptance_lengths_per_pos=[0.2900, 0.0620, 0.0115],
        id="qwen3-30b-moe-vl-eagle3",
        marks=[pytest.mark.slow_test],
        rtol=0.15,  # Higher tolerance due to small absolute values at position 2
    ),
]

# Baselines measured on MT-Bench (greedy): DeepSeek-V3 tp=8, DSV4-Flash NVFP4 tp=4.
MTP_MODEL_CONFIGS = [
    SpecDecodeModelConfig(
        method="mtp",
        model="deepseek-ai/DeepSeek-V3",
        num_speculative_tokens=1,
        expected_acceptance_length=1.882,
        id="deepseek-v3-mtp-fp8-tp8",
        tp_size=8,
        gpu_memory_utilization=0.9,
        extra_kwargs={
            "quantization": "fp8",
            "trust_remote_code": True,
            "kv_cache_dtype": "fp8",
            "block_size": None,
            "enforce_eager": True,
        },
        marks=[pytest.mark.slow_test],
        rtol=0.10,
        expected_gsm8k_accuracy=0.90,
    ),
    SpecDecodeModelConfig(
        method="mtp",
        model="nvidia/DeepSeek-V4-Flash-NVFP4",
        num_speculative_tokens=1,
        expected_acceptance_length=1.70,
        id="deepseek-v4-flash-mtp-nvfp4-tp4",
        tp_size=4,
        gpu_memory_utilization=0.9,
        extra_kwargs={
            "trust_remote_code": True,
            "kv_cache_dtype": "fp8",
            "block_size": None,
            "enforce_eager": True,
        },
        marks=[pytest.mark.slow_test],
        rtol=0.10,
    ),
]


def get_available_attention_backends() -> list[str]:
    if current_platform.is_rocm():
        return ["auto"]

    # Check if get_valid_backends is actually defined in the platform class
    # (not just returning None from __getattr__)
    get_valid_backends = getattr(current_platform.__class__, "get_valid_backends", None)
    if get_valid_backends is None:
        return ["FLASH_ATTN"]

    device_capability = current_platform.get_device_capability()
    if device_capability is None:
        return ["FLASH_ATTN"]

    attn_selector_config = AttentionSelectorConfig(
        head_size=128,
        dtype=torch.bfloat16,
        kv_cache_dtype=None,
        block_size=None,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
    )

    valid_backends, _ = current_platform.get_valid_backends(
        device_capability=device_capability,
        attn_selector_config=attn_selector_config,
    )

    return [
        candidate.backend.name
        for candidate in valid_backends
        if candidate.backend not in EXCLUDED_BACKENDS
    ]


def get_attention_backend_params() -> list[str]:
    return get_available_attention_backends()


def get_tp_size_params() -> list[pytest.param]:
    num_gpus = torch.accelerator.device_count() if torch.cuda.is_available() else 1
    return [pytest.param(tp, id=f"tp{tp}") for tp in TP_SIZES if tp <= num_gpus]


def get_mt_bench_prompts(
    tokenizer, num_prompts: int = DEFAULT_NUM_PROMPTS
) -> list[list[int]]:
    # Some checkpoints (e.g. quantized exports like the NVFP4 build) ship no
    # chat template; applying one then produces degenerate prompts and tanks
    # acceptance. Fall back to raw tokenization when no template is present.
    skip_chat_template = getattr(tokenizer, "chat_template", None) is None
    args = SimpleNamespace(
        dataset_name="hf",
        dataset_path="philschmid/mt-bench",
        num_prompts=num_prompts,
        seed=42,
        no_oversample=False,
        endpoint_type="openai-chat",
        input_len=None,
        output_len=DEFAULT_OUTPUT_LEN,
        sharegpt_output_len=DEFAULT_OUTPUT_LEN,
        hf_name=None,
        hf_split="train",
        hf_subset=None,
        hf_output_len=DEFAULT_OUTPUT_LEN,
        no_stream=True,
        disable_shuffle=False,
        skip_chat_template=skip_chat_template,
        trust_remote_code=False,
        enable_multimodal_chat=False,
        request_id_prefix="",
    )
    samples = get_samples(args, tokenizer)
    prompt_ids = [
        tokenizer.encode(sample.prompt, add_special_tokens=False) for sample in samples
    ]
    return prompt_ids


def extract_acceptance_metrics(metrics, num_spec_tokens: int) -> dict:
    num_drafts = 0
    num_accepted_tokens = 0
    acceptance_counts = [0] * num_spec_tokens

    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos in range(min(len(metric.values), num_spec_tokens)):
                acceptance_counts[pos] += metric.values[pos]

    # Calculate mean acceptance length
    # Formula: 1 + (accepted_tokens / num_drafts)
    acceptance_length = 1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1

    # Calculate per-position acceptance lengths (contribution to total)
    # Each position contributes: accepted_at_pos / num_drafts
    acceptance_lengths_per_pos = [
        count / num_drafts if num_drafts > 0 else 0.0 for count in acceptance_counts
    ]

    return {
        "acceptance_length": acceptance_length,
        "acceptance_lengths_per_pos": acceptance_lengths_per_pos,
        "num_drafts": num_drafts,
        "num_accepted_tokens": num_accepted_tokens,
    }


def _run_acceptance_length_test(
    config: SpecDecodeModelConfig,
    tp_size: int,
    attention_backend: str,
    monkeypatch: pytest.MonkeyPatch,
):
    attention_config = None
    if attention_backend != "auto":
        backend_enum = AttentionBackendEnum[attention_backend]
        if backend_enum in config.excluded_backends:
            pytest.skip(f"{attention_backend} is incompatible with {config.id}")
        attention_config = {"backend": attention_backend}

    speculative_config = {
        "method": config.method,
        "num_speculative_tokens": config.num_speculative_tokens,
    }
    if config.drafter is not None:
        speculative_config["model"] = config.drafter

    extra_kwargs = dict(config.extra_kwargs)
    # Qwen/Qwen3-30B-A3B-FP8 with TP=4 needs EP (issue #25292)
    if tp_size == 4 and "Qwen3-VL" in config.model:
        extra_kwargs["enable_expert_parallel"] = True

    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        with VllmRunner(
            model_name=config.model,
            speculative_config=speculative_config,
            attention_config=attention_config,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_log_stats=False,
            max_model_len=DEFAULT_MAX_MODEL_LEN,
            **extra_kwargs,
        ) as vllm_runner:
            tokenizer = vllm_runner.llm.get_tokenizer()
            prompt_ids = get_mt_bench_prompts(tokenizer, DEFAULT_NUM_PROMPTS)

            sampling_params = SamplingParams(
                temperature=0,
                max_tokens=DEFAULT_OUTPUT_LEN,
            )
            vllm_runner.llm.generate(
                [TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids],
                sampling_params=sampling_params,
            )

            metrics = vllm_runner.llm.get_metrics()
            results = extract_acceptance_metrics(metrics, config.num_speculative_tokens)

            actual = results["acceptance_length"]
            expected = config.expected_acceptance_length
            actual_per_pos = results["acceptance_lengths_per_pos"]
            expected_per_pos = config.expected_acceptance_lengths_per_pos
            if (
                current_platform.is_rocm()
                and config.rocm_expected_acceptance_lengths_per_pos
            ):
                expected_per_pos = config.rocm_expected_acceptance_lengths_per_pos

            rel_error = abs(actual - expected) / expected

            # Overall acceptance length always uses DEFAULT_RTOL
            assert rel_error <= DEFAULT_RTOL, (
                f"Acceptance length regression detected for {config.id}!\n"
                f"  Expected: {expected:.3f}\n"
                f"  Actual:   {actual:.3f}\n"
                f"  Relative error: {rel_error:.2%} (tolerance: {DEFAULT_RTOL:.2%})\n"
                f"  Drafts: {results['num_drafts']}, "
                f"Accepted tokens: {results['num_accepted_tokens']}"
            )

            if expected_per_pos and len(expected_per_pos) == len(actual_per_pos):
                # Per-position checks use model-specific rtol if provided
                rtol = config.rtol if config.rtol is not None else DEFAULT_RTOL
                for pos, (act, exp) in enumerate(zip(actual_per_pos, expected_per_pos)):
                    if exp > 0:
                        min_expected = exp * (1 - rtol)
                        assert act >= min_expected, (
                            f"Per-position acceptance length regression at pos {pos} "
                            f"for {config.id}!\n"
                            f"  Expected: {exp:.3f}\n"
                            f"  Actual:   {act:.3f}\n"
                            f"  Minimum:  {min_expected:.3f}\n"
                            f"  Tolerance: rtol={rtol:.2%}"
                        )

            if config.expected_gsm8k_accuracy is not None:
                accuracy = evaluate_gsm8k_offline(vllm_runner.llm)["accuracy"]
                assert accuracy >= config.expected_gsm8k_accuracy, (
                    f"GSM8k accuracy {accuracy:.3f} below "
                    f"{config.expected_gsm8k_accuracy:.3f} for {config.id}"
                )

            print(
                f"\n{config.id} [tp={tp_size}, backend={attention_backend}]: "
                f"acceptance_length={actual:.3f}"
                f" (expected={expected:.3f}, rel_error={rel_error:.2%})"
            )
            print(f"  Per-position: {[f'{v:.3f}' for v in actual_per_pos]}")
            if expected_per_pos:
                print(f"  Expected:     {[f'{v:.3f}' for v in expected_per_pos]}")


@large_gpu_mark(min_gb=40)
@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="This test is only supported on CUDA-alike platforms.",
)
@pytest.mark.parametrize(
    "model_config",
    [
        pytest.param(config, id=config.id, marks=config.marks)
        for config in EAGLE3_MODEL_CONFIGS
    ],
)
@pytest.mark.parametrize("tp_size", get_tp_size_params())
@pytest.mark.parametrize("attention_backend", get_attention_backend_params())
def test_eagle3_acceptance_length(
    model_config: SpecDecodeModelConfig,
    tp_size: int,
    attention_backend: str,
    monkeypatch: pytest.MonkeyPatch,
):
    _run_acceptance_length_test(model_config, tp_size, attention_backend, monkeypatch)


@large_gpu_mark(min_gb=80)
@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="This test is only supported on CUDA-alike platforms.",
)
@pytest.mark.parametrize(
    "model_config",
    [
        pytest.param(config, id=config.id, marks=config.marks)
        for config in MTP_MODEL_CONFIGS
    ],
)
def test_mtp_acceptance_length(
    model_config: SpecDecodeModelConfig,
    monkeypatch: pytest.MonkeyPatch,
):
    num_gpus = torch.accelerator.device_count() if torch.cuda.is_available() else 1
    if num_gpus < model_config.tp_size:
        pytest.skip(f"Need {model_config.tp_size} GPUs, only {num_gpus} available")
    _run_acceptance_length_test(model_config, model_config.tp_size, "auto", monkeypatch)
