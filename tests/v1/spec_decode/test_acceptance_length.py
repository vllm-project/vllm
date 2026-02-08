# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
EAGLE3 Acceptance Length Regression Tests.

These tests verify that acceptance lengths for EAGLE3 speculative decoding
do not regress across vLLM commits. Each test runs inference on the MT-Bench
dataset and asserts that the mean acceptance length is within tolerance of
the expected baseline.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch

from tests.conftest import VllmRunner
from tests.utils import large_gpu_mark
from vllm import SamplingParams
from vllm.benchmarks.datasets import get_samples
from vllm.inputs import TokensPrompt
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import AttentionSelectorConfig
from vllm.v1.metrics.reader import Counter, Vector


@dataclass
class Eagle3ModelConfig:
    verifier: str
    drafter: str
    expected_acceptance_length: float
    expected_acceptance_lengths_per_pos: list[float] = field(default_factory=list)
    id: str = ""
    # Backends that are incompatible with this model (will be skipped)
    excluded_backends: set[AttentionBackendEnum] = field(default_factory=set)
    # Pytest marks for this configuration
    marks: list = field(default_factory=list)
    # Custom relative tolerance (defaults to DEFAULT_RTOL if None)
    rtol: float | None = None


# Model configurations for EAGLE3 acceptance length tests.
# Expected acceptance lengths are determined by running baseline benchmarks
# using examples/offline_inference/spec_decode.py with the MT-Bench dataset.
EAGLE3_MODEL_CONFIGS = [
    Eagle3ModelConfig(
        verifier="meta-llama/Llama-3.1-8B-Instruct",
        drafter="RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3",
        expected_acceptance_length=2.60,
        expected_acceptance_lengths_per_pos=[0.7296, 0.5208, 0.3545],
        id="llama3-8b-eagle3",
    ),
    Eagle3ModelConfig(
        verifier="Qwen/Qwen3-8B",
        drafter="RedHatAI/Qwen3-8B-speculator.eagle3",
        expected_acceptance_length=2.26,
        expected_acceptance_lengths_per_pos=[0.6541, 0.3993, 0.2020],
        id="qwen3-8b-eagle3",
    ),
    Eagle3ModelConfig(
        verifier="openai/gpt-oss-20b",
        drafter="RedHatAI/gpt-oss-20b-speculator.eagle3",
        expected_acceptance_length=2.56,
        expected_acceptance_lengths_per_pos=[0.7165, 0.5120, 0.3337],
        id="gpt-oss-20b-eagle3",
        # FLASHINFER incompatible: gpt-oss-20b uses sink attention which
        # FLASHINFER does not support ("sink setting not supported")
        excluded_backends={AttentionBackendEnum.FLASHINFER},
    ),
    Eagle3ModelConfig(
        verifier="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
        drafter="nm-testing/Speculator-Qwen3-30B-MOE-VL-Eagle3",
        expected_acceptance_length=1.35,
        expected_acceptance_lengths_per_pos=[0.2900, 0.0620, 0.0115],
        id="qwen3-30b-moe-vl-eagle3",
        marks=[
            pytest.mark.slow_test,
        ],
        rtol=0.15,  # Higher tolerance due to small absolute values at position 2
    ),
]

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


def get_available_attention_backends() -> list[str]:
    # Check if get_valid_backends is actually defined in the platform class
    # (not just returning None from __getattr__)
    get_valid_backends = getattr(current_platform.__class__, "get_valid_backends", None)
    if get_valid_backends is None:
        if current_platform.is_rocm():
            # ROCm uses Triton as its default attention backend since
            # Flash Attention is not supported.
            return ["TRITON_ATTN"]
        else:
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
        backend.name
        for backend, _ in valid_backends
        if backend not in EXCLUDED_BACKENDS
    ]


def get_attention_backend_params() -> list[str]:
    return get_available_attention_backends()


def get_tp_size_params() -> list[pytest.param]:
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    return [pytest.param(tp, id=f"tp{tp}") for tp in TP_SIZES if tp <= num_gpus]


def get_mt_bench_prompts(
    tokenizer, num_prompts: int = DEFAULT_NUM_PROMPTS
) -> list[list[int]]:
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
        skip_chat_template=False,
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
        if metric.name == "vllm_spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm_spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted_tokens += metric.value
        elif metric.name == "vllm_spec_decode_num_accepted_tokens_per_pos":
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


@large_gpu_mark(min_gb=40)
@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="This test is only supported on CUDA platform.",
)
@pytest.mark.parametrize(
    "model_config",
    [
        pytest.param(config, id=config.id, marks=config.marks)
        for config in EAGLE3_MODEL_CONFIGS
    ],
)
@pytest.mark.parametrize("num_spec_tokens", [DEFAULT_NUM_SPEC_TOKENS])
@pytest.mark.parametrize("tp_size", get_tp_size_params())
@pytest.mark.parametrize("attention_backend", get_attention_backend_params())
def test_eagle3_acceptance_length(
    model_config: Eagle3ModelConfig,
    num_spec_tokens: int,
    tp_size: int,
    attention_backend: str,
    monkeypatch: pytest.MonkeyPatch,
):
    # Skip if this backend is incompatible with the model
    backend_enum = AttentionBackendEnum[attention_backend]
    if backend_enum in model_config.excluded_backends:
        pytest.skip(f"{attention_backend} is incompatible with {model_config.id}")

    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        with VllmRunner(
            model_name=model_config.verifier,
            speculative_config={
                "method": "eagle3",
                "model": model_config.drafter,
                "num_speculative_tokens": num_spec_tokens,
            },
            attention_config={"backend": attention_backend},
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=0.7,
            disable_log_stats=False,
            max_model_len=DEFAULT_MAX_MODEL_LEN,
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
            results = extract_acceptance_metrics(metrics, num_spec_tokens)

            actual_acceptance_length = results["acceptance_length"]
            expected = model_config.expected_acceptance_length
            actual_per_pos = results["acceptance_lengths_per_pos"]
            expected_per_pos = model_config.expected_acceptance_lengths_per_pos

            rel_error = abs(actual_acceptance_length - expected) / expected

            # Overall acceptance length always uses DEFAULT_RTOL
            assert rel_error <= DEFAULT_RTOL, (
                f"Acceptance length regression detected for {model_config.id}!\n"
                f"  Expected: {expected:.3f}\n"
                f"  Actual:   {actual_acceptance_length:.3f}\n"
                f"  Relative error: {rel_error:.2%} (tolerance: {DEFAULT_RTOL:.2%})\n"
                f"  Drafts: {results['num_drafts']}, "
                f"Accepted tokens: {results['num_accepted_tokens']}"
            )

            if expected_per_pos and len(expected_per_pos) == len(actual_per_pos):
                # Per-position checks use model-specific rtol if provided
                rtol = (
                    model_config.rtol if model_config.rtol is not None else DEFAULT_RTOL
                )
                for pos, (actual, exp) in enumerate(
                    zip(actual_per_pos, expected_per_pos)
                ):
                    if exp > 0:
                        pos_rel_error = abs(actual - exp) / exp
                        assert pos_rel_error <= rtol, (
                            f"Per-position acceptance length regression at pos {pos} "
                            f"for {model_config.id}!\n"
                            f"  Expected: {exp:.3f}\n"
                            f"  Actual:   {actual:.3f}\n"
                            f"  Relative error: {pos_rel_error:.2%} "
                            f"(tolerance: {rtol:.2%})"
                        )

            print(
                f"\n{model_config.id} [tp={tp_size}, backend={attention_backend}]: "
                f"acceptance_length={actual_acceptance_length:.3f}"
                f" (expected={expected:.3f}, rel_error={rel_error:.2%})"
            )
            print(f"  Per-position: {[f'{v:.3f}' for v in actual_per_pos]}")
            if expected_per_pos:
                print(f"  Expected:     {[f'{v:.3f}' for v in expected_per_pos]}")
