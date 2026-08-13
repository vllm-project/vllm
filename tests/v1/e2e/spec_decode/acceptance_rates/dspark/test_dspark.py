# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import pytest

from tests.evals.gsm8k.gsm8k_eval import (
    assert_min_accuracy,
    evaluate_gsm8k_offline,
)
from vllm.config import CompilationConfig
from vllm.platforms import current_platform

from ...utils import compute_acceptance_len, compute_acceptance_rate

REGRESSION_TOLERANCE = 0.95


@dataclass(frozen=True)
class DSparkCorrectnessConfig:
    model: str
    draft_model: str
    reference_accuracy: float
    reference_acceptance_rate: float
    reference_acceptance_len: float
    num_speculative_tokens: int = 7
    max_model_len: int = 4096
    max_num_seqs: int | None = None
    num_questions: int = 1319
    max_tokens: int = 256
    use_chat_completions: bool = False
    chat_template_kwargs: dict[str, object] | None = None
    attention_backend: str | None = None
    gpu_memory_utilization: float = 0.92
    enforce_eager: bool = False
    disable_flashinfer_sampler: bool = False
    language_model_only: bool = False


# References from 12 full GSM8K runs at temperature 1.0: accuracy 0.782-0.814,
# acceptance rate 0.418-0.434, acceptance length 3.928-4.037.
QWEN3_DSPARK_DEEPSPEC = DSparkCorrectnessConfig(
    model="Qwen/Qwen3-4B-FP8",
    draft_model="deepseek-ai/dspark_qwen3_4b_block7",
    reference_accuracy=0.801,
    reference_acceptance_rate=0.428,
    reference_acceptance_len=3.994,
    attention_backend="FLASH_ATTN",
)

# References from five 200-question runs at temperature 1.0: accuracy
# 0.900-0.955, acceptance rate 0.578-0.595, acceptance length 5.044-5.167.
GEMMA4_DSPARK_DEEPSPEC = DSparkCorrectnessConfig(
    model="RedHatAI/gemma-4-12B-it-NVFP4",
    draft_model="deepseek-ai/dspark_gemma4_12b_block7",
    reference_accuracy=0.92,
    reference_acceptance_rate=0.58,
    reference_acceptance_len=5.0,
    max_num_seqs=32,
    num_questions=200,
    use_chat_completions=True,
    gpu_memory_utilization=0.85,
    enforce_eager=True,
    disable_flashinfer_sampler=True,
    language_model_only=True,
)

# Reference from 200 GSM8K questions at temperature 1.0.
QWEN3_6_DSPARK_SPECULATORS = DSparkCorrectnessConfig(
    model="RedHatAI/Qwen3.6-35B-A3B-NVFP4",
    draft_model="RedHatAI/Qwen3.6-35B-A3B-speculator.dspark",
    reference_accuracy=0.85,
    reference_acceptance_rate=0.41,
    reference_acceptance_len=4.25,
    num_speculative_tokens=8,
    max_num_seqs=32,
    num_questions=200,
    use_chat_completions=True,
    chat_template_kwargs={"enable_thinking": False},
    gpu_memory_utilization=0.85,
    language_model_only=True,
)


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(QWEN3_DSPARK_DEEPSPEC, id="qwen3-deepspec"),
        pytest.param(GEMMA4_DSPARK_DEEPSPEC, id="gemma4-deepspec"),
        pytest.param(QWEN3_6_DSPARK_SPECULATORS, id="qwen3.6-speculators"),
    ],
)
def test_dspark_correctness_and_acceptance_rate(
    monkeypatch: pytest.MonkeyPatch,
    config: DSparkCorrectnessConfig,
    vllm_runner,
):
    """Guard GSM8K accuracy and acceptance metrics for DSpark models."""
    if config.disable_flashinfer_sampler:
        monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")

    speculative_config = {
        "method": "dspark",
        "model": config.draft_model,
        "num_speculative_tokens": config.num_speculative_tokens,
        "draft_sample_method": "probabilistic",
    }
    if config.attention_backend is not None and current_platform.is_cuda():
        speculative_config["attention_backend"] = config.attention_backend

    runner_config = {
        "block_size": None,
        "trust_remote_code": True,
        "speculative_config": speculative_config,
        "max_model_len": config.max_model_len,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "enforce_eager": config.enforce_eager,
        "enable_chunked_prefill": None,
        "enable_prefix_caching": False,
        "disable_log_stats": False,
        "compilation_config": CompilationConfig(),
    }
    if config.max_num_seqs is not None:
        runner_config["max_num_seqs"] = config.max_num_seqs
    if config.language_model_only:
        runner_config["language_model_only"] = True

    with vllm_runner(config.model, **runner_config) as spec_runner:
        spec_llm = spec_runner.llm
        results = evaluate_gsm8k_offline(
            spec_llm,
            num_questions=config.num_questions,
            max_tokens=config.max_tokens,
            temperature=1.0,
            use_chat_completions=config.use_chat_completions,
            chat_template_kwargs=config.chat_template_kwargs,
        )
        accuracy = results.accuracy
        metrics = spec_llm.get_metrics()
        acceptance_rate = compute_acceptance_rate(metrics)
        acceptance_len = compute_acceptance_len(metrics)
        context = f"DSpark target={config.model}, draft={config.draft_model}"
        metrics_summary = (
            f"gsm8k_accuracy={accuracy:.3f}, "
            f"acceptance_rate={acceptance_rate:.3f}, "
            f"acceptance_len={acceptance_len:.3f}"
        )
        print(f"{context}: {metrics_summary}")

        assert_min_accuracy(
            results,
            config.reference_accuracy * REGRESSION_TOLERANCE,
            context=context,
        )
        assert (
            acceptance_rate >= config.reference_acceptance_rate * REGRESSION_TOLERANCE
        ), f"{context}: {metrics_summary}"
        assert (
            acceptance_len >= config.reference_acceptance_len * REGRESSION_TOLERANCE
        ), f"{context}: {metrics_summary}"
