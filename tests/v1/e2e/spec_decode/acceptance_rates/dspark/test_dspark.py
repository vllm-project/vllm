# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, replace
from typing import Literal

import pytest
import torch

from tests.evals.gsm8k.gsm8k_eval import evaluate_gsm8k_offline
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
    dspark_draft_topk: int | None = None
    draft_sample_method: Literal["greedy", "probabilistic"] = "probabilistic"


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
        pytest.param(
            replace(QWEN3_DSPARK_DEEPSPEC, dspark_draft_topk=32),
            id="qwen3-deepspec-topk",
        ),
        pytest.param(
            replace(
                QWEN3_DSPARK_DEEPSPEC,
                dspark_draft_topk=32,
                draft_sample_method="greedy",
            ),
            id="qwen3-deepspec-topk-greedy",
        ),
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
    if config.dspark_draft_topk is not None:
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
        monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    reference = (
        config.reference_accuracy,
        config.reference_acceptance_rate,
        config.reference_acceptance_len,
    )
    if config.draft_sample_method == "greedy":
        reference = _evaluate_dspark(
            replace(config, dspark_draft_topk=None), vllm_runner
        )
    actual = _evaluate_dspark(config, vllm_runner)
    for metric, value, baseline in zip(
        ("accuracy", "acceptance_rate", "acceptance_len"), actual, reference
    ):
        assert value >= baseline * REGRESSION_TOLERANCE, (
            f"{metric}: {value:.3f}, reference={baseline:.3f}; {config}"
        )


@torch.inference_mode()
def _check_topk_path(worker) -> bool:
    from vllm.v1.worker.gpu.spec_decode.dspark.topk_markov import can_fuse_markov_walk

    spec = worker.model_runner.speculator
    k = spec.speculative_config.dspark_draft_topk
    assert spec._draft_topk == k
    assert spec._markov_candidate_ids.shape == (spec.vocab_size, k)
    assert spec._markov_candidate_biases.shape == (spec.vocab_size, k)
    if spec.speculative_config.draft_sample_method == "greedy":
        assert spec.draft_logits is None
        assert spec.acceptance_estimator is None and not spec.use_confidence_head
        if current_platform.is_cuda():
            hidden = torch.zeros(
                1, spec.hidden_size, dtype=spec.dtype, device=spec.device
            )
            logits = spec.model.compute_draft_logits(hidden)
            assert can_fuse_markov_walk(spec.model.model.markov_head, logits, k)
    return True


def _evaluate_dspark(
    config: DSparkCorrectnessConfig, vllm_runner
) -> tuple[float, float, float]:
    speculative_config = {
        "method": "dspark",
        "model": config.draft_model,
        "num_speculative_tokens": config.num_speculative_tokens,
        "draft_sample_method": config.draft_sample_method,
    }
    if config.dspark_draft_topk is not None:
        speculative_config["dspark_draft_topk"] = config.dspark_draft_topk
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
        if config.dspark_draft_topk is not None:
            assert all(spec_llm.collective_rpc(_check_topk_path))
        results = evaluate_gsm8k_offline(
            spec_llm,
            num_questions=config.num_questions,
            max_tokens=config.max_tokens,
            temperature=1.0,
            use_chat_completions=config.use_chat_completions,
            chat_template_kwargs=config.chat_template_kwargs,
        )
        accuracy = results["accuracy"]
        metrics = spec_llm.get_metrics()
        acceptance_rate = compute_acceptance_rate(metrics)
        acceptance_len = compute_acceptance_len(metrics)
        context = (
            f"DSpark target={config.model}, draft={config.draft_model}, "
            f"topk={config.dspark_draft_topk}, method={config.draft_sample_method}"
        )
        metrics_summary = (
            f"gsm8k_accuracy={accuracy:.3f}, "
            f"acceptance_rate={acceptance_rate:.3f}, "
            f"acceptance_len={acceptance_len:.3f}"
        )
        print(f"{context}: {metrics_summary}")

        assert acceptance_rate > 0 and acceptance_len > 1, metrics_summary
        return accuracy, acceptance_rate, acceptance_len
