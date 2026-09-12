# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import pytest

from tests.evals.gsm8k.gsm8k_eval import evaluate_gsm8k_offline
from vllm.config import CompilationConfig

from ...utils import compute_acceptance_len, compute_acceptance_rate

REGRESSION_TOLERANCE = 0.95


@dataclass(frozen=True)
class XPressCorrectnessConfig:
    model: str
    draft_model: str
    reference_accuracy: float
    reference_acceptance_rate: float
    reference_acceptance_len: float
    num_speculative_tokens: int = 15
    max_model_len: int = 4096
    max_num_seqs: int | None = None
    num_questions: int = 1319
    max_tokens: int = 256
    gpu_memory_utilization: float = 0.92
    enforce_eager: bool = False


# References from full GSM8K runs at temperature 0 on one H200: accuracy 0.889,
# acceptance rate 0.310, acceptance length 5.654.
QWEN3_XPRESS = XPressCorrectnessConfig(
    model="Qwen/Qwen3-8B",
    draft_model="UIUC-SSAIL/Qwen3-8B-XPress-b16",
    reference_accuracy=0.889,
    reference_acceptance_rate=0.310,
    reference_acceptance_len=5.654,
)


@pytest.mark.parametrize("config", [QWEN3_XPRESS], ids=["qwen3_xpress"])
def test_xpress_correctness_and_acceptance_rate(
    config: XPressCorrectnessConfig,
    vllm_runner,
):
    """Guard GSM8K accuracy and acceptance metrics for XPress models.

    Accuracy is the check that a refiner producing garbage cannot pass: NaN
    logits argmax to a fixed token, which the verifier then accepts every time,
    so acceptance alone would read as a perfect score. The upper bound on
    acceptance length trips on that same pathology directly.
    """
    speculative_config = {
        "method": "xpress",
        "model": config.draft_model,
        "num_speculative_tokens": config.num_speculative_tokens,
    }
    runner_config = {
        "block_size": None,
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

    with vllm_runner(config.model, **runner_config) as spec_runner:
        spec_llm = spec_runner.llm
        results = evaluate_gsm8k_offline(
            spec_llm,
            num_questions=config.num_questions,
            max_tokens=config.max_tokens,
            temperature=0.0,
        )
        accuracy = results["accuracy"]
        metrics = spec_llm.get_metrics()
        acceptance_rate = compute_acceptance_rate(metrics)
        acceptance_len = compute_acceptance_len(metrics)
        context = f"XPress target={config.model}, draft={config.draft_model}"
        metrics_summary = (
            f"gsm8k_accuracy={accuracy:.3f}, "
            f"acceptance_rate={acceptance_rate:.3f}, "
            f"acceptance_len={acceptance_len:.3f}"
        )
        print(f"{context}: {metrics_summary}")

        assert accuracy >= config.reference_accuracy * REGRESSION_TOLERANCE, (
            f"{context}: {metrics_summary}"
        )
        assert (
            acceptance_rate >= config.reference_acceptance_rate * REGRESSION_TOLERANCE
        ), f"{context}: {metrics_summary}"
        assert (
            acceptance_len >= config.reference_acceptance_len * REGRESSION_TOLERANCE
        ), f"{context}: {metrics_summary}"
        # Every draft token accepted on every step is not a great drafter, it is a
        # verifier rubber-stamping degenerate output.
        assert acceptance_len < config.num_speculative_tokens + 1, (
            f"{context}: {metrics_summary}"
        )
