# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses

import pytest
import torch

from tests.evals.gsm8k.gsm8k_eval import evaluate_gsm8k_offline
from tests.utils import large_gpu_mark
from vllm import LLM
from vllm.config import SpeculativeConfig
from vllm.distributed import cleanup_dist_env_and_memory


@dataclasses.dataclass
class SpeculatorTestConfig:
    model_path: str
    method: str
    display_name: str
    expected_gsm8k_accuracy: float
    accuracy_rtol: float
    expected_acceptance_len: float
    acceptance_len_rtol: float
    expected_per_pos_acceptance_rates: tuple[float, ...]
    per_pos_rtol: float
    quantization: str | None = None
    parallel_drafting: bool | None = None


DFLASH_CONFIG = SpeculatorTestConfig(
    model_path="nm-testing/dflash-qwen3-8b-speculators",
    method="dflash",
    display_name="DFlash",
    expected_gsm8k_accuracy=0.885,
    accuracy_rtol=0.03,
    expected_acceptance_len=3.45,
    acceptance_len_rtol=0.15,
    expected_per_pos_acceptance_rates=(0.795, 0.611, 0.429, 0.282),
    per_pos_rtol=0.15,
    quantization="fp8",
)

PEAGLE_CONFIG = SpeculatorTestConfig(
    model_path="nm-testing/qwen3-8b-peagle-speculators",
    method="eagle3",
    display_name="PEagle",
    expected_gsm8k_accuracy=0.88,
    accuracy_rtol=0.05,
    expected_acceptance_len=2.27,
    acceptance_len_rtol=0.20,
    expected_per_pos_acceptance_rates=(0.66, 0.36, 0.18, 0.09),
    per_pos_rtol=0.20,
    parallel_drafting=True,
)

SPECULATOR_CONFIGS = [
    pytest.param(DFLASH_CONFIG, id="dflash"),
    pytest.param(PEAGLE_CONFIG, id="peagle"),
]


def compute_spec_decode_stats(metrics) -> dict:
    """Extract all spec-decode metrics and compute derived stats."""
    name2metric = {m.name: m for m in metrics}

    n_drafts = name2metric["vllm:spec_decode_num_drafts"].value
    n_draft_tokens = name2metric["vllm:spec_decode_num_draft_tokens"].value
    n_accepted = name2metric["vllm:spec_decode_num_accepted_tokens"].value

    per_pos_vec = name2metric["vllm:spec_decode_num_accepted_tokens_per_pos"].values

    acceptance_len = 1 + (n_accepted / n_drafts) if n_drafts > 0 else 1.0
    draft_tokens_per_step = (n_draft_tokens / n_drafts) if n_drafts > 0 else 0
    overall_acceptance_rate = (n_accepted / n_draft_tokens) if n_draft_tokens > 0 else 0
    per_pos_rates = [v / n_drafts for v in per_pos_vec] if n_drafts > 0 else []

    return {
        "num_drafts": n_drafts,
        "num_draft_tokens": n_draft_tokens,
        "num_accepted_tokens": n_accepted,
        "acceptance_len": acceptance_len,
        "draft_tokens_per_step": draft_tokens_per_step,
        "overall_acceptance_rate": overall_acceptance_rate,
        "per_pos_accepted": list(per_pos_vec),
        "per_pos_acceptance_rates": per_pos_rates,
    }


def print_spec_decode_stats(stats: dict) -> None:
    """Print all spec-decode metrics and derived values."""
    print("\n===== Spec Decode Metrics =====")
    print(f"  num_drafts:              {stats['num_drafts']}")
    print(f"  num_draft_tokens:        {stats['num_draft_tokens']}")
    print(f"  num_accepted_tokens:     {stats['num_accepted_tokens']}")
    print(f"  draft_tokens_per_step:   {stats['draft_tokens_per_step']:.2f}")
    print(f"  overall_acceptance_rate: {stats['overall_acceptance_rate']:.4f}")
    print(f"  acceptance_len (1+acc/drafts): {stats['acceptance_len']:.4f}")
    print("  per-position accepted tokens:", stats["per_pos_accepted"])
    print("  per-position acceptance rates:")
    for i, rate in enumerate(stats["per_pos_acceptance_rates"]):
        print(f"    pos {i}: {rate:.4f}")
    print("===============================\n")


@pytest.mark.parametrize("config", SPECULATOR_CONFIGS)
def test_speculators_model(vllm_runner, example_prompts, monkeypatch, config):
    """
    Test speculators model properly initializes speculative decoding.

    Verifies:
    1. Speculative config is automatically initialized from speculators config
    2. Method is detected correctly
    3. parallel_drafting is set correctly (if applicable)
    4. The draft model path is correctly set
    5. Speculative tokens count is valid
    6. Text generation works with speculative decoding enabled
    """
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    runner_kwargs = dict(dtype=torch.bfloat16, enforce_eager=True)
    if config.quantization:
        runner_kwargs["quantization"] = config.quantization

    with vllm_runner(config.model_path, **runner_kwargs) as vllm_model:
        vllm_config = vllm_model.llm.llm_engine.vllm_config

        assert isinstance(vllm_config.speculative_config, SpeculativeConfig), (
            "Speculative config should be initialized for speculators model"
        )

        spec_config = vllm_config.speculative_config
        assert spec_config.method == config.method, (
            f"Expected method='{config.method}', got '{spec_config.method}'"
        )
        if config.parallel_drafting is not None:
            assert spec_config.parallel_drafting is config.parallel_drafting, (
                f"Expected parallel_drafting={config.parallel_drafting} "
                f"for {config.display_name} model"
            )
        assert spec_config.num_speculative_tokens > 0, (
            f"Expected positive speculative tokens, "
            f"got {spec_config.num_speculative_tokens}"
        )
        assert spec_config.model == config.model_path, (
            f"Draft model should be {config.model_path}, got {spec_config.model}"
        )

        vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens=20)
        assert vllm_outputs, (
            f"No outputs generated for speculators model {config.model_path}"
        )


@pytest.mark.slow_test
@large_gpu_mark(min_gb=40)
@pytest.mark.parametrize("config", SPECULATOR_CONFIGS)
def test_speculators_correctness(monkeypatch, config):
    """
    E2E correctness test via the speculators auto-detect path.

    Evaluates GSM8k accuracy to ensure the speculators-format model produces
    correct outputs, and checks that acceptance length does not collapse under
    batched inference (lm-eval style).
    """
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    spec_llm = LLM(
        model=config.model_path,
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=128,
        gpu_memory_utilization=0.85,
        enforce_eager=False,
        disable_log_stats=False,
    )

    results = evaluate_gsm8k_offline(spec_llm)
    accuracy = results["accuracy"]
    accuracy_threshold = config.expected_gsm8k_accuracy * (1 - config.accuracy_rtol)
    assert accuracy >= accuracy_threshold, (
        f"Expected GSM8K accuracy >= {accuracy_threshold:.3f}, got {accuracy:.3f}"
    )

    current_metrics = spec_llm.get_metrics()
    stats = compute_spec_decode_stats(current_metrics)
    print_spec_decode_stats(stats)

    acceptance_len = stats["acceptance_len"]
    al_threshold = config.expected_acceptance_len * (1 - config.acceptance_len_rtol)
    assert acceptance_len >= al_threshold, (
        f"{config.display_name} speculators acceptance length too low: "
        f"{acceptance_len:.2f} < {al_threshold:.2f}"
    )

    per_pos_rates = stats["per_pos_acceptance_rates"]
    for i, expected_rate in enumerate(config.expected_per_pos_acceptance_rates):
        assert i < len(per_pos_rates), (
            f"Missing per-position acceptance rate for position {i}"
        )
        threshold = expected_rate * (1 - config.per_pos_rtol)
        assert per_pos_rates[i] >= threshold, (
            f"Per-position acceptance rate at pos {i} too low: "
            f"{per_pos_rates[i]:.4f} < {threshold:.4f} "
            f"(expected ~{expected_rate:.4f})"
        )

    del spec_llm
    torch.accelerator.empty_cache()
    cleanup_dist_env_and_memory()
