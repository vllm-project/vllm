# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from tests.evals.gsm8k.gsm8k_eval import evaluate_gsm8k_offline
from tests.utils import large_gpu_mark
from vllm import LLM
from vllm.config import SpeculativeConfig
from vllm.distributed import cleanup_dist_env_and_memory

MODEL_PATH = "nm-testing/dflash-qwen3-8b-speculators"

EXPECTED_GSM8K_ACCURACY = 0.885
ACCURACY_RTOL = 0.03
EXPECTED_ACCEPTANCE_LEN = 3.45
ACCEPTANCE_LEN_RTOL = 0.15

# Expected per-position acceptance rates (accepted_at_pos / num_drafts)
# Based on GSM8K evaluation with Qwen3-8B dflash speculators.
EXPECTED_PER_POS_ACCEPTANCE_RATES = [0.795, 0.611, 0.429, 0.282]
PER_POS_RTOL = 0.15


def compute_spec_decode_stats(
    metrics,
) -> dict:
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


def test_dflash_speculators_model(vllm_runner, example_prompts, monkeypatch):
    """
    Test DFlash speculators model properly initializes speculative decoding.

    Verifies:
    1. Speculative config is automatically initialized from speculators config
    2. Method is detected as 'dflash'
    3. The draft model path is correctly set
    4. Speculative tokens count is valid (num_speculative_tokens=8)
    5. Text generation works with speculative decoding enabled
    """
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    with vllm_runner(
        MODEL_PATH,
        dtype=torch.bfloat16,
        enforce_eager=True,
        quantization="fp8",
    ) as vllm_model:
        vllm_config = vllm_model.llm.llm_engine.vllm_config

        assert isinstance(vllm_config.speculative_config, SpeculativeConfig), (
            "Speculative config should be initialized for speculators model"
        )

        spec_config = vllm_config.speculative_config
        assert spec_config.method == "dflash", (
            f"Expected method='dflash', got '{spec_config.method}'"
        )
        assert spec_config.num_speculative_tokens > 0, (
            f"Expected positive speculative tokens, "
            f"got {spec_config.num_speculative_tokens}"
        )
        assert spec_config.model == MODEL_PATH, (
            f"Draft model should be {MODEL_PATH}, got {spec_config.model}"
        )

        vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens=20)
        assert vllm_outputs, f"No outputs generated for speculators model {MODEL_PATH}"


@pytest.mark.slow_test
@large_gpu_mark(min_gb=40)
def test_dflash_speculators_correctness(monkeypatch):
    """
    E2E correctness test for DFlash via the speculators auto-detect path.

    Evaluates GSM8k accuracy to ensure the speculators-format model produces
    correct outputs, and checks that acceptance length does not collapse under
    batched inference (lm-eval style).

    Observed per-position acceptance rates on GSM8K (1319 prompts):
        pos 0: 0.795, pos 1: 0.611, pos 2: 0.429, pos 3: 0.282,
        pos 4: 0.169, pos 5: 0.093, pos 6: 0.048, pos 7: 0.023
    Observed mean AL: 3.45 (GSM8K dataset, max_num_seqs=128)
    """
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    spec_llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=128,
        gpu_memory_utilization=0.85,
        enforce_eager=False,
        disable_log_stats=False,
    )

    results = evaluate_gsm8k_offline(spec_llm)
    accuracy = results["accuracy"]
    accuracy_threshold = EXPECTED_GSM8K_ACCURACY * (1 - ACCURACY_RTOL)
    assert accuracy >= accuracy_threshold, (
        f"Expected GSM8K accuracy >= {accuracy_threshold:.3f}, got {accuracy:.3f}"
    )

    current_metrics = spec_llm.get_metrics()
    stats = compute_spec_decode_stats(current_metrics)
    print_spec_decode_stats(stats)

    acceptance_len = stats["acceptance_len"]
    al_threshold = EXPECTED_ACCEPTANCE_LEN * (1 - ACCEPTANCE_LEN_RTOL)
    assert acceptance_len >= al_threshold, (
        f"DFlash speculators acceptance length too low: "
        f"{acceptance_len:.2f} < {al_threshold:.2f}"
    )

    # Check per-position acceptance rates for the first few positions.
    per_pos_rates = stats["per_pos_acceptance_rates"]
    for i, expected_rate in enumerate(EXPECTED_PER_POS_ACCEPTANCE_RATES):
        assert i < len(per_pos_rates), (
            f"Missing per-position acceptance rate for position {i}"
        )
        threshold = expected_rate * (1 - PER_POS_RTOL)
        assert per_pos_rates[i] >= threshold, (
            f"Per-position acceptance rate at pos {i} too low: "
            f"{per_pos_rates[i]:.4f} < {threshold:.4f} "
            f"(expected ~{expected_rate:.4f})"
        )

    del spec_llm
    torch.accelerator.empty_cache()
    cleanup_dist_env_and_memory()
