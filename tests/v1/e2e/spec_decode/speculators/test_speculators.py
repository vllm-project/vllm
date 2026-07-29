# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.utils import large_gpu_mark, single_gpu_only
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

from ..utils import evaluate_llm_for_gsm8k, get_test_prompts


@pytest.mark.parametrize(
    ["model_path", "expected_accuracy_threshold"],
    [
        ("RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3", 0.7),  # ref: 75%-80%
        ("RedHatAI/Qwen3-8B-speculator.eagle3", 0.8),  # ref: 87%-92%
    ],
    ids=["llama3_eagle3_speculator", "qwen3_eagle3_speculator"],
)
@single_gpu_only
@large_gpu_mark(min_gb=24)
def test_speculators_model_integration(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_path: str,
    expected_accuracy_threshold: float,
):
    """
    Test that speculators models work with the simplified integration.

    This verifies the `vllm serve <speculator-model>` use case where
    speculative config is automatically detected from the model config
    without requiring explicit --speculative-config argument.

    Tests:
    1. Speculator model is correctly detected
    2. Verifier model is extracted from speculator config
    3. Speculative decoding is automatically enabled
    4. Text generation works correctly
    5. GSM8k accuracy of the model passes a sanity check when speculative decoding on
    6. Output matches reference (non-speculative) generation
    """
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    # Generate test prompts
    test_prompts = get_test_prompts(mm_enabled=False)

    # First run: Direct speculator model (simplified integration)
    spec_llm = LLM(model=model_path, max_model_len=4096, gpu_memory_utilization=0.92)
    evaluate_llm_for_gsm8k(
        spec_llm, expected_accuracy_threshold=expected_accuracy_threshold
    )
    spec_outputs = spec_llm.chat(test_prompts, sampling_config)

    # Verify speculative config was auto-detected
    assert spec_llm.llm_engine.vllm_config.speculative_config is not None, (
        f"Speculative config should be auto-detected for {model_path}"
    )

    spec_config = spec_llm.llm_engine.vllm_config.speculative_config
    assert spec_config.num_speculative_tokens > 0, (
        f"Expected positive speculative tokens, "
        f"got {spec_config.num_speculative_tokens}"
    )

    # Verify draft model is set to the speculator model
    assert spec_config.model == model_path, (
        f"Draft model should be {model_path}, got {spec_config.model}"
    )

    # Extract verifier model for reference run
    verifier_model = spec_llm.llm_engine.vllm_config.model_config.model

    del spec_llm
    torch.accelerator.empty_cache()
    cleanup_dist_env_and_memory()

    # Second run: Reference without speculative decoding
    ref_llm = LLM(model=verifier_model, max_model_len=4096, gpu_memory_utilization=0.92)
    ref_outputs = ref_llm.chat(test_prompts, sampling_config)
    del ref_llm
    torch.accelerator.empty_cache()
    cleanup_dist_env_and_memory()

    # Compare outputs
    matches = sum(
        1
        for ref, spec in zip(ref_outputs, spec_outputs)
        if ref.outputs[0].text == spec.outputs[0].text
    )

    # Heuristic: expect at least 66% of prompts to match exactly
    assert matches >= int(0.66 * len(ref_outputs)), (
        f"Only {matches}/{len(ref_outputs)} outputs matched. "
        f"Expected at least {int(0.66 * len(ref_outputs))} matches."
    )
