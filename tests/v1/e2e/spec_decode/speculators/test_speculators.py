# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.evals.gsm8k.gsm8k_eval import GSM8KEvalSpec, get_gsm8k_eval_spec
from tests.utils import single_gpu_only
from vllm import SamplingParams
from vllm.config import CompilationConfig

from ..utils import (
    assert_request_outputs_match,
    evaluate_llm_for_gsm8k,
    get_test_prompts,
)


@pytest.mark.parametrize(
    "gsm8k_spec",
    [
        # Measured reference: 75%-80%.
        get_gsm8k_eval_spec("spec_decode_speculators", "llama3-eagle3"),
        # Measured reference: 87%-92%.
        get_gsm8k_eval_spec("spec_decode_speculators", "qwen3-eagle3"),
    ],
    ids=["llama3_eagle3_speculator", "qwen3_eagle3_speculator"],
)
@single_gpu_only
def test_speculators_model_integration(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    gsm8k_spec: GSM8KEvalSpec,
    vllm_runner,
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
    assert gsm8k_spec.model is not None
    model_path = gsm8k_spec.model

    # Generate test prompts
    test_prompts = get_test_prompts(mm_enabled=False)

    # First run: Direct speculator model (simplified integration)
    with vllm_runner(
        model_path,
        block_size=None,
        trust_remote_code=False,
        enable_chunked_prefill=None,
        compilation_config=CompilationConfig(),
        max_model_len=4096,
        gpu_memory_utilization=0.92,
    ) as spec_runner:
        evaluate_llm_for_gsm8k(spec_runner.llm, gsm8k_spec)
        spec_outputs = spec_runner.llm.chat(test_prompts, sampling_config)

        # Verify speculative config was auto-detected
        assert spec_runner.llm.llm_engine.vllm_config.speculative_config is not None, (
            f"Speculative config should be auto-detected for {model_path}"
        )

        spec_config = spec_runner.llm.llm_engine.vllm_config.speculative_config
        assert spec_config.num_speculative_tokens > 0, (
            f"Expected positive speculative tokens, "
            f"got {spec_config.num_speculative_tokens}"
        )

        # Verify draft model is set to the speculator model
        assert spec_config.model == model_path, (
            f"Draft model should be {model_path}, got {spec_config.model}"
        )

        # Extract verifier model for reference run
        verifier_model = spec_runner.llm.llm_engine.vllm_config.model_config.model

    # Second run: Reference without speculative decoding
    with vllm_runner(
        verifier_model,
        block_size=None,
        trust_remote_code=False,
        enable_chunked_prefill=None,
        compilation_config=CompilationConfig(),
        max_model_len=4096,
        gpu_memory_utilization=0.92,
    ) as ref_runner:
        ref_outputs = ref_runner.llm.chat(test_prompts, sampling_config)

    # Heuristic: expect at least 66% of prompts to match exactly
    assert_request_outputs_match(
        ref_outputs,
        spec_outputs,
        required_matches=int(0.66 * len(ref_outputs)),
        context=f"speculator={model_path}, verifier={verifier_model}",
    )
