# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.utils import single_gpu_only
from vllm import SamplingParams
from vllm.config import CompilationConfig

from ..utils import check_spec_decode_matches_reference, get_test_prompts


@pytest.mark.parametrize(
    ["model_path", "expected_target_model", "expected_accuracy_threshold"],
    [
        # Measured reference: 75%-80%.
        (
            "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3",
            "meta-llama/Llama-3.1-8B-Instruct",
            0.72,
        ),
        # Measured reference: 87%-92%.
        ("RedHatAI/Qwen3-8B-speculator.eagle3", "Qwen/Qwen3-8B", 0.84),
    ],
    ids=["llama3_eagle3_speculator", "qwen3_eagle3_speculator"],
)
@single_gpu_only
def test_speculators_model_integration(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_path: str,
    expected_target_model: str,
    expected_accuracy_threshold: float,
    vllm_runner,
):
    """Test that speculators models work with the simplified integration.

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

    test_prompts = get_test_prompts(mm_enabled=False)
    engine_kwargs = dict(
        block_size=None,
        trust_remote_code=False,
        enable_chunked_prefill=None,
        compilation_config=CompilationConfig(),
        max_model_len=4096,
        gpu_memory_utilization=0.92,
    )
    check_spec_decode_matches_reference(
        vllm_runner,
        sampling_config,
        test_prompts,
        target_model=None,
        target_engine_kwargs=engine_kwargs,
        spec_model=model_path,
        spec_engine_kwargs=engine_kwargs,
        # Heuristic: expect at least 66% of prompts to match exactly
        prompts_required_matches=int(0.66 * len(test_prompts)),
        gsm8k_spec_accuracy_threshold=expected_accuracy_threshold,
        expected_target_model=expected_target_model,
    )
