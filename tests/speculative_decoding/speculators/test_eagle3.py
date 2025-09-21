# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.config import SpeculativeConfig
from vllm.model_executor.models.interfaces import supports_eagle3


@pytest.mark.parametrize("model_path", [
    pytest.param(
        "nm-testing/SpeculatorLlama3-1-8B-Eagle3-converted-0717-quantized",
        id="llama3-eagle3-speculator"),
    pytest.param(
        "nm-testing/Speculator-Qwen3-8B-Eagle3-converted-071-quantized",
        id="qwen3-eagle3-speculator"),
])
def test_eagle3_speculators_model(vllm_runner, example_prompts, model_path,
                                  monkeypatch):
    """
    Test Eagle3 speculators models properly initialize speculative decoding.

    This test verifies:
    1. Eagle3 support is detected for the model
    2. Speculative config is automatically initialized from embedded config
    3. The draft model path is correctly set to the speculators model
    4. Speculative tokens count is valid
    5. Text generation works with speculative decoding enabled
    """
    # Set environment variable for V1 engine serialization
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    with vllm_runner(model_path, dtype=torch.bfloat16) as vllm_model:
        # Verify Eagle3 support is detected
        eagle3_supported = vllm_model.apply_model(supports_eagle3)
        assert eagle3_supported, f"Eagle3 should be supported for {model_path}"

        vllm_config = vllm_model.llm.llm_engine.vllm_config

        assert isinstance(vllm_config.speculative_config, SpeculativeConfig), \
            "Speculative config should be initialized for speculators model"

        spec_config = vllm_config.speculative_config
        assert spec_config.num_speculative_tokens > 0, \
            (f"Expected positive speculative tokens, "
             f"got {spec_config.num_speculative_tokens}")

        assert spec_config.model == model_path, \
            f"Draft model should be {model_path}, got {spec_config.model}"

        vllm_outputs = vllm_model.generate_greedy(example_prompts,
                                                  max_tokens=20)
        assert vllm_outputs, \
            f"No outputs generated for speculators model {model_path}"
