# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch.cuda

from vllm import LLM, SamplingParams
from vllm.exceptions import VLLMValidationError
from vllm.lora.request import LoRARequest
from vllm.platforms import current_platform
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.input_processor import InputProcessor

MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"


@pytest.mark.parametrize("use_uno", [False, True])
@pytest.mark.parametrize("with_adapter", [False, True])
def test_uno_adapter_validation_precedes_engine_submission(use_uno, with_adapter):
    processor = object.__new__(InputProcessor)
    processor.speculative_config = SimpleNamespace(use_uno=lambda: use_uno)
    processor.lora_config = object()
    processor.renderer = SimpleNamespace(tokenizer=None)
    request = LoRARequest("user", 1, "test-adapter") if with_adapter else None
    if use_uno and with_adapter:
        with pytest.raises(VLLMValidationError, match="request-specific LoRA"):
            processor._validate_lora(request)
    else:
        processor._validate_lora(request)


def test_preprocess_error_handling(monkeypatch: pytest.MonkeyPatch):
    """Test that preprocessing errors are handled gracefully."""

    if current_platform.is_rocm() or current_platform.is_xpu():
        pytest.skip(
            "Skipped on ROCm/XPU: this test only works with 'fork', "
            "but ROCm/XPU uses 'spawn'."
        )

    assert not torch.cuda.is_initialized(), (
        "fork needs to be used for the engine "
        "core process and this isn't possible if cuda is already initialized"
    )

    # Store original method to call for non-failing requests
    original_preprocess = EngineCore.preprocess_add_request

    # Monkeypatch to make preprocess_add_request raise an exception
    # only for requests with "FAIL" in the first token
    def conditional_failing_preprocess(self, request: EngineCoreRequest):
        # Fail if the first token id is 333
        if request.prompt_token_ids and request.prompt_token_ids[0] == 333:
            raise ValueError("Simulated preprocessing error!")
        return original_preprocess(self, request)

    monkeypatch.setattr(
        EngineCore, "preprocess_add_request", conditional_failing_preprocess
    )

    llm = LLM(model=MODEL_NAME)

    # Create a failing request by crafting a request with an invalid token
    # We need to use a direct approach since LLM.generate tokenizes for us
    from vllm.inputs import TokensPrompt

    # This should raise an exception due to the preprocessing failure
    # Special token id to trigger the failure
    failing_prompt = TokensPrompt(prompt_token_ids=[333])
    outputs = llm.generate(failing_prompt, SamplingParams(max_tokens=10))  # type: ignore
    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].token_ids) == 0
    assert outputs[0].finished
    assert outputs[0].outputs[0].finish_reason == "error"

    # Verify the engine is still functional with a normal request
    outputs = llm.generate("Hello, my name is", SamplingParams(max_tokens=10))
    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].token_ids) > 0
    assert outputs[0].outputs[0].finish_reason in ("stop", "length")
