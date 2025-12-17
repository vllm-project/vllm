# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that we handle errors in request preprocessing gracefully."""

import pytest

from vllm import LLM, SamplingParams
from vllm.platforms import current_platform
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore

if not current_platform.is_cuda():
    pytest.skip(reason="V1 currently only supported on CUDA.", allow_module_level=True)

MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"


def test_preprocess_error_handling(monkeypatch: pytest.MonkeyPatch):
    """Test that preprocessing errors are handled gracefully."""

    # Store original method to call for non-failing requests
    original_preprocess = EngineCore.preprocess_add_request

    # Monkeypatch to make preprocess_add_request raise an exception
    # only for requests with "FAIL" in the first token
    def conditional_failing_preprocess(self, request: EngineCoreRequest):
        # Fail if the first token is very large (we'll use a special prompt for this)
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
    failing_prompt = TokensPrompt(prompt_token_ids=[333])  # Invalid large token
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
