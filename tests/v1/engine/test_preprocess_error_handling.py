# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.cuda

from vllm import LLM, RequestOutput, SamplingParams
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.platforms import current_platform
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore

MODEL_NAME = "vllm/tiny-random-LlamaForCausalLM"


def test_preprocess_error_handling(monkeypatch: pytest.MonkeyPatch):
    """Test that preprocessing errors are handled gracefully."""
    if current_platform.is_xpu():
        pytest.skip(
            "Skipped on XPU: this test only works with 'fork', but XPU uses 'spawn'."
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
        if request.prompt_token_ids is not None and request.prompt_token_ids[0] == 333:
            raise ValueError("Simulated preprocessing error!")
        return original_preprocess(self, request)

    monkeypatch.setattr(
        EngineCore, "preprocess_add_request", conditional_failing_preprocess
    )
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "fork")

    llm = LLM(model=MODEL_NAME)

    # Create a failing request by crafting a request with an invalid token
    # We need to use a direct approach since LLM.generate tokenizes for us
    from vllm.inputs import TokensPrompt

    # This should raise an exception due to the preprocessing failure
    # Special token id to trigger the failure
    failing_prompt = TokensPrompt(prompt_token_ids=[333])
    outputs = llm.generate(failing_prompt, SamplingParams(max_tokens=10))
    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].token_ids) == 0
    assert outputs[0].finished
    assert outputs[0].outputs[0].finish_reason == "error"

    # Verify the engine is still functional with a normal request
    outputs = llm.generate("Hello, my name is", SamplingParams(max_tokens=10))
    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].token_ids) > 0
    assert outputs[0].outputs[0].finish_reason in ("stop", "length")


def test_malformed_placeholder_range_fails_only_its_request(
    monkeypatch: pytest.MonkeyPatch,
):
    """An engine client can send a placeholder range that vLLM's frontends would
    never build. The engine must end that request with an error and keep
    serving, instead of failing in the scheduler."""
    # The engine core process runs the validation under test.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    llm = LLM(model=MODEL_NAME)

    empty_mask = torch.zeros(0, dtype=torch.bool)
    feature = MultiModalFeatureSpec(
        data=None,
        modality="image",
        identifier="item",
        mm_position=PlaceholderRange(offset=0, length=2, is_embed=empty_mask),
    )
    request = EngineCoreRequest(
        request_id="malformed",
        prompt_token_ids=[1, 2, 3],
        mm_features=[feature],
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    engine = llm.llm_engine
    engine.add_request(request.request_id, request, request.params)
    finish_reasons: list[str | None] = []
    while engine.has_unfinished_requests():
        for output in engine.step():
            assert isinstance(output, RequestOutput)
            if output.finished:
                finish_reasons.append(output.outputs[0].finish_reason)
    assert finish_reasons == ["error"]

    greedy = SamplingParams(temperature=0.0, max_tokens=10)
    outputs = llm.generate("Hello, my name is", greedy)
    assert outputs[0].outputs[0].finish_reason in ("stop", "length")
