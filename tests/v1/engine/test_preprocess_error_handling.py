# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import queue
import threading
from unittest.mock import MagicMock

import msgspec
import pytest
import torch.cuda

from vllm import LLM, SamplingParams
from vllm.platforms import current_platform
from vllm.v1.engine import EngineCoreRequest, EngineCoreRequestType
from vllm.v1.engine.core import EngineCore, EngineCoreProc, _decode_add_request
from vllm.v1.serial_utils import MsgpackDecoder

MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"


def test_decode_error_does_not_stop_input_processing(
    caplog: pytest.LogCaptureFixture,
):
    """The ADD decoder remains usable after dropping a malformed frame."""
    malformed_request = msgspec.msgpack.encode(
        ["bad", [1.5], None, None, None, 0.0, None, None, None]
    )
    valid_request = msgspec.msgpack.encode(
        ["good", [1], None, None, None, 0.0, None, None, None]
    )
    decoder = MsgpackDecoder(EngineCoreRequest)

    with caplog.at_level(logging.ERROR, logger="vllm.v1.engine.core"):
        assert _decode_add_request(decoder, [malformed_request]) is None

    request = _decode_add_request(decoder, [valid_request])
    assert request is not None
    assert request.request_id == "good"
    assert request.prompt_token_ids == [1]
    assert "Failed to deserialize ADD request" in caplog.text


def test_input_socket_thread_failure_is_reported():
    """An unexpected input-thread failure must reach the engine busy loop."""
    engine_core = EngineCoreProc.__new__(EngineCoreProc)
    engine_core.input_queue = queue.Queue()
    engine_core.process_input_sockets = MagicMock(
        side_effect=RuntimeError("socket failure")
    )

    engine_core._run_input_socket_thread([], None, b"engine", threading.Event())

    request_type, request = engine_core.input_queue.get_nowait()
    assert request_type == EngineCoreRequestType.INPUT_THREAD_FAILED
    assert request is None

    with pytest.raises(RuntimeError, match="Input socket thread failed"):
        EngineCoreProc._handle_client_request(
            engine_core, EngineCoreRequestType.INPUT_THREAD_FAILED, None
        )


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
