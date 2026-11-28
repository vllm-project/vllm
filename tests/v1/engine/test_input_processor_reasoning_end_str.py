# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frontend validation of the per-request `reasoning_end_str` override."""

import pytest

from vllm import SamplingParams
from vllm.config import ModelConfig, VllmConfig
from vllm.config.reasoning import ReasoningConfig
from vllm.exceptions import VLLMValidationError
from vllm.inputs.engine import tokens_input
from vllm.v1.engine.input_processor import InputProcessor

pytestmark = pytest.mark.skip_global_cleanup

MODEL = "Qwen/Qwen3-0.6B"
THINK_END_ID = 151668  # `</think>` in the Qwen3 vocabulary


def _processor(reasoning_config: ReasoningConfig | None) -> InputProcessor:
    return InputProcessor(
        VllmConfig(
            model_config=ModelConfig(model=MODEL, max_model_len=256),
            reasoning_config=reasoning_config,
        )
    )


@pytest.fixture(scope="module")
def processor() -> InputProcessor:
    return _processor(ReasoningConfig(reasoning_parser="qwen3"))


def _process(
    processor: InputProcessor, reasoning_end_str: str, **kwargs
) -> SamplingParams:
    params = SamplingParams(
        thinking_token_budget=8, reasoning_end_str=reasoning_end_str, **kwargs
    )
    request = processor.process_inputs(
        "req", tokens_input([1, 2, 3]), params, ("generate",)
    )
    assert request.sampling_params is not None
    return request.sampling_params


def test_reasoning_end_str_is_tokenized(processor):
    token_ids = _process(processor, "Time to answer.</think>").reasoning_end_token_ids
    assert token_ids is not None
    assert token_ids[-1] == THINK_END_ID
    assert THINK_END_ID not in token_ids[:-1]


@pytest.mark.parametrize(
    "reasoning_end_str,error",
    [
        ("Time to answer.", "must end with the reasoning end marker"),
        # Forcing would stop at the first marker.
        ("</think>Time to answer.</think>", "must end with the reasoning end marker"),
        # A forced start marker restarts the budget: reasoning never closes.
        ("<think>Time to answer.</think>", "must not contain the reasoning start"),
        ("go " * 200 + "</think>", "maximum is"),
    ],
)
def test_reasoning_end_str_is_rejected(processor, reasoning_end_str, error):
    with pytest.raises(VLLMValidationError, match=error):
        _process(processor, reasoning_end_str)


def test_reasoning_end_str_rejects_tokens_that_end_the_request(processor):
    """A forced EOS or stop token would end the request inside the reasoning."""
    with pytest.raises(VLLMValidationError, match="ends the request"):
        _process(processor, "Done.<|im_end|></think>")
    first_id = processor.tokenizer.encode("Time to answer.</think>")[0]
    with pytest.raises(VLLMValidationError, match="ends the request"):
        _process(processor, "Time to answer.</think>", stop_token_ids=[first_id])

    # With ignore_eos the EOS token does not stop the request.
    token_ids = _process(
        processor, "Done.<|im_end|></think>", ignore_eos=True
    ).reasoning_end_token_ids
    assert token_ids is not None
    assert token_ids[-1] == THINK_END_ID


def test_reasoning_end_str_rejects_stop_token_inside_multi_token_marker():
    """Without a parser the marker is the server's whole string, here several
    tokens: a stop token inside it ends the request before it is complete."""
    processor = _processor(
        ReasoningConfig(
            reasoning_start_str="<think>", reasoning_end_str="Done.</think>"
        )
    )
    assert processor.tokenizer is not None
    marker = processor.tokenizer.encode("Done.</think>", add_special_tokens=False)
    assert len(marker) > 1
    phrase = "Time to answer.\nDone.</think>"
    token_ids = _process(processor, phrase).reasoning_end_token_ids
    assert token_ids is not None and token_ids[-len(marker) :] == marker
    with pytest.raises(VLLMValidationError, match="ends the request"):
        _process(processor, phrase, stop_token_ids=[marker[0]])
    # Stopping right after the last forced token is allowed.
    _process(processor, phrase, stop_token_ids=[marker[-1]])


def test_reasoning_end_str_requires_reasoning_config():
    with pytest.raises(VLLMValidationError, match="reasoning_config is not"):
        _process(_processor(None), "Time to answer.</think>")


@pytest.mark.parametrize("reasoning_end_str", ["", "a" * 1024 + "</think>"])
def test_reasoning_end_str_length_is_checked_before_tokenization(reasoning_end_str):
    with pytest.raises(VLLMValidationError, match="non-empty string of at most"):
        SamplingParams(reasoning_end_str=reasoning_end_str)
