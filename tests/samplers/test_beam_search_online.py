# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import CompletionOutput, RequestOutput
from vllm.entrypoints.generate.beam_search.online import BeamSearchOnlineMixin
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.logprobs import Logprob
from vllm.sampling_params import BeamSearchParams

pytestmark = pytest.mark.cpu_test


class _Tokenizer:
    eos_token_id = 0

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


class _Renderer:
    def get_tokenizer(self) -> _Tokenizer:
        return _Tokenizer()


class _EngineClient:
    def __init__(self, token_logprobs: dict[int, float] | None = None) -> None:
        if token_logprobs is None:
            token_logprobs = {
                11: -1.0,
                12: -2.0,
                13: -3.0,
                14: -4.0,
                _Tokenizer.eos_token_id: -0.1,
            }
        self.logprobs = {
            token_id: Logprob(logprob=logprob)
            for token_id, logprob in token_logprobs.items()
        }
        self.num_calls = 0

    async def generate(self, prompt, *args, **kwargs):
        self.num_calls += 1
        yield RequestOutput(
            request_id=kwargs.get("request_id", "test-request"),
            prompt=prompt.get("prompt"),
            prompt_token_ids=prompt["prompt_token_ids"],
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text="",
                    token_ids=[],
                    cumulative_logprob=None,
                    logprobs=[self.logprobs],
                    finish_reason=None,
                )
            ],
            finished=True,
        )


class _Serving(BeamSearchOnlineMixin):
    renderer = _Renderer()

    def __init__(self, engine_client: _EngineClient | None = None) -> None:
        self.engine_client = engine_client or _EngineClient()


@pytest.mark.asyncio
async def test_beam_search_handles_extra_logprob_candidates() -> None:
    prompt = {
        "type": "token",
        "prompt": "prompt",
        "prompt_token_ids": [1],
    }
    params = BeamSearchParams(beam_width=2, max_tokens=1)

    outputs = [
        output async for output in _Serving().beam_search(prompt, "request", params)
    ]

    assert len(outputs) == 1
    assert outputs[0].outputs[0].finish_reason == "stop"
    assert outputs[0].outputs[0].token_ids == []
    assert outputs[0].outputs[0].cumulative_logprob == pytest.approx(-0.1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stop", "include_stop", "expected_text", "expected_ids"),
    [
        ("1", True, "1", [11]),
        ("11 11", False, "", [11, 11]),
    ],
)
async def test_beam_search_stops_on_stop_strings(
    stop: str,
    include_stop: bool,
    expected_text: str,
    expected_ids: list[int],
) -> None:
    prompt = {
        "type": "token",
        "prompt": "prompt",
        "prompt_token_ids": [1],
    }
    params = BeamSearchParams(
        beam_width=1,
        max_tokens=3,
        stop=[stop],
        include_stop_str_in_output=include_stop,
    )
    serving = _Serving(_EngineClient({11: -0.1}))

    outputs = [
        output async for output in serving.beam_search(prompt, "request", params)
    ]

    output = outputs[0].outputs[0]
    assert output.text == expected_text
    assert output.token_ids == expected_ids
    assert output.finish_reason == "stop"
    assert output.stop_reason == stop
    assert serving.engine_client.num_calls == len(expected_ids)


@pytest.mark.asyncio
async def test_stop_string_completes_only_matching_online_beam() -> None:
    prompt = {
        "type": "token",
        "prompt": "prompt",
        "prompt_token_ids": [1],
    }
    params = BeamSearchParams(
        beam_width=2,
        max_tokens=2,
        stop=["11 11"],
    )
    serving = _Serving(_EngineClient({11: -0.1, 12: -0.2}))

    outputs = [
        output async for output in serving.beam_search(prompt, "request", params)
    ]

    stopped = [output for output in outputs[0].outputs if output.stop_reason == "11 11"]
    active = [output for output in outputs[0].outputs if output.stop_reason is None]
    assert len(stopped) == 1
    assert len(active) == 1
    assert stopped[0].token_ids == [11, 11]
    assert stopped[0].finish_reason == "stop"
    assert active[0].finish_reason == "length"
    assert serving.engine_client.num_calls == 3


@pytest.mark.parametrize(
    ("beam_request", "expected_stop"),
    [
        (
            CompletionRequest(model="test-model", prompt="hello", stop="STOP"),
            ["STOP"],
        ),
        (
            ChatCompletionRequest(
                model="test-model",
                messages=[{"role": "user", "content": "hello"}],
                stop=["STOP", "END"],
            ),
            ["STOP", "END"],
        ),
    ],
)
def test_openai_beam_search_params_forward_stop_strings(
    beam_request: CompletionRequest | ChatCompletionRequest,
    expected_stop: list[str],
) -> None:
    params = beam_request.to_beam_search_params(
        max_tokens=3,
        default_sampling_params={},
    )

    assert params.stop_strings == expected_stop
