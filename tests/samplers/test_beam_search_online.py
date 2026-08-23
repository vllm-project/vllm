# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import CompletionOutput, RequestOutput
from vllm.entrypoints.generate.beam_search.online import BeamSearchOnlineMixin
from vllm.logprobs import Logprob
from vllm.sampling_params import BeamSearchParams


class _Tokenizer:
    eos_token_id = 0

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


class _Renderer:
    def get_tokenizer(self) -> _Tokenizer:
        return _Tokenizer()


class _EngineClient:
    def __init__(self) -> None:
        self.generate_kwargs: list[dict] = []

    async def generate(self, prompt, *args, **kwargs):
        self.generate_kwargs.append(kwargs)
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
                    logprobs=[
                        {
                            11: Logprob(logprob=-1.0),
                            12: Logprob(logprob=-2.0),
                            13: Logprob(logprob=-3.0),
                            14: Logprob(logprob=-4.0),
                            _Tokenizer.eos_token_id: Logprob(logprob=-0.1),
                        }
                    ],
                    finish_reason=None,
                )
            ],
            finished=True,
        )


class _Serving(BeamSearchOnlineMixin):
    def __init__(self) -> None:
        self.renderer = _Renderer()
        self.engine_client = _EngineClient()


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
async def test_beam_search_forwards_operation_name() -> None:
    prompt = {
        "type": "token",
        "prompt": "prompt",
        "prompt_token_ids": [1],
    }
    params = BeamSearchParams(beam_width=1, max_tokens=1)
    serving = _Serving()

    _ = [
        output
        async for output in serving.beam_search(
            prompt, "request", params, operation_name="chat"
        )
    ]

    assert serving.engine_client.generate_kwargs
    assert all(
        kwargs.get("operation_name") == "chat"
        for kwargs in serving.engine_client.generate_kwargs
    )
