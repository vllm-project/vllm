# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import CompletionOutput, RequestOutput
from vllm.entrypoints.generate.beam_search.online import BeamSearchOnlineMixin
from vllm.logprobs import Logprob, SampleLogprobs
from vllm.sampling_params import BeamSearchParams


class _Tokenizer:
    eos_token_id = 0

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


class _Renderer:
    def get_tokenizer(self) -> _Tokenizer:
        return _Tokenizer()


class _EngineClient:
    async def generate(self, prompt, *args, **kwargs):
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
    renderer = _Renderer()
    engine_client = _EngineClient()


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
@pytest.mark.parametrize(("abort_after", "prompt_token"), [(0, 1), (1, 1), (0, 0)])
@pytest.mark.parametrize("terminal_logprobs", [None, [], [{11: Logprob(-0.1)}]])
async def test_beam_search_abort_preserves_prefixes(
    monkeypatch,
    abort_after: int,
    prompt_token: int,
    terminal_logprobs: SampleLogprobs | None,
) -> None:
    """An aborted child ends the search, including siblings that returned a token."""
    calls = 0

    async def generate(prompt, *args, **kwargs):
        nonlocal calls
        calls += 1
        depth = len(prompt["prompt_token_ids"]) - 1
        assert depth <= abort_after, "Beam search continued after abort"
        result = await anext(_EngineClient().generate(prompt, *args, **kwargs))
        output = result.outputs[0]
        output.token_ids = [11]
        output.logprobs = [{11: Logprob(-0.1), 12: Logprob(-0.2)}]
        output.finish_reason = "length"
        if depth == abort_after and prompt["prompt_token_ids"][-1] != 12:
            output.token_ids = []
            output.logprobs = terminal_logprobs
            output.finish_reason = "abort"
        yield result

    serving = _Serving()
    monkeypatch.setattr(serving.engine_client, "generate", generate)
    prompt = {"type": "token", "prompt": "prompt", "prompt_token_ids": [prompt_token]}
    outputs = [
        output
        async for output in serving.beam_search(
            prompt, "request", BeamSearchParams(beam_width=2, max_tokens=4)
        )
    ]

    assert calls == 1 + 2 * abort_after
    assert len(outputs) == 1
    assert outputs[0].finished
    assert outputs[0].request_id == "request"
    expected_tokens = [[11], [12]] if abort_after else [[]]
    assert [output.token_ids for output in outputs[0].outputs] == expected_tokens
    for output in outputs[0].outputs:
        assert output.finish_reason == "abort"
        assert output.text == _Tokenizer().decode(list(output.token_ids))
        assert output.logprobs is not None
        assert len(output.logprobs) == abort_after
    assert outputs[0].outputs[0].cumulative_logprob == pytest.approx(-0.1 * abort_after)
