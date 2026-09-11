# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import CompletionOutput, RequestOutput
from vllm.config import VllmConfig
from vllm.entrypoints.generate.beam_search.offline import BeamSearchOfflineMixin
from vllm.entrypoints.generate.beam_search.online import BeamSearchOnlineMixin
from vllm.logprobs import Logprob
from vllm.sampling_params import BeamSearchParams


class _Tokenizer:
    eos_token_id = 0
    special_token_id = eos_token_id

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        if skip_special_tokens:
            token_ids = [
                token_id for token_id in token_ids if token_id != self.special_token_id
            ]
        return " ".join(str(token_id) for token_id in token_ids)


class _Renderer:
    def get_tokenizer(self) -> _Tokenizer:
        return _Tokenizer()


class _VllmConfig:
    watermark_config = object()
    speculative_config = None
    _check_supports_watermarking = VllmConfig._check_supports_watermarking


class _EngineClient:
    vllm_config = _VllmConfig()

    async def generate(self, prompt, *args, **kwargs):
        assert not args[0].watermarking
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


class _AsyncServing(BeamSearchOnlineMixin):
    renderer = _Renderer()
    engine_client = _EngineClient()


class _OfflineServing(BeamSearchOfflineMixin):
    renderer = _Renderer()
    llm_engine = type("Engine", (), {"vllm_config": _VllmConfig()})()

    def _preprocess_cmpl(self, prompts):
        return prompts

    def _lora_request_to_seq(self, lora_request, num_requests):
        return [None] * num_requests

    def _beam_search_step(self, **kwargs):
        assert not kwargs["base_sampling_params"].watermarking
        return True


@pytest.mark.asyncio
async def test_beam_search_handles_extra_logprob_candidates() -> None:
    prompt = {
        "type": "token",
        "prompt": "prompt",
        "prompt_token_ids": [1],
    }
    params = BeamSearchParams(beam_width=2, max_tokens=1, watermarking=False)

    outputs = [
        output
        async for output in _AsyncServing().beam_search(prompt, "request", params)
    ]

    assert len(outputs) == 1
    assert outputs[0].outputs[0].finish_reason == "stop"
    assert outputs[0].outputs[0].token_ids == []
    assert outputs[0].outputs[0].cumulative_logprob == pytest.approx(-0.1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("skip_special_tokens", "expected_text"),
    [
        pytest.param(True, "", id="skip"),
        pytest.param(False, "0", id="keep"),
    ],
)
async def test_beam_search_respects_skip_special_tokens(
    skip_special_tokens: bool, expected_text: str
) -> None:
    prompt = {
        "type": "token",
        "prompt": "prompt",
        "prompt_token_ids": [1],
    }
    params = BeamSearchParams(
        beam_width=1,
        max_tokens=1,
        ignore_eos=True,
        skip_special_tokens=skip_special_tokens,
        watermarking=False,
    )

    outputs = [
        output
        async for output in _AsyncServing().beam_search(prompt, "request", params)
    ]

    assert outputs[0].outputs[0].text == expected_text
    assert outputs[0].outputs[0].token_ids == [_Tokenizer.special_token_id]


@pytest.mark.asyncio
async def test_beam_search_rejects_watermarking() -> None:
    prompt = {
        "type": "token",
        "prompt": "prompt",
        "prompt_token_ids": [1],
    }
    params = BeamSearchParams(beam_width=2, max_tokens=1)

    with pytest.raises(ValueError, match="Beam search is not supported"):
        await anext(_AsyncServing().beam_search(prompt, "request", params))


def test_offline_beam_search_rejects_watermarking() -> None:
    with pytest.raises(ValueError, match="Beam search is not supported"):
        _OfflineServing().beam_search(
            [{"type": "token", "prompt_token_ids": [1]}],
            BeamSearchParams(beam_width=2, max_tokens=1),
        )


def test_offline_beam_search_disables_internal_watermarking() -> None:
    outputs = _OfflineServing().beam_search(
        [{"type": "token", "prompt_token_ids": [1]}],
        BeamSearchParams(beam_width=2, max_tokens=1, watermarking=False),
    )

    assert len(outputs) == 1
