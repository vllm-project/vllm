# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
import torch

from vllm import CompletionOutput, RequestOutput, SamplingParams
from vllm import logger as vllm_logger
from vllm.config import VllmConfig
from vllm.entrypoints.generate.beam_search.offline import BeamSearchOfflineMixin
from vllm.entrypoints.generate.beam_search.online import BeamSearchOnlineMixin
from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence
from vllm.logprobs import Logprob, SampleLogprobs
from vllm.sampling_params import BeamSearchParams


@pytest.fixture
def reset_warning_once():
    vllm_logger._print_warning_once.cache_clear()
    yield
    vllm_logger._print_warning_once.cache_clear()


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


class _InputProcessor:
    vllm_config = _VllmConfig()

    def resolve_watermarking(self, params):
        return self.vllm_config._check_supports_watermarking(params)


class _EngineClient:
    input_processor = _InputProcessor()

    async def generate(self, prompt, *args, **kwargs):
        assert args[0].watermarking is False
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
    llm_engine = _EngineClient()

    def _preprocess_cmpl(self, prompts):
        return prompts

    def _lora_request_to_seq(self, lora_request, num_requests):
        return [None] * num_requests

    def _beam_search_step(self, **kwargs):
        assert kwargs["base_sampling_params"].watermarking is False
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
@pytest.mark.parametrize(("abort_after", "prompt_token"), [(0, 1), (1, 1), (0, 0)])
@pytest.mark.parametrize("terminal_logprobs", [None, [], [{11: Logprob(-0.1)}]])
async def test_beam_search_abort_returns_partial_outputs(
    monkeypatch,
    abort_after: int,
    prompt_token: int,
    terminal_logprobs: SampleLogprobs | None,
) -> None:
    """Abort returns each beam's tokens and scores from the last completed step."""
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

    serving = _AsyncServing()
    monkeypatch.setattr(serving.engine_client, "generate", generate)
    prompt = {"type": "token", "prompt": "prompt", "prompt_token_ids": [prompt_token]}
    outputs = [
        output
        async for output in serving.beam_search(
            prompt,
            "request",
            BeamSearchParams(beam_width=2, max_tokens=4, watermarking=False),
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


@pytest.mark.asyncio
async def test_beam_search_warns_and_disables_watermarking(
    caplog_vllm, reset_warning_once
) -> None:
    prompt = {
        "type": "token",
        "prompt": "prompt",
        "prompt_token_ids": [1],
    }
    params = BeamSearchParams(beam_width=2, max_tokens=1)

    with caplog_vllm.at_level("WARNING"):
        output = await anext(_AsyncServing().beam_search(prompt, "request", params))

    assert "beam search requests will run without watermarking" in caplog_vllm.text
    # The caller's params object is never rewritten by admission.
    assert params.watermarking is None
    assert output.finished


def test_offline_beam_search_warns_and_disables_watermarking(
    caplog_vllm, reset_warning_once
) -> None:
    params = BeamSearchParams(beam_width=2, max_tokens=1)

    with caplog_vllm.at_level("WARNING"):
        outputs = _OfflineServing().beam_search(
            [{"type": "token", "prompt_token_ids": [1]}], params
        )

    assert "beam search requests will run without watermarking" in caplog_vllm.text
    # The caller's params object is never rewritten by admission.
    assert params.watermarking is None
    assert len(outputs) == 1


def test_offline_beam_search_disables_internal_watermarking() -> None:
    outputs = _OfflineServing().beam_search(
        [{"type": "token", "prompt_token_ids": [1]}],
        BeamSearchParams(beam_width=2, max_tokens=1, watermarking=False),
    )

    assert len(outputs) == 1


def test_offline_structured_beam_search_disables_internal_watermarking() -> None:
    grammar = MagicMock()
    grammar.is_terminated.return_value = False
    grammar.fill_bitmask.side_effect = lambda bitmask, index: bitmask[index].fill_(1)
    backend = MagicMock()
    backend.compile_grammar.return_value = grammar
    serving = _OfflineServing()
    serving.model_config = MagicMock()
    serving.model_config.get_vocab_size.return_value = 32
    beam = BeamSearchSequence(
        orig_prompt={"type": "token", "prompt_token_ids": [1]},
        tokens=[1],
        logprobs=[],
    )

    base_params = SamplingParams(
        logprobs=2,
        watermarking=False,
        skip_clone=True,
    )
    entries = serving._build_beam_sampling_params(
        [beam],
        base_params,
        backend,
        ("regex", ".*"),
        torch.zeros((1, 1), dtype=torch.int32),
    )

    assert entries[0] is not None
    beam_params = entries[0][0]
    assert beam_params is not base_params
    assert beam_params.logprobs == base_params.logprobs
    assert beam_params.watermarking is False
    assert beam_params.allowed_token_ids == [0]
    assert base_params.allowed_token_ids is None
