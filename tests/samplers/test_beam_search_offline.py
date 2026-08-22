# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only unit tests for offline beam search."""

import pytest

from vllm import CompletionOutput, RequestOutput
from vllm.entrypoints.generate.beam_search.offline import BeamSearchOfflineMixin
from vllm.entrypoints.generate.beam_search.utils import check_beam_search_stop
from vllm.inputs import EncoderDecoderInput, tokens_input
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


class _OfflineBeamSearch(BeamSearchOfflineMixin):
    renderer = _Renderer()

    def __init__(self, token_logprobs: dict[int, float] | None = None) -> None:
        if token_logprobs is None:
            token_logprobs = {11: -0.1}
        self.logprobs = {
            token_id: Logprob(logprob=logprob)
            for token_id, logprob in token_logprobs.items()
        }
        self.num_calls = 0

    def _preprocess_cmpl(self, prompts):
        return [
            {
                "type": "token",
                "prompt": prompt,
                "prompt_token_ids": [1],
            }
            for prompt in prompts
        ]

    def _render_and_run_requests(self, prompts, *args, **kwargs):
        self.num_calls += 1
        return [
            RequestOutput(
                request_id="test-request",
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
            for prompt in prompts
        ]


@pytest.mark.parametrize(
    ("stop", "include_stop", "expected_text", "expected_tokens", "expected_calls"),
    [
        ("1", True, "1 1", [1, 11], 1),
        ("11 11", False, "1 ", [1, 11, 11], 2),
    ],
)
def test_offline_beam_search_stops_on_stop_strings(
    stop: str,
    include_stop: bool,
    expected_text: str,
    expected_tokens: list[int],
    expected_calls: int,
) -> None:
    params = BeamSearchParams(
        beam_width=1,
        max_tokens=3,
        stop=[stop],
        include_stop_str_in_output=include_stop,
    )
    serving = _OfflineBeamSearch()

    outputs = serving.beam_search(["prompt"], params)

    sequence = outputs[0].sequences[0]
    assert sequence.text == expected_text
    assert sequence.tokens == expected_tokens
    assert sequence.finish_reason == "stop"
    assert sequence.stop_reason == stop
    assert serving.num_calls == expected_calls


def test_stop_string_completes_only_matching_offline_beam() -> None:
    params = BeamSearchParams(
        beam_width=2,
        max_tokens=2,
        stop=["11 11"],
    )
    serving = _OfflineBeamSearch({11: -0.1, 12: -0.2})

    outputs = serving.beam_search(["prompt"], params)

    stopped = [
        sequence for sequence in outputs[0].sequences if sequence.stop_reason == "11 11"
    ]
    active = [
        sequence for sequence in outputs[0].sequences if sequence.stop_reason is None
    ]
    assert len(stopped) == 1
    assert len(active) == 1
    assert stopped[0].tokens == [1, 11, 11]
    assert stopped[0].finish_reason == "stop"
    assert serving.num_calls == 2


def test_stop_check_uses_decoder_prompt_length() -> None:
    prompt = EncoderDecoderInput(
        type="enc_dec",
        encoder_prompt=tokens_input([99], prompt="encoder"),
        decoder_prompt=tokens_input([1, 2], prompt="decoder"),
    )
    token_ids = [1, 2, 11]

    assert check_beam_search_stop(_Tokenizer(), prompt, token_ids, ["2"], False) is None
    stop_result = check_beam_search_stop(_Tokenizer(), prompt, token_ids, ["11"], False)

    assert stop_result is not None
    assert stop_result.output_text == ""
    assert stop_result.stop_reason == "11"
