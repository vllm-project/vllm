# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.sampling_params import (
    RepetitionDetectionParams,
    RequestOutputKind,
    SamplingParams,
)
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.detokenizer import (
    REPETITION_DETECTED_STOP_REASON,
    BaseIncrementalDetokenizer,
    DetokenizerFinish,
)
from vllm.v1.engine.output_processor import OutputProcessor


class _CharacterDetokenizer(BaseIncrementalDetokenizer):
    def decode_next(self, next_token_id: int) -> str:
        return chr(next_token_id)


def _make_request() -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id="test-internal",
        external_req_id="test",
        prompt_token_ids=[],
        mm_features=None,
        sampling_params=SamplingParams(
            max_tokens=256,
            output_kind=RequestOutputKind.FINAL_ONLY,
            repetition_detection=RepetitionDetectionParams(
                min_pattern_size=8,
                max_pattern_size=8,
                min_count=2,
                mode="word_anywhere",
            ),
        ),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def _encode(text: str) -> list[int]:
    return [ord(char) for char in text]


def _repeated_text() -> str:
    return (
        "One, TWO three four five six seven eight. gap "
        "one two three four five six seven eight!"
    )


def test_word_repetition_waits_for_last_word_boundary_and_preserves_raw_text():
    request = _make_request()
    detokenizer = _CharacterDetokenizer(request)
    raw_text = _repeated_text()

    assert detokenizer.update(_encode(raw_text), stop_terminated=False) is None

    finish = detokenizer.update(_encode(" "), stop_terminated=False)
    assert finish == DetokenizerFinish(
        FinishReason.REPETITION,
        REPETITION_DETECTED_STOP_REASON,
    )
    assert detokenizer.output_text == raw_text + " "


def test_word_repetition_does_not_override_core_stop():
    request = _make_request()
    detokenizer = _CharacterDetokenizer(request)
    raw_text = _repeated_text() + " "

    finish = detokenizer.update(
        _encode(raw_text + "Z"),
        stop_terminated=True,
    )

    assert finish is None
    assert detokenizer.output_text == raw_text


def test_word_repetition_truncates_later_text_from_the_same_step():
    request = _make_request()
    detokenizer = _CharacterDetokenizer(request)
    raw_text = _repeated_text() + " "

    finish = detokenizer.update(
        _encode(raw_text + "this text arrived in the same step"),
        stop_terminated=False,
    )

    assert finish == DetokenizerFinish(
        FinishReason.REPETITION,
        REPETITION_DETECTED_STOP_REASON,
    )
    assert detokenizer.output_text == raw_text


def test_output_processor_maps_word_repetition_and_aborts_core():
    request = _make_request()
    detokenizer = _CharacterDetokenizer(request)
    output_processor = OutputProcessor(tokenizer=None, log_stats=False)
    output_processor.add_request(request, prompt="")
    output_processor.request_states[request.request_id].detokenizer = detokenizer
    raw_text = _repeated_text() + " "

    processed = output_processor.process_outputs(
        [
            EngineCoreOutput(
                request_id=request.request_id,
                new_token_ids=_encode(raw_text),
            )
        ]
    )

    assert processed.reqs_to_abort == [request.request_id]
    assert len(processed.request_outputs) == 1
    request_output = processed.request_outputs[0]
    assert request_output.finished
    completion = request_output.outputs[0]
    assert completion.text == raw_text
    assert completion.finish_reason == "repetition"
    assert completion.stop_reason == REPETITION_DETECTED_STOP_REASON
    assert not output_processor.has_unfinished_requests()
