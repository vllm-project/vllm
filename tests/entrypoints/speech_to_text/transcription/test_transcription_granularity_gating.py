# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``timestamp_granularities`` decides which machinery a request pays for.

A ``verbose_json`` request that only asks for ``word`` needs none of the
segment-level work: no logprobs (only ``avg_logprob`` reads them), no segment
cutting, and no ``<|notimestamps|>`` -> ``<|0.00|>`` prompt swap (timestamp
tokens are what segments are cut on; word onsets come from cross-attention DTW).
These tests pin that gating, and pin the regression it is easy to cause: a plain
``verbose_json`` request sends ``timestamp_granularities=[]``, not
``["segment"]``, so a naive ``"segment" in granularities`` check would silently
drop its segments.
"""

import io
import logging

import pytest
from fastapi import UploadFile

from vllm.entrypoints.speech_to_text.base.serving import SpeechToTextBaseServing
from vllm.entrypoints.speech_to_text.transcription.protocol import (
    TranscriptionRequest,
    TranscriptionSegment,
)
from vllm.entrypoints.speech_to_text.translation.protocol import (
    TranslationRequest,
    TranslationSegment,
)
from vllm.sampling_params import SamplingParams

# Fake token ids: 0 is a special (eos), 100+ are timestamp tokens, the rest text.
_TS_BEGIN = 100
_PIECES = {10: " Mary", 11: " had", 12: " a", 13: " lamb"}


class _FakeTokenizer:
    all_special_ids = {0}
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        assert text == "<|0.00|>"
        return [_TS_BEGIN]

    def convert_tokens_to_ids(self, token):
        assert token == "<|0.00|>"
        return _TS_BEGIN

    def decode(self, ids):
        return "".join(_PIECES[int(i)] for i in ids)


class _FakeModelCls:
    supports_word_timestamp = True
    supports_segment_timestamp = True

    @staticmethod
    def post_process_output(text: str) -> str:
        return f"post({text})"


class _FakeServing:
    """Just the format-gating slice of the serving object, with the real methods.

    Everything these methods touch is provided here; nothing else of the serving
    stack (engine, renderer, model config) is needed to decide what a request
    format requires, which is the point of keeping that decision in one place.
    """

    _wants_word_timestamps = SpeechToTextBaseServing._wants_word_timestamps
    _needs_segments = SpeechToTextBaseServing._needs_segments
    _apply_verbose_params = SpeechToTextBaseServing._apply_verbose_params
    _parse_generation_prompt = SpeechToTextBaseServing._parse_generation_prompt
    _preprocess_verbose_prompt = SpeechToTextBaseServing._preprocess_verbose_prompt
    _collect_output_parts = SpeechToTextBaseServing._collect_output_parts
    _collect_words = SpeechToTextBaseServing._collect_words
    _count_word_tokens = SpeechToTextBaseServing._count_word_tokens
    _get_verbose_segments = SpeechToTextBaseServing._get_verbose_segments
    _group_words = SpeechToTextBaseServing._group_words

    def __init__(self, task_type: str = "transcribe", supports_words: bool = True):
        self.task_type = task_type
        self.tokenizer = _FakeTokenizer()
        self._words_missing = 0

        class _ModelCls(_FakeModelCls):
            supports_word_timestamp = supports_words

        self.model_cls = _ModelCls
        # Whisper is an encoder-decoder model; only that flag is read here.
        self.model_config = type("_Cfg", (), {"is_encoder_decoder": True})()


class _FakeLogprob:
    def __init__(self, logprob: float):
        self.logprob = logprob


class _FakeOutput:
    """The single ``CompletionOutput`` slice the response formatter reads."""

    def __init__(self, token_ids, text="raw text", logprobs=None, word_align=None):
        self.token_ids = token_ids
        self.text = text
        self.logprobs = logprobs
        self.word_align = word_align


# A realistic pair: the onsets cover *every* decoder position, so they are longer
# than the generated tokens by the decoder prompt, and the generated stream ends
# with a timestamp token plus eos the way Whisper's does. ``_group_words`` stops
# one position short of the end (a token's end time is the next position's onset),
# which is why those two trailing non-word tokens matter: they absorb the stop.
_TOKEN_IDS = [10, 11, 101, 0]
_ONSETS = [0.0, 0.0, 0.0, 0.5, 0.9, 1.2, 1.4]


def _upload_file() -> UploadFile:
    return UploadFile(file=io.BytesIO(b""), filename="audio.wav")


def _request(granularities: list[str] | None = None, **kwargs) -> TranscriptionRequest:
    # The field is alias-only ("timestamp_granularities[]"), which is how the
    # multipart form actually names it; passing the python name would land in
    # extra fields and leave the real one at its default.
    if granularities is not None:
        kwargs["timestamp_granularities[]"] = granularities
    return TranscriptionRequest(file=_upload_file(), **kwargs)


def _word_only() -> TranscriptionRequest:
    return _request(response_format="verbose_json", granularities=["word"])


# --------------------------------------------------------------------------
# Which machinery does this request format need?
# --------------------------------------------------------------------------


def test_plain_verbose_json_still_needs_segments():
    """The OpenAI default is empty, but plain verbose_json still means segments.

    This is the regression gate: ``timestamp_granularities`` defaults to ``[]``,
    so gating on ``"segment" in granularities`` would drop segments from every
    plain ``verbose_json`` request.
    """
    request = _request(response_format="verbose_json")
    assert request.timestamp_granularities == []
    serving = _FakeServing()
    assert serving._needs_segments(request) is True
    assert serving._wants_word_timestamps(request) is False


@pytest.mark.parametrize(
    "granularities", [["segment"], ["word", "segment"], ["segment", "word"]]
)
def test_explicit_segment_granularity_needs_segments(granularities):
    serving = _FakeServing()
    request = _request(response_format="verbose_json", granularities=granularities)
    assert serving._needs_segments(request) is True


def test_word_only_skips_segments():
    serving = _FakeServing()
    request = _word_only()
    assert serving._wants_word_timestamps(request) is True
    assert serving._needs_segments(request) is False


def test_word_only_falls_back_to_segments_when_words_are_unsupported():
    """Never return an empty response: if words cannot be produced, cut segments."""
    serving = _FakeServing(supports_words=False)
    request = _word_only()
    assert serving._wants_word_timestamps(request) is False
    assert serving._needs_segments(request) is True


def test_word_only_falls_back_to_segments_for_translation():
    """Translation has no word path, and its request model has no granularities."""
    serving = _FakeServing(task_type="translate")
    request = TranslationRequest(file=_upload_file(), response_format="verbose_json")
    assert not hasattr(request, "timestamp_granularities")
    assert serving._wants_word_timestamps(request) is False
    assert serving._needs_segments(request) is True


@pytest.mark.parametrize("response_format", ["json", "text"])
def test_non_verbose_formats_need_nothing(response_format):
    serving = _FakeServing()
    request = _request(
        response_format=response_format, granularities=["word", "segment"]
    )
    assert serving._needs_segments(request) is False
    assert serving._wants_word_timestamps(request) is False


# --------------------------------------------------------------------------
# (a) logprobs are only asked for when something reads them
# --------------------------------------------------------------------------


def test_word_only_request_does_not_request_logprobs():
    serving = _FakeServing()
    params = SamplingParams()
    serving._apply_verbose_params(_word_only(), params)
    assert params.logprobs is None
    assert params.extra_args == {"word_timestamps": True}


def test_plain_verbose_json_requests_logprobs():
    """Segment ``avg_logprob`` is the only consumer, so it must still get them."""
    serving = _FakeServing()
    params = SamplingParams()
    serving._apply_verbose_params(_request(response_format="verbose_json"), params)
    assert params.logprobs == 1
    assert not params.extra_args


def test_word_and_segment_request_gets_both():
    serving = _FakeServing()
    params = SamplingParams()
    request = _request(
        response_format="verbose_json", granularities=["word", "segment"]
    )
    serving._apply_verbose_params(request, params)
    assert params.logprobs == 1
    assert params.extra_args == {"word_timestamps": True}


def test_json_request_gets_neither():
    serving = _FakeServing()
    params = SamplingParams()
    serving._apply_verbose_params(_request(response_format="json"), params)
    assert params.logprobs is None
    assert not params.extra_args


# --------------------------------------------------------------------------
# (b) the prompt is only rewritten when timestamp tokens are needed
# --------------------------------------------------------------------------

_PROMPT = {
    "encoder_prompt": {"prompt": "", "multi_modal_data": {"audio": None}},
    "decoder_prompt": {
        "prompt": "<|startoftranscript|><|tr|><|transcribe|><|notimestamps|>"
    },
}


def test_word_only_request_keeps_notimestamps():
    """Word onsets come from cross-attention, not from timestamp tokens.

    Asking for timestamp tokens pushes a model that was not trained to emit them
    out of its normal decoding regime, so a word-only request must leave the
    prompt exactly as the model built it.
    """
    serving = _FakeServing()
    parsed = serving._parse_generation_prompt(_word_only(), dict(_PROMPT))
    assert "<|notimestamps|>" in parsed["decoder_prompt"]["prompt"]
    assert "<|0.00|>" not in parsed["decoder_prompt"]["prompt"]


def test_word_only_prompt_is_identical_to_the_json_prompt():
    """The fast path must not become its own prompt variant.

    ``_group_words`` recovers the decoder-prompt length by differencing, so the
    prompt is allowed to change length -- but there is no reason for a word-only
    request to differ from a plain ``json`` one, and keeping them identical is
    what makes the two comparable in a measurement.
    """
    serving = _FakeServing()
    word_only = serving._parse_generation_prompt(_word_only(), dict(_PROMPT))
    plain = serving._parse_generation_prompt(
        _request(response_format="json"), dict(_PROMPT)
    )
    assert word_only == plain


def test_segment_request_still_asks_for_timestamp_tokens():
    serving = _FakeServing()
    parsed = serving._parse_generation_prompt(
        _request(response_format="verbose_json"), dict(_PROMPT)
    )
    assert "<|notimestamps|>" not in parsed["decoder_prompt"]["prompt"]
    assert "<|0.00|>" in parsed["decoder_prompt"]["prompt"]


# --------------------------------------------------------------------------
# (c)/(d)/(e) the response body
# --------------------------------------------------------------------------


def _collect(serving, request, output, start_time=0.0):
    return serving._collect_output_parts(
        output,
        request=request,
        segment_class=TranscriptionSegment,
        start_time=start_time,
        need_segments=serving._needs_segments(request),
        want_words=serving._wants_word_timestamps(request),
    )


def test_word_only_response_text_comes_from_the_raw_output():
    """``verbose_json`` normally rebuilds ``text`` by joining its segments.

    With no segments to join, ``text`` has to come from the raw output the way
    ``json`` gets it -- otherwise the response carries word timestamps and an
    empty ``text``, which every downstream text consumer silently drops.
    """
    serving = _FakeServing()
    output = _FakeOutput(
        token_ids=_TOKEN_IDS,
        text=" Mary had",
        logprobs=None,
        word_align=_ONSETS,
    )
    segments, text_parts, words = _collect(serving, _word_only(), output)

    assert segments == []
    assert text_parts == ["post( Mary had)"]
    assert [w.word for w in words] == ["Mary", "had"]


def test_plain_verbose_json_still_builds_segments_and_avg_logprob():
    """Regression gate for the segment path, logprobs included."""
    serving = _FakeServing()
    # " Mary had" <|1|><|1|> " a lamb" <|2|>, then eos.
    token_ids = [10, 11, 101, 101, 12, 13, 102, 0]
    logprobs = [{tid: _FakeLogprob(-0.5)} for tid in token_ids]
    output = _FakeOutput(token_ids=token_ids, text="unused", logprobs=logprobs)

    segments, text_parts, words = _collect(
        serving, _request(response_format="verbose_json"), output, start_time=30.0
    )

    assert [s.text for s in segments] == [" Mary had", " a lamb"]
    # ``text`` is joined from the segments, not from the raw output.
    assert text_parts == [" Mary had", " a lamb"]
    assert words == []
    assert [s.start for s in segments] == [30.0, 30.02]
    for segment in segments:
        assert segment.avg_logprob == pytest.approx(-0.5 * 3 / 4)


def test_word_and_segment_request_gets_both_in_the_body():
    serving = _FakeServing()
    token_ids = [10, 11, 101, 101, 12, 13, 102, 0]
    logprobs = [{tid: _FakeLogprob(-0.5)} for tid in token_ids]
    output = _FakeOutput(
        token_ids=token_ids,
        text="unused",
        logprobs=logprobs,
        word_align=[0.0, 0.0, 0.0] + [0.2 * i for i in range(1, 9)],
    )
    request = _request(
        response_format="verbose_json", granularities=["word", "segment"]
    )
    segments, text_parts, words = _collect(serving, request, output)

    assert len(segments) == 2
    assert text_parts == [" Mary had", " a lamb"]
    assert [w.word for w in words] == ["Mary", "had", "a", "lamb"]


def test_json_request_text_is_unchanged_by_the_gating():
    serving = _FakeServing()
    output = _FakeOutput(token_ids=[10, 11], text=" Mary had")
    segments, text_parts, words = _collect(
        serving, _request(response_format="json"), output
    )
    assert (segments, text_parts, words) == ([], ["post( Mary had)"], [])


# --------------------------------------------------------------------------
# Asking for words and getting none must not be silent
# --------------------------------------------------------------------------


def test_missing_word_align_warns(caplog):
    """The engine dropped the request: no capture slot, or no readout at all."""
    serving = _FakeServing()
    output = _FakeOutput(token_ids=_TOKEN_IDS, text=" Mary had", word_align=None)
    with caplog.at_level(logging.WARNING):
        _, _, words = _collect(serving, _word_only(), output)
    assert words == []
    assert "no word alignment" in caplog.text
    # The output held two real word tokens, which is what makes this a failure.
    assert "2 word token(s)" in caplog.text
    assert serving._words_missing == 1


def test_ungroupable_onsets_warn_differently(caplog):
    """Onsets came back but no words were built from them.

    Points somewhere completely different from a missing readout -- at the token
    stream, not at the capture -- so the two must be distinguishable in the log.
    """
    serving = _FakeServing()
    # Two word tokens, but a single onset: a token's end time is the next
    # position's onset, so there is nothing to pair the first token with.
    output = _FakeOutput(token_ids=[10, 11], text="", word_align=[0.5])
    with caplog.at_level(logging.WARNING):
        _, _, words = _collect(serving, _word_only(), output)
    assert words == []
    assert "no words could be grouped" in caplog.text
    assert "1 onset(s)" in caplog.text
    assert "no word alignment" not in caplog.text


def test_a_silent_output_does_not_warn(caplog):
    """A timestamp token and eos is silence, not a failure.

    Warning on "no words" alone floods the log on every non-speech chunk, which is
    exactly what made the real coverage loss impossible to see.
    """
    serving = _FakeServing()
    output = _FakeOutput(token_ids=[101, 0], text="", word_align=[0.0, 0.1, 0.2])
    with caplog.at_level(logging.WARNING):
        _, _, words = _collect(serving, _word_only(), output)
    assert words == []
    assert caplog.text == ""
    assert serving._words_missing == 0


def test_repeated_failures_are_logged_sparsely(caplog):
    """Bounded logging: one line at 1, 2, 4, 8 ... never one per request."""
    serving = _FakeServing()
    output = _FakeOutput(token_ids=_TOKEN_IDS, text=" Mary had", word_align=None)
    with caplog.at_level(logging.WARNING):
        for _ in range(20):
            _collect(serving, _word_only(), output)
    assert serving._words_missing == 20
    # 1, 2, 4, 8, 16 -> 5 lines for 20 failures.
    assert caplog.text.count("Word timestamps were requested but not produced") == 5


def test_no_warning_when_words_were_produced(caplog):
    serving = _FakeServing()
    output = _FakeOutput(token_ids=_TOKEN_IDS, text=" Mary had", word_align=_ONSETS)
    with caplog.at_level(logging.WARNING):
        _, _, words = _collect(serving, _word_only(), output)
    assert len(words) == 2
    assert caplog.text == ""


def test_no_warning_when_words_were_not_requested(caplog):
    serving = _FakeServing()
    output = _FakeOutput(token_ids=_TOKEN_IDS, text=" Mary had", word_align=None)
    with caplog.at_level(logging.WARNING):
        _collect(serving, _request(response_format="json"), output)
    assert caplog.text == ""


def test_translation_segments_are_untouched():
    """The translate path shares the collector; its segment class must still work."""
    serving = _FakeServing(task_type="translate")
    token_ids = [10, 11, 101, 101, 12, 13, 102, 0]
    logprobs = [{tid: _FakeLogprob(-0.25)} for tid in token_ids]
    request = TranslationRequest(file=_upload_file(), response_format="verbose_json")
    segments, text_parts, words = serving._collect_output_parts(
        _FakeOutput(token_ids=token_ids, text="unused", logprobs=logprobs),
        request=request,
        segment_class=TranslationSegment,
        start_time=0.0,
        need_segments=True,
        want_words=False,
    )
    assert [s.text for s in segments] == [" Mary had", " a lamb"]
    assert text_parts == [" Mary had", " a lamb"]
    assert words == []
