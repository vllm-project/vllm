# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.model_executor.models.interfaces import SupportsRealtime
from vllm.model_executor.models.qwen3_asr_realtime import Qwen3ASRRealtimeGeneration

pytestmark = pytest.mark.skip_global_cleanup

clean = Qwen3ASRRealtimeGeneration.clean_realtime_text


def test_single_segment_is_stripped_and_language_detected() -> None:
    assert clean("language English<asr_text>Hello there") == ("Hello there", "en")


def test_multi_segment_last_language_wins() -> None:
    text, lang = clean(
        "language English<asr_text>Hello\nlanguage Chinese<asr_text>\u4f60\u597d"
    )

    assert (text, lang) == ("Hello\n\u4f60\u597d", "zh")
    assert "<asr_text>" not in text


def test_partial_language_name_is_withheld() -> None:
    assert clean("language English<asr_text>Hi\nlanguage Chi") == ("Hi\n", "en")


def test_partial_asr_text_tag_is_withheld() -> None:
    assert clean("language English<asr_text>Hi\nlanguage Chinese<asr_te") == (
        "Hi\n",
        "en",
    )


def test_literal_language_word_in_speech_is_flushed() -> None:
    assert clean("language English<asr_text>I study language models") == (
        "I study language models",
        "en",
    )


def test_forced_mode_stream_without_preamble_passthrough() -> None:
    assert clean("Hello there") == ("Hello there", None)


def test_default_hook_is_passthrough_for_other_models() -> None:
    class _DummyRealtime(SupportsRealtime):
        pass

    raw = "language English<asr_text>untouched"
    assert _DummyRealtime.clean_realtime_text(raw) == (raw, None)
