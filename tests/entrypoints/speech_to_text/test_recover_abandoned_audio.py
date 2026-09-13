# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for recovering audio abandoned inside a chunk.

The numbers in these tests are measurements taken from a running
whisper-large-v3-turbo deployment (see the issue this change fixes), not values
chosen to make the assertions pass: a fixture invented from memory would exercise
a server that does not exist.
"""

import pytest

from vllm.entrypoints.speech_to_text.base.serving import (
    SPARSE_MIN_DURATION_S,
    is_sparse_recovery,
    last_decoded_timestamp,
    strip_seam_overlap,
)


class TestLastDecodedTimestamp:
    """Whisper states how far it got through its timestamp tokens."""

    def test_reads_the_final_timestamp(self):
        init = 50365  # any base; only the offset from it matters
        # <|0.00|> text <|5.82|> <|5.82|> text <|21.56|>
        tokens = (init, 1, 2, init + 291, init + 291, 3, 4, init + 1078)
        assert last_decoded_timestamp(tokens, init) == pytest.approx(21.56)

    def test_returns_none_without_timestamps(self):
        assert last_decoded_timestamp((1, 2, 3), 50365) is None


class TestIsSparseRecovery:
    """A few words spread over many seconds are not speech, whatever they say."""

    @pytest.mark.parametrize(
        "text,duration",
        [
            (" Thank you.", 13.32),  # measured phantom over silence
            (" Let's go.", 13.82),  # measured, and in no blocklist anywhere
        ],
    )
    def test_drops_measured_phantoms(self, text, duration):
        assert is_sparse_recovery(text, duration) is True

    def test_keeps_speech_at_a_real_speaking_rate(self):
        speech = (
            "the first words I spoke in the original phonograph a little piece "
            "of practical poetry mary had a little lamb its fleece was white as snow"
        )
        assert is_sparse_recovery(speech, 15.1) is False

    def test_does_not_judge_a_short_region_by_density(self):
        assert is_sparse_recovery("done", SPARSE_MIN_DURATION_S - 0.1) is False


class TestStripSeamOverlap:
    """The pre-roll buys context at the price of a repeated word or two."""

    def test_removes_the_repetition_the_preroll_caused(self):
        assert strip_seam_overlap("and then the kitchen", "the kitchen has a stove") == (
            "has a stove"
        )

    def test_ignores_case_and_punctuation(self):
        assert strip_seam_overlap("finishing up", "Finishing up, that is all") == "that is all"

    def test_leaves_a_single_shared_word_alone(self):
        # Speech shares words all the time; one match is not evidence.
        assert strip_seam_overlap("twenty seconds", "seconds later we start") == (
            "seconds later we start"
        )

    def test_leaves_a_long_repetition_alone(self):
        # Half a second cannot have produced five words: this is the speaker
        # repeating himself, and the transcript keeps what he said.
        assert strip_seam_overlap("one two three four five", "one two three four five six") == (
            "one two three four five six"
        )
