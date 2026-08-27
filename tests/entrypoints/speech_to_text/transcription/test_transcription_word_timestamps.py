# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for word-level timestamps on the transcription API.

Requires the server to be started with ``--enable-word-timestamps``; without it
``words`` stays ``None`` even when the client asks for word granularity.
"""

import asyncio
from types import SimpleNamespace

import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.speech_to_text.base.serving import SpeechToTextBaseServing

MODEL_NAME = "openai/whisper-large-v3-turbo"


@pytest.fixture(scope="module")
def server():
    with RemoteOpenAIServer(MODEL_NAME, ["--enable-word-timestamps"]) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def whisper_client(server):
    async with server.get_async_client() as async_client:
        yield async_client


async def _transcribe_words(client, audio_file, language: str = "en", **kwargs):
    return await client.audio.transcriptions.create(
        model=MODEL_NAME,
        file=audio_file,
        language=language,
        response_format="verbose_json",
        timestamp_granularities=["word"],
        temperature=0.0,
        **kwargs,
    )


def _assert_word_list_is_sane(transcription, max_start: float | None = None):
    words = transcription.words
    assert words, "expected a non-empty words list"

    duration = float(transcription.duration)
    for word in words:
        assert word.word.strip() == word.word
        assert word.word != ""
        assert 0.0 <= word.start <= word.end <= duration + 0.5

    starts = [w.start for w in words]
    assert starts == sorted(starts), "word onsets must be non-decreasing"

    # The words must reconstruct the transcript, modulo whitespace.
    joined = "".join(w.word for w in words)
    assert joined == transcription.text.replace(" ", "")

    if max_start is not None:
        assert words[0].start <= max_start, (
            f"first word starts at {words[0].start}s, expected <= {max_start}s"
        )


@pytest.mark.asyncio
async def test_word_timestamps(whisper_client, mary_had_lamb):
    transcription = await _transcribe_words(whisper_client, mary_had_lamb)
    assert "Mary had a little lamb," in transcription.text
    # Speech starts almost immediately in this clip.
    _assert_word_list_is_sane(transcription, max_start=1.0)


@pytest.mark.asyncio
async def test_word_timestamps_not_requested(whisper_client, mary_had_lamb):
    """Word alignment is opt-in per request, not just per server."""
    transcription = await whisper_client.audio.transcriptions.create(
        model=MODEL_NAME,
        file=mary_had_lamb,
        language="en",
        response_format="verbose_json",
        temperature=0.0,
    )
    assert transcription.words is None
    assert transcription.segments is not None


@pytest.mark.asyncio
async def test_word_only_request_omits_segments(whisper_client, mary_had_lamb):
    """Asking only for ``word`` should not pay for segments.

    Skipping them is what lets the request skip ``logprobs`` and the
    ``<|notimestamps|>`` -> ``<|0.00|>`` prompt swap as well, so ``segments``
    coming back ``None`` is the observable half of the fast path.
    """
    transcription = await _transcribe_words(whisper_client, mary_had_lamb)
    assert transcription.words
    assert transcription.segments is None


@pytest.mark.asyncio
async def test_word_only_request_still_carries_text(whisper_client, mary_had_lamb):
    """``text`` must survive the segment skip, and must match the json answer.

    ``verbose_json`` builds ``text`` by joining its segments, so a word-only
    request has to take it from the raw output instead. It is also the tighter
    assertion: the two formats now send the *same* decoder prompt, so a fine-tune
    that never learned to emit timestamp tokens no longer transcribes differently
    (or not at all) depending on the response format. That divergence has been
    measured -- one Whisper fine-tune returned 271 characters as ``json`` and 0 as
    ``verbose_json`` for the same audio, with no warning anywhere.
    """
    word_only = await _transcribe_words(whisper_client, mary_had_lamb)
    mary_had_lamb.seek(0)
    plain = await whisper_client.audio.transcriptions.create(
        model=MODEL_NAME,
        file=mary_had_lamb,
        language="en",
        response_format="json",
        temperature=0.0,
    )
    assert word_only.text.strip()
    assert word_only.text.strip() == plain.text.strip()


@pytest.mark.asyncio
async def test_word_timestamps_batched(
    whisper_client, mary_had_lamb, winning_call, foscolo
):
    """Concurrent requests must each get their own timeline.

    A shared capture buffer would let one request's cross-attention leak into
    another's, which shows up as onsets shifted into a later request's clip.
    """
    transcriptions = await asyncio.gather(
        _transcribe_words(whisper_client, mary_had_lamb),
        _transcribe_words(whisper_client, winning_call),
        _transcribe_words(whisper_client, foscolo, language="it"),
    )
    for transcription in transcriptions:
        _assert_word_list_is_sane(transcription, max_start=2.0)

    # Sanity check that the batch really was three distinct clips.
    assert len({t.text for t in transcriptions}) == 3


def test_group_words_builds_words_from_onsets():
    """``_group_words`` turns per-token onsets into words offset by the chunk."""

    class _FakeTokenizer:
        # 0/1 are specials, 100+ are timestamp tokens, the rest are text pieces.
        _pieces = {10: " Mary", 11: " had", 12: " a", 13: " lit", 14: "tle"}
        all_special_ids = {0, 1}

        def convert_tokens_to_ids(self, token):
            assert token == "<|0.00|>"
            return 100

        def decode(self, ids):
            return self._pieces[ids[0]]

    token_ids = [1, 10, 11, 12, 13, 14, 100, 0]
    # Onsets cover every decoder position, so they are longer than the generated
    # tokens by the decoder prompt; the leading 0.0 here stands for that prompt.
    token_times = [0.0, 0.10, 0.40, 0.70, 0.90, 1.10, 1.50, 1.60, 1.70]

    words = SpeechToTextBaseServing._group_words(
        SimpleNamespace(tokenizer=_FakeTokenizer()), token_ids, token_times, 30.0
    )

    assert [w.word for w in words] == ["Mary", "had", "a", "little"]
    # Special and timestamp tokens are skipped; the chunk offset is applied.
    assert [w.start for w in words] == [30.40, 30.70, 30.90, 31.10]
    assert [w.end for w in words] == [30.70, 30.90, 31.10, 31.60]
