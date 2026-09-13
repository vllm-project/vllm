# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for realtime segment timestamps.

The timestamper turns output-token positions into audio time, which only
works because Voxtral realtime emits exactly one token per 80 ms frame. Two
things can silently produce plausible-but-wrong timestamps, and both are
pinned here:

- the boundary marker precedes the subwords of its own emission group
  (arXiv 2602.11298 section 3.1), so pairing it with the text that follows it
  shifts every segment by one full group;
- the frame offset is derived from the *audio* length of the left pad, not
  from the pad token count, so a tokenizer change cannot shift every
  timestamp by the pad duration without failing loud.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import regex as re
from mistral_common.tokens.tokenizers.base import SpecialTokens

from vllm.entrypoints.speech_to_text.realtime.connection import RealtimeConnection
from vllm.entrypoints.speech_to_text.realtime.protocol import (
    SessionUpdate,
    TranscriptionDone,
    TranscriptionSegmentTimestamp,
)
from vllm.model_executor.models import voxtral_realtime
from vllm.model_executor.models.voxtral_realtime import (
    VoxtralRealtimeGeneration,
    VoxtralRealtimeSegmentTimestamper,
)

BOUNDARY = 33  # [STREAMING_WORD]
PAD = 32  # [STREAMING_PAD]
FRAME_MS = 80.0

# Stands in for `tokenizer.decode(ids, skip_special_tokens=True)`: real
# words for real ids, and nothing at all for control tokens.
VOCAB = {1: "Mary", 2: " had", 4: " lit", 5: "tle", 6: " lamb"}


def _decode(token_ids: list[int]) -> str:
    return "".join(VOCAB.get(token_id, "") for token_id in token_ids)


def _timestamper(offset_frames: int = 1) -> VoxtralRealtimeSegmentTimestamper:
    return VoxtralRealtimeSegmentTimestamper(
        boundary_token_id=BOUNDARY,
        frame_duration_ms=FRAME_MS,
        offset_frames=offset_frames,
        decode=_decode,
    )


def _feed(
    timestamper: VoxtralRealtimeSegmentTimestamper,
    deltas: list[list[int]],
) -> list[tuple[str, float]]:
    segments: list[tuple[str, float]] = []
    for delta in deltas:
        segments.extend(timestamper.process_token_ids(delta))
    return segments


@pytest.mark.parametrize(
    ("deltas", "expected"),
    [
        # A boundary opens a segment, it does not close one.
        pytest.param([[BOUNDARY]], [], id="boundary-opens-nothing"),
        pytest.param([[BOUNDARY], [1], [BOUNDARY]], [("Mary", 0.08)], id="single"),
        pytest.param(
            [[BOUNDARY], [4], [5], [BOUNDARY]],
            [(" little", 0.08)],
            id="multi-token",
        ),
        # One boundary per emission group, so words the model emitted
        # together stay together (arXiv 2602.11298 section 3.1).
        pytest.param(
            [[BOUNDARY], [1], [2], [BOUNDARY]],
            [("Mary had", 0.08)],
            id="grouped-words",
        ),
        # Padding before the first boundary belongs to no segment, but still
        # advances the clock.
        pytest.param(
            [[PAD], [PAD], [BOUNDARY], [1], [BOUNDARY]],
            [("Mary", 0.24)],
            id="leading-pads-dropped",
        ),
        pytest.param([[BOUNDARY], [BOUNDARY]], [], id="empty-segment-skipped"),
        # Silence inside an utterance decodes to nothing and is not reported.
        pytest.param(
            [[BOUNDARY], [PAD], [PAD], [BOUNDARY]],
            [],
            id="silence-segment-skipped",
        ),
        pytest.param([[1]], [], id="text-before-first-boundary"),
        pytest.param(
            [[BOUNDARY], [1], [BOUNDARY], [2], [BOUNDARY]],
            [("Mary", 0.08), (" had", 0.24)],
            id="inverted-pairing-guard",
        ),
    ],
)
def test_segments(deltas, expected):
    assert _feed(_timestamper(), deltas) == expected


def test_trailing_segment_is_flushed():
    timestamper = _timestamper()
    assert _feed(timestamper, [[BOUNDARY], [1], [6]]) == []
    assert timestamper.flush() == [("Mary lamb", 0.08)]


def test_flush_does_not_re_emit_a_closed_segment():
    timestamper = _timestamper()
    assert _feed(timestamper, [[BOUNDARY], [1], [BOUNDARY]]) == [("Mary", 0.08)]
    assert timestamper.flush() == []
    assert timestamper.flush() == []


def test_merged_delta_matches_token_at_a_time():
    """Defensive: identical results however tokens are batched into deltas.

    Unreachable today because ``realtime_max_tokens = 1`` and the lockstep
    stops the producer from running ahead, so ``RequestOutputCollector`` has
    nothing to merge. Segmenting on token ids rather than on delta text keeps
    it that way if the lockstep is ever relaxed.
    """
    one_at_a_time = _feed(
        _timestamper(), [[BOUNDARY], [1], [BOUNDARY], [2], [BOUNDARY]]
    )
    merged = _feed(_timestamper(), [[BOUNDARY, 1, BOUNDARY, 2], [BOUNDARY]])
    assert merged == one_at_a_time


def test_offset_frames_shifts_every_segment():
    assert _feed(_timestamper(offset_frames=0), [[BOUNDARY], [1], [BOUNDARY]]) == [
        ("Mary", 0.0)
    ]


class _FakeTokenizer:
    """The slice of the Mistral tokenizer the timestamper reads."""

    def __init__(self, n_pad_audio_frames: int = 32, has_boundary: bool = True):
        audio_config = SimpleNamespace(
            frame_rate=12.5,
            raw_audio_length_per_tok=1280,
            n_left_pad_tokens=32,
            get_num_delay_tokens=lambda: 6,
        )
        left_pad = SimpleNamespace(
            audio_array=np.zeros(n_pad_audio_frames * 1280, dtype=np.float32)
        )
        audio_encoder = SimpleNamespace(
            audio_config=audio_config,
            get_padding_audio=lambda: (left_pad, left_pad),
            # 1 BOS + 32 left pad + 6 delay = 39
            encode_streaming_tokens=lambda: [PAD] * 32 + [0] * 6,
        )
        self.instruct = SimpleNamespace(
            start=lambda: [1],
            audio_encoder=audio_encoder,
        )
        self.tokenizer = SimpleNamespace(
            is_special=lambda token: has_boundary,
            get_special_token=lambda token: BOUNDARY,
        )

    def decode(self, token_ids, skip_special_tokens=False):
        return _decode(list(token_ids))


@pytest.fixture
def fake_tokenizer(monkeypatch):
    def _install(**kwargs):
        tokenizer = _FakeTokenizer(**kwargs)
        monkeypatch.setattr(
            voxtral_realtime, "cached_tokenizer_from_config", lambda _: tokenizer
        )
        return tokenizer

    return _install


def test_offset_frames_tracks_the_pad_audio(fake_tokenizer):
    """The left pad is audio; the delay tokens are not.

    Subtracting only the delay - as the issue's formula does - leaves the
    32-frame left pad inside the prompt length and puts every timestamp
    2.56 s late.
    """
    fake_tokenizer()
    timestamper = VoxtralRealtimeGeneration.get_realtime_segment_timestamper(
        SimpleNamespace(model="fake-model")
    )
    assert timestamper._offset_frames == 1
    assert timestamper.process_token_ids([BOUNDARY, 1, BOUNDARY]) == [("Mary", 0.08)]


def test_pad_audio_diverging_from_pad_tokens_fails_loud(fake_tokenizer):
    """Never silently keep an offset the pad audio no longer justifies."""
    fake_tokenizer(n_pad_audio_frames=38)
    with pytest.raises(ValueError, match="38 frames"):
        VoxtralRealtimeGeneration.get_realtime_segment_timestamper(
            SimpleNamespace(model="fake-model")
        )


def test_missing_boundary_token_fails_loud(fake_tokenizer):
    fake_tokenizer(has_boundary=False)
    with pytest.raises(ValueError, match=re.escape(SpecialTokens.streaming_word.value)):
        VoxtralRealtimeGeneration.get_realtime_segment_timestamper(
            SimpleNamespace(model="fake-model")
        )


@pytest.mark.parametrize(
    "key", ["timestamp_granularities", "timestamp_granularities[]"]
)
def test_session_update_accepts_both_granularity_spellings(key):
    """`POST /v1/audio/transcriptions` uses the bracketed form."""
    session = SessionUpdate(
        **{"type": "session.update", "model": "m", key: ["segment"]}
    )
    assert session.timestamp_granularities == ["segment"]


class _FakeWebSocket:
    def __init__(self):
        self.sent: list[dict] = []

    async def send_text(self, data: str) -> None:
        self.sent.append(json.loads(data))


def _connection(
    supports: bool,
    stream_interval: int = 1,
    outputs: list[list[int]] | None = None,
) -> RealtimeConnection:
    """A connection over a fake engine that replays ``outputs`` as deltas."""
    model_cls = SimpleNamespace(
        supports_realtime_segment_timestamps=supports,
        get_realtime_segment_timestamper=lambda _: _timestamper(),
        realtime_max_tokens=1,
    )

    async def generate(prompt=None, sampling_params=None, request_id=None):
        for index, token_ids in enumerate(outputs or []):
            yield SimpleNamespace(
                prompt_token_ids=[0] * 39 if index == 0 else None,
                outputs=[SimpleNamespace(text=_decode(token_ids), token_ids=token_ids)],
            )

    serving = SimpleNamespace(
        model_cls=model_cls,
        model_config=SimpleNamespace(model="fake-model"),
        engine_client=SimpleNamespace(
            generate=generate,
            vllm_config=SimpleNamespace(
                scheduler_config=SimpleNamespace(stream_interval=stream_interval)
            ),
        ),
        transcribe_realtime=lambda audio_stream, input_stream: None,
        _is_model_supported=lambda model: model == "fake-model",
    )
    return RealtimeConnection(_FakeWebSocket(), serving)


@pytest.mark.asyncio
async def test_opt_in_is_acknowledged():
    conn = _connection(supports=True)
    await conn.handle_event(
        {
            "type": "session.update",
            "model": "fake-model",
            "timestamp_granularities": ["segment"],
        }
    )
    ack = conn.websocket.sent[-1]
    assert ack["type"] == "session.updated"
    assert ack["timestamp_granularities"] == ["segment"]
    assert conn._segment_timestamps

    # A later plain update turns it back off, rather than accumulating.
    await conn.handle_event({"type": "session.update", "model": "fake-model"})
    assert conn.websocket.sent[-1]["timestamp_granularities"] == []
    assert not conn._segment_timestamps


@pytest.mark.asyncio
async def test_unsupported_model_rejects_and_explains_at_commit():
    """Covers realtime models without a token-per-frame lockstep."""
    conn = _connection(supports=False)
    await conn.handle_event(
        {
            "type": "session.update",
            "model": "fake-model",
            "timestamp_granularities": ["segment"],
        }
    )
    error = conn.websocket.sent[-1]
    assert error["code"] == "unsupported_timestamp_granularity"
    assert not conn._is_model_validated
    assert not conn._segment_timestamps

    await conn.handle_event({"type": "input_audio_buffer.commit"})
    commit_error = conn.websocket.sent[-1]
    assert commit_error["code"] == "model_not_validated"
    assert "does not support segment timestamps" in commit_error["error"]


@pytest.mark.asyncio
async def test_word_granularity_is_rejected():
    conn = _connection(supports=True)
    await conn.handle_event(
        {
            "type": "session.update",
            "model": "fake-model",
            "timestamp_granularities": ["word"],
        }
    )
    assert conn.websocket.sent[-1]["code"] == "invalid_timestamp_granularity"


@pytest.mark.asyncio
async def test_model_advertising_support_without_implementing_it_is_rejected():
    """Rejected at negotiation, not unwound to the connection loop.

    The `SupportsRealtime` default raises a bare `NotImplementedError`, and a
    duck-typed out-of-tree model raises `AttributeError`. Either escaping
    would leave the session configured behind the client's back.
    """

    def _unimplemented(_):
        raise NotImplementedError

    conn = _connection(supports=True)
    conn.serving.model_cls.get_realtime_segment_timestamper = _unimplemented
    await conn.handle_event(
        {
            "type": "session.update",
            "model": "fake-model",
            "timestamp_granularities": ["segment"],
        }
    )

    error = conn.websocket.sent[-1]
    assert error["code"] == "segment_timestamps_unavailable"
    assert "cannot build a segment timestamper" in error["error"]
    assert not conn._is_model_validated
    assert not conn._segment_timestamps


@pytest.mark.asyncio
async def test_batched_streaming_is_rejected():
    """`--stream-interval > 1` stalls the realtime lockstep entirely.

    Rejected at negotiation so the opt-in fails loud instead of hanging. The
    server is unusable for realtime either way, which is why the error must
    not offer dropping the opt-in as a workaround.
    """
    conn = _connection(supports=True, stream_interval=4)
    await conn.handle_event(
        {
            "type": "session.update",
            "model": "fake-model",
            "timestamp_granularities": ["segment"],
        }
    )
    error = conn.websocket.sent[-1]
    assert error["code"] == "unsupported_timestamp_granularity"
    assert "stream-interval" in error["error"]

    # The gate is scoped to the opt-in: a plain session.update on the same
    # server is still accepted, exactly as before this feature.
    conn = _connection(supports=True, stream_interval=4)
    await conn.handle_event({"type": "session.update", "model": "fake-model"})
    assert conn.websocket.sent[-1]["type"] == "session.updated"


@pytest.mark.asyncio
@pytest.mark.parametrize("opted_in", [True, False])
async def test_generation_streams_segments_and_flushes_the_trailing_one(opted_in):
    """The delta/done wiring, which the unit tests above cannot reach.

    ``done`` must repeat what the deltas already sent, in order, and add
    exactly the trailing segment that generation ended before any delta could
    carry - no duplicate, no drop.
    """
    conn = _connection(
        supports=True, outputs=[[BOUNDARY], [1], [2], [BOUNDARY], [4], [5], [6]]
    )
    conn._is_connected = True

    update: dict = {"type": "session.update", "model": "fake-model"}
    if opted_in:
        update["timestamp_granularities"] = ["segment"]
    await conn.handle_event(update)

    await conn.start_generation()
    await conn.generation_task

    events = [e for e in conn.websocket.sent if e["type"].startswith("transcription")]
    deltas, done = events[:-1], events[-1]
    assert done["type"] == "transcription.done"
    assert done["text"] == "Mary had little lamb"

    if not opted_in:
        assert all("segments" not in event for event in events)
        return

    streamed = [segment for delta in deltas for segment in delta["segments"]]
    assert streamed == [{"text": "Mary had", "end": 0.08}]
    assert done["segments"] == streamed + [{"text": " little lamb", "end": 0.32}]

    # The segment rides the delta *after* the text it describes, and that
    # delta carries no text of its own.
    assert deltas[3]["delta"] == ""
    assert "".join(s["text"] for s in done["segments"]) == done["text"]


def test_transcription_done_keys_unchanged_without_opt_in():
    """A client that did not opt in must see byte-identical payloads.

    ``segments`` defaults to ``None`` and would otherwise serialize as
    ``"segments": null``, adding a key to every event. Removing or adding
    keys breaks statically typed clients.
    """
    done = TranscriptionDone(text=" Mary had a little lamb", usage=None)
    payload = json.loads(done.model_dump_json(exclude={"segments"}))
    assert set(payload) == {"type", "text", "usage"}

    opted_in = TranscriptionDone(
        text=" Mary had a little lamb",
        usage=None,
        segments=[TranscriptionSegmentTimestamp(text="Mary", end=0.08)],
    )
    assert json.loads(opted_in.model_dump_json())["segments"] == [
        {"text": "Mary", "end": 0.08}
    ]
