# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Server-side turn detection for ``/v1/realtime``.

``server_vad`` ends a turn after a fixed silence, judged frame by frame by
Silero VAD. ``semantic_vad`` uses a short, eagerness-dependent silence only as a
candidate pause and asks Smart Turn v3.2 whether the speech so far is a
complete turn; an incomplete turn keeps listening until speech resumes or the
silence reaches the eagerness limit.

Both models run in ``vllm._rust_turn_detection`` (no ONNX Runtime). Their files
are downloaded from the Hugging Face Hub at pinned revisions and checked
against pinned sha256 digests when loaded. The state machine and its defaults
follow vLLM-Omni's duplex turn detection (``vllm_omni/engine/duplex/vad.py`` and
``turn_detection.py``); vLLM-Omni scores pauses with the int8 Smart Turn graph,
this module with the fp32 one.
"""

import functools
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from vllm.logger import init_logger

from .protocol import SemanticVad, ServerVad

logger = init_logger(__name__)

SAMPLE_RATE_HZ = 16_000
FRAME_SAMPLES = 512
#: Smart Turn sees at most the last 8 s of the turn.
SMART_TURN_WINDOW_SAMPLES = 8 * SAMPLE_RATE_HZ
#: Speech must last this long before it starts a turn.
MIN_SPEECH_DURATION_MS = 96
#: Width of the hysteresis band below ``threshold``: scores inside it keep a
#: pending silence running but can neither start nor end it.
HYSTERESIS = 0.15

#: Silence that makes a pause a candidate end of turn, by eagerness. The
#: default favors keeping clause pauses inside a turn.
SEMANTIC_VAD_CANDIDATE_SILENCE_MS = {
    "high": 300,
    "medium": 1200,
    "auto": 1200,
    "low": 1500,
}
#: Longest silence ``semantic_vad`` waits through after Smart Turn calls the
#: turn incomplete, measured from the start of the pause.
SEMANTIC_VAD_MAX_SILENCE_MS = {"high": 2000, "medium": 4000, "auto": 4000, "low": 8000}
#: Smart Turn probability at or above which a candidate pause ends the turn.
SEMANTIC_VAD_COMPLETION_THRESHOLD = 0.5

FrameScorer = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class TurnEvent:
    """A turn boundary on the session's audio clock (ms of audio received)."""

    kind: Literal["speech_started", "speech_stopped"]
    audio_ms: int


def _ms(samples: int) -> int:
    return round(samples * 1000 / SAMPLE_RATE_HZ)


def _samples(ms: int) -> int:
    return max(1, round(ms * SAMPLE_RATE_HZ / 1000))


class _Model:
    """One Rust model, downloaded and loaded once per process on first use."""

    def __init__(self, cls_name: str, prefix: str) -> None:
        self._cls_name = cls_name
        self._prefix = prefix
        self._lock = threading.Lock()
        self._model = None

    def get(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    import vllm._rust_turn_detection as rt
                    from vllm.transformers_utils.repo_utils import hf_api

                    path = hf_api().hf_hub_download(
                        getattr(rt, f"{self._prefix}_REPO"),
                        getattr(rt, f"{self._prefix}_FILE"),
                        revision=getattr(rt, f"{self._prefix}_REVISION"),
                    )
                    self._model = getattr(rt, self._cls_name)(path)
        return self._model


SILERO = _Model("SileroVad", "SILERO")
SMART_TURN = _Model("SmartTurn", "SMART_TURN")


def _silero_stream(model) -> FrameScorer:
    """A fresh Silero stream: scores consecutive 512-sample frames."""
    state = model.new_state()

    def score(frame: np.ndarray) -> float:
        nonlocal state
        probability, state = model.infer(frame, state)
        return probability

    return score


class _SileroEndpointer:
    """Speech start and end on Silero's 512-sample frame grid.

    Speech starts once ``MIN_SPEECH_DURATION_MS`` of frames score at or above
    ``threshold``; the reported start reaches back ``prefix_padding_ms``, but
    not past the previous reset. Speech ends after ``silence_duration_ms`` of
    frames below ``threshold - HYSTERESIS``.
    """

    def __init__(
        self,
        new_frame_scorer: Callable[[], FrameScorer],
        *,
        start_sample: int,
        threshold: float,
        prefix_padding_ms: int,
        silence_duration_ms: int,
    ) -> None:
        self._new_frame_scorer = new_frame_scorer
        self._threshold = threshold
        self._negative_threshold = max(threshold - HYSTERESIS, 0.01)
        self._prefix_padding_ms = prefix_padding_ms
        self._min_speech = _samples(MIN_SPEECH_DURATION_MS)
        self._min_silence = _samples(silence_duration_ms)
        #: Samples scored so far: the session's audio clock.
        self.processed = start_sample
        self.reset()

    def reset(self) -> None:
        """Start a new Silero stream; the audio clock keeps running."""
        self._score = self._new_frame_scorer()
        self.speech_active = False
        self._candidate = 0
        self._candidate_start = 0
        self._silence = 0
        #: Where the silence that last ended speech began.
        self.silence_start_ms = 0
        self._stream_start_ms = _ms(self.processed)

    @property
    def speech_pending(self) -> bool:
        """An onset is being confirmed against ``MIN_SPEECH_DURATION_MS``."""
        return self._candidate > 0

    def frame(self, frame: np.ndarray) -> TurnEvent | None:
        """Score one frame; return the boundary it completes, if any."""
        frame_start = self.processed
        self.processed += FRAME_SAMPLES
        probability = min(1.0, max(0.0, float(self._score(frame))))

        if self.speech_active:
            if probability >= self._threshold:
                # Confirmed speech cancels a pending end. Otherwise a brief
                # pause could age through resumed speech and split the turn.
                self._silence = 0
                return None
            below = probability < self._negative_threshold
            if not below and self._silence == 0:
                return None
            self._silence += FRAME_SAMPLES
            if not below or self._silence < self._min_silence:
                return None
            self.silence_start_ms = _ms(self.processed - self._silence)
            self.speech_active = False
            self._silence = 0
            # As in OpenAI's Realtime API, audio_end_ms includes the silence.
            return TurnEvent("speech_stopped", _ms(self.processed))

        if probability < self._threshold:
            self._candidate = 0
            return None
        if self._candidate == 0:
            self._candidate_start = frame_start
        self._candidate += FRAME_SAMPLES
        if self._candidate < self._min_speech:
            return None
        self.speech_active = True
        self._candidate = self._silence = 0
        start_ms = _ms(self._candidate_start) - self._prefix_padding_ms
        return TurnEvent("speech_started", max(self._stream_start_ms, start_ms, 0))


class _Ring:
    """The most recent ``SMART_TURN_WINDOW_SAMPLES`` of audio."""

    def __init__(self) -> None:
        self._buffer = np.zeros(SMART_TURN_WINDOW_SAMPLES, dtype=np.float32)
        self._write = 0

    def append(self, samples: np.ndarray) -> None:
        # Callers append one frame at a time, which never exceeds the buffer.
        head = min(samples.size, self._buffer.size - self._write)
        self._buffer[self._write : self._write + head] = samples[:head]
        self._buffer[: samples.size - head] = samples[head:]
        self._write = (self._write + samples.size) % self._buffer.size

    def last(self, count: int) -> np.ndarray:
        """The newest ``count`` samples (at most the window), oldest first."""
        count = min(max(0, count), self._buffer.size)
        start = self._write - count
        if start >= 0:
            return self._buffer[start : self._write].copy()
        return np.concatenate((self._buffer[start:], self._buffer[: self._write]))


class TurnDetector:
    """One per session: feed it the session's audio, get turn boundaries back.

    Audio is 16 kHz float32. Calls must be serialized; Smart Turn takes tens of
    milliseconds on a CPU core, so callers run ``process`` off the event loop.
    Constructing one may download and load the models, so do that off the
    event loop too.

    Args:
        config: The session's ``turn_detection`` setting.
        start_sample: Session audio already received, where this detector's
            clock starts.
        new_frame_scorer: Starts a Silero stream (the default) or a stand-in.
        turn_scorer: Smart Turn (the default for ``semantic_vad``) or a
            stand-in.

    """

    def __init__(
        self,
        config: ServerVad | SemanticVad,
        *,
        start_sample: int = 0,
        new_frame_scorer: Callable[[], FrameScorer] | None = None,
        turn_scorer: FrameScorer | None = None,
    ) -> None:
        self.config = config
        if new_frame_scorer is None:
            new_frame_scorer = functools.partial(_silero_stream, SILERO.get())
        self._turn_scorer: FrameScorer | None = None
        if isinstance(config, SemanticVad):
            # The Silero knobs are not on the wire for semantic_vad; use the
            # server_vad defaults.
            defaults = ServerVad(type="server_vad")
            threshold, prefix_padding_ms = (
                defaults.threshold,
                defaults.prefix_padding_ms,
            )
            silence_ms = SEMANTIC_VAD_CANDIDATE_SILENCE_MS[config.eagerness]
            self._max_silence_ms = SEMANTIC_VAD_MAX_SILENCE_MS[config.eagerness]
            self._turn_scorer = turn_scorer or SMART_TURN.get().predict
            self._ring = _Ring()
        else:
            threshold, prefix_padding_ms = config.threshold, config.prefix_padding_ms
            silence_ms = config.silence_duration_ms
        self._endpointer = _SileroEndpointer(
            new_frame_scorer,
            start_sample=start_sample,
            threshold=threshold,
            prefix_padding_ms=prefix_padding_ms,
            silence_duration_ms=silence_ms,
        )
        self._pending = np.empty(0, dtype=np.float32)
        self._turn_active = False
        self._turn_start_ms = 0
        self._pause_start_ms: int | None = None

    @property
    def in_turn(self) -> bool:
        """A ``speech_started`` has not been followed by ``speech_stopped``."""
        if self._turn_scorer is None:
            return self._endpointer.speech_active
        return self._turn_active

    def process(self, audio: np.ndarray) -> list[TurnEvent]:
        """Advance over ``audio``; return the boundaries it completes, in order.

        A partial frame at the end is kept for the next call, so boundaries
        fall on the same frame grid however the audio is chunked.
        """
        samples = np.concatenate((self._pending, np.asarray(audio, dtype=np.float32)))
        events: list[TurnEvent] = []
        offset = 0
        while samples.size - offset >= FRAME_SAMPLES:
            frame = samples[offset : offset + FRAME_SAMPLES]
            offset += FRAME_SAMPLES
            if self._turn_scorer is None:
                event = self._endpointer.frame(frame)
                if event is not None and event.kind == "speech_stopped":
                    self._endpointer.reset()
            else:
                self._ring.append(frame)
                event = self._semantic_frame(frame)
            if event is not None:
                events.append(event)
        self._pending = samples[offset:].copy()
        return events

    def _semantic_frame(self, frame: np.ndarray) -> TurnEvent | None:
        """Silero hears pauses; Smart Turn decides whether they end the turn.

        A candidate pause never reaches the client. While paused the turn stays
        active, and speech that resumes belongs to the same turn.
        """
        inner = self._endpointer.frame(frame)
        if inner is not None and inner.kind == "speech_started":
            self._pause_start_ms = None
            if self._turn_active:
                return None
            self._turn_active = True
            self._turn_start_ms = inner.audio_ms
            return inner
        if not self._turn_active:
            return None

        complete = False
        if inner is not None:
            # A candidate pause. Score the whole turn so far; a turn that
            # resumed and paused again is scored with more context.
            self._pause_start_ms = self._endpointer.silence_start_ms
            complete = self._turn_complete()
        timed_out = (
            self._pause_start_ms is not None
            # Let an onset finish its confirmation before timing out.
            and not self._endpointer.speech_pending
            and _ms(self._endpointer.processed) - self._pause_start_ms
            >= self._max_silence_ms
        )
        if not (complete or timed_out):
            return None
        self._turn_active = False
        self._pause_start_ms = None
        self._endpointer.reset()
        return TurnEvent("speech_stopped", _ms(self._endpointer.processed))

    def _turn_complete(self) -> bool:
        assert self._turn_scorer is not None
        turn_ms = _ms(self._endpointer.processed) - self._turn_start_ms
        try:
            samples = turn_ms * SAMPLE_RATE_HZ // 1000
            probability = self._turn_scorer(self._ring.last(samples))
        except Exception:
            # Fall back to the eagerness silence limit rather than drop or
            # prematurely end the turn.
            logger.warning(
                "Smart Turn inference failed; treating the pause as incomplete",
                exc_info=True,
            )
            return False
        return probability >= SEMANTIC_VAD_COMPLETION_THRESHOLD
