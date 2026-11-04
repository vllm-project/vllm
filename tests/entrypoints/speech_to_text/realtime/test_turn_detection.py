# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Turn detection for ``/v1/realtime``: the endpointing state machine on
scripted Silero scores, and the Rust models against ONNX Runtime reference
scores. The WebSocket path is tested in test_realtime_validation.py."""

import numpy as np
import pytest

from vllm.entrypoints.speech_to_text.realtime import turn_detection as td
from vllm.entrypoints.speech_to_text.realtime.protocol import SemanticVad, ServerVad

FRAME = td.FRAME_SAMPLES
FRAME_MS = 32


def scripted(probabilities: list[float]):
    """A Silero stand-in: consecutive frames get these scores, across resets."""
    scores = iter(probabilities)
    return lambda: lambda frame: next(scores)


def frames(n: int) -> np.ndarray:
    return np.zeros(n * FRAME, dtype=np.float32)


def events(detector: td.TurnDetector, audio: np.ndarray, chunk: int) -> list:
    out = []
    for offset in range(0, audio.size, chunk):
        out += detector.process(audio[offset : offset + chunk])
    return [(e.kind, e.audio_ms) for e in out]


# 10 silent frames, 20 speech frames (onset at 320 ms), then silence.
SPEECH = [0.0] * 10 + [0.9] * 20


@pytest.mark.parametrize("chunk", [7, FRAME, 1600, 100 * FRAME])
def test_server_vad_boundaries_do_not_depend_on_chunking(chunk):
    detector = td.TurnDetector(
        ServerVad(type="server_vad"), new_frame_scorer=scripted(SPEECH + [0.0] * 30)
    )
    # Start: onset 320 ms minus 300 ms prefix padding. Stop: after 500 ms of
    # silence (16 frames, the 16th ends at frame 46), trailing silence included.
    assert events(detector, frames(60), chunk) == [
        ("speech_started", 20),
        ("speech_stopped", 46 * FRAME_MS),
    ]


def test_server_vad_scores_in_the_hysteresis_band_do_not_end_the_turn():
    # 0.4 is below threshold 0.5 but above 0.5 - 0.15: it neither starts the
    # silence timer nor ends the turn; the 16 silent frames after it do.
    probabilities = SPEECH + [0.4] * 30 + [0.0] * 16
    detector = td.TurnDetector(
        ServerVad(type="server_vad"), new_frame_scorer=scripted(probabilities)
    )
    assert events(detector, frames(76), FRAME) == [
        ("speech_started", 20),
        ("speech_stopped", 76 * FRAME_MS),
    ]


def test_server_vad_padding_never_reaches_into_the_previous_turn():
    # The second onset (frame 48, 1536 ms) is 64 ms after the first turn
    # ended at 1472 ms, closer than the 300 ms prefix padding.
    probabilities = SPEECH + [0.0] * 18 + [0.9] * 5
    detector = td.TurnDetector(
        ServerVad(type="server_vad"), new_frame_scorer=scripted(probabilities)
    )
    assert events(detector, frames(53), FRAME) == [
        ("speech_started", 20),
        ("speech_stopped", 46 * FRAME_MS),
        ("speech_started", 46 * FRAME_MS),
    ]


def test_detector_clock_starts_where_it_was_enabled():
    # Enabled after 1 s of session audio.
    detector = td.TurnDetector(
        ServerVad(type="server_vad"),
        start_sample=td.SAMPLE_RATE_HZ,
        new_frame_scorer=scripted(SPEECH + [0.0] * 30),
    )
    assert events(detector, frames(60), FRAME) == [
        ("speech_started", 1000 + 20),
        ("speech_stopped", 1000 + 46 * FRAME_MS),
    ]


class TurnScorer:
    def __init__(self, *probabilities: float):
        self.probabilities = list(probabilities)
        self.windows: list[int] = []

    def __call__(self, audio: np.ndarray) -> float:
        self.windows.append(audio.size)
        return self.probabilities.pop(0)


def semantic(probabilities: list[float], scorer: TurnScorer) -> td.TurnDetector:
    return td.TurnDetector(
        SemanticVad(type="semantic_vad", eagerness="high"),
        new_frame_scorer=scripted(probabilities),
        turn_scorer=scorer,
    )


def test_semantic_vad_ends_a_complete_turn_at_the_candidate_pause():
    scorer = TurnScorer(0.9)
    detector = semantic(SPEECH + [0.0] * 30, scorer)
    # eagerness=high: a 300 ms pause (10 frames, ending at frame 40) is a
    # candidate; Smart Turn calls it complete.
    assert events(detector, frames(60), FRAME) == [
        ("speech_started", 20),
        ("speech_stopped", 40 * FRAME_MS),
    ]
    # Smart Turn saw the turn from its padded start to the end of the pause.
    assert scorer.windows == [(40 * FRAME_MS - 20) * 16]


def test_semantic_vad_waits_out_an_incomplete_turn():
    detector = semantic(SPEECH + [0.0] * 70, TurnScorer(0.1))
    # Incomplete: the turn ends once the pause (from 960 ms) reaches the
    # 2000 ms limit for eagerness=high, at the first frame ending >= 2960 ms.
    assert events(detector, frames(100), FRAME) == [
        ("speech_started", 20),
        ("speech_stopped", 93 * FRAME_MS),
    ]


def test_semantic_vad_resumed_speech_stays_in_the_same_turn():
    scorer = TurnScorer(0.1, 0.9)
    probabilities = SPEECH + [0.0] * 12 + [0.9] * 20 + [0.0] * 20
    detector = semantic(probabilities, scorer)
    # One turn: the first pause is incomplete, speech resumes, the second
    # pause (frames 62-71) is complete.
    assert events(detector, frames(82), FRAME) == [
        ("speech_started", 20),
        ("speech_stopped", 72 * FRAME_MS),
    ]
    assert scorer.windows == [(40 * FRAME_MS - 20) * 16, (72 * FRAME_MS - 20) * 16]


def test_semantic_vad_survives_a_failing_turn_scorer():
    def failing(audio):
        raise RuntimeError("boom")

    detector = td.TurnDetector(
        SemanticVad(type="semantic_vad", eagerness="high"),
        new_frame_scorer=scripted(SPEECH + [0.0] * 70),
        turn_scorer=failing,
    )
    # Treated as incomplete: the eagerness limit still ends the turn.
    assert events(detector, frames(100), FRAME) == [
        ("speech_started", 20),
        ("speech_stopped", 93 * FRAME_MS),
    ]


# ONNX Runtime 1.31.1 on the same pinned files, through vLLM-Omni's Silero and
# Smart Turn backends: every 16th Silero frame of mary_had_lamb, and Smart Turn
# on the first 3 s, the first 6 s and the whole clip.
ORT_SILERO_EVERY_16TH = [
    0.005679, 0.007078, 0.798167, 0.249491, 0.793731, 0.942799, 0.995527,
    0.092292, 0.997883, 0.998938, 0.444304, 0.286369, 0.965487, 0.945619,
    0.986361, 0.973971, 0.970086, 0.644264, 0.993253, 0.988209, 0.995311,
    0.922999, 0.952017, 0.998408, 0.999279, 0.99772, 0.993466, 0.998234,
    0.993555, 0.949062, 0.088787, 0.007788,
]  # fmt: skip
ORT_SMART_TURN = {
    ("mary_had_lamb", 3): 0.021028,
    ("mary_had_lamb", 6): 0.925185,
    ("mary_had_lamb", None): 0.058758,
    ("winning_call", 3): 0.033202,
    ("winning_call", 6): 0.051862,
    ("winning_call", None): 0.009897,
}


def asset_audio(name: str) -> np.ndarray:
    from vllm.assets.audio import AudioAsset
    from vllm.multimodal.media.audio import load_audio

    audio, _ = load_audio(str(AudioAsset(name).get_local_path()), sr=16000, mono=True)
    return np.ascontiguousarray(audio, dtype=np.float32)


@pytest.fixture(scope="module")
def rust_models():
    return td.SILERO.get(), td.SMART_TURN.get()


def test_silero_matches_onnxruntime(rust_models):
    silero, _ = rust_models
    audio = asset_audio("mary_had_lamb")
    state = silero.new_state()
    probabilities = []
    for offset in range(0, audio.size - FRAME + 1, FRAME):
        probability, state = silero.infer(audio[offset : offset + FRAME], state)
        probabilities.append(probability)
    np.testing.assert_allclose(
        probabilities[::16], ORT_SILERO_EVERY_16TH, rtol=0, atol=5e-5
    )


def test_smart_turn_matches_onnxruntime(rust_models):
    _, smart_turn = rust_models
    for (name, seconds), expected in ORT_SMART_TURN.items():
        audio = asset_audio(name)
        if seconds is not None:
            audio = np.ascontiguousarray(audio[: seconds * td.SAMPLE_RATE_HZ])
        assert smart_turn.predict(audio) == pytest.approx(expected, abs=1e-4), name
