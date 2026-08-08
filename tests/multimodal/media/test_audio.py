# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pybase64 as base64
import pytest

from vllm.multimodal.media import AudioMediaIO
from vllm.multimodal.media.audio import AUDIO_LOADER_REGISTRY, load_audio

from ...conftest import AudioTestAssets

pytestmark = pytest.mark.cpu_test

ASSETS_DIR = Path(__file__).parent.parent / "assets"
assert ASSETS_DIR.exists()

# Backends that decode compressed audio without extra system dependencies.
# "auto" is the default (soundfile, falling back to PyAV).
AUDIO_BACKENDS = ["auto", "soundfile", "pyav"]


@pytest.fixture
def dummy_audio():
    return np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=float)


@pytest.fixture
def dummy_audio_bytes(audio_assets: AudioTestAssets):
    with open(audio_assets[0].get_local_path(), "rb") as f:
        return f.read()


def test_audio_media_io_load_bytes(dummy_audio_bytes):
    audio_io = AudioMediaIO()
    out = audio_io.load_bytes(dummy_audio_bytes)
    assert isinstance(out[0], np.ndarray)
    assert out[1] == 16000


def test_audio_media_io_load_base64(dummy_audio_bytes):
    audio_io = AudioMediaIO()
    encoded = base64.b64encode(dummy_audio_bytes).decode("utf-8")
    out = audio_io.load_base64("audio/wav", encoded)
    assert isinstance(out[0], np.ndarray)
    assert out[1] == 16000


def test_audio_media_io_load_file(audio_assets: AudioTestAssets):
    audio_io = AudioMediaIO()
    path = audio_assets[0].get_local_path()
    out = audio_io.load_file(path)
    assert isinstance(out[0], np.ndarray)
    assert out[1] == 16000


def test_audio_media_io_encode_base64(dummy_audio):
    audio_io = AudioMediaIO()
    media = (dummy_audio, 16000)
    with patch("soundfile.write") as mock_write:

        def write_to_buffer(buffer, *_args, **_kwargs):
            buffer.write(b"dummy_wav_data")

        mock_write.side_effect = write_to_buffer

        out = audio_io.encode_base64(media)
        decoded = base64.b64decode(out)
        assert decoded == b"dummy_wav_data"
        mock_write.assert_called_once()


def test_load_audio_max_duration_respected(dummy_audio_bytes):
    """Valid audio within the duration limit should load successfully."""
    from io import BytesIO

    y, sr = load_audio(BytesIO(dummy_audio_bytes), sr=None, max_duration_s=3600)
    assert isinstance(y, np.ndarray)
    assert len(y) > 0


def test_load_audio_max_duration_rejected(dummy_audio_bytes):
    """Audio exceeding the duration limit must be rejected during decode."""
    from io import BytesIO

    with pytest.raises(ValueError, match="exceeds maximum allowed duration"):
        load_audio(BytesIO(dummy_audio_bytes), sr=None, max_duration_s=0.0001)


def test_audio_media_io_from_video(video_assets):
    audio_io = AudioMediaIO()
    video_path = video_assets[0].video_path
    with open(video_path, "rb") as f:
        audio, sr = audio_io.load_bytes(f.read())
    audio_ref, sr_ref = load_audio(video_path, sr=None)
    assert sr == sr_ref
    np.testing.assert_allclose(audio_ref, audio, atol=1e-4)


def test_audio_backend_registry_contents():
    """All shipped backends must be registered under stable names."""
    assert set(AUDIO_LOADER_REGISTRY.name2class) >= {
        "auto",
        "soundfile",
        "pyav",
        "torchcodec",
    }


def test_audio_media_io_default_backend_is_auto():
    """Omitting audio_backend must preserve the historical behaviour."""
    assert AudioMediaIO().audio_backend == "auto"


def test_audio_media_io_unknown_backend_rejected():
    """An unknown backend must fail loudly instead of silently degrading."""
    with pytest.raises(ValueError, match="Unknown audio_backend"):
        AudioMediaIO(audio_backend="not_a_backend")


@pytest.mark.parametrize("backend", AUDIO_BACKENDS)
def test_audio_media_io_backend_load_bytes(backend, dummy_audio_bytes):
    audio_io = AudioMediaIO(audio_backend=backend)
    assert audio_io.audio_backend == backend
    audio, sr = audio_io.load_bytes(dummy_audio_bytes)
    assert isinstance(audio, np.ndarray)
    assert sr == 16000


@pytest.mark.parametrize("backend", AUDIO_BACKENDS)
def test_audio_backend_matches_default(backend, dummy_audio_bytes):
    """Every backend must decode to the same samples as the default."""
    ref_audio, ref_sr = AudioMediaIO().load_bytes(dummy_audio_bytes)
    audio, sr = AudioMediaIO(audio_backend=backend).load_bytes(dummy_audio_bytes)
    assert sr == ref_sr
    np.testing.assert_allclose(ref_audio, audio, atol=1e-4)


def test_torchcodec_backend_matches_default(dummy_audio_bytes):
    """torchcodec must be sample-identical to the default backend."""
    pytest.importorskip("torchcodec")

    ref_audio, ref_sr = AudioMediaIO().load_bytes(dummy_audio_bytes)
    audio, sr = AudioMediaIO(audio_backend="torchcodec").load_bytes(dummy_audio_bytes)
    assert sr == ref_sr
    np.testing.assert_allclose(ref_audio, audio, atol=1e-4)


@pytest.mark.parametrize("mono", [True, False])
def test_torchcodec_bit_exact_at_native_rate(mono, dummy_audio_bytes):
    """torchcodec must be bit-exact with pyav on the path vLLM actually uses.

    `AudioMediaIO` always decodes at the native sample rate, and there the two
    backends must agree exactly rather than approximately. An explicit `sr`
    resamples at a different stage in each backend, so that path is not
    expected to be bit-exact and is deliberately not asserted here.
    """
    pytest.importorskip("torchcodec")

    from io import BytesIO

    from vllm.multimodal.media.audio import (
        load_audio_pyav,
        load_audio_torchcodec,
    )

    ref_audio, ref_sr = load_audio_pyav(BytesIO(dummy_audio_bytes), sr=None, mono=mono)
    audio, sr = load_audio_torchcodec(BytesIO(dummy_audio_bytes), sr=None, mono=mono)
    assert sr == ref_sr
    np.testing.assert_array_equal(ref_audio, audio)


def test_torchcodec_backend_extracts_audio_from_video(video_assets):
    """Extracting an audio track from a video must match the default backend."""
    pytest.importorskip("torchcodec")

    video_path = video_assets[0].video_path
    with open(video_path, "rb") as f:
        video_bytes = f.read()

    ref_audio, ref_sr = AudioMediaIO().load_bytes(video_bytes)
    audio, sr = AudioMediaIO(audio_backend="torchcodec").load_bytes(video_bytes)
    assert sr == ref_sr
    # Video containers are the motivating case for this backend, and
    # AudioMediaIO decodes at the native rate, so require bit-exactness.
    np.testing.assert_array_equal(ref_audio, audio)


def test_torchcodec_max_duration_rejected(dummy_audio_bytes):
    """The decompression-bomb guard must hold for torchcodec too."""
    pytest.importorskip("torchcodec")

    from io import BytesIO

    from vllm.multimodal.media.audio import load_audio_torchcodec

    with pytest.raises(ValueError, match="exceeds maximum allowed duration"):
        load_audio_torchcodec(
            BytesIO(dummy_audio_bytes), sr=None, max_duration_s=0.0001
        )


def test_torchcodec_max_duration_respected(dummy_audio_bytes):
    """Audio within the duration limit must still load."""
    pytest.importorskip("torchcodec")

    from io import BytesIO

    from vllm.multimodal.media.audio import load_audio_torchcodec

    audio, _ = load_audio_torchcodec(
        BytesIO(dummy_audio_bytes), sr=None, max_duration_s=3600
    )
    assert isinstance(audio, np.ndarray)
    assert len(audio) > 0


def test_torchcodec_max_duration_does_not_alter_output(dummy_audio_bytes):
    """A generous limit must not change the decoded samples.

    Enforcing the limit switches the decode to a bounded range request, so
    check that path returns exactly what the unbounded one does.
    """
    pytest.importorskip("torchcodec")

    from io import BytesIO

    from vllm.multimodal.media.audio import load_audio_torchcodec

    unbounded, sr_unbounded = load_audio_torchcodec(
        BytesIO(dummy_audio_bytes), sr=None, max_duration_s=None
    )
    bounded, sr_bounded = load_audio_torchcodec(
        BytesIO(dummy_audio_bytes), sr=None, max_duration_s=3600
    )
    assert sr_bounded == sr_unbounded
    np.testing.assert_array_equal(unbounded, bounded)


def test_torchcodec_max_duration_rejects_under_reported_duration(dummy_audio_bytes):
    """A container that under-reports its duration must still be rejected.

    Container metadata is attacker-controlled, so passing the header check
    cannot be enough to admit an input: the decoded sample count has to be
    re-checked. Simulate a lying header while leaving the real decode intact.
    """
    pytest.importorskip("torchcodec")

    from io import BytesIO

    from vllm.multimodal.media import audio as audio_module

    real_decoder_cls = audio_module.AudioDecoder
    true_duration = real_decoder_cls(
        BytesIO(dummy_audio_bytes)
    ).metadata.duration_seconds
    limit = true_duration / 2

    class UnderReportingMetadata:
        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return getattr(self._real, name)

        @property
        def duration_seconds(self):
            # Claims to fit within the limit while the stream does not.
            return limit / 2

    class UnderReportingDecoder:
        def __init__(self, *args, **kwargs):
            self._real = real_decoder_cls(*args, **kwargs)

        @property
        def metadata(self):
            return UnderReportingMetadata(self._real.metadata)

        def __getattr__(self, name):
            return getattr(self._real, name)

    with (
        patch.object(audio_module, "AudioDecoder", UnderReportingDecoder),
        pytest.raises(ValueError, match="exceeds maximum allowed duration"),
    ):
        audio_module.load_audio_torchcodec(
            BytesIO(dummy_audio_bytes), sr=None, max_duration_s=limit
        )
