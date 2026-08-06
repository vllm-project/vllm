# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import subprocess
import sys
import textwrap
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pybase64 as base64
import pytest
import soundfile as sf

from vllm.multimodal.media import AudioMediaIO
from vllm.multimodal.media.audio import (
    AUDIO_LOADER_REGISTRY,
    load_audio,
    load_audio_soundfile,
)

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
    y, sr = load_audio(BytesIO(dummy_audio_bytes), sr=None, max_duration_s=3600)
    assert isinstance(y, np.ndarray)
    assert len(y) > 0


def test_load_audio_max_duration_rejected(dummy_audio_bytes):
    """Audio exceeding the duration limit must be rejected during decode."""
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


def _make_flac_bytes(frames: int, channels: int, samplerate: int) -> bytes:
    """Create a minimal FLAC file in memory for testing."""
    data = np.zeros((frames, channels), dtype=np.int16)
    buf = BytesIO()
    sf.write(buf, data, samplerate, format="FLAC")
    return buf.getvalue()


def test_small_file_passes_memory_guard():
    """A small valid file should pass both duration and memory guards."""
    payload = _make_flac_bytes(frames=16000, channels=1, samplerate=16000)
    y, sr = load_audio_soundfile(
        BytesIO(payload),
        sr=None,
        max_duration_s=600,
        max_decode_bytes=256 * 1024 * 1024,
    )
    assert isinstance(y, np.ndarray)
    assert len(y) == 16000


def test_memory_guard_rejects_large_allocation():
    """A file whose frames*channels*4 exceeds the byte limit must be
    rejected before allocating the buffer."""
    # 100_000 frames * 8 channels * 4 bytes = 3.2 MB
    payload = _make_flac_bytes(frames=100_000, channels=8, samplerate=48000)
    # Set limit to 1 MiB — should reject
    with pytest.raises(ValueError, match="VLLM_MAX_AUDIO_DECODE_BYTES"):
        load_audio_soundfile(
            BytesIO(payload),
            sr=None,
            max_duration_s=600,
            max_decode_bytes=1 * 1024 * 1024,
        )


def test_forged_samplerate_rejected_by_memory_guard():
    """The PoC scenario: high sample rate fools the duration guard but
    the memory guard catches the large frame*channel allocation."""
    # Forged high sample rate: 655350 Hz, 8 channels, 1M frames
    # Duration guard sees: 1_000_000 / 655_350 = 1.5s → passes
    # Memory: 1_000_000 * 8 * 4 = 32 MB
    payload = _make_flac_bytes(frames=1_000_000, channels=8, samplerate=655350)
    # Set memory limit to 16 MiB — below the 32 MB allocation
    with pytest.raises(ValueError, match="VLLM_MAX_AUDIO_DECODE_BYTES"):
        load_audio_soundfile(
            BytesIO(payload),
            sr=None,
            max_duration_s=600,
            max_decode_bytes=16 * 1024 * 1024,
        )


def test_load_audio_threads_max_decode_bytes():
    """Verify load_audio passes max_decode_bytes through to backend."""
    # 50_000 frames * 4 channels * 4 bytes = 800 KB
    payload = _make_flac_bytes(frames=50_000, channels=4, samplerate=44100)
    # Limit of 512 KB should reject
    with pytest.raises(ValueError, match="VLLM_MAX_AUDIO_DECODE_BYTES"):
        load_audio(
            BytesIO(payload),
            sr=None,
            max_duration_s=600,
            max_decode_bytes=512 * 1024,
        )


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


def test_torchcodec_resamples_without_pyav(dummy_audio_bytes):
    """torchcodec must resample itself, not fall back to PyAV.

    Passing an ``sr`` that differs from the native rate exercises the
    torchcodec-internal resampler. The assertion is not just that the output
    rate matches ``sr`` — ``resample_audio_pyav`` is patched to fail if called,
    so a regression that routes resampling back through PyAV turns this test
    red rather than silently producing the right sample rate through PyAV.
    """
    pytest.importorskip("torchcodec")

    from io import BytesIO

    from vllm.multimodal.media import audio as audio_module
    from vllm.multimodal.media.audio import load_audio_torchcodec

    def _fail_if_resample_called(*args, **kwargs):
        raise AssertionError(
            "torchcodec backend must not fall back to resample_audio_pyav"
        )

    with patch.object(audio_module, "resample_audio_pyav", _fail_if_resample_called):
        audio, sr = load_audio_torchcodec(BytesIO(dummy_audio_bytes), sr=8000)

    assert sr == 8000
    assert isinstance(audio, np.ndarray)
    assert len(audio) > 0


def _under_reporting_decoder_cls(real_decoder_cls, fake_duration):
    """Wrap a real AudioDecoder so metadata reports ``fake_duration``.

    The decode itself is left intact — only ``metadata.duration_seconds``
    lies. This forces the duration guard's decision onto the post-decode
    re-check of the actual sample count, which is what both tests below
    exercise (one with ``sr=None``, one with an explicit ``sr``).
    """

    class UnderReportingMetadata:
        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return getattr(self._real, name)

        @property
        def duration_seconds(self):
            # Claims to fit within the limit while the stream does not.
            return fake_duration

    class UnderReportingDecoder:
        def __init__(self, *args, **kwargs):
            self._real = real_decoder_cls(*args, **kwargs)

        @property
        def metadata(self):
            return UnderReportingMetadata(self._real.metadata)

        def __getattr__(self, name):
            return getattr(self._real, name)

    return UnderReportingDecoder


def test_torchcodec_duration_guard_uses_output_rate(dummy_audio_bytes):
    """The post-decode duration guard must use the *output* sample rate.

    With ``sr`` set, torchcodec returns samples at that rate, so comparing
    against a threshold derived from the native rate would let an over-long
    stream through. A lying header (under-reported duration) lets the metadata
    pre-check pass, forcing the decision onto the post-decode re-check — which
    must reject, and can only reject if the sample-count threshold is derived
    from the output rate.
    """
    pytest.importorskip("torchcodec")

    from io import BytesIO

    from vllm.multimodal.media import audio as audio_module

    real_decoder_cls = audio_module.AudioDecoder
    true_duration = real_decoder_cls(
        BytesIO(dummy_audio_bytes)
    ).metadata.duration_seconds
    limit = true_duration / 2
    fake_duration = limit / 2
    under_reporting_decoder = _under_reporting_decoder_cls(
        real_decoder_cls, fake_duration
    )

    with (
        patch.object(audio_module, "AudioDecoder", under_reporting_decoder),
        pytest.raises(ValueError, match="exceeds maximum allowed duration"),
    ):
        audio_module.load_audio_torchcodec(
            BytesIO(dummy_audio_bytes), sr=8000, max_duration_s=limit
        )


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


def test_torchcodec_max_duration_at_boundary_not_rejected(dummy_audio_bytes):
    """A stream whose duration equals the limit must not be rejected.

    Mirrors load_audio_pyav, which compares the decoded sample count against
    ``int(sr * max_duration_s)`` with strict ``>``: the ``int()`` truncation
    gives a one-sample grace so a stream landing exactly on the limit is
    admitted rather than rejected by rounding.
    """
    pytest.importorskip("torchcodec")

    from io import BytesIO

    from vllm.multimodal.media.audio import load_audio_torchcodec

    # Use the container's own reported duration as the limit, so the stream
    # lands exactly on the boundary rather than just under it.
    from torchcodec.decoders import AudioDecoder

    true_duration = AudioDecoder(BytesIO(dummy_audio_bytes)).metadata.duration_seconds
    audio, sr = load_audio_torchcodec(
        BytesIO(dummy_audio_bytes), sr=None, max_duration_s=true_duration
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
    under_reporting_decoder = _under_reporting_decoder_cls(
        real_decoder_cls, limit / 2
    )

    with (
        patch.object(audio_module, "AudioDecoder", under_reporting_decoder),
        pytest.raises(ValueError, match="exceeds maximum allowed duration"),
    ):
        audio_module.load_audio_torchcodec(
            BytesIO(dummy_audio_bytes), sr=None, max_duration_s=limit
        )


def test_torchcodec_max_decode_bytes_rejected(dummy_audio_bytes):
    """The memory guard must hold for torchcodec too.

    Mirrors the soundfile/pyav ``max_decode_bytes`` tests: a limit below the
    stream's real PCM size must be rejected before the decoded buffer is
    handed back.
    """
    pytest.importorskip("torchcodec")

    from io import BytesIO

    from vllm.multimodal.media.audio import load_audio_torchcodec

    # First decode unbounded to learn the real PCM footprint, then set the
    # limit just below it so the guard must trip.
    audio, sr = load_audio_torchcodec(BytesIO(dummy_audio_bytes), sr=None)
    real_bytes = audio.nbytes
    assert real_bytes > 0

    with pytest.raises(ValueError, match="VLLM_MAX_AUDIO_DECODE_BYTES"):
        load_audio_torchcodec(
            BytesIO(dummy_audio_bytes),
            sr=None,
            max_decode_bytes=real_bytes - 1,
        )


def test_torchcodec_backend_works_without_audio_extras(audio_assets):
    """The torchcodec backend must work with no PyAV/libsndfile installed.

    Masking ``av`` and ``soundfile`` in-process is meaningless: ``audio.py``
    imports them at module load under ``try/except``, which has already run by
    the time any test starts. Run a fresh interpreter where both look absent,
    so the only way to decode is the self-contained torchcodec path. This is
    the property the PR sells: audio support without ``pip install vllm[audio]``.
    """
    pytest.importorskip("torchcodec")

    asset_path = audio_assets[0].get_local_path()
    code = textwrap.dedent(
        """
        import pathlib
        import sys

        # Make PyAV and libsndfile look absent so only torchcodec can serve.
        for name in ("av", "soundfile"):
            sys.modules[name] = None

        from vllm.multimodal.media import AudioMediaIO

        io = AudioMediaIO(audio_backend="torchcodec")
        audio, sr = io.load_file(pathlib.Path(sys.argv[1]))
        assert audio.size > 0, audio.size
        assert sr > 0, sr
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(asset_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"torchcodec backend failed with no PyAV/libsndfile:\n"
        f"{result.stderr}"
    )
