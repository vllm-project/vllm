# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import av
import numpy as np
import pybase64 as base64
import pytest
import soundfile as sf
from torch.profiler import ProfilerActivity, profile

import vllm.envs as envs
from vllm.multimodal.media import AudioMediaIO
from vllm.multimodal.media import audio as audio_module
from vllm.multimodal.media.audio import (
    load_audio,
    load_audio_soundfile,
    load_audio_torchcodec,
)
from vllm.multimodal.media.base import MediaDecodedBudget

from ...conftest import AudioTestAssets

pytestmark = pytest.mark.cpu_test

ASSETS_DIR = Path(__file__).parent.parent / "assets"
assert ASSETS_DIR.exists()


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


def test_audio_media_io_load_base64_rejects_malformed(dummy_audio_bytes):
    """Malformed base64 must surface as a ValueError so the server answers 400.
    Without strict decoding the bad characters are dropped, the garbage reaches
    libsndfile, and the client gets a 500 instead."""
    encoded = base64.b64encode(dummy_audio_bytes).decode("utf-8")
    malformed = encoded[:8] + "!!!@@@###" + encoded[8:]

    audio_io = AudioMediaIO()
    with pytest.raises(ValueError):
        audio_io.load_base64("audio/wav", malformed)


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


@pytest.mark.parametrize("backend", ["soundfile", "pyav", "torchcodec"])
@pytest.mark.parametrize(
    "duration_limit,decode_limit",
    [(0, 256 * 1024 * 1024), (600, 0), (0, 0)],
    ids=["duration", "decoded-bytes", "both"],
)
def test_load_audio_zero_decode_limits_are_disabled(
    backend, duration_limit, decode_limit, dummy_audio_bytes
):
    """Zero decode limits are disabled consistently across backends."""
    audio, sample_rate = load_audio(
        BytesIO(dummy_audio_bytes),
        sr=None,
        max_duration_s=duration_limit,
        max_decode_bytes=decode_limit,
        backend=backend,
    )

    assert audio.size > 0
    assert sample_rate == 16000


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


@pytest.mark.parametrize(
    ("declared_frames", "max_duration_s", "sample_rate"),
    [(0, None, None), (0, 600, 16000), (1, 600, None), (1, None, 16000)],
)
def test_torchcodec_bounds_allocation_with_unreliable_duration(
    declared_frames, max_duration_s, sample_rate
):
    """Missing or understated duration must not allow a full PCM allocation."""
    payload = bytearray(_make_flac_bytes(768000, 8, 192000))
    stream_info = int.from_bytes(payload[18:26], "big")
    stream_info = (stream_info & ~((1 << 36) - 1)) | declared_frames
    payload[18:26] = stream_info.to_bytes(8, "big")
    payload[26:42] = bytes(16)
    load_audio_torchcodec(BytesIO(_make_flac_bytes(32, 1, 16000)))

    with (
        profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof,
        pytest.raises(ValueError, match="VLLM_MAX_AUDIO_DECODE_BYTES"),
    ):
        load_audio_torchcodec(
            BytesIO(payload),
            sr=sample_rate,
            max_duration_s=max_duration_s,
            max_decode_bytes=64 * 1024,
        )

    # Allow codec-frame buffers and concatenation, but not the full 2–24 MiB PCM.
    largest_allocation = max(event.cpu_memory_usage for event in prof.events())
    assert 0 < largest_allocation < 1024 * 1024


@pytest.mark.parametrize("mono", [False, True])
def test_torchcodec_unknown_duration_preserves_audio_at_byte_limit(mono):
    """A short high-rate stream remains valid even without a declared length."""
    payload = bytearray(_make_flac_bytes(4096, 8, 192000))
    payload[21] &= 0xF0
    payload[22:26] = bytes(4)
    payload[26:42] = bytes(16)
    expected, expected_sr = load_audio_torchcodec(BytesIO(payload), sr=None, mono=mono)

    actual, actual_sr = load_audio_torchcodec(
        BytesIO(payload),
        sr=None,
        mono=mono,
        max_duration_s=600,
        max_decode_bytes=4096 * 8 * 4,
    )

    assert actual_sr == expected_sr == 192000
    np.testing.assert_array_equal(actual, expected)


def _make_timestamped_flac_bytes(
    frames: int,
    *,
    sample_rate: int = 16000,
    start_samples: int = 0,
    constant_pts: bool = False,
) -> bytes:
    buffer = BytesIO()
    with av.open(
        buffer,
        mode="w",
        format="matroska",
        options={"live": "1", "avoid_negative_ts": "disabled"},
    ) as container:
        stream = container.add_stream("flac", rate=sample_rate)
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(
            np.arange(frames, dtype=np.int16).reshape(1, -1),
            format="s16",
            layout="mono",
        )
        frame.sample_rate = sample_rate
        frame.pts = start_samples
        frame.time_base = Fraction(1, sample_rate)
        for packet in stream.encode(frame):
            if constant_pts:
                packet.pts = packet.dts = 0
            container.mux(packet)
        for packet in stream.encode(None):
            if constant_pts:
                packet.pts = packet.dts = 0
            container.mux(packet)
    return buffer.getvalue()


@pytest.mark.parametrize("start_samples", [4000, -4000])
@pytest.mark.parametrize("over_limit", [False, True])
@pytest.mark.parametrize("output_rate", [None, 22050])
def test_torchcodec_byte_limit_preserves_timestamped_audio(
    start_samples, over_limit, output_rate
):
    """Keep native playback trimming before any bounded sample-rate conversion."""
    payload = _make_timestamped_flac_bytes(12800, start_samples=start_samples)
    metadata = audio_module.AudioDecoder(BytesIO(payload)).metadata
    assert metadata.duration_seconds is None
    assert metadata.begin_stream_seconds == start_samples / 16000
    expected, expected_sr = load_audio_torchcodec(BytesIO(payload), sr=output_rate)
    if output_rate is not None and start_samples < 0:
        # Native trimming has different boundaries from integrated resampling.
        native, native_sr = load_audio_torchcodec(BytesIO(payload), sr=None)
        expected = audio_module.resample_audio_pyav(
            native, orig_sr=native_sr, target_sr=output_rate
        )
    # The budget also counts the frames discarded before playback time zero.
    decode_limit = max(12800 * 4, expected.nbytes)
    decode_limit -= 4 if over_limit else 0
    if over_limit:
        with pytest.raises(ValueError, match="VLLM_MAX_AUDIO_DECODE_BYTES"):
            load_audio_torchcodec(
                BytesIO(payload), sr=output_rate, max_decode_bytes=decode_limit
            )
    else:
        actual, actual_sr = load_audio_torchcodec(
            BytesIO(payload), sr=output_rate, max_decode_bytes=decode_limit
        )
        assert actual_sr == expected_sr
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("output_rate", [8000, 48000])
@pytest.mark.parametrize("mono", [False, True])
def test_torchcodec_byte_limit_preserves_resampled_audio(output_rate, mono):
    """Ordinary resampling and downmixing match the unlimited decoder exactly."""
    buffer = BytesIO()
    sf.write(
        buffer,
        np.arange(8192, dtype=np.int16).reshape(4096, 2),
        16000,
        format="FLAC",
    )
    payload = buffer.getvalue()
    expected, expected_sr = load_audio_torchcodec(
        BytesIO(payload), sr=output_rate, mono=mono
    )
    # Both channels are allocated before downmixing.
    decode_limit = max(4096, 4096 * output_rate // 16000) * 2 * 4
    actual, actual_sr = load_audio_torchcodec(
        BytesIO(payload), sr=output_rate, mono=mono, max_decode_bytes=decode_limit
    )
    assert actual_sr == expected_sr == output_rate
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="VLLM_MAX_AUDIO_DECODE_BYTES"):
        load_audio_torchcodec(
            BytesIO(payload),
            sr=output_rate,
            mono=mono,
            max_decode_bytes=decode_limit - 1,
        )


@pytest.mark.parametrize("output_rate", [8000, 48000])
def test_torchcodec_resampling_budget_includes_short_input_padding(output_rate):
    """Reject padded resampler allocations even when the final waveform fits."""
    payload = _make_timestamped_flac_bytes(64)
    native, native_sr = load_audio_torchcodec(BytesIO(payload), sr=None)
    expected = audio_module.resample_audio_pyav(
        native, orig_sr=native_sr, target_sr=output_rate
    )
    decode_limit = 1024 * output_rate // native_sr * 4
    assert expected.nbytes < decode_limit
    with pytest.raises(ValueError, match="Audio resampling would exceed"):
        load_audio_torchcodec(
            BytesIO(payload), sr=output_rate, max_decode_bytes=decode_limit - 1
        )
    actual, actual_sr = load_audio_torchcodec(
        BytesIO(payload), sr=output_rate, max_decode_bytes=decode_limit
    )
    assert actual_sr == output_rate
    assert actual.shape == (64 * output_rate // native_sr,)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "sample_rate,output_rate,constant_pts",
    [(16000, None, True), (1, 16000, False)],
)
def test_torchcodec_bounds_allocation_before_accumulation_or_resampling(
    sample_rate, output_rate, constant_pts
):
    """Neither packet timestamps nor resampling may bypass the PCM budget."""
    frames = 640000 if constant_pts else 64
    payload = _make_timestamped_flac_bytes(
        frames, sample_rate=sample_rate, constant_pts=constant_pts
    )
    load_audio_torchcodec(BytesIO(_make_flac_bytes(32, 1, 16000)))

    with (
        profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof,
        pytest.raises(ValueError, match="VLLM_MAX_AUDIO_DECODE_BYTES"),
    ):
        load_audio_torchcodec(
            BytesIO(payload), sr=output_rate, max_decode_bytes=64 * 1024
        )

    largest_allocation = max(event.cpu_memory_usage for event in prof.events())
    assert 0 < largest_allocation < 1024 * 1024


def test_torchcodec_preflight_respects_duration_limit():
    """A generous byte budget must not defer the duration rejection."""
    payload = _make_timestamped_flac_bytes(64000, constant_pts=True)
    with (
        profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof,
        pytest.raises(ValueError, match="VLLM_MAX_AUDIO_DECODE_DURATION_S"),
    ):
        load_audio_torchcodec(
            BytesIO(payload),
            sr=None,
            max_duration_s=0.1,
            max_decode_bytes=1024 * 1024,
        )
    largest_allocation = max(event.cpu_memory_usage for event in prof.events())
    assert 0 < largest_allocation < 64 * 1024


@pytest.mark.skip_global_cleanup
def test_request_budget_remains_enforced_when_audio_decode_limit_disabled(
    monkeypatch,
):
    monkeypatch.setattr(envs, "VLLM_MAX_AUDIO_DECODE_DURATION_S", 0)
    monkeypatch.setattr(envs, "VLLM_MAX_AUDIO_DECODE_BYTES", 0)
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", 384)
    clip = _make_flac_bytes(frames=64, channels=1, samplerate=16000)
    audio_io = AudioMediaIO(
        audio_backend="soundfile", decoded_budget=MediaDecodedBudget()
    )

    audio, sample_rate = audio_io.load_bytes(clip)
    assert sample_rate == 16000
    assert audio.nbytes == 256

    with pytest.raises(ValueError, match="VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES"):
        audio_io.load_bytes(clip)


@pytest.mark.skip_global_cleanup
def test_request_budget_allows_parallel_audio_decodes(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", 512)
    monkeypatch.setattr(envs, "VLLM_MAX_AUDIO_DECODE_BYTES", 256)
    budget = MediaDecodedBudget()
    audio_io = AudioMediaIO(decoded_budget=budget)
    decoding = threading.Barrier(2)

    def decode(*args, **kwargs):
        assert kwargs["max_decode_bytes"] == 256
        decoding.wait(timeout=5)
        return np.zeros(32, dtype=np.float32), 16000

    with (
        patch.object(audio_module, "load_audio", side_effect=decode),
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        results = list(pool.map(audio_io.load_bytes, [b"a", b"b"]))

    assert len(results) == 2
    assert budget.used == 256


@pytest.mark.skip_global_cleanup
def test_request_budget_waits_for_inflight_audio_decode(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", 256)
    monkeypatch.setattr(envs, "VLLM_MAX_AUDIO_DECODE_BYTES", 256)
    budget = MediaDecodedBudget()
    audio_io = AudioMediaIO(decoded_budget=budget)
    decoding = threading.Event()
    finish_decoding = threading.Event()
    second_reservation = threading.Event()
    reserve = budget.reserve
    decode_calls = 0

    def track_reservation(max_bytes):
        if decoding.is_set():
            second_reservation.set()
        return reserve(max_bytes)

    def decode(*args, **kwargs):
        nonlocal decode_calls
        decode_calls += 1
        decoding.set()
        assert finish_decoding.wait(timeout=5)
        return np.zeros(64, dtype=np.float32), 16000

    with (
        patch.object(budget, "reserve", side_effect=track_reservation),
        patch.object(audio_module, "load_audio", side_effect=decode),
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        first = pool.submit(audio_io.load_bytes, b"a")
        assert decoding.wait(timeout=5)
        second = pool.submit(audio_io.load_bytes, b"b")
        try:
            assert second_reservation.wait(timeout=5)
        finally:
            finish_decoding.set()
        first.result(timeout=5)
        with pytest.raises(ValueError, match="VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES"):
            second.result(timeout=5)

    assert decode_calls == 1
    assert budget.used == 256


@pytest.mark.skip_global_cleanup
def test_request_budget_releases_failed_audio_reservation(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", 256)
    monkeypatch.setattr(envs, "VLLM_MAX_AUDIO_DECODE_BYTES", 256)
    budget = MediaDecodedBudget()
    audio_io = AudioMediaIO(decoded_budget=budget)
    audio = (np.zeros(64, dtype=np.float32), 16000)

    with patch.object(
        audio_module, "load_audio", side_effect=[ValueError("invalid audio"), audio]
    ):
        with pytest.raises(ValueError, match="invalid audio"):
            audio_io.load_bytes(b"a")
        assert audio_io.load_bytes(b"b")[0].nbytes == 256

    assert budget.used == 256


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


@pytest.mark.parametrize("backend", ["soundfile", "pyav", "torchcodec"])
def test_load_audio_backend_matches_default(backend, dummy_audio_bytes):
    """Every explicit backend must decode to the same samples as `auto`."""
    if backend == "torchcodec":
        pytest.importorskip("torchcodec")
    ref_audio, ref_sr = load_audio(BytesIO(dummy_audio_bytes), sr=None)
    audio, sr = load_audio(BytesIO(dummy_audio_bytes), sr=None, backend=backend)
    assert sr == ref_sr
    if backend == "torchcodec":
        # Vorbis encoder padding is trimmed, so torchcodec must match the
        # soundfile reference exactly, including the waveform length.
        assert audio.shape == ref_audio.shape
        np.testing.assert_allclose(ref_audio, audio, atol=1e-4)
        return
    # Decoders disagree only on codec encoder-delay/padding, so PyAV may
    # emit a few extra trailing samples (e.g. ~192 for Ogg Vorbis). Compare
    # the overlapping region, which must agree to float32 precision.
    n = min(ref_audio.shape[-1], audio.shape[-1])
    assert n > 0
    np.testing.assert_allclose(ref_audio[:n], audio[:n], atol=1e-4)


def test_load_audio_default_preserves_vorbis_length(dummy_audio_bytes):
    """Default decoding must not append Vorbis padding to the speech waveform."""
    expected, expected_sr = load_audio_soundfile(BytesIO(dummy_audio_bytes), sr=None)
    audio, sr = load_audio(BytesIO(dummy_audio_bytes), sr=None)
    assert sr == expected_sr
    assert audio.shape == expected.shape


def test_load_audio_torchcodec_preserves_vorbis_length(dummy_audio_bytes):
    """Torchcodec must trim Vorbis trailing padding even on FFmpeg < 5.0,
    matching the soundfile reference exactly."""
    pytest.importorskip("torchcodec")
    expected, expected_sr = load_audio_soundfile(BytesIO(dummy_audio_bytes), sr=None)
    audio, sr = load_audio_torchcodec(BytesIO(dummy_audio_bytes), sr=None)
    assert sr == expected_sr
    assert audio.shape == expected.shape
    np.testing.assert_allclose(audio, expected, atol=1e-4)


@pytest.mark.parametrize("sample_rate", [16000, 22050])
def test_torchcodec_byte_limit_preserves_vorbis_resampling(
    dummy_audio_bytes, sample_rate
):
    """Budget checks preserve Vorbis trimming at the requested output rate."""
    expected, expected_sr = load_audio_torchcodec(
        BytesIO(dummy_audio_bytes), sr=sample_rate
    )
    audio, audio_sr = load_audio_torchcodec(
        BytesIO(dummy_audio_bytes), sr=sample_rate, max_decode_bytes=256 * 1024 * 1024
    )
    assert audio_sr == expected_sr
    np.testing.assert_array_equal(audio, expected)


def test_load_audio_unknown_backend_rejected(dummy_audio_bytes):
    """An unknown backend must fail loudly instead of silently degrading."""
    with pytest.raises(ValueError, match="Unknown audio backend"):
        load_audio(BytesIO(dummy_audio_bytes), sr=None, backend="not_a_backend")


def test_load_audio_auto_falls_back_without_torchcodec(dummy_audio_bytes):
    """`auto` must fall back to the soundfile → PyAV chain when torchcodec
    is not importable."""
    ref_audio, ref_sr = load_audio_soundfile(BytesIO(dummy_audio_bytes), sr=None)
    with patch.object(audio_module, "load_audio_torchcodec", side_effect=ImportError):
        audio, sr = load_audio(BytesIO(dummy_audio_bytes), sr=None, backend="auto")
    assert sr == ref_sr
    np.testing.assert_array_equal(ref_audio, audio)


def test_load_audio_auto_falls_back_without_ffmpeg(dummy_audio_bytes):
    """Torchcodec installed but system ffmpeg missing (`AudioDecoder is None`)
    must surface as ImportError so `auto` falls back to soundfile → PyAV."""
    ref_audio, ref_sr = load_audio_soundfile(BytesIO(dummy_audio_bytes), sr=None)
    with patch.object(audio_module, "AudioDecoder", None):
        with pytest.raises(ImportError, match="torchcodec audio backend"):
            load_audio_torchcodec(BytesIO(dummy_audio_bytes), sr=None)
        audio, sr = load_audio(BytesIO(dummy_audio_bytes), sr=None, backend="auto")
    assert sr == ref_sr
    np.testing.assert_array_equal(ref_audio, audio)


def test_load_audio_auto_falls_back_when_libtorchcodec_unloadable(
    dummy_audio_bytes,
):
    """Torchcodec loads its ffmpeg-backed core lazily at decoder construction;
    a "Could not load libtorchcodec" RuntimeError there (no system ffmpeg)
    must also surface as ImportError so `auto` falls back."""

    class _UnloadableDecoder:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Could not load libtorchcodec. Likely causes: ...")

    ref_audio, ref_sr = load_audio_soundfile(BytesIO(dummy_audio_bytes), sr=None)
    with patch.object(audio_module, "AudioDecoder", _UnloadableDecoder):
        with pytest.raises(ImportError, match="torchcodec audio backend"):
            load_audio_torchcodec(BytesIO(dummy_audio_bytes), sr=None)
        audio, sr = load_audio(BytesIO(dummy_audio_bytes), sr=None, backend="auto")
    assert sr == ref_sr
    np.testing.assert_array_equal(ref_audio, audio)


def test_audio_media_io_audio_backend_kwarg(dummy_audio_bytes):
    """`audio_backend` selects the backend; unknown values fail at init."""
    audio, sr = AudioMediaIO(audio_backend="pyav").load_bytes(dummy_audio_bytes)
    assert isinstance(audio, np.ndarray)
    assert sr == 16000
    with pytest.raises(ValueError, match="Unknown audio_backend"):
        AudioMediaIO(audio_backend="not_a_backend")


def test_torchcodec_max_duration_rejected(dummy_audio_bytes):
    """The decompression-bomb duration guard must hold for torchcodec too."""
    pytest.importorskip("torchcodec")
    with pytest.raises(ValueError, match="exceeds maximum allowed duration"):
        load_audio_torchcodec(
            BytesIO(dummy_audio_bytes), sr=None, max_duration_s=0.0001
        )


def test_torchcodec_max_decode_bytes_rejected(dummy_audio_bytes):
    """The decompression-bomb memory guard must hold for torchcodec too."""
    pytest.importorskip("torchcodec")
    audio, _ = load_audio_torchcodec(BytesIO(dummy_audio_bytes), sr=None)
    with pytest.raises(ValueError, match="VLLM_MAX_AUDIO_DECODE_BYTES"):
        load_audio_torchcodec(
            BytesIO(dummy_audio_bytes),
            sr=None,
            max_decode_bytes=audio.nbytes - 1,
        )
