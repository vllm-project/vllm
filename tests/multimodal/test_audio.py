# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# test_audio.py
import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vllm.multimodal.audio import (
    MONO_AUDIO_SPEC,
    PASSTHROUGH_AUDIO_SPEC,
    AudioMediaIO,
    AudioResampler,
    AudioSpec,
    ChannelReduction,
    get_audio_spec,
    get_target_channels,
    normalize_audio,
    resample_audio_librosa,
    resample_audio_scipy,
)


@pytest.fixture
def dummy_audio():
    return np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=float)


def test_resample_audio_librosa(dummy_audio):
    with patch("vllm.multimodal.audio.librosa.resample") as mock_resample:
        mock_resample.return_value = dummy_audio * 2
        out = resample_audio_librosa(dummy_audio, orig_sr=44100, target_sr=22050)
        mock_resample.assert_called_once_with(
            dummy_audio, orig_sr=44100, target_sr=22050
        )
        assert np.all(out == dummy_audio * 2)


def test_resample_audio_scipy(dummy_audio):
    out_down = resample_audio_scipy(dummy_audio, orig_sr=4, target_sr=2)
    out_up = resample_audio_scipy(dummy_audio, orig_sr=2, target_sr=4)
    out_same = resample_audio_scipy(dummy_audio, orig_sr=4, target_sr=4)

    assert len(out_down) == 3
    assert len(out_up) == 10
    assert np.all(out_same == dummy_audio)


@pytest.mark.xfail(reason="resample_audio_scipy is buggy for non-integer ratios")
def test_resample_audio_scipy_non_integer_ratio(dummy_audio):
    out = resample_audio_scipy(dummy_audio, orig_sr=5, target_sr=3)

    expected_len = int(round(len(dummy_audio) * 3 / 5))
    assert len(out) == expected_len

    assert isinstance(out, np.ndarray)
    assert np.isfinite(out).all()


def test_audio_resampler_librosa_calls_resample(dummy_audio):
    resampler = AudioResampler(target_sr=22050, method="librosa")
    with patch("vllm.multimodal.audio.resample_audio_librosa") as mock_resample:
        mock_resample.return_value = dummy_audio
        out = resampler.resample(dummy_audio, orig_sr=44100)
        mock_resample.assert_called_once_with(
            dummy_audio, orig_sr=44100, target_sr=22050
        )
        assert np.all(out == dummy_audio)


def test_audio_resampler_scipy_calls_resample(dummy_audio):
    resampler = AudioResampler(target_sr=22050, method="scipy")
    with patch("vllm.multimodal.audio.resample_audio_scipy") as mock_resample:
        mock_resample.return_value = dummy_audio
        out = resampler.resample(dummy_audio, orig_sr=44100)
        mock_resample.assert_called_once_with(
            dummy_audio, orig_sr=44100, target_sr=22050
        )
        assert np.all(out == dummy_audio)


def test_audio_resampler_invalid_method(dummy_audio):
    resampler = AudioResampler(target_sr=22050, method="invalid")
    with pytest.raises(ValueError):
        resampler.resample(dummy_audio, orig_sr=44100)


def test_audio_resampler_no_target_sr(dummy_audio):
    resampler = AudioResampler(target_sr=None)
    with pytest.raises(RuntimeError):
        resampler.resample(dummy_audio, orig_sr=44100)


@pytest.fixture
def dummy_audio_bytes():
    return b"FAKEAUDIOBYTES"


def test_audio_media_io_load_bytes(dummy_audio_bytes):
    audio_io = AudioMediaIO()
    with patch("vllm.multimodal.audio.librosa.load") as mock_load:
        mock_load.return_value = (np.array([0.1, 0.2]), 16000)
        out = audio_io.load_bytes(dummy_audio_bytes)
        mock_load.assert_called_once()
        assert isinstance(out[0], np.ndarray)
        assert out[1] == 16000


def test_audio_media_io_load_base64(dummy_audio_bytes):
    audio_io = AudioMediaIO()
    encoded = base64.b64encode(dummy_audio_bytes).decode("utf-8")
    with patch.object(AudioMediaIO, "load_bytes") as mock_load_bytes:
        mock_load_bytes.return_value = (np.array([0.1, 0.2]), 16000)
        out = audio_io.load_base64("audio/wav", encoded)
        mock_load_bytes.assert_called_once()
        assert isinstance(out[0], np.ndarray)
        assert out[1] == 16000


def test_audio_media_io_load_file():
    audio_io = AudioMediaIO()
    path = Path("/fake/path.wav")
    with patch("vllm.multimodal.audio.librosa.load") as mock_load:
        mock_load.return_value = (np.array([0.1, 0.2]), 16000)
        out = audio_io.load_file(path)
        mock_load.assert_called_once_with(path, sr=None)
        assert isinstance(out[0], np.ndarray)
        assert out[1] == 16000


def test_audio_media_io_encode_base64(dummy_audio):
    audio_io = AudioMediaIO()
    media = (dummy_audio, 16000)
    with patch("vllm.multimodal.audio.soundfile.write") as mock_write:

        def write_to_buffer(buffer, *_args, **_kwargs):
            buffer.write(b"dummy_wav_data")

        mock_write.side_effect = write_to_buffer

        out = audio_io.encode_base64(media)
        decoded = base64.b64decode(out)
        assert decoded == b"dummy_wav_data"
        mock_write.assert_called_once()


# ============================================================
# Tests for ChannelReduction enum
# ============================================================


class TestChannelReduction:
    """Tests for ChannelReduction enum."""

    def test_channel_reduction_values(self):
        """Test that ChannelReduction has expected values."""
        assert ChannelReduction.MEAN.value == "mean"
        assert ChannelReduction.FIRST.value == "first"
        assert ChannelReduction.MAX.value == "max"
        assert ChannelReduction.SUM.value == "sum"

    def test_channel_reduction_is_string_enum(self):
        """Test that ChannelReduction members are also strings."""
        assert isinstance(ChannelReduction.MEAN, str)
        assert ChannelReduction.MEAN == "mean"


# ============================================================
# Tests for AudioSpec dataclass
# ============================================================


class TestAudioSpec:
    """Tests for AudioSpec dataclass."""

    def test_default_values(self):
        """Test that AudioSpec has sensible defaults."""
        spec = AudioSpec()
        assert spec.target_channels == 1
        assert spec.channel_reduction == ChannelReduction.MEAN
        assert spec.needs_normalization is True

    def test_mono_spec(self):
        """Test mono audio spec configuration."""
        spec = AudioSpec(target_channels=1, channel_reduction=ChannelReduction.MEAN)
        assert spec.target_channels == 1
        assert spec.needs_normalization is True

    def test_passthrough_spec(self):
        """Test passthrough audio spec (no normalization)."""
        spec = AudioSpec(target_channels=None)
        assert spec.target_channels is None
        assert spec.needs_normalization is False

    def test_stereo_spec(self):
        """Test stereo audio spec configuration."""
        spec = AudioSpec(target_channels=2)
        assert spec.target_channels == 2
        assert spec.needs_normalization is True

    def test_repr_passthrough(self):
        """Test repr for passthrough spec."""
        spec = AudioSpec(target_channels=None)
        assert "passthrough" in repr(spec)

    def test_repr_mono(self):
        """Test repr for mono spec."""
        spec = AudioSpec(target_channels=1, channel_reduction=ChannelReduction.MEAN)
        assert "channels=1" in repr(spec)
        assert "mean" in repr(spec)

    def test_predefined_mono_spec(self):
        """Test that MONO_AUDIO_SPEC is configured correctly."""
        assert MONO_AUDIO_SPEC.target_channels == 1
        assert MONO_AUDIO_SPEC.channel_reduction == ChannelReduction.MEAN

    def test_predefined_passthrough_spec(self):
        """Test that PASSTHROUGH_AUDIO_SPEC is configured correctly."""
        assert PASSTHROUGH_AUDIO_SPEC.target_channels is None
        assert PASSTHROUGH_AUDIO_SPEC.needs_normalization is False


# ============================================================
# Tests for get_audio_spec function
# ============================================================


class TestGetAudioSpec:
    """Tests for get_audio_spec detection function."""

    def test_whisper_feature_extractor_returns_mono_spec(self):
        """WhisperFeatureExtractor should return MONO_AUDIO_SPEC."""
        from transformers.models.whisper import WhisperFeatureExtractor

        feature_extractor = WhisperFeatureExtractor()
        spec = get_audio_spec(feature_extractor)
        assert spec.target_channels == 1
        assert spec.channel_reduction == ChannelReduction.MEAN

    def test_unknown_feature_extractor_returns_passthrough(self):
        """Unknown feature extractors should return PASSTHROUGH_AUDIO_SPEC."""
        unknown_extractor = MagicMock()
        unknown_extractor.__class__.__name__ = "SomeUnknownExtractor"

        spec = get_audio_spec(unknown_extractor)
        assert spec.target_channels is None
        assert spec.needs_normalization is False

    def test_none_returns_passthrough(self):
        """None should return PASSTHROUGH_AUDIO_SPEC."""
        spec = get_audio_spec(None)
        assert spec.target_channels is None

    def test_string_returns_passthrough(self):
        """Non-object types should return PASSTHROUGH_AUDIO_SPEC."""
        spec = get_audio_spec("not a feature extractor")
        assert spec.target_channels is None


# ============================================================
# Tests for get_target_channels function
# ============================================================


class TestGetTargetChannels:
    """Tests for get_target_channels convenience function."""

    def test_whisper_feature_extractor_returns_1(self):
        """WhisperFeatureExtractor should return 1 (mono)."""
        from transformers.models.whisper import WhisperFeatureExtractor

        feature_extractor = WhisperFeatureExtractor()
        result = get_target_channels(feature_extractor)
        assert result == 1

    def test_unknown_feature_extractor_returns_none(self):
        """Unknown feature extractors should return None (passthrough)."""
        unknown_extractor = MagicMock()
        unknown_extractor.__class__.__name__ = "SomeUnknownExtractor"

        result = get_target_channels(unknown_extractor)
        assert result is None

    def test_none_returns_none(self):
        """None should return None."""
        result = get_target_channels(None)
        assert result is None

    def test_consistency_with_get_audio_spec(self):
        """get_target_channels should be consistent with get_audio_spec."""
        from transformers.models.whisper import WhisperFeatureExtractor

        feature_extractor = WhisperFeatureExtractor()
        spec = get_audio_spec(feature_extractor)
        target_channels = get_target_channels(feature_extractor)
        assert spec.target_channels == target_channels

    def test_consistency_with_mono_check(self):
        """get_target_channels == 1 should indicate mono requirement."""
        from transformers.models.whisper import WhisperFeatureExtractor

        feature_extractor = WhisperFeatureExtractor()
        assert get_target_channels(feature_extractor) == 1

        unknown = MagicMock()
        assert get_target_channels(unknown) is None


# ============================================================
# Tests for normalize_audio function
# ============================================================


class TestNormalizeAudio:
    """Tests for normalize_audio function with different specs."""

    def test_passthrough_preserves_audio(self):
        """Passthrough spec should not modify audio."""
        stereo = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        result = normalize_audio(stereo, PASSTHROUGH_AUDIO_SPEC)
        np.testing.assert_array_equal(result, stereo)

    def test_mono_spec_with_numpy_stereo(self):
        """Mono spec should reduce stereo numpy array to 1D."""
        stereo = np.array([[1.0, 2.0], [-1.0, 0.0]], dtype=np.float32)
        result = normalize_audio(stereo, MONO_AUDIO_SPEC)
        assert result.ndim == 1
        np.testing.assert_array_almost_equal(result, [0.0, 1.0])

    def test_mono_spec_with_torch_stereo(self):
        """Mono spec should reduce stereo torch tensor to 1D."""
        stereo = torch.tensor([[1.0, 2.0], [-1.0, 0.0]])
        result = normalize_audio(stereo, MONO_AUDIO_SPEC)
        assert result.ndim == 1
        torch.testing.assert_close(result, torch.tensor([0.0, 1.0]))

    def test_mono_passthrough_for_1d_numpy(self):
        """1D numpy array should pass through unchanged with mono spec."""
        mono = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = normalize_audio(mono, MONO_AUDIO_SPEC)
        assert result.ndim == 1
        np.testing.assert_array_equal(result, mono)

    def test_mono_passthrough_for_1d_torch(self):
        """1D torch tensor should pass through unchanged with mono spec."""
        mono = torch.tensor([1.0, 2.0, 3.0])
        result = normalize_audio(mono, MONO_AUDIO_SPEC)
        assert result.ndim == 1
        torch.testing.assert_close(result, mono)

    def test_first_channel_reduction(self):
        """FIRST reduction should take only the first channel."""
        spec = AudioSpec(target_channels=1, channel_reduction=ChannelReduction.FIRST)
        stereo = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = normalize_audio(stereo, spec)
        np.testing.assert_array_equal(result, [1.0, 2.0])

    def test_max_channel_reduction(self):
        """MAX reduction should take max across channels."""
        spec = AudioSpec(target_channels=1, channel_reduction=ChannelReduction.MAX)
        stereo = np.array([[1.0, 4.0], [3.0, 2.0]], dtype=np.float32)
        result = normalize_audio(stereo, spec)
        np.testing.assert_array_equal(result, [3.0, 4.0])

    def test_sum_channel_reduction(self):
        """SUM reduction should sum across channels."""
        spec = AudioSpec(target_channels=1, channel_reduction=ChannelReduction.SUM)
        stereo = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = normalize_audio(stereo, spec)
        np.testing.assert_array_equal(result, [4.0, 6.0])

    def test_invalid_3d_array_raises(self):
        """3D arrays should raise ValueError."""
        audio_3d = np.random.randn(2, 3, 4).astype(np.float32)
        with pytest.raises(ValueError, match="Unsupported audio"):
            normalize_audio(audio_3d, MONO_AUDIO_SPEC)

    def test_channel_expansion_raises(self):
        """Expanding from mono to stereo should raise ValueError."""
        mono = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        spec = AudioSpec(target_channels=2)
        with pytest.raises(ValueError, match="Cannot expand"):
            normalize_audio(mono, spec)

    def test_time_channels_format_numpy(self):
        """Audio in (time, channels) format should be transposed to (channels, time).

        This handles the case where audio loaders like soundfile return
        (time, channels) format instead of (channels, time) like torchaudio.
        """
        # Create audio in (time, channels) format: 1000 samples, 2 channels
        audio_time_channels = np.array(
            [[1.0, -1.0]] * 1000,  # 1000 time steps, 2 channels
            dtype=np.float32,
        )
        assert audio_time_channels.shape == (1000, 2)  # (time, channels)

        result = normalize_audio(audio_time_channels, MONO_AUDIO_SPEC)

        # Should be reduced to mono 1D
        assert result.ndim == 1
        assert result.shape == (1000,)
        # Mean of [1.0, -1.0] at each time step should be 0.0
        np.testing.assert_array_almost_equal(result, np.zeros(1000))

    def test_time_channels_format_torch(self):
        """Torch tensor in (time, channels) format should be transposed."""
        # Create audio in (time, channels) format: 1000 samples, 2 channels
        audio_time_channels = torch.tensor(
            [[1.0, -1.0]] * 1000,  # 1000 time steps, 2 channels
        )
        assert audio_time_channels.shape == (1000, 2)  # (time, channels)

        result = normalize_audio(audio_time_channels, MONO_AUDIO_SPEC)

        # Should be reduced to mono 1D
        assert result.ndim == 1
        assert result.shape == (1000,)
        # Mean of [1.0, -1.0] at each time step should be 0.0
        torch.testing.assert_close(result, torch.zeros(1000))

    def test_channels_time_format_preserved(self):
        """Audio already in (channels, time) format should work correctly."""
        # Create audio in standard (channels, time) format: 2 channels, 1000 samples
        audio_channels_time = np.array(
            [[1.0] * 1000, [-1.0] * 1000],  # 2 channels, 1000 time steps
            dtype=np.float32,
        )
        assert audio_channels_time.shape == (2, 1000)  # (channels, time)

        result = normalize_audio(audio_channels_time, MONO_AUDIO_SPEC)

        # Should be reduced to mono 1D
        assert result.ndim == 1
        assert result.shape == (1000,)
        # Mean of [1.0, -1.0] at each time step should be 0.0
        np.testing.assert_array_almost_equal(result, np.zeros(1000))

    def test_ambiguous_square_audio_numpy(self):
        """Square audio arrays (N, N) should use shape[0] > shape[1] heuristic.

        For a square array, shape[0] == shape[1], so no transpose happens
        and we assume (channels, time) format.
        """
        # Create square audio: 4 channels, 4 samples
        audio_square = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
            dtype=np.float32,
        )
        assert audio_square.shape == (4, 4)

        result = normalize_audio(audio_square, MONO_AUDIO_SPEC)

        # Should be reduced to mono 1D with mean across channels (axis 0)
        assert result.ndim == 1
        assert result.shape == (4,)
        # Mean across 4 channels: [1+5+9+13, 2+6+10+14, ...] / 4
        expected = np.array([7.0, 8.0, 9.0, 10.0])
        np.testing.assert_array_almost_equal(result, expected)


# ============================================================
# Tests for MultiModalDataParser integration with target_channels
# ============================================================


class TestMultiModalDataParserChannelNormalization:
    """Tests for MultiModalDataParser.target_channels integration.

    These tests verify that the target_channels parameter is properly used
    in the _parse_audio_data method to normalize audio channels.
    """

    def test_parser_normalizes_stereo_to_mono(self):
        """Parser should normalize stereo to mono when target_channels=1."""
        from vllm.multimodal.parse import MultiModalDataParser

        # Create parser with mono normalization enabled
        parser = MultiModalDataParser(
            target_sr=16000,
            target_channels=1,
        )

        # Create stereo audio (simulating torchaudio output)
        stereo_audio = np.array(
            [[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]],  # 2 channels, 3 samples
            dtype=np.float32,
        )

        # Parse audio data
        result = parser._parse_audio_data((stereo_audio, 16000))

        # Check that result is mono (1D)
        audio_item = result.get(0)
        assert audio_item.ndim == 1, f"Expected 1D mono audio, got {audio_item.ndim}D"
        assert audio_item.shape == (3,), f"Expected shape (3,), got {audio_item.shape}"
        # Channel average of [1, 1, 1] and [-1, -1, -1] should be [0, 0, 0]
        np.testing.assert_array_almost_equal(audio_item, np.zeros(3))

    def test_parser_preserves_stereo_when_target_channels_none(self):
        """Parser should preserve stereo when target_channels=None."""
        from vllm.multimodal.parse import MultiModalDataParser

        # Create parser without channel normalization
        parser = MultiModalDataParser(
            target_sr=16000,
            target_channels=None,
        )

        # Create stereo audio
        stereo_audio = np.array(
            [[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]],
            dtype=np.float32,
        )

        # Parse audio data
        result = parser._parse_audio_data((stereo_audio, 16000))

        # Check that result preserves original shape (after resampling)
        audio_item = result.get(0)
        # When target_channels=None, stereo audio should be preserved
        assert audio_item.ndim == 2, f"Expected 2D stereo audio, got {audio_item.ndim}D"

    def test_parser_mono_passthrough_when_target_channels_1(self):
        """Parser should pass through mono audio unchanged when target_channels=1."""
        from vllm.multimodal.parse import MultiModalDataParser

        # Create parser with mono normalization enabled
        parser = MultiModalDataParser(
            target_sr=16000,
            target_channels=1,
        )

        # Create mono audio (already 1D)
        mono_audio = np.random.randn(16000).astype(np.float32)

        # Parse audio data
        result = parser._parse_audio_data((mono_audio, 16000))

        # Check that result is still mono (1D)
        audio_item = result.get(0)
        assert audio_item.ndim == 1
        assert audio_item.shape == (16000,)

    def test_parser_with_target_channels_2(self):
        """Parser should reduce 6-channel to 2-channel when target_channels=2."""
        from vllm.multimodal.parse import MultiModalDataParser

        # Create parser with stereo target
        parser = MultiModalDataParser(
            target_sr=16000,
            target_channels=2,
        )

        # Create 6-channel audio (5.1 surround)
        surround_audio = np.random.randn(6, 1000).astype(np.float32)

        # Parse audio data
        result = parser._parse_audio_data((surround_audio, 16000))

        # Check that result is stereo (2 channels)
        audio_item = result.get(0)
        assert audio_item.ndim == 2
        assert audio_item.shape[0] == 2  # 2 channels


# ============================================================
# Model Integration Tests (Unit Tests)
# ============================================================
#
# NOTE: These are *unit tests* for the AudioSpec framework integration.
# They use mocks to verify that model processors correctly configure
# target_channels without requiring GPU or model weights.
#
# For full end-to-end inference tests with actual models, see:
# - tests/models/multimodal/generation/test_ultravox.py
# - tests/models/multimodal/generation/test_whisper.py
# ============================================================


class TestModelProcessorIntegration:
    """Unit tests verifying models correctly configure target_channels.

    These tests verify that model processors (Qwen2 Audio, Ultravox, Whisper,
    Qwen2.5 Omni) correctly use get_target_channels() to configure the
    MultiModalDataParser for audio channel normalization.

    Note: These are lightweight unit tests using mocks. For full inference
    tests with actual model weights, see tests/models/multimodal/generation/.
    """

    def test_qwen2_audio_processor_uses_target_channels(self):
        """Verify Qwen2AudioMultiModalProcessor configures target_channels=1."""
        from unittest.mock import MagicMock, patch

        from vllm.model_executor.models.qwen2_audio import (
            Qwen2AudioMultiModalProcessor,
            Qwen2AudioProcessingInfo,
        )

        # Mock the processing info
        mock_info = MagicMock(spec=Qwen2AudioProcessingInfo)
        mock_feature_extractor = MagicMock()
        mock_feature_extractor.sampling_rate = 16000
        mock_info.get_feature_extractor.return_value = mock_feature_extractor

        # Patch get_target_channels to return 1 (mono) for Whisper-based models
        with patch(
            "vllm.model_executor.models.qwen2_audio.get_target_channels",
            return_value=1,
        ):
            processor = Qwen2AudioMultiModalProcessor.__new__(
                Qwen2AudioMultiModalProcessor
            )
            processor.info = mock_info

            # Get the data parser
            data_parser = processor._get_data_parser()

            # Verify target_channels is set to 1 (mono)
            assert data_parser.target_channels == 1

    def test_ultravox_processor_uses_target_channels(self):
        """Verify UltravoxMultiModalProcessor configures target_channels=1."""
        from unittest.mock import MagicMock, patch

        from vllm.model_executor.models.ultravox import (
            UltravoxMultiModalProcessor,
            UltravoxProcessingInfo,
        )

        # Mock the processing info
        mock_info = MagicMock(spec=UltravoxProcessingInfo)
        mock_feature_extractor = MagicMock()
        mock_feature_extractor.sampling_rate = 16000
        mock_info.get_feature_extractor.return_value = mock_feature_extractor

        # Patch get_target_channels to return 1 (mono) for Whisper-based models
        with patch(
            "vllm.model_executor.models.ultravox.get_target_channels",
            return_value=1,
        ):
            processor = UltravoxMultiModalProcessor.__new__(UltravoxMultiModalProcessor)
            processor.info = mock_info

            # Get the data parser
            data_parser = processor._get_data_parser()

            # Verify target_channels is set to 1 (mono)
            assert data_parser.target_channels == 1

    def test_whisper_processor_uses_target_channels(self):
        """Verify WhisperMultiModalProcessor configures target_channels=1."""
        from unittest.mock import MagicMock, patch

        from vllm.model_executor.models.whisper import (
            WhisperMultiModalProcessor,
            WhisperProcessingInfo,
        )

        # Mock the processing info
        mock_info = MagicMock(spec=WhisperProcessingInfo)
        mock_feature_extractor = MagicMock()
        mock_feature_extractor.sampling_rate = 16000
        mock_info.get_feature_extractor.return_value = mock_feature_extractor

        # Patch get_target_channels to return 1 (mono) for Whisper-based models
        with patch(
            "vllm.model_executor.models.whisper.get_target_channels",
            return_value=1,
        ):
            processor = WhisperMultiModalProcessor.__new__(WhisperMultiModalProcessor)
            processor.info = mock_info

            # Get the data parser
            data_parser = processor._get_data_parser()

            # Verify target_channels is set to 1 (mono)
            assert data_parser.target_channels == 1

    def test_qwen2_5_omni_processor_uses_target_channels(self):
        """Verify Qwen2_5OmniThinkerMultiModalProcessor configures target_channels=1."""
        from unittest.mock import MagicMock, patch

        from vllm.model_executor.models.qwen2_5_omni_thinker import (
            Qwen2_5OmniThinkerMultiModalProcessor,
            Qwen2_5OmniThinkerProcessingInfo,
        )

        # Mock the processing info
        mock_info = MagicMock(spec=Qwen2_5OmniThinkerProcessingInfo)
        mock_feature_extractor = MagicMock()
        mock_feature_extractor.sampling_rate = 16000
        mock_info.get_feature_extractor.return_value = mock_feature_extractor

        # Patch get_target_channels to return 1 (mono) for Whisper-based models
        with patch(
            "vllm.model_executor.models.qwen2_5_omni_thinker.get_target_channels",
            return_value=1,
        ):
            processor = Qwen2_5OmniThinkerMultiModalProcessor.__new__(
                Qwen2_5OmniThinkerMultiModalProcessor
            )
            processor.info = mock_info

            # Get the data parser
            data_parser = processor._get_data_parser()

            # Verify target_channels is set to 1 (mono)
            assert data_parser.target_channels == 1

    def test_unknown_feature_extractor_passthrough(self):
        """Verify unknown feature extractors result in target_channels=None."""
        from vllm.multimodal.audio import get_target_channels

        # Create a mock unknown feature extractor
        unknown_extractor = MagicMock()
        unknown_extractor.__class__.__name__ = "UnknownFeatureExtractor"

        # get_target_channels should return None for unknown extractors
        result = get_target_channels(unknown_extractor)
        assert result is None


# ============================================================
# End-to-End Audio Pipeline Tests
# ============================================================


class TestAudioPipelineE2E:
    """End-to-end tests for audio normalization in the full pipeline.

    These tests verify the complete flow from raw audio input through
    the MultiModalDataParser, simulating different audio loader formats.
    """

    def test_stereo_audio_normalized_to_mono_e2e(self):
        """Full pipeline: stereo audio (torchaudio format) → mono output."""
        from vllm.multimodal.parse import MultiModalDataParser

        # Simulate torchaudio output: (channels, time) format
        # Stereo audio with left channel = 1.0, right channel = -1.0
        stereo_torchaudio = np.array(
            [[1.0] * 16000, [-1.0] * 16000],  # 2 channels, 1 second at 16kHz
            dtype=np.float32,
        )
        assert stereo_torchaudio.shape == (2, 16000)

        # Create parser with mono normalization (like Whisper models)
        parser = MultiModalDataParser(
            target_sr=16000,
            target_channels=1,
        )

        # Process audio through the parser
        result = parser._parse_audio_data((stereo_torchaudio, 16000))
        audio_output = result.get(0)

        # Verify output is mono 1D
        assert audio_output.ndim == 1, f"Expected 1D, got {audio_output.ndim}D"
        assert audio_output.shape == (16000,)

        # Verify channel averaging: mean of [1.0, -1.0] = 0.0
        np.testing.assert_array_almost_equal(audio_output, np.zeros(16000), decimal=5)

    def test_soundfile_format_normalized_to_mono_e2e(self):
        """Full pipeline: soundfile format (time, channels) → mono output."""
        from vllm.multimodal.parse import MultiModalDataParser

        # Simulate soundfile output: (time, channels) format
        # 16000 samples, 2 channels
        stereo_soundfile = np.array(
            [[0.5, -0.5]] * 16000,  # Each row is [left, right]
            dtype=np.float32,
        )
        assert stereo_soundfile.shape == (16000, 2)

        # Create parser with mono normalization
        parser = MultiModalDataParser(
            target_sr=16000,
            target_channels=1,
        )

        # Process audio through the parser
        result = parser._parse_audio_data((stereo_soundfile, 16000))
        audio_output = result.get(0)

        # Verify output is mono 1D
        assert audio_output.ndim == 1, f"Expected 1D, got {audio_output.ndim}D"
        assert audio_output.shape == (16000,)

        # Verify channel averaging: mean of [0.5, -0.5] = 0.0
        np.testing.assert_array_almost_equal(audio_output, np.zeros(16000), decimal=5)

    def test_librosa_mono_passthrough_e2e(self):
        """Full pipeline: librosa mono format → preserved as mono."""
        from vllm.multimodal.parse import MultiModalDataParser

        # Simulate librosa output: already mono (time,) format
        mono_librosa = np.random.randn(16000).astype(np.float32)
        assert mono_librosa.shape == (16000,)

        # Create parser with mono normalization
        parser = MultiModalDataParser(
            target_sr=16000,
            target_channels=1,
        )

        # Process audio through the parser
        result = parser._parse_audio_data((mono_librosa, 16000))
        audio_output = result.get(0)

        # Verify output is still mono 1D
        assert audio_output.ndim == 1
        assert audio_output.shape == (16000,)

        # Verify audio content is preserved
        np.testing.assert_array_almost_equal(audio_output, mono_librosa)

    def test_multichannel_5_1_surround_to_mono_e2e(self):
        """Full pipeline: 5.1 surround (6 channels) → mono output."""
        from vllm.multimodal.parse import MultiModalDataParser

        # Simulate 5.1 surround audio: 6 channels
        surround_audio = np.array(
            [
                [1.0] * 8000,  # Front Left
                [2.0] * 8000,  # Front Right
                [3.0] * 8000,  # Center
                [4.0] * 8000,  # LFE (subwoofer)
                [5.0] * 8000,  # Rear Left
                [6.0] * 8000,  # Rear Right
            ],
            dtype=np.float32,
        )
        assert surround_audio.shape == (6, 8000)

        # Create parser with mono normalization
        parser = MultiModalDataParser(
            target_sr=16000,
            target_channels=1,
        )

        # Process audio through the parser
        result = parser._parse_audio_data((surround_audio, 16000))
        audio_output = result.get(0)

        # Verify output is mono 1D
        assert audio_output.ndim == 1

        # Verify channel averaging: mean of [1,2,3,4,5,6] = 3.5
        expected_value = (1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0) / 6
        np.testing.assert_array_almost_equal(
            audio_output, np.full(8000, expected_value), decimal=5
        )

    def test_torch_tensor_input_e2e(self):
        """Full pipeline: torch.Tensor stereo input → mono numpy output."""
        from vllm.multimodal.parse import MultiModalDataParser

        # Simulate torch tensor input (from torchaudio)
        stereo_torch = torch.tensor(
            [[1.0] * 8000, [-1.0] * 8000],  # 2 channels
            dtype=torch.float32,
        )
        assert stereo_torch.shape == (2, 8000)

        # Create parser with mono normalization
        parser = MultiModalDataParser(
            target_sr=16000,
            target_channels=1,
        )

        # Process audio through the parser
        # Note: Parser expects numpy, so we convert first (simulating real usage)
        result = parser._parse_audio_data((stereo_torch.numpy(), 16000))
        audio_output = result.get(0)

        # Verify output is mono 1D numpy array
        assert audio_output.ndim == 1
        assert isinstance(audio_output, np.ndarray)

        # Verify channel averaging
        np.testing.assert_array_almost_equal(audio_output, np.zeros(8000), decimal=5)

    def test_passthrough_preserves_stereo_e2e(self):
        """Full pipeline: stereo with target_channels=None → stereo preserved."""
        from vllm.multimodal.parse import MultiModalDataParser

        # Stereo audio
        stereo_audio = np.array(
            [[1.0] * 8000, [-1.0] * 8000],
            dtype=np.float32,
        )

        # Create parser WITHOUT mono normalization (passthrough)
        parser = MultiModalDataParser(
            target_sr=16000,
            target_channels=None,  # Passthrough - no normalization
        )

        # Process audio through the parser
        result = parser._parse_audio_data((stereo_audio, 16000))
        audio_output = result.get(0)

        # Verify output preserves stereo (2D)
        assert audio_output.ndim == 2
        assert audio_output.shape == (2, 8000)

    def test_resampling_with_channel_normalization_e2e(self):
        """Full pipeline: resample + channel normalize in single pass."""
        from vllm.multimodal.parse import MultiModalDataParser

        # Stereo audio at 48kHz (common recording rate)
        stereo_48k = np.array(
            [[1.0] * 48000, [-1.0] * 48000],  # 1 second at 48kHz
            dtype=np.float32,
        )

        # Create parser with both resampling and mono normalization
        parser = MultiModalDataParser(
            target_sr=16000,  # Resample to 16kHz
            target_channels=1,  # Normalize to mono
        )

        # Process audio through the parser
        result = parser._parse_audio_data((stereo_48k, 48000))
        audio_output = result.get(0)

        # Verify output is mono 1D at target sample rate
        assert audio_output.ndim == 1
        # After resampling from 48kHz to 16kHz, length should be ~16000
        assert audio_output.shape[0] == 16000

    def test_very_short_audio_e2e(self):
        """Full pipeline: very short audio (< 1 frame) handled correctly."""
        from vllm.multimodal.parse import MultiModalDataParser

        # Very short stereo audio (10 samples)
        short_stereo = np.array(
            [[1.0] * 10, [-1.0] * 10],
            dtype=np.float32,
        )

        parser = MultiModalDataParser(
            target_sr=16000,
            target_channels=1,
        )

        result = parser._parse_audio_data((short_stereo, 16000))
        audio_output = result.get(0)

        # Should still produce mono output
        assert audio_output.ndim == 1
        assert audio_output.shape == (10,)
        np.testing.assert_array_almost_equal(audio_output, np.zeros(10))
