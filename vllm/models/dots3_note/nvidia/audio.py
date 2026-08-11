# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn
from transformers import WhisperConfig
from transformers.audio_utils import mel_filter_bank

from .audio_encoder import DotsSpeechEncoder

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
DEFAULT_CHUNK_LENGTH_S = 60
DEFAULT_MERGE_FACTOR = 1
N_SAMPLES = DEFAULT_CHUNK_LENGTH_S * SAMPLE_RATE


class Dots3NoteAudioConfig:
    def __init__(self, **kwargs):
        self.encoder_type = kwargs.get("encoder_type", "dots")
        self.whisper_config = kwargs.get("whisper_config", {})
        self.whisper_adapter_in_dim = kwargs.get(
            "whisper_adapter_in_dim", kwargs.get("adapter_in_dim", 1280)
        )
        self.whisper_adapter_out_dim = kwargs.get(
            "whisper_adapter_out_dim", kwargs.get("adapter_out_dim", 2048)
        )
        self.merge_factor = kwargs.get("merge_factor", DEFAULT_MERGE_FACTOR)
        self.chunk_seconds = kwargs.get("chunk_seconds", DEFAULT_CHUNK_LENGTH_S)
        self.use_conv2d_stem = kwargs.get("use_conv2d_stem", True)
        self.use_rope = kwargs.get("use_rope", True)
        self.use_rms_norm = kwargs.get("use_rms_norm", True)
        self.use_causal = kwargs.get("use_causal", False)
        self.downsample_hidden_size = kwargs.get("downsample_hidden_size", 480)
        self.conv_chunksize = kwargs.get("conv_chunksize", 500)
        self.conv_stem_gradient_checkpointing = kwargs.get(
            "conv_stem_gradient_checkpointing", False
        )
        self.conv_bucket_step = kwargs.get("conv_bucket_step")
        self.conv_bucket_max_elements = kwargs.get("conv_bucket_max_elements")
        self.rope_parameters = kwargs.get(
            "rope_parameters",
            {
                "partial_rotary_factor": 0.5,
                "rope_theta": 10000.0,
                "rope_type": "default",
            },
        )

    @property
    def conv_temporal_stride(self) -> int:
        return 8 if self.use_conv2d_stem else 2

    @property
    def chunk_samples(self) -> int:
        return int(self.chunk_seconds * SAMPLE_RATE)

    @property
    def chunk_mel_frames(self) -> int:
        return int(self.chunk_seconds * 100)


def pad_or_trim(array, length=N_SAMPLES, axis=-1):
    if array.shape[axis] > length:
        array = array.index_select(
            dim=axis, index=torch.arange(length, device=array.device)
        )
    if array.shape[axis] < length:
        pad_widths = [(0, 0)] * array.ndim
        pad_widths[axis] = (0, length - array.shape[axis])
        array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    return array


@lru_cache(maxsize=4)
def _mel_filters(device, n_mels=128):
    filters = mel_filter_bank(
        num_frequency_bins=1 + N_FFT // 2,
        num_mel_filters=n_mels,
        min_frequency=0.0,
        max_frequency=float(SAMPLE_RATE) / 2.0,
        sampling_rate=SAMPLE_RATE,
        norm="slaney",
        mel_scale="slaney",
    )
    return torch.from_numpy(filters).T.contiguous().float().to(device)


@lru_cache(maxsize=4)
def _hann_window(device):
    # Generate on CPU (default) then move, to preserve the exact reference
    # window values. Direct on-device generation changes numerics slightly.
    return torch.hann_window(N_FFT).to(device)


def log_mel_spectrogram(audio, n_mels=128):
    window = _hann_window(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    filters = _mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


class DotsEncoderWithMask(nn.Module):
    def __init__(self, config: Dots3NoteAudioConfig):
        super().__init__()
        whisper_config = WhisperConfig(**config.whisper_config)
        whisper_config.use_rope = config.use_rope
        whisper_config.rope_parameters = config.rope_parameters
        whisper_config.use_rms_norm = config.use_rms_norm
        whisper_config.use_causal = config.use_causal
        whisper_config.use_conv2d_stem = config.use_conv2d_stem
        whisper_config.downsample_hidden_size = config.downsample_hidden_size
        whisper_config.conv_chunksize = config.conv_chunksize
        whisper_config.conv_stem_gradient_checkpointing = (
            config.conv_stem_gradient_checkpointing
        )
        whisper_config.conv_bucket_step = config.conv_bucket_step
        whisper_config.conv_bucket_max_elements = config.conv_bucket_max_elements

        self.speech_encoder = DotsSpeechEncoder(whisper_config)
        self.merge_factor = config.merge_factor
        self.chunk_seconds = config.chunk_seconds
        self.chunk_samples = config.chunk_samples
        self.chunk_mel_frames = config.chunk_mel_frames
        self.conv_temporal_stride = config.conv_temporal_stride

    @property
    def device(self):
        return next(self.speech_encoder.parameters()).device

    def _forward_speech_encoder(
        self,
        mel_features: torch.Tensor,
        input_seq_lens: torch.Tensor,
        audio_sample_lens: list[int],
    ) -> torch.Tensor:
        """Run the eager speech encoder without server-side slicing/batching."""
        mel_features = mel_features.to(dtype=torch.bfloat16, device=self.device)
        return self.speech_encoder(
            mel_features,
            return_dict=True,
            input_seq_lens=input_seq_lens,
            audio_sample_lens=audio_sample_lens,
        ).last_hidden_state

    def encode_waveform(self, audio_waveform: torch.Tensor) -> torch.Tensor:
        segments = []
        time_step = 0
        while time_step * SAMPLE_RATE < audio_waveform.shape[0]:
            segments.append(
                audio_waveform[
                    time_step * SAMPLE_RATE : (time_step + self.chunk_seconds)
                    * SAMPLE_RATE
                ]
            )
            time_step += self.chunk_seconds

        mel_features = []
        token_lens = []
        audio_sample_lens = []
        for audio_segment in segments:
            segment_length = audio_segment.shape[0]
            token_len = (segment_length - 1) // (
                HOP_LENGTH * self.conv_temporal_stride * self.merge_factor
            ) + 1
            pad_audio = pad_or_trim(audio_segment.flatten(), length=self.chunk_samples)
            mel = log_mel_spectrogram(pad_audio)
            assert mel.shape[1] == self.chunk_mel_frames
            mel_features.append(mel)
            token_lens.append(token_len)
            audio_sample_lens.append(segment_length)

        mel_features = torch.stack(mel_features, dim=0)
        # Keep input_seq_lens on CPU: the conv2d bucket path reads it via
        # ``.item()`` and CPU scalars avoid device->host syncs. The encoder's
        # varlen path moves it to the device itself.
        input_seq_lens = torch.tensor(token_lens, dtype=torch.long) * self.merge_factor
        audio_embedding = self._forward_speech_encoder(
            mel_features, input_seq_lens, audio_sample_lens
        )

        chunk_embeddings = []
        for idx, token_len in enumerate(token_lens):
            chunk_embeddings.append(
                audio_embedding[idx, : token_len * self.merge_factor, :]
            )
        return torch.cat(chunk_embeddings, dim=0).unsqueeze(0)


class AudioAdapter(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.proj(x)


class Dots3NoteAudioModel(nn.Module):
    def __init__(self, config: Dots3NoteAudioConfig):
        super().__init__()
        if config.encoder_type != "dots":
            raise ValueError("Dots3Note only supports encoder_type='dots'")
        self.merge_factor = config.merge_factor
        self.audio_adapter = AudioAdapter(
            config.whisper_adapter_in_dim,
            config.whisper_adapter_out_dim,
        )

        self.dots_encoder = DotsEncoderWithMask(config)

    @property
    def device(self):
        return self.dots_encoder.device

    def _merge_embeddings(self, embedding: torch.Tensor) -> torch.Tensor:
        if self.merge_factor <= 1:
            return embedding
        return embedding.reshape(
            embedding.shape[0],
            embedding.shape[1] // self.merge_factor,
            embedding.shape[2] * self.merge_factor,
        )

    def _encode_single_audio(self, audio_waveform):
        embedding = self.dots_encoder.encode_waveform(audio_waveform)
        embedding = self._merge_embeddings(embedding)
        embedding = self.audio_adapter(embedding)
        return embedding.squeeze(0)

    def _split_waveforms(self, audio_inputs, lengths):
        # Convert lengths to CPU ints once so the per-audio slicing below stays
        # on the host and does not trigger a device->host sync per audio.
        if isinstance(audio_inputs, list):
            return audio_inputs
        lengths_list = lengths.tolist()
        waveforms = []
        audio_start = 0
        for length in lengths_list:
            waveforms.append(audio_inputs[audio_start : audio_start + length])
            audio_start += length
        return waveforms

    def forward(self, audio_inputs, lengths):
        waveforms = self._split_waveforms(audio_inputs, lengths)
        all_embeddings = []
        token_lengths = []
        for waveform in waveforms:
            embedding = self._encode_single_audio(waveform)
            all_embeddings.append(embedding)
            token_lengths.append(embedding.shape[0])
        return torch.cat(all_embeddings, dim=0), token_lengths
