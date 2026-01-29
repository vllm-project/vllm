# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch
import torch.nn as nn
import torchaudio.compliance.kaldi as kaldi
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoFeatureExtractor,
    AutoProcessor,
    BatchFeature,
)
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.processing_utils import ProcessorMixin
from transformers.utils import TensorType

from vllm.logger import init_logger

logger = init_logger(__name__)


def apply_cmvn(inputs, cmvn):  # noqa
    """
    Apply CMVN with mvn data
    """

    device = inputs.device
    # dtype = inputs.dtype
    frame, dim = inputs.shape

    means = cmvn[0:1, :dim]
    vars = cmvn[1:2, :dim]
    inputs += means.to(device)
    inputs *= vars.to(device)

    return inputs.type(torch.float32)


def apply_lfr(inputs, lfr_m, lfr_n):
    # LFR_inputs = []
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / lfr_n))
    left_padding = inputs[0].repeat((lfr_m - 1) // 2, 1)
    inputs = torch.vstack((left_padding, inputs))
    T = T + (lfr_m - 1) // 2
    feat_dim = inputs.shape[-1]
    strides = (lfr_n * feat_dim, 1)
    sizes = (T_lfr, lfr_m * feat_dim)
    last_idx = (T - lfr_m) // lfr_n + 1
    num_padding = lfr_m - (T - last_idx * lfr_n)
    if num_padding > 0:
        num_padding = (
            (2 * lfr_m - 2 * T + (T_lfr - 1 + last_idx) * lfr_n)
            / 2
            * (T_lfr - last_idx)
        )
        inputs = torch.vstack([inputs] + [inputs[-1:]] * int(num_padding))
    LFR_outputs = inputs.as_strided(sizes, strides)
    return LFR_outputs.clone().type(torch.float32)


def load_cmvn(cmvn_file):
    with open(cmvn_file, encoding="utf-8") as f:
        lines = f.readlines()
    means_list = []
    vars_list = []
    for i in range(len(lines)):
        line_item = lines[i].split()
        if line_item[0] == "<AddShift>":
            line_item = lines[i + 1].split()
            if line_item[0] == "<LearnRateCoef>":
                add_shift_line = line_item[3 : (len(line_item) - 1)]
                means_list = list(add_shift_line)
                continue
        elif line_item[0] == "<Rescale>":
            line_item = lines[i + 1].split()
            if line_item[0] == "<LearnRateCoef>":
                rescale_line = line_item[3 : (len(line_item) - 1)]
                vars_list = list(rescale_line)
                continue
    means = np.array(means_list).astype(np.float32)
    vars = np.array(vars_list).astype(np.float32)
    cmvn = np.array([means, vars])
    cmvn = torch.as_tensor(cmvn, dtype=torch.float32)
    return cmvn


class WavFrontend(nn.Module):
    """Conventional frontend structure for ASR."""

    def __init__(
        self,
        cmvn_file: str = "null",
        fs: int = 16000,
        window: str = "hamming",
        n_mels: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        filter_length_min: int = -1,
        filter_length_max: int = -1,
        lfr_m: int = 1,
        lfr_n: int = 1,
        dither: float = 1.0,
        snip_edges: bool = True,
        upsacle_samples: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.fs = fs
        self.window = window
        self.n_mels = n_mels
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.filter_length_min = filter_length_min
        self.filter_length_max = filter_length_max
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        self.cmvn_file = cmvn_file
        self.dither = dither
        self.snip_edges = snip_edges
        self.upsacle_samples = upsacle_samples
        self.cmvn = None if self.cmvn_file is None else load_cmvn(self.cmvn_file)

    def output_size(self) -> int:
        return self.n_mels * self.lfr_m

    def forward(
        self,
        input: torch.Tensor,
        input_lengths,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = input.size(0)
        feats = []
        feats_lens = []
        for i in range(batch_size):
            waveform_length = input_lengths[i]
            waveform = input[i][:waveform_length]
            if self.upsacle_samples:
                waveform = waveform * (1 << 15)
            waveform = waveform.unsqueeze(0)
            mat = kaldi.fbank(
                waveform,
                num_mel_bins=self.n_mels,
                frame_length=min(self.frame_length, waveform_length / self.fs * 1000),
                frame_shift=self.frame_shift,
                dither=self.dither,
                energy_floor=0.0,
                window_type=self.window,
                sample_frequency=self.fs,
                snip_edges=self.snip_edges,
            )

            if self.lfr_m != 1 or self.lfr_n != 1:
                mat = apply_lfr(mat, self.lfr_m, self.lfr_n)
            if self.cmvn is not None:
                mat = apply_cmvn(mat, self.cmvn)
            feat_length = mat.size(0)
            feats.append(mat)
            feats_lens.append(feat_length)

        feats_lens = torch.as_tensor(feats_lens)
        if batch_size == 1:
            feats_pad = feats[0][None, :, :]
        else:
            feats_pad = pad_sequence(feats, batch_first=True, padding_value=0.0)
        return feats_pad, feats_lens

    def forward_fbank(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = input.size(0)
        feats = []
        feats_lens = []
        for i in range(batch_size):
            waveform_length = input_lengths[i]
            waveform = input[i][:waveform_length]
            waveform = waveform * (1 << 15)
            waveform = waveform.unsqueeze(0)
            mat = kaldi.fbank(
                waveform,
                num_mel_bins=self.n_mels,
                frame_length=self.frame_length,
                frame_shift=self.frame_shift,
                dither=self.dither,
                energy_floor=0.0,
                window_type=self.window,
                sample_frequency=self.fs,
            )

            feat_length = mat.size(0)
            feats.append(mat)
            feats_lens.append(feat_length)

        feats_lens = torch.as_tensor(feats_lens)
        feats_pad = pad_sequence(feats, batch_first=True, padding_value=0.0)
        return feats_pad, feats_lens

    def forward_lfr_cmvn(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = input.size(0)
        feats = []
        feats_lens = []
        for i in range(batch_size):
            mat = input[i, : input_lengths[i], :]
            if self.lfr_m != 1 or self.lfr_n != 1:
                mat = apply_lfr(mat, self.lfr_m, self.lfr_n)
            if self.cmvn is not None:
                mat = apply_cmvn(mat, self.cmvn)
            feat_length = mat.size(0)
            feats.append(mat)
            feats_lens.append(feat_length)

        feats_lens = torch.as_tensor(feats_lens)
        feats_pad = pad_sequence(feats, batch_first=True, padding_value=0.0)
        return feats_pad, feats_lens


class FunASRFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a FunASR feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_
        utils.SequenceFeatureExtractor`] which contains most of the main
        methods. Users should refer to this superclass for more information
        regarding those methods.

    This class extracts mel-filter bank features from raw speech using a custom
    numpy implementation of the `Short Time Fourier Transform` which should
    match pytorch's `torch.stft` equivalent.

    Args:
        feature_size (`int`, *optional*, defaults to 80):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio files should be digitalized
            expressed in hertz (Hz).
        hop_length (`int`, *optional*, defaults to 160):
            Length of the overlapping windows for the STFT used to obtain the
            Mel Frequency coefficients.
        chunk_length (`int`, *optional*, defaults to 30):
            The maximum number of chunks of `sampling_rate` samples used to
            trim and pad longer or shorter audio sequences.
        n_fft (`int`, *optional*, defaults to 400):
            Size of the Fourier transform.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio. Should correspond to silences.
        dither (`float`, *optional*, defaults to 0.0):
            Adds dithering. In other words, adds a small Gaussian noise to each frame.
            E.g. use 0.0001 to add dithering with a normal distribution centered
            around 0.0 with standard deviation 0.0001 (assuming [-1,+1] range
            of raw_speech). The value 0.0 means no dithering.
            Dithering has similar effect as `spectrogram(mel_floor=...)`. It reduces
            the high log_mel_fbank values for signals with hard-zero sections,
            when VAD cutoff is present in the signal.
    """

    model_input_names = ["input_features"]

    def __init__(
        self,
        feature_size=80,
        sampling_rate=16000,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
        padding_value=0.0,
        dither=0.0,
        return_attention_mask=False,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        self.frontend_conf = kwargs.get("frontend_conf", {})
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_samples = chunk_length * sampling_rate
        self.nb_max_frames = self.n_samples // hop_length
        self.sampling_rate = sampling_rate
        self.dither = dither

    def extract_fbank(
        self, data, data_len=None, data_type: str = "sound", frontend=None, **kwargs
    ):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
            if len(data.shape) < 2:
                data = data[None, :]  # data: [batch, N]
            data_len = [data.shape[1]] if data_len is None else data_len
        elif isinstance(data, torch.Tensor):
            if len(data.shape) < 2:
                data = data[None, :]  # data: [batch, N]
            data_len = [data.shape[1]] if data_len is None else data_len
        elif isinstance(data, (list, tuple)):
            data_list, data_len = [], []
            for data_i in data:
                if isinstance(data_i, np.ndarray):
                    data_i = torch.from_numpy(data_i)
                data_list.append(data_i)
                data_len.append(data_i.shape[0])
            data = pad_sequence(data_list, batch_first=True)  # data: [batch, N]

        data, data_len = frontend(data, data_len, **kwargs)

        if isinstance(data_len, (list, tuple)):
            data_len = torch.tensor([data_len])
        return data.to(torch.float32), data_len.to(torch.int32)

    def __call__(
        self,
        raw_speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        truncation: bool = True,
        pad_to_multiple_of: int | None = None,
        return_tensors: str | TensorType | None = None,
        return_attention_mask: bool | None = None,
        padding: str | None = "max_length",
        max_length: int | None = None,
        sampling_rate: int | None = None,
        do_normalize: bool | None = None,
        device: str | None = "cpu",
        return_token_timestamps: bool | None = None,
        **kwargs,
    ) -> BatchFeature:
        self.frontend = WavFrontend(**self.frontend_conf)
        speech, speech_lengths = self.extract_fbank(
            raw_speech,
            data_type=kwargs.get("data_type", "sound"),
            frontend=self.frontend,
            is_final=True,
        )  # speech: [b, T, d]
        olens = 1 + (speech_lengths - 3 + 2 * 1) // 2
        olens = 1 + (olens - 3 + 2 * 1) // 2
        fake_token_len = (olens - 1) // 2 + 1
        batched_speech = BatchFeature(
            {
                "input_features": speech,
                "speech_lengths": speech_lengths,
                "fake_token_len": fake_token_len,
            }
        )
        return batched_speech


class FunASRProcessor(ProcessorMixin):
    r"""
    Constructs a FunASR processor which wraps a FunASR feature extractor and
    a FunASR tokenizer into a single processor.

    [`FunASRProcessor`] offers all the functionalities of
    [`FunASRFeatureExtractor`] and [`Qwen2Tokenizer`]. See the
    [`~FunASRProcessor.__call__`] and [`~FunASRProcessor.decode`] for more
    information.

    Args:
        feature_extractor (`FunASRFeatureExtractor`): An instance of
            [`FunASRFeatureExtractor`].
            The feature extractor is a required input.
        tokenizer (`Qwen2Tokenizer`):
            An instance of [`Qwen2Tokenizer`]. The tokenizer is a required
            input.
    """

    feature_extractor_class = "FunASRFeatureExtractor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        audio_token="<|AUDIO|>",
    ):
        super().__init__(feature_extractor, tokenizer)
        self.current_processor = self.feature_extractor
        self._in_target_context_manager = False
        self.audio_token = (
            tokenizer.audio_token if hasattr(tokenizer, "audio_token") else audio_token
        )
        self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token)

    def get_decoder_prompt_ids(self, task=None, language=None, no_timestamps=True):
        return self.tokenizer.get_decoder_prompt_ids(
            task=task, language=language, no_timestamps=no_timestamps
        )

    def __call__(self, *args, **kwargs):
        """
        Forwards the `audio` argument to FunASRFeatureExtractor's
        [`~FunASRFeatureExtractor.__call__`] and the `text` argument to
        [`~Qwen2Tokenizer.__call__`]. Please refer to the docstring of the
        above two methods for more information.
        """
        # For backward compatibility
        if self._in_target_context_manager:
            return self.current_processor(*args, **kwargs)

        audio = kwargs.pop("audio", None)
        sampling_rate = kwargs.pop("sampling_rate", None)
        text = kwargs.pop("text", None)
        if len(args) > 0:
            audio = args[0]
            args = args[1:]

        # if audio is None and text is None:
        #    raise ValueError(
        #        "You need to specify either an `audio` or `text` input to process."
        #    )
        if text is None:
            raise ValueError("You need to specify `text` input to process.")
        elif isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError(
                "Invalid input text. Please provide a string, or a list of strings"
            )

        if audio is not None:
            # ensure we have as much audios as audio tokens
            num_audio_tokens = sum(sample.count(self.audio_token) for sample in text)
            num_audios = 1 if type(audio) is np.ndarray else len(audio)
            if num_audio_tokens != num_audios:
                raise ValueError(
                    f"Found {num_audio_tokens} {self.audio_token} token{'s' if num_audio_tokens > 1 else ''} in provided text but received {num_audios} audio{'s' if num_audios > 1 else ''}"  # noqa: E501
                )
            inputs = self.feature_extractor(
                audio, *args, sampling_rate=sampling_rate, **kwargs
            )

            expanded_text = []
            for sample in text:
                replace_str = []
                while self.audio_token in sample:
                    num_audio_tokens = inputs["fake_token_len"].item()

                    expanded_audio_token = self.audio_token * num_audio_tokens

                    replace_str.append(expanded_audio_token)
                    sample = sample.replace(self.audio_token, "<placeholder>", 1)

                while "<placeholder>" in sample:
                    sample = sample.replace("<placeholder>", replace_str.pop(0), 1)
                expanded_text.append(sample)
            text = expanded_text

        if text is not None:
            encodings = self.tokenizer(text, **kwargs)

        if text is None:
            return inputs

        elif audio is None:
            return encodings
        else:
            inputs["labels"] = encodings["input_ids"]

            return inputs

    def get_prompt_ids(self, text: str, return_tensors="np"):
        return self.tokenizer.get_prompt_ids(text, return_tensors=return_tensors)


AutoFeatureExtractor.register("FunASRFeatureExtractor", FunASRFeatureExtractor)
AutoProcessor.register("FunASRProcessor", FunASRProcessor)
