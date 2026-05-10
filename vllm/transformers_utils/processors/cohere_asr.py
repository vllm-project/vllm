# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchaudio.functional import melscale_fbanks
from transformers import AutoFeatureExtractor, AutoProcessor, BatchFeature
from transformers.feature_extraction_sequence_utils import (
    SequenceFeatureExtractor,
)
from transformers.processing_utils import ProcessorMixin

logger = logging.getLogger(__name__)

CONSTANT = 1e-5
INF_VAL = 10000.0


class FilterbankFeatures(nn.Module):
    """Featurizer that converts wavs to Mel Spectrograms.
    See AudioToMelSpectrogramPreprocessor for args.
    """

    window: torch.Tensor
    fb: torch.Tensor

    def __init__(
        self,
        sample_rate=16000,
        n_window_size=320,
        n_window_stride=160,
        window="hann",
        normalize="per_feature",
        n_fft=None,
        preemph=0.97,
        nfilt=64,
        lowfreq=0,
        highfreq=None,
        log=True,
        log_zero_guard_type="add",
        log_zero_guard_value=2**-24,
        dither=CONSTANT,
        pad_to=16,
        max_duration=30,
        frame_splicing=1,
        exact_pad=False,
        pad_value=0,
        mag_power=2.0,
        use_grads=False,
        rng=None,
        nb_augmentation_prob=0.0,
        nb_max_freq=4000,
        mel_norm="slaney",
        stft_exact_pad=False,
        stft_conv=False,
        device="cpu",
    ):
        super().__init__()
        if stft_conv or stft_exact_pad:
            logger.warning(
                "Using torch_stft is deprecated and has been removed. "
                "The values have been forcibly set to False for "
                "FilterbankFeatures and AudioToMelSpectrogramPreprocessor. "
                "Please set exact_pad to True as needed."
            )
        if exact_pad and n_window_stride % 2 == 1:
            raise NotImplementedError(
                f"{self} received exact_pad == True, but hop_size was odd. "
                "If audio_length % hop_size == 0, the returned spectrogram "
                "would not be of length audio_length // hop_size. "
                "Please use an even hop_size."
            )
        self.log_zero_guard_value = log_zero_guard_value
        if (
            n_window_size is None
            or n_window_stride is None
            or not isinstance(n_window_size, int)
            or not isinstance(n_window_stride, int)
            or n_window_size <= 0
            or n_window_stride <= 0
        ):
            raise ValueError(
                f"{self} got an invalid value for either n_window_size or "
                f"n_window_stride. Both must be positive ints."
            )

        self.sample_rate = sample_rate
        self.win_length = n_window_size
        self.hop_length = n_window_stride
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))
        self.stft_pad_amount = (
            (self.n_fft - self.hop_length) // 2 if exact_pad else None
        )
        self.exact_pad = exact_pad
        self.sample_rate = sample_rate
        self.max_duration = max_duration

        if exact_pad:
            logger.info("STFT using exact pad")
        torch_windows = {
            "hann": torch.hann_window,
            "hamming": torch.hamming_window,
            "blackman": torch.blackman_window,
            "bartlett": torch.bartlett_window,
            "none": None,
        }
        window_fn = torch_windows.get(window)
        window_tensor = (
            window_fn(self.win_length, periodic=False) if window_fn else None
        )
        self.register_buffer("window", window_tensor)

        self.normalize = normalize
        self.log = log
        self.dither = dither
        self.frame_splicing = frame_splicing
        self.nfilt = nfilt
        self.preemph = preemph
        self.pad_to = pad_to
        highfreq = highfreq or sample_rate / 2
        self.sample_rate = sample_rate
        # disable pad min duration
        # self.pad_min_duration = 1.0
        self.pad_min_duration = 0.0
        self.pad_direction = "both"

        filterbanks = melscale_fbanks(
            n_freqs=self.n_fft // 2 + 1,
            f_min=lowfreq,
            f_max=highfreq,
            n_mels=nfilt,
            sample_rate=sample_rate,
            norm=mel_norm,
            mel_scale="slaney",
        ).T.unsqueeze(0)
        self.register_buffer("fb", filterbanks)

        # Calculate maximum sequence length
        max_length = self.get_seq_len(
            torch.tensor(max_duration * sample_rate, dtype=torch.float)
        )
        max_pad = pad_to - (max_length % pad_to) if pad_to > 0 else 0
        self.max_length = max_length + max_pad
        self.pad_value = pad_value
        self.mag_power = mag_power

        # We want to avoid taking the log of zero
        # There are two options: either adding or clamping to a small value
        if log_zero_guard_type not in ["add", "clamp"]:
            raise ValueError(
                f"{self} received {log_zero_guard_type} for the "
                f"log_zero_guard_type parameter. It must be either 'add' or "
                f"'clamp'."
            )

        self.use_grads = use_grads
        if not use_grads:
            self.forward = torch.no_grad()(self.forward)
        self._rng = random.Random() if rng is None else rng
        self.nb_augmentation_prob = nb_augmentation_prob
        if self.nb_augmentation_prob > 0.0:
            if nb_max_freq >= sample_rate / 2:
                self.nb_augmentation_prob = 0.0
            else:
                self._nb_max_fft_bin = int((nb_max_freq / sample_rate) * n_fft)

        # log_zero_guard_value is the the small we want to use, we support
        # an actual number, or "tiny", or "eps"
        self.log_zero_guard_type = log_zero_guard_type

        assert self.window is not None
        assert self.fb is not None
        self.window = self.window.to(dtype=torch.bfloat16)
        self.fb = self.fb.to(dtype=torch.bfloat16)

        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(0)

    @torch._dynamo.disable
    def stft(self, x):
        # disable autocast to get full range of stft values
        with torch.amp.autocast(x.device.type, enabled=False):
            return torch.stft(
                x,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                center=not self.exact_pad,
                window=self.window.to(dtype=torch.float, device=x.device),
                return_complex=True,
                pad_mode="constant",
            )

    def log_zero_guard_value_fn(self, x):
        if isinstance(self.log_zero_guard_value, str):
            if self.log_zero_guard_value == "tiny":
                return torch.finfo(x.dtype).tiny
            elif self.log_zero_guard_value == "eps":
                return torch.finfo(x.dtype).eps
            else:
                raise ValueError(
                    f"{self} received {self.log_zero_guard_value} for the "
                    f"log_zero_guard_type parameter. It must be either a "
                    f"number, 'tiny', or 'eps'"
                )
        else:
            return self.log_zero_guard_value

    def get_seq_len(self, seq_len):
        # Assuming that center is True is stft_pad_amount = 0
        pad_amount = (
            self.stft_pad_amount * 2
            if self.stft_pad_amount is not None
            else self.n_fft // 2 * 2
        )
        seq_len = torch.floor_divide(
            (seq_len + pad_amount - self.n_fft), self.hop_length
        )
        return seq_len.to(dtype=torch.long)

    @property
    def filter_banks(self):
        return self.fb

    def splice_frames(self, x, frame_splicing):
        """Stacks frames together across feature dim

        input is batch_size, feature_dim, num_frames
        output is batch_size, feature_dim*frame_splicing, num_frames

        """
        seq = [x]
        for n in range(1, frame_splicing):
            seq.append(torch.cat([x[:, :, :n], x[:, :, n:]], dim=2))
        return torch.cat(seq, dim=1)

    def normalize_batch(self, x, seq_len, normalize_type):
        x_mean = None
        x_std = None
        if normalize_type == "per_feature":
            batch_size = x.shape[0]
            max_time = x.shape[2]

            # When doing stream capture to a graph, item() is not allowed
            # because it calls cudaStreamSynchronize(). Therefore, we are
            # sacrificing some error checking when running with cuda graphs.
            # if (
            #     torch.cuda.is_available()
            #     and not torch.cuda.is_current_stream_capturing()
            #     and torch.any(seq_len == 1).item()
            # ):
            #     raise ValueError(
            #         "normalize_batch with `per_feature` normalize_type "
            #         "received a tensor of length 1. This will result in "
            #         "torch.std() returning nan. Make sure your audio length "
            #         "has enough samples for a single feature (ex. at least "
            #         "`hop_length` for Mel Spectrograms)."
            #     )
            time_steps = (
                torch.arange(max_time, device=x.device)
                .unsqueeze(0)
                .expand(batch_size, max_time)
            )
            valid_mask = time_steps < seq_len.unsqueeze(1)
            x_mean_numerator = torch.where(valid_mask.unsqueeze(1), x, 0.0).sum(axis=2)
            x_mean_denominator = valid_mask.sum(axis=1)
            x_mean = x_mean_numerator / x_mean_denominator.unsqueeze(1)

            # Subtract 1 in the denominator to correct for the bias.
            x_std = torch.sqrt(
                torch.sum(
                    torch.where(valid_mask.unsqueeze(1), x - x_mean.unsqueeze(2), 0.0)
                    ** 2,
                    axis=2,
                )
                / (x_mean_denominator.unsqueeze(1) - 1.0)
            )
            x_std = x_std.masked_fill(
                x_std.isnan(), 0.0
            )  # edge case: only 1 frame in denominator
            # make sure x_std is not zero
            x_std += CONSTANT
            return (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2), x_mean, x_std
        elif normalize_type == "all_features":
            x_mean = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
            x_std = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
            for i in range(x.shape[0]):
                x_mean[i] = x[i, :, : seq_len[i].item()].mean()
                x_std[i] = x[i, :, : seq_len[i].item()].std()
            # make sure x_std is not zero
            x_std += CONSTANT
            return (x - x_mean.view(-1, 1, 1)) / x_std.view(-1, 1, 1), x_mean, x_std
        elif "fixed_mean" in normalize_type and "fixed_std" in normalize_type:
            x_mean = torch.tensor(normalize_type["fixed_mean"], device=x.device)
            x_std = torch.tensor(normalize_type["fixed_std"], device=x.device)
            return (
                (x - x_mean.view(x.shape[0], x.shape[1]).unsqueeze(2))
                / x_std.view(x.shape[0], x.shape[1]).unsqueeze(2),
                x_mean,
                x_std,
            )
        else:
            return x, x_mean, x_std

    @torch.compile
    def forward(self, x, seq_len, linear_spec=False):
        if x.shape[1] < self.sample_rate * self.pad_min_duration:
            pad_amount = int(self.sample_rate * self.pad_min_duration) - x.shape[1]
            if self.pad_direction == "right":
                x = F.pad(x, (0, pad_amount), value=self.pad_value)
            elif self.pad_direction == "left":
                x = F.pad(x, (pad_amount, 0), value=self.pad_value)
            elif self.pad_direction == "both":
                left_pad = pad_amount // 2
                right_pad = pad_amount - left_pad
                x = F.pad(x, (left_pad, right_pad), value=self.pad_value)
            else:
                raise ValueError(
                    f"{self} received an invalid pad_direction: {self.pad_direction}. "
                    f"It must be one of 'left', 'right', or 'both'."
                )
            seq_len = torch.tensor([x.shape[1]], dtype=torch.float, device=x.device)

        seq_len_time = seq_len
        seq_len_unfixed = self.get_seq_len(seq_len)

        # fix for seq_len = 0 for streaming; if size was 0, it is always padded
        # to 1, and normalizer fails
        seq_len = torch.where(
            seq_len == 0, torch.zeros_like(seq_len_unfixed), seq_len_unfixed
        )

        if self.stft_pad_amount is not None:
            x = torch.nn.functional.pad(
                x.unsqueeze(1), (self.stft_pad_amount, self.stft_pad_amount), "constant"
            ).squeeze(1)

        # use dither for inference as well
        if self.dither > 0:
            x += self.dither * torch.randn(
                x.shape, dtype=x.dtype, device=x.device, generator=self.generator
            )

        # do preemphasis
        if self.preemph is not None:
            timemask = torch.arange(x.shape[1], device=x.device).unsqueeze(
                0
            ) < seq_len_time.unsqueeze(1)
            x = torch.cat(
                (x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]), dim=1
            )

            x = x.masked_fill(~timemask, 0.0)

        x = self.stft(x)

        # torch stft returns complex tensor (of shape [B,N,T]); so convert to magnitude
        # guard is needed for sqrt if grads are passed through
        guard = 0 if not self.use_grads else CONSTANT
        x = torch.view_as_real(x)
        x = torch.sqrt(x.pow(2).sum(-1) + guard)

        # get power spectrum
        if self.mag_power != 1.0:
            x = x.pow(self.mag_power)

        # return plain spectrogram if required
        if linear_spec:
            return x, seq_len

        # disable autocast, otherwise it might be automatically casted to fp16
        # on fp16 compatible GPUs and get NaN values for input value of 65520
        with torch.amp.autocast(x.device.type, enabled=False):
            # dot with filterbank energies
            x = torch.matmul(self.fb.to(x.dtype), x)

        # log features if required
        if self.log:
            if self.log_zero_guard_type == "add":
                x = torch.log(x + self.log_zero_guard_value_fn(x))
            elif self.log_zero_guard_type == "clamp":
                x = torch.log(torch.clamp(x, min=self.log_zero_guard_value_fn(x)))
            else:
                raise ValueError("log_zero_guard_type was not understood")

        # frame splicing if required
        if self.frame_splicing > 1:
            x = self.splice_frames(x, self.frame_splicing)

        # normalize if required
        if self.normalize:
            x, _, _ = self.normalize_batch(x, seq_len, normalize_type=self.normalize)

        # mask to zero any values beyond seq_len in batch, pad to multiple of
        # `pad_to` (for efficiency)
        max_len = x.size(-1)
        mask = torch.arange(max_len, device=x.device)
        mask = mask.repeat(x.size(0), 1) >= seq_len.unsqueeze(1)
        x = x.masked_fill(
            mask.unsqueeze(1).type(torch.bool).to(device=x.device), self.pad_value
        )

        del mask
        pad_to = self.pad_to
        if pad_to == "max":
            x = nn.functional.pad(
                x, (0, self.max_length - x.size(-1)), value=self.pad_value
            )
        elif pad_to > 0:
            pad_amt = x.size(-1) % pad_to
            if pad_amt != 0:
                x = nn.functional.pad(x, (0, pad_to - pad_amt), value=self.pad_value)

        return x, seq_len


class CohereASRFeatureExtractor(SequenceFeatureExtractor):
    """HF-compatible feature extractor wrapping FilterbankFeatures."""

    model_input_names = ["input_features"]

    def __init__(
        self,
        feature_size=64,
        sampling_rate=16000,
        padding_value=0.0,
        max_duration=30,
        n_window_size=320,
        n_window_stride=160,
        window="hann",
        normalize="per_feature",
        n_fft=None,
        preemph=0.97,
        lowfreq=0,
        highfreq=None,
        log=True,
        log_zero_guard_type="add",
        log_zero_guard_value=2**-24,
        dither=CONSTANT,
        pad_to=16,
        frame_splicing=1,
        exact_pad=False,
        mag_power=2.0,
        nb_augmentation_prob=0.0,
        nb_max_freq=4000,
        mel_norm="slaney",
        stft_exact_pad=False,
        stft_conv=False,
        device="cpu",
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )
        self.max_duration = max_duration
        self.hop_length = n_window_stride
        self._device = torch.device(device)
        self._fb_config = dict(
            sample_rate=sampling_rate,
            n_window_size=n_window_size,
            n_window_stride=n_window_stride,
            window=window,
            normalize=normalize,
            n_fft=n_fft,
            preemph=preemph,
            nfilt=feature_size,
            lowfreq=lowfreq,
            highfreq=highfreq,
            log=log,
            log_zero_guard_type=log_zero_guard_type,
            log_zero_guard_value=log_zero_guard_value,
            dither=dither,
            pad_to=pad_to,
            max_duration=max_duration,
            frame_splicing=frame_splicing,
            exact_pad=exact_pad,
            pad_value=padding_value,
            mag_power=mag_power,
            nb_augmentation_prob=nb_augmentation_prob,
            nb_max_freq=nb_max_freq,
            mel_norm=mel_norm,
            stft_exact_pad=stft_exact_pad,
            stft_conv=stft_conv,
            device=device,
        )
        self._filterbank: FilterbankFeatures | None = None

    @property
    def filterbank(self) -> FilterbankFeatures:
        if self._filterbank is None:
            fb = FilterbankFeatures(**self._fb_config)
            fb.eval()
            self._filterbank = fb.to(self._device)
        return self._filterbank

    def get_seq_len(self, seq_len):
        return self.filterbank.get_seq_len(seq_len)

    def __call__(
        self,
        raw_speech,
        sampling_rate=None,
        return_tensors=None,
        **kwargs,
    ) -> BatchFeature:
        if isinstance(raw_speech, np.ndarray):
            raw_speech = [raw_speech]

        seq_len = torch.tensor([s.shape[0] for s in raw_speech])

        max_len = max(s.shape[0] for s in raw_speech)
        padded = np.zeros((len(raw_speech), max_len), dtype=np.float32)
        for i, s in enumerate(raw_speech):
            padded[i, : s.shape[0]] = s

        audio_tensor = torch.from_numpy(padded).to(self._device)
        seq_len = seq_len.to(self._device)

        with torch.no_grad():
            input_features, length = self.filterbank(audio_tensor, seq_len)

        result = BatchFeature(
            {"input_features": input_features.cpu(), "length": length.cpu()}
        )
        if return_tensors is not None:
            result = result.convert_to_tensors(return_tensors)
        return result


class CohereASRProcessor(ProcessorMixin):
    """HF-compatible processor combining CohereASRFeatureExtractor and a
    tokenizer."""

    feature_extractor_class = "CohereASRFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    def __call__(
        self,
        text=None,
        audio=None,
        sampling_rate=None,
        return_tensors=None,
        **kwargs,
    ):
        if audio is not None:
            result = self.feature_extractor(
                audio,
                sampling_rate=sampling_rate,
                return_tensors=return_tensors,
            )
        else:
            result = BatchFeature()

        if text is not None:
            text_inputs = self.tokenizer(text, return_tensors=return_tensors, **kwargs)
            result["input_ids"] = text_inputs["input_ids"]

        return result


AutoFeatureExtractor.register("CohereASRFeatureExtractor", CohereASRFeatureExtractor)
AutoProcessor.register("CohereASRProcessor", CohereASRProcessor)
