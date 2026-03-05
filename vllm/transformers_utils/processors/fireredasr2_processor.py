# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoFeatureExtractor,
    AutoProcessor,
    BatchFeature,
)
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.processing_utils import ProcessorMixin
from transformers.utils import TensorType

from vllm.logger import init_logger
from vllm.utils.import_utils import LazyLoader

if TYPE_CHECKING:
    import kaldi_native_fbank as knf
else:
    knf = LazyLoader("knf", globals(), "kaldi_native_fbank")


logger = init_logger(__name__)


class CMVN:
    def __init__(self, dim, means, inverse_std_variences):
        self.dim, self.means, self.inverse_std_variences = (
            dim,
            np.array(means),
            np.array(inverse_std_variences),
        )

    def __call__(self, x):
        assert x.shape[-1] == self.dim, "CMVN dim mismatch"
        out = x - self.means
        out = out * self.inverse_std_variences
        return out


class KaldifeatFbank:
    def __init__(self, num_mel_bins=80, frame_length=25, frame_shift=10, dither=1.0):
        self.dither = dither
        opts = knf.FbankOptions()
        opts.frame_opts.dither = dither
        opts.mel_opts.num_bins = num_mel_bins
        opts.frame_opts.snip_edges = True
        opts.mel_opts.debug_mel = False
        self.opts = opts

    def __call__(self, sample_rate, wav_np, is_train=False):
        dither = self.dither if is_train else 0.0
        self.opts.frame_opts.dither = dither
        fbank = knf.OnlineFbank(self.opts)

        fbank.accept_waveform(sample_rate, wav_np.tolist())
        feat = []
        for i in range(fbank.num_frames_ready):
            feat.append(fbank.get_frame(i))
        if len(feat) == 0:
            print("Check data, len(feat) == 0", wav_np, flush=True)
            return np.zeros((0, self.opts.mel_opts.num_bins))
        feat = np.vstack(feat)
        return feat


class FireRedASR2FeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a FireRedASR2 feature extractor.

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
        chunk_length (`int`, *optional*, defaults to 30):
            The maximum number of chunks of `sampling_rate` samples used to
            trim and pad longer or shorter audio sequences.
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
        chunk_length=30,
        padding_value=0.0,
        return_attention_mask=False,
        dim=80,
        means=None,
        inverse_std_variences=None,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        max_length=3000,
        downsample_rate=2,
        left_context=3,
        right_context=3,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        self.chunk_length = chunk_length
        self.max_length = max_length
        self.dim = dim
        self.means = means
        self.inverse_std_variences = inverse_std_variences
        self.num_mel_bins = num_mel_bins
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.dither = dither
        self.sampling_rate = sampling_rate
        self.downsample_rate = downsample_rate
        self.context = left_context + 1 + right_context

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
        **kwargs,
    ) -> BatchFeature:
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                f"The model corresponding to this feature extractor: "
                f"{self.__class__.__name__} was trained using a sampling "
                f"rate of {self.sampling_rate}. Please make sure that the "
                f"provided `raw_speech` input was sampled with "
                f"{self.sampling_rate} and not {sampling_rate}."
            )

        def padding_position_is_0(padded_input, input_lengths):
            N, T = padded_input.size()[:2]
            mask = torch.ones((N, T)).to(padded_input.device)
            for i in range(N):
                mask[i, input_lengths[i] :] = 0
            mask = mask.unsqueeze(dim=1)
            return mask.to(torch.uint8)

        # initialize the CMVN and Fbank objects
        self.cmvn = CMVN(self.dim, self.means, self.inverse_std_variences)
        self.fbank = KaldifeatFbank(
            num_mel_bins=self.num_mel_bins,
            frame_length=self.frame_length,
            frame_shift=self.frame_shift,
            dither=self.dither,
        )

        feats = []
        speech_lengths = []
        fake_token_lengths = []
        for speech in raw_speech:
            """
            We must multiply by 32768 here because FireRedASR2 loads audio data
            using kaldiio.load_mat, while vLLM loads audio data using librosa.
            """
            speech = speech * 32768
            fbank = self.fbank(sampling_rate, speech)
            fbank = self.cmvn(fbank)
            fbank = torch.from_numpy(fbank).float()
            length = fbank.size(0)
            feats.append(fbank)
            speech_lengths.append(length)
            padded_input2 = fbank
            padded_input2 = F.pad(
                padded_input2, (0, 0, 0, self.context - 1), "constant", 0.0
            )
            src_mask = padding_position_is_0(
                padded_input2[None, :, :], torch.tensor([length], dtype=torch.int32)
            )
            x_mask = src_mask
            mask = x_mask[:, :, :-2:2][:, :, :-2:2]
            input_lengths = mask[:, -1, :].sum(dim=-1)
            input_lengths = input_lengths // self.downsample_rate
            fake_token_len = torch.clamp(input_lengths, min=1)
            fake_token_lengths.append(fake_token_len)

        feats = torch.stack(feats, dim=0)
        batched_speech = self.pad(
            BatchFeature({"input_features": feats}),
            padding=padding,
            max_length=max_length if max_length else self.max_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask or do_normalize,
        )

        if return_tensors is not None:
            batched_speech = batched_speech.convert_to_tensors(return_tensors)

        batched_speech["speech_lengths"] = torch.tensor(speech_lengths)
        batched_speech["fake_token_lengths"] = torch.concat(fake_token_lengths)
        return batched_speech


class FireRedASR2Processor(ProcessorMixin):
    r"""
    Constructs a FireRedASR2 processor which wraps a FireRedASR2 feature extractor and
    a FireRedASR2 tokenizer into a single processor.

    [`FireRedASR2Processor`] offers all the functionalities of
    [`FireRedASR2FeatureExtractor`] and [`Qwen2Tokenizer`]. See the
    [`~FireRedASR2Processor.__call__`] and [`~FireRedASR2Processor.decode`] for more
    information.

    Args:
        feature_extractor (`FireRedASR2FeatureExtractor`): An instance of
            [`FireRedASR2FeatureExtractor`].
            The feature extractor is a required input.
        tokenizer (`Qwen2Tokenizer`):
            An instance of [`Qwen2Tokenizer`]. The tokenizer is a required
            input.
    """

    feature_extractor_class = "FireRedASR2FeatureExtractor"
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
        Forwards the `audio` argument to FireRedASR2FeatureExtractor's
        [`~FireRedASR2FeatureExtractor.__call__`] and the `text` argument to
        [`~Qwen2Tokenizer.__call__`]. Please refer to the docstring of the
        above two methods for more information.
        """
        if self._in_target_context_manager:
            return self.current_processor(*args, **kwargs)

        audio = kwargs.pop("audio", None)
        sampling_rate = kwargs.pop("sampling_rate", None)
        text = kwargs.pop("text", None)
        if len(args) > 0:
            audio = args[0]
            args = args[1:]

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
                    num_audio_tokens = int(inputs["fake_token_lengths"].item())

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


AutoFeatureExtractor.register(
    "FireRedASR2FeatureExtractor", FireRedASR2FeatureExtractor
)
AutoProcessor.register("FireRedASR2Processor", FireRedASR2Processor)
