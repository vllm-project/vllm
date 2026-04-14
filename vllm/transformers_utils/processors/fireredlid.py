# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FireRedLID feature extractor and processor.

The FeatureExtractor handles:
  - Raw waveform → 80-dim log-mel filterbank (via kaldi_native_fbank)
  - CMVN normalization (means / inverse_std_variences from preprocessor_config)
  - Padding + length tracking

The Processor wraps the FeatureExtractor and a tokenizer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoFeatureExtractor,
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


# ---------------------------------------------------------------------------
# Helpers (shared with FireRedASR2 processor)
# ---------------------------------------------------------------------------


class CMVN:
    def __init__(self, dim, means, inverse_std_variences):
        self.dim = dim
        self.means = np.array(means)
        self.inverse_std_variences = np.array(inverse_std_variences)

    def __call__(self, x):
        assert x.shape[-1] == self.dim, "CMVN dim mismatch"
        out = x - self.means
        out = out * self.inverse_std_variences
        return out


class KaldifeatFbank:
    def __init__(
        self,
        num_mel_bins: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        dither: float = 0.0,
    ):
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
            return np.zeros((0, self.opts.mel_opts.num_bins))
        return np.vstack(feat)


# ---------------------------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------------------------


class FireRedLIDFeatureExtractor(SequenceFeatureExtractor):
    """
    Extracts 80-dim log-mel filterbank features from raw waveforms,
    applies CMVN, and returns padded feature tensors with lengths.

    Also computes ``fake_token_lengths`` — the actual encoder output
    length for each audio — so that vLLM can allocate the correct
    number of cross-attention KV cache slots.
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
        self.dim = dim
        self.means = means
        self.inverse_std_variences = inverse_std_variences
        self.num_mel_bins = num_mel_bins
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.dither = dither
        self.sampling_rate = sampling_rate
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
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"FireRedLIDFeatureExtractor expects sampling_rate="
                f"{self.sampling_rate}, got {sampling_rate}."
            )

        # Initialize helpers
        cmvn = CMVN(self.dim, self.means, self.inverse_std_variences)
        fbank = KaldifeatFbank(
            num_mel_bins=self.num_mel_bins,
            frame_length=self.frame_length,
            frame_shift=self.frame_shift,
            dither=self.dither,
        )

        def padding_position_is_0(padded_input, input_lengths):
            N, T = padded_input.size()[:2]
            mask = torch.ones((N, T)).to(padded_input.device)
            for i in range(N):
                mask[i, input_lengths[i] :] = 0
            mask = mask.unsqueeze(dim=1)
            return mask.to(torch.uint8)

        feats = []
        speech_lengths = []
        fake_token_lengths = []

        for speech in raw_speech:
            # vLLM loads audio via librosa (float32 in [-1,1]),
            # but kaldi_native_fbank expects int16-scale values.
            speech_scaled = speech * 32768
            feat = fbank(self.sampling_rate, speech_scaled)
            feat = cmvn(feat)
            feat = torch.from_numpy(feat).float()
            length = feat.size(0)
            feats.append(feat)
            speech_lengths.append(length)

            # Compute the actual Conv2dSubsampling output length.
            # This mirrors the mask logic in Conv2dSubsampling.forward:
            #   pad context frames, then mask[:, :, :-2:2][:, :, :-2:2].sum()
            padded_input = F.pad(feat, (0, 0, 0, self.context - 1), "constant", 0.0)
            src_mask = padding_position_is_0(
                padded_input[None, :, :],
                torch.tensor([length], dtype=torch.int32),
            )
            mask = src_mask[:, :, :-2:2][:, :, :-2:2]
            enc_len = mask[:, -1, :].sum(dim=-1)
            fake_token_len = torch.clamp(enc_len, min=1)
            fake_token_lengths.append(fake_token_len)

        if len(feats) == 0:
            return BatchFeature()

        # Pad to uniform length
        max_feat_len = max(f.size(0) for f in feats)
        padded = feats[0].new_zeros(len(feats), max_feat_len, feats[0].size(1))
        for i, feat in enumerate(feats):
            padded[i, : feat.size(0)] = feat

        result = BatchFeature({"input_features": padded})

        if return_tensors is not None:
            result = result.convert_to_tensors(return_tensors)

        result["speech_lengths"] = torch.tensor(speech_lengths, dtype=torch.long)
        result["fake_token_lengths"] = torch.concat(fake_token_lengths)
        return result


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


class FireRedLIDProcessor(ProcessorMixin):
    """
    Wraps FireRedLIDFeatureExtractor + a tokenizer.
    """

    feature_extractor_class = "FireRedLIDFeatureExtractor"
    tokenizer_class = ("PreTrainedTokenizer", "PreTrainedTokenizerFast")

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)
        self.current_processor = self.feature_extractor
        self._in_target_context_manager = False

    def __call__(self, *args, **kwargs):
        if self._in_target_context_manager:
            return self.current_processor(*args, **kwargs)

        audio = kwargs.pop("audio", None)
        sampling_rate = kwargs.pop("sampling_rate", None)
        text = kwargs.pop("text", None)
        if len(args) > 0:
            audio = args[0]
            args = args[1:]

        if audio is not None:
            inputs = self.feature_extractor(
                audio, *args, sampling_rate=sampling_rate, **kwargs
            )
        else:
            inputs = BatchFeature()

        if text is not None:
            if isinstance(text, str):
                text = [text]
            encodings = self.tokenizer(text, **kwargs)
            if audio is not None:
                inputs["labels"] = encodings["input_ids"]
            else:
                return encodings

        return inputs


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

AutoFeatureExtractor.register("FireRedLIDFeatureExtractor", FireRedLIDFeatureExtractor)
