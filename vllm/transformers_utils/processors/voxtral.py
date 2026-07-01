# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from math import ceil

import numpy as np
import torch
from mistral_common.tokens.tokenizers.audio import AudioEncoder
from transformers import BatchFeature, ProcessorMixin, TensorType
from transformers.audio_utils import AudioInput

from vllm.tokenizers.mistral import MistralTokenizer


class MistralCommonFeatureExtractor:
    """
    Provide a HF-compatible interface for
    `mistral_common.tokens.tokenizers.multimodal.AudioEncoder`.
    """

    def __init__(self, audio_encoder: AudioEncoder) -> None:
        self.audio_encoder = audio_encoder

    @property
    def sampling_rate(self):
        return self.audio_encoder.audio_config.sampling_rate

    @property
    def frame_rate(self):
        return self.audio_encoder.audio_config.frame_rate

    def __call__(
        self,
        audios: AudioInput,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        audios_lst = [audios] if not isinstance(audios, list) else audios

        audios_processed = list[torch.Tensor]()

        for audio in audios_lst:
            audio = np.asarray(audio, dtype=np.float32).ravel()
            if not self.audio_encoder.audio_config.is_streaming:
                audio = self.audio_encoder.pad(audio, self.sampling_rate)

            audios_processed.append(torch.from_numpy(audio))

        return BatchFeature(
            {"audio_arrays": audios_processed}, tensor_type=return_tensors
        )

    def get_num_audio_tokens(self, audio_length: int) -> int:
        return ceil(audio_length / (self.sampling_rate // self.frame_rate))

    def fetch_audio(self, audio_url_or_urls, sampling_rate=None):
        """HF-compatible duck-typed ``fetch_audio``.

        Mirrors :meth:`transformers.SequenceFeatureExtractor.fetch_audio` so
        :class:`transformers.ProcessorMixin.prepare_inputs_layout` (added in
        transformers 5.10) works on this duck-typed feature extractor. Older
        transformers versions never invoke this method, so the addition is a
        no-op there.

        Accepts the same shapes as ``SequenceFeatureExtractor.fetch_audio``:

        * ``np.ndarray`` / ``torch.Tensor`` — returned as-is.
        * ``list[float]`` — returned as-is (a single audio sample).
        * ``str`` URL or path — delegated to
          :func:`transformers.audio_utils.load_audio`.
        * ``list`` of any of the above — recursed element-wise.

        ``ProcessorMixin.prepare_inputs_layout`` always passes already-decoded
        audio (numpy array or torch tensor), so the str / list-of-str branches
        exist only to keep the contract identical to the upstream method.

        The semantics of ``transformers.audio_utils.is_valid_audio`` differ
        between transformers versions (5.9 only accepts ndarray/tensor; 5.10
        also accepts ``list[float]``). We detect ``list[float]`` explicitly to
        keep behavior identical across versions.
        """
        from transformers.audio_utils import is_valid_audio

        sampling_rate = sampling_rate if sampling_rate else self.sampling_rate
        if is_valid_audio(audio_url_or_urls):
            return audio_url_or_urls
        if isinstance(audio_url_or_urls, (list, tuple)):
            if audio_url_or_urls and isinstance(audio_url_or_urls[0], float):
                # A single audio represented as ``list[float]``.
                return audio_url_or_urls
            return [
                self.fetch_audio(x, sampling_rate=sampling_rate)
                for x in audio_url_or_urls
            ]
        if isinstance(audio_url_or_urls, str):
            from transformers.audio_utils import load_audio

            return load_audio(audio_url_or_urls, sampling_rate=sampling_rate)
        raise TypeError(
            "only a numpy array, torch tensor, str URL/path, or list of those "
            f"is supported but got type={type(audio_url_or_urls)}"
        )


class MistralCommonVoxtralProcessor(ProcessorMixin):
    attributes = ["feature_extractor", "tokenizer"]

    def __init__(
        self,
        tokenizer: MistralTokenizer,
        feature_extractor: MistralCommonFeatureExtractor,
    ) -> None:
        self.tokenizer = tokenizer.transformers_tokenizer
        self.feature_extractor = feature_extractor

        audio_special_ids = self.feature_extractor.audio_encoder.special_ids
        self.audio_token_id = audio_special_ids.audio
        self.begin_audio_token_id = audio_special_ids.begin_audio
