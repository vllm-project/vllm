# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoProcessor, BatchFeature
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.processing_utils import ProcessorMixin

NORMALIZATION_EPS = 1e-7


class OmniASRFeatureExtractor(SequenceFeatureExtractor):
    def __init__(
        self, sampling_rate=16000, padding_value=0.0, feature_size=1, **kwargs
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )

    def __call__(
        self,
        audio: list[np.ndarray] | np.ndarray,
        sampling_rate: int | None = None,
        **kwargs,
    ) -> BatchFeature:
        data = {}
        features = []
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"Expected sampling_rate={self.sampling_rate}, got {sampling_rate}. "
                f"OmniASR requires 16kHz audio."
            )
        if not isinstance(audio, list):
            audio = [audio]
        for a in audio:
            t = torch.tensor(a, dtype=torch.float32)
            t = (t - t.mean()) / torch.sqrt(t.var() + NORMALIZATION_EPS)
            features.append(t)
        data["input_features"] = features
        return BatchFeature(data=data)


class OmniASRProcessor(ProcessorMixin):
    """HF-compatible processor combining OmniASRFeatureExtractor and a tokenizer."""

    feature_extractor_class = "OmniASRFeatureExtractor"
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
            )
        else:
            result = BatchFeature()
        if text is not None:
            text_input = self.tokenizer(text, return_tensors=return_tensors, **kwargs)
            result["input_ids"] = text_input["input_ids"]

        return result


AutoFeatureExtractor.register("OmniASRFeatureExtractor", OmniASRFeatureExtractor)
AutoProcessor.register("OmniASRProcessor", OmniASRProcessor)
