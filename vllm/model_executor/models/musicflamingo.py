# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""MusicFlamingo model adapter.

MusicFlamingo shares the AudioFlamingo3 architecture, so we reuse the same
implementation and multimodal processor, while accepting MusicFlamingo config
and processor classes when available.
"""

from collections.abc import Mapping

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing import BaseProcessingInfo
from vllm.transformers_utils.configs.musicflamingo import (
    MusicFlamingoConfig,
    MusicFlamingoProcessor,
)

from .audioflamingo3 import (
    AudioFlamingo3DummyInputsBuilder,
    AudioFlamingo3ForConditionalGeneration,
    AudioFlamingo3MultiModalDataParser,
    AudioFlamingo3MultiModalProcessor,
)


class MusicFlamingoProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(MusicFlamingoConfig)

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(MusicFlamingoProcessor, **kwargs)

    def get_feature_extractor(self, **kwargs: object):
        hf_processor = self.get_hf_processor(**kwargs)
        return hf_processor.feature_extractor

    def get_data_parser(self):
        feature_extractor = self.get_feature_extractor()

        return AudioFlamingo3MultiModalDataParser(
            target_sr=feature_extractor.sampling_rate,
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}


class MusicFlamingoDummyInputsBuilder(AudioFlamingo3DummyInputsBuilder):
    pass


@MULTIMODAL_REGISTRY.register_processor(
    AudioFlamingo3MultiModalProcessor,
    info=MusicFlamingoProcessingInfo,
    dummy_inputs=MusicFlamingoDummyInputsBuilder,
)
class MusicFlamingoForConditionalGeneration(AudioFlamingo3ForConditionalGeneration):
    """MusicFlamingo model for conditional generation."""
