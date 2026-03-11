# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""MusicFlamingo model adapter.

MusicFlamingo shares the AudioFlamingo3 architecture, so we reuse the same
implementation and multimodal processor, while accepting MusicFlamingo config
and processor classes when available.
"""

from collections.abc import Mapping

from transformers.models.audioflamingo3 import (
    AudioFlamingo3Config,
    AudioFlamingo3Processor,
)

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing import BaseProcessingInfo

from .audioflamingo3 import (
    AudioFlamingo3DummyInputsBuilder,
    AudioFlamingo3ForConditionalGeneration,
    AudioFlamingo3MultiModalProcessor,
)

try:
    # Optional dependency: use MusicFlamingo classes when transformers provides them.
    from transformers.models.musicflamingo import (
        MusicFlamingoConfig,
        MusicFlamingoProcessor,
    )
except Exception:  # pragma: no cover - optional dependency
    MusicFlamingoConfig = None
    MusicFlamingoProcessor = None


class MusicFlamingoProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        if MusicFlamingoConfig is None:
            return self.ctx.get_hf_config(AudioFlamingo3Config)
        return self.ctx.get_hf_config((MusicFlamingoConfig, AudioFlamingo3Config))

    def get_hf_processor(self, **kwargs: object):
        if MusicFlamingoProcessor is None:
            return self.ctx.get_hf_processor(AudioFlamingo3Processor, **kwargs)
        # Tuple triggers AutoProcessor path and accepts either processor class.
        return self.ctx.get_hf_processor(
            (MusicFlamingoProcessor, AudioFlamingo3Processor), **kwargs
        )

    def get_feature_extractor(self, **kwargs: object):
        hf_processor = self.get_hf_processor(**kwargs)
        return hf_processor.feature_extractor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}


class MusicFlamingoDummyInputsBuilder(AudioFlamingo3DummyInputsBuilder):
    pass


@MULTIMODAL_REGISTRY.register_processor(
    AudioFlamingo3MultiModalProcessor,
    info=MusicFlamingoProcessingInfo,
    dummy_inputs=MusicFlamingoDummyInputsBuilder,
)
class MusicFlamingoForConditionalGeneration(AudioFlamingo3ForConditionalGeneration):
    """MusicFlamingo model for conditional generation."""
