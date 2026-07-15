# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm.models.inkling.common.mm_preprocess import InklingMultiModalDataParser
from vllm.models.inkling.configs import (
    InklingAudioConfig,
    InklingVisionConfig,
)


@pytest.mark.parametrize(
    ("config_cls", "kwargs", "missing"),
    [
        (InklingAudioConfig, {"decoder_dmodel": 16}, "n_mel_bins"),
        (InklingVisionConfig, {"decoder_dmodel": 16}, "vision_encoder_type"),
    ],
)
def test_enabled_tower_requires_architecture_fields(config_cls, kwargs, missing):
    with pytest.raises(ValueError, match=missing):
        config_cls(**kwargs)


def test_inkling_raw_2d_audio_is_rejected_as_ambiguous():
    parser = InklingMultiModalDataParser(target_sr=16_000, target_channels=1)
    with pytest.raises(ValueError, match="ambiguous channel layout"):
        parser._parse_audio_data(np.zeros((2, 100), dtype=np.float32))
