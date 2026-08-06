# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.model_executor.models.fireredasr2 import FireRedASR2ProcessingInfo
from vllm.transformers_utils.processors.fireredasr2 import FireRedASR2FeatureExtractor

pytestmark = pytest.mark.skip_global_cleanup


def _feature_extractor(max_length: int) -> FireRedASR2FeatureExtractor:
    extractor = object.__new__(FireRedASR2FeatureExtractor)
    extractor.max_length = max_length
    return extractor


def test_fireredasr2_request_max_length_cannot_exceed_server_limit():
    info = object.__new__(FireRedASR2ProcessingInfo)

    def get_hf_processor(**kwargs):
        max_length = int(kwargs.get("max_length", 3000))
        return SimpleNamespace(feature_extractor=_feature_extractor(max_length))

    info.get_hf_processor = get_hf_processor

    with pytest.raises(ValueError, match="may not exceed"):
        info.get_feature_extractor(max_length=3001)


def test_fireredasr2_request_max_length_can_reduce_server_limit():
    info = object.__new__(FireRedASR2ProcessingInfo)

    def get_hf_processor(**kwargs):
        max_length = int(kwargs.get("max_length", 3000))
        return SimpleNamespace(feature_extractor=_feature_extractor(max_length))

    info.get_hf_processor = get_hf_processor

    extractor = info.get_feature_extractor(max_length=1024)

    assert extractor.max_length == 1024
