# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The engine runs `Platform.validate_request` on incoming requests, so a
platform's checks also apply to frontends that bypass `InputProcessor`."""

from unittest.mock import patch

import numpy as np
import pytest

from vllm import SamplingParams
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.platforms.interface import Platform
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import _validate_request_on_platform


def make_request(**kwargs) -> EngineCoreRequest:
    fields = dict(
        request_id="req",
        prompt_token_ids=np.array([1, 2, 3, 4]),
        mm_features=None,
        sampling_params=SamplingParams(),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    fields.update(kwargs)
    return EngineCoreRequest(**fields)


class RecordingPlatform(Platform):
    seen: list = []
    error: Exception | None = None

    @classmethod
    def validate_request(cls, processed_inputs, params) -> None:
        cls.seen.append((processed_inputs, params))
        if cls.error is not None:
            raise cls.error


@pytest.fixture
def platform():
    RecordingPlatform.seen = []
    RecordingPlatform.error = None
    with patch("vllm.platforms.current_platform", RecordingPlatform):
        yield RecordingPlatform


def test_token_request(platform):
    request = make_request()
    _validate_request_on_platform(request)
    ((engine_input, params),) = platform.seen
    assert engine_input == {"type": "token", "prompt_token_ids": [1, 2, 3, 4]}
    assert params is request.sampling_params


def test_multimodal_request(platform):
    position = PlaceholderRange(offset=1, length=2)
    feature = MultiModalFeatureSpec(
        data=None, modality="image", identifier="img0", mm_position=position
    )
    _validate_request_on_platform(make_request(mm_features=[feature]))
    ((engine_input, _),) = platform.seen
    assert engine_input["type"] == "multimodal"
    assert engine_input["prompt_token_ids"] == [1, 2, 3, 4]
    assert engine_input["mm_placeholders"] == {"image": [position]}
    assert engine_input["mm_hashes"] == {"image": ["img0"]}


@pytest.mark.parametrize(
    "error",
    [VLLMValidationError("unsupported option"), ValueError("unsupported option")],
)
def test_rejection_is_a_validation_error(platform, error):
    platform.error = error
    with pytest.raises(VLLMValidationError, match="unsupported option"):
        _validate_request_on_platform(make_request())
