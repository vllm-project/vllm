# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the token_in_token_out request/response protocol.

These tests intentionally avoid spinning up a server — they exercise the
pydantic validators on ``GenerateRequest`` directly so they run fast and
fail loudly if the validator semantics ever drift.
"""

import json

import pytest

from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    GenerateRequest,
    MultiModalFeatures,
)
from vllm.sampling_params import SamplingParams


def _base_payload() -> dict:
    return {"token_ids": [1, 2, 3], "sampling_params": {}}


def test_omitted_max_tokens_is_not_provided():
    """Body without ``max_tokens`` must surface as 'not provided' so the
    server can apply its own default instead of the dataclass 16."""
    req = GenerateRequest.model_validate(_base_payload())
    # SamplingParams' dataclass default leaks through the parsed instance —
    # this is exactly the bug the server-side defaulting works around.
    assert req.sampling_params.max_tokens == 16
    assert not req.is_sampling_param_provided("max_tokens")


def test_explicit_max_tokens_is_provided():
    """Even when the client picks the same value as the dataclass default,
    it must register as explicitly set so the server won't override it."""
    payload = _base_payload()
    payload["sampling_params"] = {"max_tokens": 16}
    req = GenerateRequest.model_validate(payload)
    assert req.sampling_params.max_tokens == 16
    assert req.is_sampling_param_provided("max_tokens")

    payload["sampling_params"] = {"max_tokens": 256}
    req = GenerateRequest.model_validate(payload)
    assert req.sampling_params.max_tokens == 256
    assert req.is_sampling_param_provided("max_tokens")


def test_other_fields_tracked_independently():
    payload = _base_payload()
    payload["sampling_params"] = {"temperature": 0.0}
    req = GenerateRequest.model_validate(payload)
    assert not req.is_sampling_param_provided("max_tokens")
    assert req.is_sampling_param_provided("temperature")


def test_json_roundtrip_preserves_provided_keys():
    payload = _base_payload()
    payload["sampling_params"] = {"temperature": 0.5}
    req = GenerateRequest.model_validate_json(json.dumps(payload))
    assert not req.is_sampling_param_provided("max_tokens")
    assert req.is_sampling_param_provided("temperature")


def test_internal_instance_construction_treats_all_as_provided():
    """When internal callers build ``GenerateRequest`` from a pre-resolved
    ``SamplingParams`` instance, every field is considered explicitly set
    so server-side defaulting can't clobber values resolved upstream."""
    sp = SamplingParams(max_tokens=500, temperature=0.0)
    req = GenerateRequest(token_ids=[1, 2, 3], sampling_params=sp)
    assert req.is_sampling_param_provided("max_tokens")
    assert req.is_sampling_param_provided("temperature")
    # And keys we never touched should also count as provided in this path.
    assert req.is_sampling_param_provided("top_p")


def _features_payload(**overrides) -> dict:
    payload = {
        "mm_hashes": {"image": ["h0"]},
        "mm_placeholders": {"image": [{"offset": 3, "length": 4}]},
    }
    payload.update(overrides)
    return payload


def test_raw_images_and_kwargs_data_are_mutually_exclusive():
    """The two payload carriers describe the same items in different forms;
    accepting both would leave the consumer no way to pick a winner."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        MultiModalFeatures.model_validate(
            _features_payload(kwargs_data={"image": ["x"]}, raw_images={"image": ["y"]})
        )


def test_raw_images_must_align_with_mm_hashes():
    """`raw_images[m][i]` is positional against `mm_hashes[m][i]`, so a
    length or key disagreement means the consumer would mispair items."""
    with pytest.raises(ValueError, match="raw_images"):
        MultiModalFeatures.model_validate(
            _features_payload(raw_images={"image": ["a", "b"]})
        )

    with pytest.raises(ValueError, match="absent from mm_hashes"):
        MultiModalFeatures.model_validate(
            _features_payload(raw_images={"video": ["a"]})
        )


def test_raw_images_alone_is_accepted():
    features = MultiModalFeatures.model_validate(
        _features_payload(raw_images={"image": ["a"]})
    )
    assert features.kwargs_data is None
    assert features.raw_images == {"image": ["a"]}
