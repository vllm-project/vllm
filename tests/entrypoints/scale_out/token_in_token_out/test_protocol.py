# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the token_in_token_out request/response protocol.

These tests intentionally avoid spinning up a server — they exercise the
pydantic validators on ``GenerateRequest`` directly so they run fast and
fail loudly if the validator semantics ever drift.
"""

import json

from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    GenerateRequest,
    MultiModalFeatures,
    PlaceholderRangeInfo,
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


# ---------------------------------------------------------------------------
# Metadata-only payload tests
# ---------------------------------------------------------------------------


def _mm_features(kwargs_data: dict | None = None) -> MultiModalFeatures:
    return MultiModalFeatures(
        mm_hashes={"image": ["hash0"]},
        mm_placeholders={"image": [PlaceholderRangeInfo(offset=0, length=10)]},
        kwargs_data=kwargs_data,
    )


def _mm_payload(
    kwargs_data: dict | None = None,
    kv_transfer_params: dict | None = None,
    ec_transfer_params: dict | None = None,
) -> dict:
    features = _mm_features(kwargs_data)
    payload = _base_payload()
    payload["features"] = features.model_dump()
    if kv_transfer_params is not None:
        payload["kv_transfer_params"] = kv_transfer_params
    if ec_transfer_params is not None:
        payload["ec_transfer_params"] = ec_transfer_params
    return payload


def test_hash_only_decode_with_kv_transfer_params_accepted():
    """Metadata-only request with kv_transfer_params (remote prefill decode)
    is legal — embeddings arrive out-of-band via the EC connector."""
    payload = _mm_payload(
        kwargs_data=None,
        kv_transfer_params={"do_remote_prefill": True},
    )
    req = GenerateRequest.model_validate(payload)
    assert req.features.kwargs_data is None
    assert req.kv_transfer_params == {"do_remote_prefill": True}


def test_hash_only_decode_with_ec_transfer_params_accepted():
    """Metadata-only request with ec_transfer_params is legal — embeddings
    arrive via the encoder-cache transfer path."""
    payload = _mm_payload(
        kwargs_data=None,
        ec_transfer_params={"hash0": {"metadata": {}}},
    )
    req = GenerateRequest.model_validate(payload)
    assert req.features.kwargs_data is None
    assert req.ec_transfer_params is not None


def test_metadata_only_without_transfer_params_accepted():
    """Metadata-only request with no transfer params stays valid. Whether
    it can be served is a runtime cache-residency question resolved by the
    generate worker — a hit uses the receiver cache, a miss raises
    MultiModalCacheMissError and the caller retries with data."""
    payload = _mm_payload(kwargs_data=None)
    req = GenerateRequest.model_validate(payload)
    assert req.features.kwargs_data is None
    assert req.kv_transfer_params is None
    assert req.ec_transfer_params is None


def test_full_payload_with_kwargs_data_accepted_without_transfer_params():
    """Full payload (kwargs_data present) needs no transfer params — the
    generate worker has the tensor data inline."""
    payload = _mm_payload(
        kwargs_data={"image": ["base64encodeddata"]},
    )
    req = GenerateRequest.model_validate(payload)
    assert req.features.kwargs_data is not None
    assert req.kv_transfer_params is None
    assert req.ec_transfer_params is None


def test_no_features_accepted_without_transfer_params():
    """Text-only request (no features) needs no transfer params."""
    req = GenerateRequest.model_validate(_base_payload())
    assert req.features is None


def test_hash_only_decode_json_roundtrip():
    """Hash-only decode request survives JSON serialization roundtrip."""
    payload = _mm_payload(
        kwargs_data=None,
        kv_transfer_params={"do_remote_prefill": True, "remote_engine_id": 42},
    )
    json_str = json.dumps(payload)
    req = GenerateRequest.model_validate_json(json_str)
    assert req.features.kwargs_data is None
    assert req.features.mm_hashes == {"image": ["hash0"]}
    assert req.kv_transfer_params["remote_engine_id"] == 42
