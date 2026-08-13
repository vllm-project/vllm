# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

from vllm.v1.worker.adaptive_verification_profile_cache import (
    PROFILE_CACHE_SCHEMA_VERSION,
    _cache_path,
    _digest,
    load_profile_cache,
    profile_cache_fingerprint,
    save_profile_cache,
    sentinel_batches,
    validate_profile_sentinel,
)


def _factors() -> dict:
    return {
        "schema": PROFILE_CACHE_SCHEMA_VERSION,
        "model": {"revision": "abc"},
        "hardware": {"sm": "12.1"},
        "profile": {"k": 5, "capture_sizes": [1, 8]},
    }


def _curves() -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    return [(1, 0.5), (4, 0.8)], [(1, 1.25), (8, 2.5)]


def test_profile_cache_exact_round_trip(tmp_path) -> None:
    factors = _factors()
    curves = _curves()

    save_profile_cache(factors, curves, str(tmp_path))

    assert load_profile_cache(factors, str(tmp_path)) == curves
    changed = {**factors, "profile": {**factors["profile"], "k": 4}}
    assert load_profile_cache(changed, str(tmp_path)) is None


def test_profile_cache_rejects_corruption(tmp_path) -> None:
    factors = _factors()
    save_profile_cache(factors, _curves(), str(tmp_path))
    path = _cache_path(profile_cache_fingerprint(factors), str(tmp_path))
    with open(path) as file:
        payload = json.load(file)
    payload["draft_curve"][0][1] = 999.0
    with open(path, "w") as file:
        json.dump(payload, file)

    assert load_profile_cache(factors, str(tmp_path)) is None


def test_profile_cache_rejects_other_schema_version(tmp_path) -> None:
    factors = _factors()
    save_profile_cache(factors, _curves(), str(tmp_path))
    path = _cache_path(profile_cache_fingerprint(factors), str(tmp_path))
    with open(path) as file:
        payload = json.load(file)
    payload.pop("checksum")
    payload["schema"] = PROFILE_CACHE_SCHEMA_VERSION + 1
    payload["checksum"] = _digest(payload)
    with open(path, "w") as file:
        json.dump(payload, file)

    assert load_profile_cache(factors, str(tmp_path)) is None


def test_profile_sentinel_requires_matching_graph_points(monkeypatch) -> None:
    monkeypatch.setattr(
        "vllm.envs.VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN", 4096
    )
    assert sentinel_batches([1, 8]) == [
        {"num_tokens": 8, "context_len": 4096},
        {"num_tokens": 8, "context_len": 4096},
    ]
    samples = [
        SimpleNamespace(
            forward_ms=2.0,
            drafter_ms=0.75,
            num_target_tokens=8,
            num_reqs=4,
            full_cudagraph=True,
        ),
        SimpleNamespace(
            forward_ms=2.2,
            drafter_ms=0.70,
            num_target_tokens=8,
            num_reqs=4,
            full_cudagraph=True,
        ),
    ]
    assert validate_profile_sentinel(samples, _curves())
    assert not validate_profile_sentinel(
        [*samples[:1], SimpleNamespace(**{**vars(samples[1]), "forward_ms": 9.0})],
        _curves(),
    )
    assert not validate_profile_sentinel(
        [
            *samples[:1],
            SimpleNamespace(**{**vars(samples[1]), "full_cudagraph": False}),
        ],
        _curves(),
    )
