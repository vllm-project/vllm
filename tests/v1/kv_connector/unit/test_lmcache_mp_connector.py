# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request

pytest.importorskip("lmcache")
_lmcache_mp_connector = pytest.importorskip(
    "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector"
)
LMCacheMPRequestTracker = _lmcache_mp_connector.LMCacheMPRequestTracker
_make_lmcache_mp_cache_salt = _lmcache_mp_connector._make_lmcache_mp_cache_salt


def _make_request_stub(
    cache_salt: str | None = None,
    lora_name: str | None = None,
) -> SimpleNamespace:
    lora_request = None
    if lora_name is not None:
        lora_request = LoRARequest(
            lora_name=lora_name,
            lora_int_id=1,
            lora_path=f"/adapters/{lora_name}",
        )
    return SimpleNamespace(cache_salt=cache_salt, lora_request=lora_request)


def test_lmcache_mp_cache_salt_includes_lora_name():
    salt_a = _make_lmcache_mp_cache_salt(
        _make_request_stub(cache_salt="shared/salt", lora_name="org/adapter_a")
    )
    salt_b = _make_lmcache_mp_cache_salt(
        _make_request_stub(cache_salt="shared/salt", lora_name="org/adapter_b")
    )

    assert salt_a != salt_b
    assert salt_a.startswith("vllm_lmcache_mp_v1_lora_")
    assert "/" not in salt_a
    assert "/" not in salt_b


def test_lmcache_mp_cache_salt_domain_separates_user_salt():
    lora_salt = _make_lmcache_mp_cache_salt(
        _make_request_stub(cache_salt="shared", lora_name="adapter_a")
    )
    plain_salt = _make_lmcache_mp_cache_salt(
        _make_request_stub(cache_salt=lora_salt, lora_name=None)
    )

    assert plain_salt != lora_salt


def test_lmcache_mp_cache_salt_keeps_empty_non_lora_salt_empty():
    assert _make_lmcache_mp_cache_salt(_make_request_stub()) == ""


def test_lmcache_mp_request_tracker_uses_lora_cache_salt():
    lora_request = LoRARequest(
        lora_name="adapter_a",
        lora_int_id=1,
        lora_path="/adapters/adapter_a",
    )
    request = Request(
        request_id="req",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        lora_request=lora_request,
        cache_salt="shared",
    )

    tracker = LMCacheMPRequestTracker(request)

    assert tracker.cache_salt == _make_lmcache_mp_cache_salt(request)
    assert tracker.cache_salt != "shared"
