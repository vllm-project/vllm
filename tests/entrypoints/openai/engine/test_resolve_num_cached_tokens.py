# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.engine.serving import resolve_num_cached_tokens


def test_resolve_prefers_pd_prefill_cache_stats():
    kv_params = {"do_remote_prefill": True, "num_cached_tokens": 1280}
    assert resolve_num_cached_tokens(kv_params, 0) == 1280


def test_resolve_falls_back_without_kv_transfer_params():
    assert resolve_num_cached_tokens(None, 42) == 42


def test_resolve_falls_back_when_not_remote_prefill():
    kv_params = {"do_remote_decode": True, "num_cached_tokens": 1280}
    assert resolve_num_cached_tokens(kv_params, 42) == 42


def test_resolve_falls_back_when_num_cached_tokens_missing():
    kv_params = {"do_remote_prefill": True}
    assert resolve_num_cached_tokens(kv_params, 42) == 42
