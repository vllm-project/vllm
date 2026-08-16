# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layout resolution across all of a model's attention backends.

resolve_kv_cache_layout sees every backend at once, so its answer -- and the errors it
raises -- must not depend on the order the backends were selected in.
"""

import pytest

import vllm.v1.attention.backends.utils as attn_backend_utils
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionBackend
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout
from vllm.v1.kv_cache_interface import KVCacheLayout


class _RequiresLHBNC(TritonAttentionBackend):
    """Stands in for ROCM_ATTN: requires a layout its peers cannot read."""

    @classmethod
    def get_name(cls) -> str:
        return "REQUIRES_LHBNC"

    @classmethod
    def get_required_kv_cache_layout(cls) -> str:
        return "LHBNC"


class _RequiresLBHNC(TritonAttentionBackend):
    @classmethod
    def get_name(cls) -> str:
        return "REQUIRES_LBHNC"

    @classmethod
    def get_required_kv_cache_layout(cls) -> str:
        return "LBHNC"


@pytest.fixture(autouse=True)
def clear_resolved_layout(monkeypatch):
    monkeypatch.setattr(attn_backend_utils, "_RESOLVED_KV_CACHE_LAYOUT", None)
    monkeypatch.setattr(attn_backend_utils, "_KV_CACHE_LAYOUT_OVERRIDE", None)
    monkeypatch.setattr(attn_backend_utils.envs, "VLLM_KV_CACHE_LAYOUT", None)


def test_required_layout_wins_over_peers_in_either_order():
    for backends in (
        [_RequiresLBHNC, TritonAttentionBackend],
        [TritonAttentionBackend, _RequiresLBHNC],
    ):
        assert resolve_kv_cache_layout(backends) is KVCacheLayout.LBHNC


def test_peer_that_cannot_read_the_required_layout_is_rejected():
    # TRITON_ATTN does not support LHBNC; resolving late must not hide that.
    for backends in (
        [_RequiresLHBNC, TritonAttentionBackend],
        [TritonAttentionBackend, _RequiresLHBNC],
    ):
        with pytest.raises(ValueError, match="not supported by the TRITON_ATTN"):
            resolve_kv_cache_layout(backends)


def test_conflicting_requirements_are_rejected():
    for backends in (
        [_RequiresLHBNC, _RequiresLBHNC],
        [_RequiresLBHNC, _RequiresLHBNC],
    ):
        with pytest.raises(ValueError, match="conflicting layouts"):
            resolve_kv_cache_layout(backends)


def test_ssm_backends_do_not_veto_the_layout():
    # Mamba pages are single-head and single-state, so SSM backends leave
    # supports_kv_cache_layout() at the head-inside-block default.
    assert not Mamba2AttentionBackend.supports_kv_cache_layout(KVCacheLayout.LHBNC)
    layout = resolve_kv_cache_layout([_RequiresLHBNC, Mamba2AttentionBackend])
    assert layout is KVCacheLayout.LHBNC
