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


class _OnlyLHBNC(TritonAttentionBackend):
    """Stands in for ROCM_ATTN: supports a layout its peers cannot read."""

    @classmethod
    def get_name(cls) -> str:
        return "ONLY_LHBNC"

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.LHBNC,)


class _OnlyLBHNC(TritonAttentionBackend):
    @classmethod
    def get_name(cls) -> str:
        return "ONLY_LBHNC"

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.LBHNC,)


class _BlockOuter(TritonAttentionBackend):
    """Stands in for the DeepSeek sparse indexer: overlaid groups need the
    layer dim inside the block dim."""

    @classmethod
    def get_name(cls) -> str:
        return "BLOCK_OUTER"

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.BLHNC, KVCacheLayout.BLNHC)


@pytest.fixture(autouse=True)
def clear_resolved_layout(monkeypatch):
    monkeypatch.setattr(attn_backend_utils, "_RESOLVED_KV_CACHE_LAYOUT", None)
    monkeypatch.setattr(attn_backend_utils, "_KV_CACHE_LAYOUT_OVERRIDE", None)
    monkeypatch.setattr(attn_backend_utils.envs, "VLLM_KV_CACHE_LAYOUT", None)


def test_narrow_supported_set_wins_over_peers_in_either_order():
    for backends in (
        [_OnlyLBHNC, TritonAttentionBackend],
        [TritonAttentionBackend, _OnlyLBHNC],
    ):
        assert resolve_kv_cache_layout(backends) is KVCacheLayout.LBHNC


def test_peer_that_cannot_read_the_narrowed_layout_is_rejected():
    # TRITON_ATTN does not support LHBNC; resolving late must not hide that.
    for backends in (
        [_OnlyLHBNC, TritonAttentionBackend],
        [TritonAttentionBackend, _OnlyLHBNC],
    ):
        with pytest.raises(ValueError, match="No KV cache layout satisfies"):
            resolve_kv_cache_layout(backends)


def test_disjoint_supported_sets_are_rejected():
    for backends in (
        [_OnlyLHBNC, _OnlyLBHNC],
        [_OnlyLBHNC, _OnlyLHBNC],
    ):
        with pytest.raises(ValueError, match="No KV cache layout satisfies"):
            resolve_kv_cache_layout(backends)


def test_block_outer_declaration_narrows_the_default():
    # DeepSeek-style overlay models land block-outermost with no env var.
    for backends in (
        [_BlockOuter, TritonAttentionBackend],
        [TritonAttentionBackend, _BlockOuter],
    ):
        assert resolve_kv_cache_layout(backends) is KVCacheLayout.BLHNC


def test_explicit_user_layout_conflicting_with_supported_set_fails(monkeypatch):
    monkeypatch.setattr(attn_backend_utils.envs, "VLLM_KV_CACHE_LAYOUT", "LBNHC")
    with pytest.raises(ValueError, match="VLLM_KV_CACHE_LAYOUT=LBNHC"):
        resolve_kv_cache_layout([_OnlyLBHNC, TritonAttentionBackend])


def test_explicit_user_layout_in_supported_set_is_honored(monkeypatch):
    monkeypatch.setattr(attn_backend_utils.envs, "VLLM_KV_CACHE_LAYOUT", "BLNHC")
    layout = resolve_kv_cache_layout([_BlockOuter, TritonAttentionBackend])
    assert layout is KVCacheLayout.BLNHC


def test_ssm_backends_do_not_veto_the_layout():
    # Mamba pages are single-head and single-state, so SSM backends leave
    # supported_kv_cache_layouts() at the head-inside-block default.
    assert (
        KVCacheLayout.LHBNC not in Mamba2AttentionBackend.supported_kv_cache_layouts()
    )
    layout = resolve_kv_cache_layout([_OnlyLHBNC, Mamba2AttentionBackend])
    assert layout is KVCacheLayout.LHBNC
