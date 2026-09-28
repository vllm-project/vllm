# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing/selection tests for the AITER ASM cprr DCP-verify route.

Companion to the upstream segmented-verify tests. Everything here is pure CPU:
it pins the *decisions* (which route a KV group takes, which head count the
kernel runs at, which configurations are refused) rather than kernel numerics.
The numerics parity test against the segmented route needs a GPU and both
routes built; it is marked and skipped when unavailable.
"""

from __future__ import annotations

import pytest

rocm_aiter_mla = pytest.importorskip(
    "vllm.v1.attention.backends.mla.rocm_aiter_mla",
    reason="ROCm AITER MLA backend not available",
)

_heads = rocm_aiter_mla._asm_dcp_verify_heads
_configured = rocm_aiter_mla._asm_dcp_verify_configured
_select_route = rocm_aiter_mla._select_dcp_decode_route
Route = rocm_aiter_mla._DCPDecodeRoute
NATIVE = rocm_aiter_mla._NATIVE_CPRR_HEADS
MIN_QLEN = rocm_aiter_mla._MIN_CPRR_QLEN


def test_dcp_verify_env_defaults_to_segmented(monkeypatch):
    monkeypatch.delenv("VLLM_ROCM_AITER_MLA_DCP_VERIFY", raising=False)
    assert rocm_aiter_mla.envs.VLLM_ROCM_AITER_MLA_DCP_VERIFY == "segmented"


def test_dcp_verify_env_rejects_unknown_route(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_AITER_MLA_DCP_VERIFY", "unknown")
    with pytest.raises(ValueError, match="Invalid value"):
        _ = rocm_aiter_mla.envs.VLLM_ROCM_AITER_MLA_DCP_VERIFY


# --------------------------------------------------------------------------
# _asm_dcp_verify_heads -- which head count the cprr kernel runs at
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", NATIVE)
def test_native_head_counts_are_used_as_is(n):
    assert _heads(n) == n


def test_non_native_head_count_pads_up_to_the_next_native():
    """K3 at TP8/DCP8 gathers 96 heads for the target; the kernel has no 96
    variant, so it must run at 128, not silently fall out of the asm path."""
    assert _heads(96) == 128
    assert _heads(1) == 16
    assert _heads(17) == 32
    assert _heads(65) == 128


def test_head_count_above_the_largest_native_is_unservable():
    """0 means 'cannot serve'; anything else would route a shape the kernel
    has no build for."""
    assert _heads(max(NATIVE) + 1) == 0
    assert _heads(256) == 0


# --------------------------------------------------------------------------
# _asm_dcp_verify_configured -- reachability
# --------------------------------------------------------------------------


@pytest.fixture
def on_gfx950(monkeypatch):
    """The cprr kernels are built only for gfx950, so the gate checks the arch.
    Patched rather than detected, so both branches run on any runner."""
    import vllm.platforms.rocm as rocm

    monkeypatch.setattr(rocm, "on_gfx950", lambda: True)


def test_route_needs_dcp(on_gfx950):
    assert _configured(dcp_world_size=1, cp_interleave=1, multi_token_decode=True) is (
        False
    )
    assert _configured(dcp_world_size=8, cp_interleave=1, multi_token_decode=True) is (
        True
    )


def test_round_robin_interleave_other_than_one_is_excluded(on_gfx950):
    """The kernel's global-position causal window assumes interleave 1, the
    same restriction the segmented gate carries."""
    assert _configured(dcp_world_size=8, cp_interleave=2, multi_token_decode=True) is (
        False
    )
    assert _configured(dcp_world_size=8, cp_interleave=16, multi_token_decode=True) is (
        False
    )


def test_route_needs_multi_token_decode(on_gfx950):
    """Single-token decode is already served. Enabling the route for it would
    pad the head count on every step and buy nothing."""
    assert _configured(dcp_world_size=8, cp_interleave=1, multi_token_decode=False) is (
        False
    )


def test_route_needs_gfx950(monkeypatch):
    """hsa/gfx942/mla and hsa/gfx1250/mla carry no cprr rows, so off gfx950 the
    kernel lookup fails at the first verify step rather than falling back."""
    import vllm.platforms.rocm as rocm

    monkeypatch.setattr(rocm, "on_gfx950", lambda: False)
    assert _configured(dcp_world_size=8, cp_interleave=1, multi_token_decode=True) is (
        False
    )


# --------------------------------------------------------------------------
# qlen floor
# --------------------------------------------------------------------------


def test_min_cprr_qlen_is_above_two():
    """Qlen 2 is below CPRR's floor and must use segmented verification."""
    assert MIN_QLEN > 2


@pytest.mark.parametrize(
    "supports,causal,qlen,asm,expected",
    [
        (True, True, 1, True, Route.PLAIN),
        (True, True, 2, True, Route.SEGMENTED),
        (True, True, 3, True, Route.CPRR),
        (True, True, 4, True, Route.CPRR),
        (True, True, 4, False, Route.SEGMENTED),
        (True, False, 2, True, Route.PLAIN),
        (True, False, 4, False, Route.PLAIN),
        (False, True, 3, True, Route.CPRR),
    ],
)
def test_dcp_decode_route(supports, causal, qlen, asm, expected):
    assert _select_route(supports, causal, qlen, asm) is expected


def test_causal_multi_token_batch_without_a_valid_route_fails():
    with pytest.raises(RuntimeError, match="requires either segmented MLA"):
        _select_route(False, True, 2, True)


# Kernel numerics live in test_rocm_aiter_mla_dcp_cprr_numerics.py, which runs
# the real builder and the real asm kernel on a gfx950 GPU.
