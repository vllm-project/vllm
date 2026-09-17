# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing/selection tests for the AITER ASM cprr DCP-verify route.

Companion to the upstream segmented-verify tests. Everything here is pure CPU:
it pins the *decisions* (which route a KV group takes, which head count the
kernel runs at, which configurations are refused) rather than kernel numerics.
The numerics parity test against the segmented route needs a GPU and both
routes built; it is marked and skipped when unavailable.

Destined for tests/v1/attention/test_rocm_aiter_mla_dcp_cprr.py alongside
test_rocm_aiter_mla_fp8_decode_routing.py.
"""

from __future__ import annotations

import pytest

rocm_aiter_mla = pytest.importorskip(
    "vllm.v1.attention.backends.mla.rocm_aiter_mla",
    reason="ROCm AITER MLA backend not available",
)

_parse = rocm_aiter_mla._parse_dcp_verify_env
_heads = rocm_aiter_mla._asm_dcp_verify_heads
_selected = rocm_aiter_mla._asm_dcp_verify_selected
_configured = rocm_aiter_mla._asm_dcp_verify_configured
NATIVE = rocm_aiter_mla._NATIVE_CPRR_HEADS
MIN_QLEN = rocm_aiter_mla._MIN_CPRR_QLEN

ENV = "VLLM_ROCM_AITER_MLA_DCP_VERIFY"


# --------------------------------------------------------------------------
# _parse_dcp_verify_env
# --------------------------------------------------------------------------


def test_default_route_is_asm(monkeypatch):
    """Unset must mean asm: upstream's only DCP-verify route is Triton
    segmented, and under DSpark every step has qlen > 1, so an accidental
    default of 'segmented' silently moves ALL decode onto Triton."""
    monkeypatch.delenv(ENV, raising=False)
    assert _parse() == ("asm", frozenset())


@pytest.mark.parametrize(
    "raw,route,heads",
    [
        ("asm", "asm", frozenset()),
        ("segmented", "segmented", frozenset()),
        ("ASM", "asm", frozenset()),  # case-insensitive
        ("  segmented  ", "segmented", frozenset()),  # surrounding space
        ("segmented:64", "segmented", frozenset({64})),
        ("asm:64,128", "asm", frozenset({64, 128})),
        ("segmented:16,32,64,128", "segmented", frozenset({16, 32, 64, 128})),
    ],
)
def test_parse_accepted_forms(monkeypatch, raw, route, heads):
    monkeypatch.setenv(ENV, raw)
    assert _parse() == (route, heads)


@pytest.mark.parametrize("raw", ["segmented:", "asm:"])
def test_bare_trailing_colon_is_rejected(monkeypatch, raw):
    """A trailing ':' read as 'no filter' would flip EVERY group instead of
    the one that was meant -- the opposite of the intent, silently."""
    monkeypatch.setenv(ENV, raw)
    with pytest.raises(ValueError, match="ends in ':'"):
        _parse()


@pytest.mark.parametrize("raw", ["segmented:,", "asm:64,", "segmented:64,,128"])
def test_empty_head_list_entry_is_rejected(monkeypatch, raw):
    """Dropping empty entries silently is the same failure as a trailing ':'.
    'segmented:,' leaves an EMPTY filter, which applies the route to every
    group: the opposite of what was asked."""
    monkeypatch.setenv(ENV, raw)
    with pytest.raises(ValueError, match="empty entry"):
        _parse()


@pytest.mark.parametrize("raw", ["asm:0", "segmented:-64", "asm:64,0"])
def test_non_positive_head_count_is_rejected(monkeypatch, raw):
    """A count that can never match routes every group to the other path,
    inverting the request rather than failing."""
    monkeypatch.setenv(ENV, raw)
    with pytest.raises(ValueError, match="must be positive"):
        _parse()


@pytest.mark.parametrize("raw", ["triton", "", "asm segmented", "seg"])
def test_unknown_route_is_rejected(monkeypatch, raw):
    monkeypatch.setenv(ENV, raw)
    with pytest.raises(ValueError):
        _parse()


@pytest.mark.parametrize("raw", ["segmented:abc", "asm:64,x", "asm:64.5"])
def test_non_integer_head_list_is_rejected(monkeypatch, raw):
    monkeypatch.setenv(ENV, raw)
    with pytest.raises(ValueError, match="comma-separated list"):
        _parse()


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
# _asm_dcp_verify_selected -- per-KV-group routing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("heads", [64, 96, 128])
def test_no_filter_applies_the_route_to_every_group(monkeypatch, heads):
    monkeypatch.setenv(ENV, "asm")
    assert _selected(heads) is True
    monkeypatch.setenv(ENV, "segmented")
    assert _selected(heads) is False


def test_head_filter_splits_target_and_draft(monkeypatch):
    """The reason the per-group form exists: at TP8/DCP8 a DSpark target
    gathers 96 heads and its draft 64, so 'segmented:64' must mean
    'draft on segmented, target still on asm' -- the bisection knob."""
    monkeypatch.setenv(ENV, "segmented:64")
    assert _selected(64) is False  # draft -> segmented
    assert _selected(96) is True  # target -> asm


def test_head_filter_inverts_with_the_route(monkeypatch):
    """'asm:64' is the mirror image: the listed group goes asm, the rest
    segmented."""
    monkeypatch.setenv(ENV, "asm:64")
    assert _selected(64) is True
    assert _selected(96) is False


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
    """Qlen 2 is the steady state at num_speculative_tokens=1. The per-step
    gate omits the global-position window below _MIN_CPRR_QLEN, which is
    correct for qlen 1 (a decode row sees every local token) but WRONG for
    qlen 2, where a row can still be causally truncated. The builder must
    refuse it at boot rather than degrade acceptance silently."""
    assert MIN_QLEN > 2


# Kernel numerics live in test_rocm_aiter_mla_dcp_cprr_numerics.py, which runs
# the real builder and the real asm kernel on a gfx950 GPU.
