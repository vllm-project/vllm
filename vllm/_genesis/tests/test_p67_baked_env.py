# SPDX-License-Identifier: Apache-2.0
"""TDD test for H2 fix — P67 should bake env reads at module load
instead of doing per-dispatch os.environ.get(...) in the hot path.

Quality audit (research agent aae2c26c, 2026-04-28) measured ~0.5% TPS
overhead from per-dispatch env reads on K=3 spec-decode (~800
dispatches/sec). Fix: snapshot env values at module load + emit them as
literal Python in the text-patch body.

Trade-off: GENESIS_P67_MAX_PRIOR_LEN and GENESIS_P67_DEBUG_COMPARE are
container-launch-time tunables now. Operators set them the same way
(env vars at start), the snapshot just happens earlier.
"""
from __future__ import annotations

import re


from vllm._genesis.wiring.spec_decode.patch_67_tq_multi_query_kernel import (
    P67_NEW,
    _BAKED_MAX_PRIOR,
    _BAKED_DEBUG_COMPARE,
)


def test_p67_emit_has_no_per_dispatch_env_reads():
    """H2 fix: hot-path emit must NOT contain os.environ.get('GENESIS_P67_*')
    patterns. Those would re-read env on every call (~800/sec).
    """
    forbidden = re.compile(
        r"environ\.get\(\s*[\"']GENESIS_P67_(MAX_PRIOR_LEN|DEBUG_COMPARE)[\"']"
    )
    assert not forbidden.search(P67_NEW), (
        "P67 emit contains per-dispatch env read for GENESIS_P67_*. "
        "These should be baked at module load (see _BAKED_* constants)."
    )


def test_p67_emit_has_baked_max_prior_literal():
    """H2 fix: emit should contain literal `_genesis_p67_max_prior = <int>`."""
    expected = f"_genesis_p67_max_prior = {_BAKED_MAX_PRIOR}"
    assert expected in P67_NEW, (
        f"Expected baked literal {expected!r} in P67_NEW emit"
    )


def test_p67_emit_has_baked_debug_compare_literal():
    """H2 fix: emit should contain literal `_debug_compare = True/False`."""
    expected = f"_debug_compare = {_BAKED_DEBUG_COMPARE}"
    assert expected in P67_NEW, (
        f"Expected baked literal {expected!r} in P67_NEW emit"
    )


def test_p67_baked_constants_are_typed_correctly():
    """Defense: the snapshot must be the right type for the emit substitution
    to produce valid Python."""
    assert isinstance(_BAKED_MAX_PRIOR, int), (
        "_BAKED_MAX_PRIOR must be int for f-string literal emit"
    )
    assert isinstance(_BAKED_DEBUG_COMPARE, bool), (
        "_BAKED_DEBUG_COMPARE must be bool for f-string literal emit"
    )
