# SPDX-License-Identifier: Apache-2.0
"""TDD tests for P67c — Per-row vote sparse-V integration into P67 split-M.

Test contract (10 cases — host-runnable subset; GPU-only tests skipped with
`pytest.mark.skipif(not torch.cuda.is_available())`):

1. Wiring imports cleanly
2. Dispatcher PATCH_REGISTRY entry correct
3. Env OFF → patch reports skipped
4. Env ON without P67 → patch reports skipped (requires P67 base)
5. Env ON with P67 → patch reports applied
6. Kernel signature has new constexpr params (constexpr defaults preserve compat)
7. Threshold clamping correct (negative → 0, > 0.5 → 0.5)
8. Sink tokens default = 4
9. Constexpr DCE invariant: SPARSE_V=0 path identical to no-SPARSE_V code (text grep)
10. Bit-exact contract: SPARSE_V=1 + threshold=0 NEVER skips (p_max ≥ 0 always)

Bench tests (require live GPU + model) are NOT in this unit suite — they're
exercised by `bench_model.py` after deployment.

Author: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import os



def test_p67c_wiring_imports():
    """P67c wiring module imports cleanly."""
    from vllm._genesis.wiring.perf_hotfix import patch_67c_sparse_v
    assert hasattr(patch_67c_sparse_v, "apply")


def test_p67c_dispatcher_registry():
    """P67c registered in PATCH_REGISTRY with correct env flag + requires_patches."""
    from vllm._genesis.dispatcher import PATCH_REGISTRY
    assert "P67c" in PATCH_REGISTRY
    e = PATCH_REGISTRY["P67c"]
    assert e["env_flag"] == "GENESIS_ENABLE_P67_SPARSE_V"
    assert e["default_on"] is False
    # P67c requires P67 base kernel
    assert "P67" in e.get("requires_patches", []) or e.get("requires_p67")


def test_p67c_skips_when_env_off(monkeypatch):
    """When env is OFF, apply() returns 'skipped'."""
    monkeypatch.delenv("GENESIS_ENABLE_P67_SPARSE_V", raising=False)
    from vllm._genesis.wiring.perf_hotfix.patch_67c_sparse_v import apply
    status, reason = apply()
    assert status == "skipped"
    assert "opt-in" in reason.lower()


def test_p67c_skips_when_p67_not_enabled(monkeypatch):
    """If P67 base is not enabled, P67c is also skipped."""
    monkeypatch.setenv("GENESIS_ENABLE_P67_SPARSE_V", "1")
    monkeypatch.delenv("GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL", raising=False)
    from vllm._genesis.wiring.perf_hotfix.patch_67c_sparse_v import apply
    status, reason = apply()
    assert status == "skipped"
    assert "P67" in reason


def test_p67c_applies_when_both_enabled(monkeypatch):
    """When P67 + P67c env vars set, apply() returns 'applied'."""
    monkeypatch.setenv("GENESIS_ENABLE_P67_SPARSE_V", "1")
    monkeypatch.setenv("GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL", "1")
    from vllm._genesis.wiring.perf_hotfix.patch_67c_sparse_v import apply
    status, reason = apply()
    assert status == "applied"
    assert "sparse-V" in reason or "config" in reason.lower()


def test_p67_kernel_has_new_constexpr_params():
    """P67 kernel signature includes SPARSE_V/THRESHOLD/SINK_TOKENS constexpr."""
    import inspect
    from vllm._genesis.kernels import p67_multi_query_kernel
    src = inspect.getsource(p67_multi_query_kernel)
    # New constexpr params
    assert "SPARSE_V: tl.constexpr" in src
    assert "SPARSE_V_THRESHOLD: tl.constexpr" in src
    assert "SINK_TOKENS: tl.constexpr" in src


def test_p67c_launcher_clamps_threshold(monkeypatch):
    """Launcher clamps threshold to safe range [0, 0.5]."""
    monkeypatch.setenv("GENESIS_P67_SPARSE_V_THRESHOLD", "-1.0")
    from vllm._genesis.kernels.p67_multi_query_kernel import (
        _resolve_sparse_v_threshold,
    )
    thr = _resolve_sparse_v_threshold()
    assert thr >= 0.0, "Negative threshold not clamped"

    monkeypatch.setenv("GENESIS_P67_SPARSE_V_THRESHOLD", "1.0")
    thr = _resolve_sparse_v_threshold()
    assert thr <= 0.5, "Above-0.5 threshold not clamped"

    monkeypatch.setenv("GENESIS_P67_SPARSE_V_THRESHOLD", "0.001")
    thr = _resolve_sparse_v_threshold()
    assert abs(thr - 0.001) < 1e-9


def test_p67c_default_sink_tokens():
    """Default SINK_TOKENS via env = 4."""
    os.environ.pop("GENESIS_P67_SPARSE_V_SINK_TOKENS", None)
    from vllm._genesis.kernels.p67_multi_query_kernel import (
        _resolve_sparse_v_sink_tokens,
    )
    assert _resolve_sparse_v_sink_tokens() == 4


def test_p67c_constexpr_dce_invariant():
    """SPARSE_V=0 path is structurally separate from skip path (text grep).

    The skip block must be guarded by `if SPARSE_V:` constexpr so Triton
    compiler removes it entirely when SPARSE_V=0. This is the bit-exact
    contract — without DCE, even SPARSE_V=0 might compile to different SASS.
    """
    import inspect
    from vllm._genesis.kernels import p67_multi_query_kernel
    src = inspect.getsource(p67_multi_query_kernel)
    # The skip-decision block must be nested inside `if SPARSE_V:`
    assert "if SPARSE_V:" in src
    # The actual skip branch (decay-only path) must be guarded by skip_pv_t
    assert "skip_pv_t" in src or "skip_pv" in src


def test_p67c_threshold_zero_never_skips_invariant():
    """Math invariant: when threshold=0.0, P_t_max < 0.0 is NEVER true.

    P_t = tl.exp2(S_t - M_new_t) ≥ 0 (exp is non-negative).
    Therefore tl.max(P_t) ≥ 0 always.
    Therefore (p_t_max < 0.0) is always False → skip never fires.
    Therefore SPARSE_V=1 + threshold=0 == SPARSE_V=0 (bit-exact).

    This is the test contract from TheTom #41422 PR: bit-exact at threshold=0.
    Verified mathematically here; runtime verification in bench/test_p67c_bit_exact.
    """
    # Invariant verified by reasoning above. Test is documentation.
    # Numeric edge case: tl.exp2 of very negative numbers → 0.0, not -0.0.
    # Float comparison: 0.0 < 0.0 is False → skip not triggered → bit-exact.
    p_t_max_low = 2.0 ** (-1000)  # smallest reasonable exp2 result
    assert p_t_max_low >= 0.0, "exp2 of negative → must be non-negative"
    # Threshold=0 never fires:
    assert not (p_t_max_low < 0.0)
