# SPDX-License-Identifier: Apache-2.0
"""Unit tests for P103 — FLA Cliff 2 chunked fwd_h+fwd_o orchestrator.

CPU-only smoke tests — verify dispatcher metadata, wiring import, env-gate
behaviour without actually invoking the GPU kernels. Numerical correctness
test (which requires GPU + triton) is in a separate gpu_test_p103.py.
"""
from __future__ import annotations

import unittest.mock as mock



def test_p103_in_dispatcher():
    """P103 must be registered in PATCH_REGISTRY with the expected schema."""
    from vllm._genesis.dispatcher import PATCH_REGISTRY
    assert "P103" in PATCH_REGISTRY
    meta = PATCH_REGISTRY["P103"]
    assert meta["env_flag"] == "GENESIS_ENABLE_P103"
    assert meta["default_on"] is False
    assert meta["category"] == "memory_hotfix"
    assert "Cliff 2" in meta["title"]
    assert "fwd_h" in meta["credit"] or "fwd_o" in meta["credit"]


def test_p103_wiring_module_imports():
    """The wiring module must import cleanly (no syntax errors)."""
    from vllm._genesis.wiring.hybrid import patch_103_fla_cliff2_chunked as p103
    assert hasattr(p103, "apply")
    assert hasattr(p103, "is_applied")
    assert hasattr(p103, "should_apply")


def test_p103_apply_register_in_apply_all():
    """P103 must have a wrapper function registered via @register_patch."""
    from vllm._genesis.patches import apply_all
    assert hasattr(apply_all, "apply_patch_103_fla_cliff2_chunked")


def test_p103_should_apply_off_by_default(monkeypatch):
    """Without GENESIS_ENABLE_P103=1, should_apply() must return False."""
    from vllm._genesis.wiring.hybrid import patch_103_fla_cliff2_chunked as p103
    monkeypatch.delenv("GENESIS_ENABLE_P103", raising=False)
    assert p103.should_apply() is False


def test_p103_should_apply_recognizes_truthy_env(monkeypatch):
    """should_apply() must accept all truthy env values."""
    from vllm._genesis.wiring.hybrid import patch_103_fla_cliff2_chunked as p103
    # Mock platform checks since this test runs CPU-only
    with mock.patch.object(p103, "is_nvidia_cuda", return_value=True), \
         mock.patch.object(p103, "is_sm_at_least", return_value=True):
        for v in ("1", "true", "yes", "on", "True", "YES"):
            monkeypatch.setenv("GENESIS_ENABLE_P103", v)
            assert p103.should_apply() is True, f"{v!r} should activate P103"
        for v in ("0", "", "off", "no", "False"):
            monkeypatch.setenv("GENESIS_ENABLE_P103", v)
            assert p103.should_apply() is False, f"{v!r} should NOT activate P103"


def test_p103_apply_fails_soft_when_module_missing(monkeypatch):
    """If FLA module is unavailable, apply() must return ('skipped', ...)
    not raise."""
    from vllm._genesis.wiring.hybrid import patch_103_fla_cliff2_chunked as p103
    monkeypatch.setenv("GENESIS_ENABLE_P103", "1")
    with mock.patch.object(p103, "is_nvidia_cuda", return_value=True), \
         mock.patch.object(p103, "is_sm_at_least", return_value=True):
        # Simulate FLA missing
        import importlib
        original_import = importlib.import_module

        def _fail_for_chunk_module(name, *a, **kw):
            if name == p103._TARGET_MODULE:
                raise ImportError("simulated: chunk module not available")
            return original_import(name, *a, **kw)

        with mock.patch.object(importlib, "import_module",
                               side_effect=_fail_for_chunk_module):
            status, reason = p103.apply()
            assert status == "skipped"
            assert "FLA module" in reason or "not available" in reason


def test_p103_marker_attr_consistent():
    """The wrapper marker attribute name must match between apply and is_applied."""
    from vllm._genesis.wiring.hybrid import patch_103_fla_cliff2_chunked as p103
    assert p103._GENESIS_P103_MARKER_ATTR == "_genesis_p103_chunked_wrap"


def test_p103_max_t_env_default():
    """MAX_T defaults to 16384 when env unset, rounded down to FLA_CHUNK_SIZE multiple."""
    # We can't easily test the actual wrapper without FLA loaded, but we
    # can verify the default value is in the code (defensive sanity).
    import inspect
    from vllm._genesis.wiring.hybrid import patch_103_fla_cliff2_chunked as p103
    src = inspect.getsource(p103._make_chunked_wrapper)
    assert '"16384"' in src
    assert "GENESIS_FLA_FWD_H_MAX_T" in src
    # rounding to FLA_CHUNK_SIZE multiple
    assert "_MAX_T // fla_chunk_size" in src or "// fla_chunk_size) * fla_chunk_size" in src


def test_p103_kda_path_not_covered_documented():
    """The patch deliberately doesn't cover kda.py path; this should be
    documented in the wiring module docstring."""
    from vllm._genesis.wiring.hybrid import patch_103_fla_cliff2_chunked as p103
    assert "KDA" in p103.__doc__ or "kda" in p103.__doc__.lower()


# ─────────────────────────────────────────────────────────────────
# v7.69 — self-install at module-import time (club-3090#19 finding 2)
# ─────────────────────────────────────────────────────────────────


def test_p103_self_install_helper_exists():
    """v7.69: `_genesis_p103_install_at_import` must be exposed at module
    top so the text-patched chunk.py can `from ... import` it."""
    from vllm._genesis.wiring.hybrid import patch_103_fla_cliff2_chunked as p103
    assert hasattr(p103, "_genesis_p103_install_at_import"), (
        "v7.69 helper missing — text-patched chunk.py won't be able to "
        "import the install function"
    )


def test_p103_self_install_helper_signature():
    """Helper must accept a single dict-like (module globals)."""
    import inspect
    from vllm._genesis.wiring.hybrid import patch_103_fla_cliff2_chunked as p103
    sig = inspect.signature(p103._genesis_p103_install_at_import)
    params = list(sig.parameters.values())
    assert len(params) == 1, (
        f"helper must take exactly one arg (module globals), got "
        f"{len(params)}: {[p.name for p in params]}"
    )


def test_p103_self_install_returns_false_when_env_off(monkeypatch):
    """env-off path: helper must short-circuit cleanly (no side effects)."""
    monkeypatch.delenv("GENESIS_ENABLE_P103", raising=False)
    from vllm._genesis.wiring.hybrid import patch_103_fla_cliff2_chunked as p103

    fake_globals = {"chunk_gated_delta_rule_fwd": lambda *a, **kw: None}
    result = p103._genesis_p103_install_at_import(fake_globals)
    assert result is False
    # globals must be untouched
    assert fake_globals["chunk_gated_delta_rule_fwd"].__name__ != (
        "chunk_gated_delta_rule_fwd"
    ) or not hasattr(
        fake_globals["chunk_gated_delta_rule_fwd"],
        "_genesis_p103_chunked_wrap",
    )


def test_p103_self_install_no_op_if_already_wrapped(monkeypatch):
    """Idempotency: helper called twice on same module dict returns
    True both times, second call is no-op."""
    monkeypatch.setenv("GENESIS_ENABLE_P103", "1")
    from vllm._genesis.wiring.hybrid import patch_103_fla_cliff2_chunked as p103

    # Pre-mark the function as already wrapped
    def fake_fn():
        pass
    fake_fn._genesis_p103_chunked_wrap = True
    fake_globals = {"chunk_gated_delta_rule_fwd": fake_fn}
    result = p103._genesis_p103_install_at_import(fake_globals)
    assert result is True
    # Same wrapper still in place
    assert fake_globals["chunk_gated_delta_rule_fwd"] is fake_fn


def test_p103_self_install_returns_false_on_missing_deps(monkeypatch):
    """Soft failure: missing closure dep must NOT raise — return False."""
    monkeypatch.setenv("GENESIS_ENABLE_P103", "1")
    from vllm._genesis.wiring.hybrid import patch_103_fla_cliff2_chunked as p103

    # Provide chunk_gated_delta_rule_fwd but NO closure deps — helper
    # should fall through softly.
    fake_globals = {"chunk_gated_delta_rule_fwd": lambda *a, **kw: None}
    result = p103._genesis_p103_install_at_import(fake_globals)
    assert result is False, (
        "missing closure deps must produce False, not raise — chunk.py "
        "import must always succeed"
    )


def test_p103_self_install_succeeds_with_mock_chunk_globals(monkeypatch):
    """Full happy path: helper installs wrapper into a synthetic chunk.py
    globals dict that has all expected symbols."""
    monkeypatch.setenv("GENESIS_ENABLE_P103", "1")
    from vllm._genesis.wiring.hybrid import patch_103_fla_cliff2_chunked as p103

    def orig_fwd(q, k, v, g, beta, scale, initial_state, output_final_state,
                 cu_seqlens=None, chunk_indices=None, chunk_offsets=None):
        return None
    fake_globals = {
        "chunk_gated_delta_rule_fwd": orig_fwd,
        "chunk_local_cumsum": lambda **kw: None,
        "chunk_scaled_dot_kkt_fwd": lambda **kw: None,
        "solve_tril": lambda **kw: None,
        "recompute_w_u_fwd": lambda **kw: None,
        "chunk_gated_delta_rule_fwd_h": lambda **kw: None,
        "chunk_fwd_o": lambda **kw: None,
        "FLA_CHUNK_SIZE": 64,
        "SUPPRESS_LEVEL": 0,
    }
    result = p103._genesis_p103_install_at_import(fake_globals)
    assert result is True

    new_fn = fake_globals["chunk_gated_delta_rule_fwd"]
    assert new_fn is not orig_fwd
    assert getattr(new_fn, "_genesis_p103_chunked_wrap", False) is True


def test_p103_text_patch_block_includes_env_check_and_helper_call():
    """The text-patch block appended to chunk.py must contain BOTH the
    env-flag check AND the call to _genesis_p103_install_at_import.

    Regression guard: if someone refactors the block string and forgets
    the env check, P103 would always fire even without opt-in.
    """
    from vllm._genesis.wiring.hybrid import patch_103_fla_cliff2_chunked as p103

    block = p103._P103_SELF_INSTALL_BLOCK
    assert "GENESIS_ENABLE_P103" in block, "env-flag check missing"
    assert "_genesis_p103_install_at_import" in block, "helper import missing"
    assert "globals()" in block, (
        "must call install with globals() — needs the chunk.py module dict"
    )
    assert "try:" in block and "except" in block, (
        "must wrap in try/except so chunk.py import survives any failure"
    )


def test_p103_text_patch_anchor_matches_real_chunk_py_pattern():
    """Anchor must match the EXACT end-of-file pattern of vllm's chunk.py
    (chunk_gated_delta_rule's final `return o, final_state` block).
    """
    from vllm._genesis.wiring.hybrid import patch_103_fla_cliff2_chunked as p103

    anchor = p103._P103_SELF_INSTALL_ANCHOR
    # Anchor must end with the high-level function's return
    assert "return o, final_state" in anchor
    # And include the autograd Function.apply call (so we don't anchor
    # on a generic `return o, final_state` elsewhere in the file)
    assert "ChunkGatedDeltaRuleFunction.apply" in anchor


def test_p103_self_install_text_patcher_builds_with_specific_drift_marker():
    """Drift markers on the chunk.py text-patch must NOT include generic
    prefixes that could collide with other patches in the same file."""
    import os
    import tempfile

    from vllm._genesis.wiring.hybrid import patch_103_fla_cliff2_chunked as p103
    import vllm._genesis.guards as guards

    with tempfile.TemporaryDirectory() as td:
        ops_dir = os.path.join(
            td, "model_executor", "layers", "fla", "ops"
        )
        os.makedirs(ops_dir)
        with open(os.path.join(ops_dir, "chunk.py"), "w") as f:
            f.write("# placeholder\n")

        orig = guards.vllm_install_root
        guards.vllm_install_root = lambda: td
        try:
            patcher = p103._make_self_install_text_patcher()
        finally:
            guards.vllm_install_root = orig

    assert patcher is not None

    # Drift markers must be specific (no generic '[Genesis P103' that
    # would collide with future patches' insertions in chunk.py).
    for m in patcher.upstream_drift_markers:
        # Must include version/feature qualifier
        assert "v7.69" in m or "self-install" in m, (
            f"drift marker {m!r} too generic — risk of collision with "
            f"sibling patches"
        )


def test_p103_apply_attempts_text_patch_first(monkeypatch):
    """apply() in v7.69 must call _make_self_install_text_patcher BEFORE
    the legacy setattr step — text-patch is the durable mechanism that
    survives `exec vllm serve` + worker spawn."""
    import inspect
    from vllm._genesis.wiring.hybrid import patch_103_fla_cliff2_chunked as p103

    src = inspect.getsource(p103.apply)
    # text-patch step must appear before the importlib.import_module call
    text_patch_pos = src.find("_make_self_install_text_patcher")
    setattr_pos = src.find("setattr(chunk_mod, _FN_NAME, wrapper)")
    assert text_patch_pos > 0, "text-patch step missing"
    assert setattr_pos > 0, "setattr step missing"
    assert text_patch_pos < setattr_pos, (
        "text-patch step must run BEFORE setattr step (text-patch is the "
        "durable mechanism; setattr is defense-in-depth for current process)"
    )


def test_p103_module_docstring_explains_v7_69_install_model():
    """v7.69 install-model rationale must be documented at module top."""
    from vllm._genesis.wiring.hybrid import patch_103_fla_cliff2_chunked as p103
    doc = p103.__doc__ or ""
    assert "exec vllm serve" in doc, (
        "module docstring must explain why v7.69 was needed (entrypoint "
        "exec pattern)"
    )
    assert "self-install" in doc.lower() or "module-import" in doc.lower()
