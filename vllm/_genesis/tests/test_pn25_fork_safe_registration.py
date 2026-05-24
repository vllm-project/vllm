# SPDX-License-Identifier: Apache-2.0
"""TDD tests for issue #16 — PN25 fork-safe op registration.

Bug context (noonghunna, 2026-05-01):
- vLLM workers spawn via VLLM_WORKER_MULTIPROC_METHOD=spawn (fresh interpreter)
- Each worker re-imports silu_and_mul_customop.py → module-level
  `_op_registered = False`
- First FFN forward in worker calls _register_op_once()
- Local flag False → @custom_op decoration runs again
- That triggers torch.library.infer_schema() inside Dynamo trace
- Dynamo refuses to trace infer_schema → engine crash

Fix: check torch.ops.genesis.silu_and_mul_pooled registry FIRST. If
op already globally registered (C++ state survives spawn), sync local
flag and skip re-decoration entirely.

This test simulates the worker-spawn condition by:
1. Calling _register_op_once() (parent — registers globally)
2. Resetting module-level _op_registered to False (mimics spawn)
3. Calling _register_op_once() again — must NOT trigger infer_schema

Author: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa.
Bug: https://github.com/Sandermage/genesis-vllm-patches/issues/16
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch



def test_register_op_once_idempotent_in_same_process():
    """Two consecutive calls in same process: second is fast-path."""
    from vllm._genesis.kernels import silu_and_mul_customop as mod
    # Reset state for clean test
    mod._op_registered = False
    # First call may register (or fail if torch < 2.4 / no CUDA)
    result1 = mod._register_op_once()
    # Second call: must return same result without re-decorating
    result2 = mod._register_op_once()
    assert result1 == result2


def test_post_spawn_state_check_does_not_redecorate():
    """Simulate worker spawn: reset _op_registered to False, ensure
    second registration call doesn't invoke @custom_op decorator
    when the op is already globally registered.

    This is the EXACT bug from issue #16 — the fix is to short-circuit
    on global registry presence BEFORE attempting decoration.
    """
    from vllm._genesis.kernels import silu_and_mul_customop as mod
    import torch

    # Pre-condition: simulate "op was registered by parent process".
    # We mock torch.ops.genesis.silu_and_mul_pooled as existing.
    fake_genesis = MagicMock()
    fake_genesis.silu_and_mul_pooled = MagicMock()

    # Reset local flag (simulates fresh worker interpreter)
    mod._op_registered = False

    # Patch torch.ops.genesis to appear registered
    with patch.object(torch.ops, "genesis", fake_genesis, create=True):
        # Mock torch.library.custom_op so we can detect if it's called
        # (which would mean our short-circuit failed)
        with patch("torch.library.custom_op") as mock_custom_op:
            result = mod._register_op_once()

            # Must return True (op IS registered globally)
            assert result is True
            # Must NOT have called @custom_op (avoiding infer_schema)
            assert not mock_custom_op.called, (
                "Bug #16: _register_op_once called @custom_op even though "
                "op was already globally registered — would trigger "
                "infer_schema and crash worker"
            )
            # Local flag synced
            assert mod._op_registered is True


def test_handles_missing_genesis_namespace_gracefully():
    """When torch.ops.genesis doesn't exist (cold import), fall through
    to registration path normally — no crash from AttributeError.
    """
    from vllm._genesis.kernels import silu_and_mul_customop as mod
    import torch

    # Reset
    mod._op_registered = False

    # Remove torch.ops.genesis attribute if present
    original_genesis = getattr(torch.ops, "genesis", None)
    try:
        if hasattr(torch.ops, "genesis"):
            delattr(torch.ops, "genesis")
    except (AttributeError, RuntimeError):
        pass

    try:
        # Should not crash; will either register normally or fail gracefully
        result = mod._register_op_once()
        assert isinstance(result, bool)
    finally:
        # Restore original state if we removed it
        if original_genesis is not None:
            try:
                setattr(torch.ops, "genesis", original_genesis)
            except (AttributeError, RuntimeError):
                pass


def test_except_clause_handles_attribute_error():
    """Source has try/except guarding the torch.ops.genesis check.

    Verified by source inspection — running a real RuntimeError mock
    is hard because torch.library internals also use hasattr() so we
    can't safely globally patch it. Instead verify the source guard
    is present.
    """
    import inspect
    from vllm._genesis.kernels import silu_and_mul_customop as mod
    src = inspect.getsource(mod._register_op_once)
    # try/except around the genesis registry probe
    assert "try:" in src
    assert "except (AttributeError, RuntimeError)" in src


def test_fix_documented_in_source():
    """Fix is documented in source with #16 issue reference."""
    import inspect
    from vllm._genesis.kernels import silu_and_mul_customop as mod
    src = inspect.getsource(mod._register_op_once)
    assert "#16" in src, "Issue #16 reference missing from fix"
    assert "infer_schema" in src, (
        "Bug mechanism (infer_schema Dynamo trace) not documented"
    )
    # Fork-safe / spawn-safe semantics documented
    assert "spawn" in src.lower() or "fork" in src.lower()


def test_global_check_runs_before_custom_op_call():
    """The torch.ops.genesis check must appear BEFORE the
    @custom_op decoration in source order. If the order is wrong,
    the fix doesn't help.
    """
    import inspect
    from vllm._genesis.kernels import silu_and_mul_customop as mod
    src = inspect.getsource(mod._register_op_once)
    # Find positions of the genesis check and the custom_op call
    genesis_check_pos = src.find("torch.ops.genesis")
    custom_op_call_pos = src.find("custom_op(_OP_QUALNAME")
    if custom_op_call_pos == -1:
        # Variant: ".register_op_once" or other forms — find @custom_op decorator
        custom_op_call_pos = src.find("@custom_op(")
    assert genesis_check_pos > 0, "Genesis registry check missing"
    assert custom_op_call_pos > 0, "custom_op decoration missing"
    assert genesis_check_pos < custom_op_call_pos, (
        "BUG: genesis registry check must run BEFORE @custom_op call"
    )


# ─────────────────────────────────────────────────────────────────
# Sister-kernel coverage: gdn_dual_stream_customop.py (P7b)
# Same bug class as PN25 silu_and_mul_pooled. Preventive fix applied
# 2026-05-02 to avoid identical crash on 1×3090 spawn configs.
# ─────────────────────────────────────────────────────────────────


def test_p7b_dual_stream_has_global_registry_check():
    """P7b's _register_op_once also has the spawn-safety guard."""
    import inspect
    from vllm._genesis.kernels import gdn_dual_stream_customop as mod
    src = inspect.getsource(mod._register_op_once)
    assert "torch.ops.genesis" in src
    assert "dual_linear_parallel" in src
    assert "#16" in src or "spawn" in src.lower()


def test_p7b_post_spawn_state_check_does_not_redecorate():
    """P7b: same spawn-safety test as PN25."""
    from vllm._genesis.kernels import gdn_dual_stream_customop as mod
    import torch

    fake_genesis = MagicMock()
    fake_genesis.dual_linear_parallel = MagicMock()

    mod._op_registered = False

    with patch.object(torch.ops, "genesis", fake_genesis, create=True):
        with patch("torch.library.custom_op") as mock_custom_op:
            result = mod._register_op_once()
            assert result is True
            assert not mock_custom_op.called, (
                "P7b same bug as #16: re-decoration triggers infer_schema"
            )
            assert mod._op_registered is True


def test_p7b_global_check_runs_before_custom_op_call():
    """P7b: source order — global check before @custom_op."""
    import inspect
    from vllm._genesis.kernels import gdn_dual_stream_customop as mod
    src = inspect.getsource(mod._register_op_once)
    genesis_check_pos = src.find("torch.ops.genesis")
    custom_op_call_pos = src.find("custom_op(_OP_QUALNAME")
    if custom_op_call_pos == -1:
        custom_op_call_pos = src.find("@custom_op(")
    assert genesis_check_pos > 0
    assert custom_op_call_pos > 0
    assert genesis_check_pos < custom_op_call_pos
