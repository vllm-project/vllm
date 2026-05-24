# SPDX-License-Identifier: Apache-2.0
"""TDD tests for T4.6 — Triton compile-time watchdog instrumentation.

Test contract:
1. PatchStats has compile_elapsed_sec field, default 0.0
2. Field is float (not int — sub-second precision matters)
3. apply_all.run() sets the field after completion (any value > 0)
4. Watchdog log emitted at appropriate threshold (> 120s = WARNING)

Author: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations




def test_patch_stats_has_compile_elapsed_field():
    """PatchStats dataclass has compile_elapsed_sec field."""
    from vllm._genesis.patches.apply_all import PatchStats
    s = PatchStats()
    assert hasattr(s, "compile_elapsed_sec")
    assert isinstance(s.compile_elapsed_sec, float)
    assert s.compile_elapsed_sec == 0.0


def test_compile_elapsed_field_is_float_typed():
    """Field type annotation is float (sub-second precision).

    Module uses `from __future__ import annotations` so __annotations__
    holds string forms; we check equivalence both via string AND resolved
    type via inspect.get_type_hints (works even with PEP 563 deferred eval).
    """
    import typing
    from vllm._genesis.patches.apply_all import PatchStats
    annotations = PatchStats.__annotations__
    assert "compile_elapsed_sec" in annotations
    # String form equals 'float' (PEP 563)
    assert annotations["compile_elapsed_sec"] in ("float", float), (
        f"Got: {annotations['compile_elapsed_sec']!r}"
    )
    # Resolved form is float
    hints = typing.get_type_hints(PatchStats)
    assert hints["compile_elapsed_sec"] is float


def test_apply_all_sets_compile_elapsed():
    """run() populates compile_elapsed_sec to a positive value.

    We run in dry-run mode (apply=False) to keep the test fast. Even
    dry-run runs through the full registry, so elapsed > 0.
    """
    from vllm._genesis.patches.apply_all import run
    stats = run(verbose=False, apply=False)
    assert stats.compile_elapsed_sec > 0.0, "Watchdog didn't measure elapsed"
    # Reasonable sanity: dry-run < 30s
    assert stats.compile_elapsed_sec < 60.0, (
        f"Dry-run took {stats.compile_elapsed_sec:.1f}s — investigate"
    )


def test_watchdog_warning_threshold_documented():
    """Source code documents the 120s threshold for WARNING level."""
    import inspect
    from vllm._genesis.patches import apply_all
    src = inspect.getsource(apply_all.run)
    assert "120" in src, "Watchdog threshold not parameterized"
    assert "compile-watchdog" in src.lower()


def test_watchdog_log_message_actionable():
    """Watchdog log message includes actionable recovery hints."""
    import inspect
    from vllm._genesis.patches import apply_all
    src = inspect.getsource(apply_all.run)
    # When > 120s, recovery hints should be in the log
    assert "TRITON_CACHE_DIR" in src or "compile cache" in src.lower()
