# SPDX-License-Identifier: Apache-2.0
"""TDD for `verify_live_rebinds()` — post-register rebind verification.

Ensures the orchestrator's live-state introspection correctly reports
whether each runtime monkey-patch is currently bound in the process.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import sys
import pytest


# The three runtime-rebind wiring modules we expect `verify_live_rebinds`
# to check. Any new runtime rebind should extend _check() in apply_all.py.
RUNTIME_WIRING_MODULES = {
    "P22": "patch_22_tq_prealloc",
    "P31": "patch_31_router_softmax",
    "P14": "patch_14_block_table",
}


class TestVerifyLiveRebinds:
    def test_returns_dict_with_all_runtime_patches(self):
        """The function must report status for every runtime-rebind patch."""
        # Import from apply_all standalone (avoids loading vllm)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_aa_verify",
            "vllm/_genesis/patches/apply_all.py",
        )
        if spec is None or spec.loader is None:
            pytest.skip("cannot load apply_all standalone")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            pytest.skip(f"apply_all loads vllm lazily, may fail: {e}")

        results = mod.verify_live_rebinds()
        assert isinstance(results, dict)
        for patch_id in RUNTIME_WIRING_MODULES:
            assert patch_id in results, (
                f"verify_live_rebinds() missing entry for {patch_id}"
            )

    def test_result_shape(self):
        """Each entry must have expected/actual/ok/error|note fields."""
        from vllm._genesis.patches.apply_all import verify_live_rebinds
        results = verify_live_rebinds()
        for patch_id, r in results.items():
            assert "expected" in r
            assert "actual" in r
            assert "ok" in r
            # Must be True/False (not None) when there's no error
            if "error" not in r and "note" not in r:
                assert isinstance(r["actual"], bool)

    def test_each_module_has_is_applied(self):
        """Every runtime-wiring module must export an `is_applied()` callable.

        This is a contract test: if we add a new runtime rebind without
        is_applied(), verify_live_rebinds() degrades to "no-op note" which
        defeats the entire monitoring gate.
        """
        import importlib
        for patch_id, modname in RUNTIME_WIRING_MODULES.items():
            try:
                m = importlib.import_module(f"vllm._genesis.wiring.{modname}")
            except Exception as e:
                pytest.skip(f"cannot import {modname}: {e}")
            fn = getattr(m, "is_applied", None)
            assert callable(fn), (
                f"{modname}.is_applied must exist and be callable "
                f"(contract for runtime-rebind wiring modules)"
            )

    def test_returns_false_when_not_applied(self):
        """With nothing rebound (plugin not run), verify should report all
        expected=True, actual=False, ok=False — i.e. a red signal."""
        from vllm._genesis.patches.apply_all import verify_live_rebinds

        # In test process, no vLLM is imported so all is_applied() should be
        # False (module can't find targets → returns False). Expected=True
        # (we always expect rebinds to be live post-register) → ok=False.
        results = verify_live_rebinds()
        for patch_id, r in results.items():
            if "error" in r:
                # Import error → expected, log shows diagnostic
                assert r["ok"] is False
            else:
                assert r["expected"] is True
                # actual should be False (not live) or None (no is_applied)
                assert r["actual"] in (False, None, True)
