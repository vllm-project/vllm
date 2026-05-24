# SPDX-License-Identifier: Apache-2.0
"""TDD for the attribute-rebind wiring primitive.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import types
import pytest


@pytest.fixture
def fake_module():
    """A throwaway module object we can rebind attrs on."""
    mod = types.ModuleType("fake_target_mod")

    def original_fn(x):
        return x * 2

    mod.original_fn = original_fn
    return mod


def my_replacement(x):
    return x * 100


class TestAttributeRebinder:
    def test_apply_sets_new_attr(self, fake_module):
        from vllm._genesis.wiring import AttributeRebinder

        r = AttributeRebinder(
            patch_name="test-basic",
            target_module=fake_module,
            target_attr="original_fn",
            replacement=my_replacement,
        )
        assert r.apply() is True
        assert fake_module.original_fn is my_replacement
        assert fake_module.original_fn(5) == 500

    def test_is_applied_reflects_live_binding(self, fake_module):
        from vllm._genesis.wiring import AttributeRebinder

        r = AttributeRebinder(
            patch_name="t",
            target_module=fake_module,
            target_attr="original_fn",
            replacement=my_replacement,
        )
        assert r.is_applied() is False
        r.apply()
        assert r.is_applied() is True

        # Simulate external reversion
        def other(x):
            return x
        fake_module.original_fn = other
        assert r.is_applied() is False

    def test_revert_restores_original(self, fake_module):
        from vllm._genesis.wiring import AttributeRebinder

        original_before = fake_module.original_fn

        r = AttributeRebinder(
            patch_name="t",
            target_module=fake_module,
            target_attr="original_fn",
            replacement=my_replacement,
        )
        r.apply()
        assert fake_module.original_fn is my_replacement

        r.revert()
        assert fake_module.original_fn is original_before
        assert fake_module.original_fn(3) == 6  # original behavior

    def test_missing_target_skips_cleanly(self, fake_module):
        from vllm._genesis.wiring import AttributeRebinder

        r = AttributeRebinder(
            patch_name="t",
            target_module=fake_module,
            target_attr="this_attr_does_not_exist",
            replacement=my_replacement,
        )
        # Should NOT raise — logs a warning, returns False
        assert r.apply() is False
        assert not r.is_applied()

    def test_idempotent_reapply(self, fake_module):
        from vllm._genesis.wiring import AttributeRebinder

        r = AttributeRebinder(
            patch_name="t",
            target_module=fake_module,
            target_attr="original_fn",
            replacement=my_replacement,
        )
        assert r.apply() is True
        # Second apply is a no-op (already applied)
        assert r.apply() is False
        # But is_applied still True
        assert r.is_applied() is True

    def test_already_our_function_is_idempotent(self, fake_module):
        """Fresh rebinder sees our function already bound → considered applied."""
        from vllm._genesis.wiring import AttributeRebinder

        fake_module.original_fn = my_replacement  # pre-bound

        r = AttributeRebinder(
            patch_name="t",
            target_module=fake_module,
            target_attr="original_fn",
            replacement=my_replacement,
        )
        # apply returns False because no change was made, but is_applied True
        assert r.apply() is False
        assert r.is_applied() is True

    def test_assert_applied_raises_when_not(self, fake_module):
        from vllm._genesis.wiring import AttributeRebinder

        r = AttributeRebinder(
            patch_name="t",
            target_module=fake_module,
            target_attr="original_fn",
            replacement=my_replacement,
        )
        with pytest.raises(AssertionError, match="NOT bound"):
            r.assert_applied()


class TestWiringRegistry:
    def test_registry_tracks_rebinds(self, fake_module):
        from vllm._genesis.wiring import AttributeRebinder, WiringRegistry

        WiringRegistry.clear_for_tests()
        assert len(WiringRegistry.all()) == 0

        r = AttributeRebinder(
            patch_name="reg-test",
            target_module=fake_module,
            target_attr="original_fn",
            replacement=my_replacement,
        )
        r.apply()

        assert len(WiringRegistry.all()) == 1
        s = WiringRegistry.summary()
        assert s["total"] == 1
        assert s["applied"] == 1
        assert s["pending_or_reverted"] == 0
        WiringRegistry.clear_for_tests()

    def test_clear_for_tests_reverts_all(self, fake_module):
        from vllm._genesis.wiring import AttributeRebinder, WiringRegistry

        WiringRegistry.clear_for_tests()
        original = fake_module.original_fn

        r = AttributeRebinder(
            patch_name="r1",
            target_module=fake_module,
            target_attr="original_fn",
            replacement=my_replacement,
        )
        r.apply()
        assert fake_module.original_fn is my_replacement

        WiringRegistry.clear_for_tests()
        assert fake_module.original_fn is original
        assert len(WiringRegistry.all()) == 0
