# SPDX-License-Identifier: Apache-2.0
"""TDD tests for Patch 40 — TurboQuant GQA-grouped decode stage1.

Covers:
- Module imports cleanly on CPU-only (Triton kernel build lazy)
- `should_apply()` respects env + platform
- `should_use_grouped_kernel()` dispatcher correctness:
  * opt-in required
  * only key_fp8 routes to grouped
  * only VQB==4 routes to grouped
  * kv_group_size > 1 required (MHA falls through)
- Constants match upstream PR #40792 (BLOCK_H=16, BLOCK_KV=16, etc.)
- Wiring apply/is_applied/revert surface
- Wiring self-retires when upstream symbol `_tq_grouped_decode_stage1`
  appears on target module
- Fallback to original on non-k8v4 / non-GQA calls

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations


import pytest


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    """Default env has P40 disabled; individual tests flip on via
    monkeypatch.setenv + module re-read."""
    monkeypatch.delenv("GENESIS_ENABLE_P40", raising=False)
    yield


def _reload_kernel_module():
    """Re-import the kernel module so `_ENABLED_AT_IMPORT` re-reads env."""
    import importlib
    from vllm._genesis.kernels import tq_grouped_decode
    return importlib.reload(tq_grouped_decode)


class TestP40ModuleImport:
    def test_import_succeeds_on_cpu(self):
        """Module must be importable without CUDA/triton."""
        from vllm._genesis.kernels import tq_grouped_decode
        assert hasattr(tq_grouped_decode, "should_apply")
        assert hasattr(tq_grouped_decode, "get_grouped_kernel")
        assert hasattr(tq_grouped_decode, "should_use_grouped_kernel")

    def test_constants_match_upstream_pr_40792(self):
        from vllm._genesis.kernels import tq_grouped_decode as k
        assert k.BLOCK_H == 16
        assert k.BLOCK_KV == 16
        assert k.NUM_WARPS == 4
        assert k.NUM_STAGES == 2


class TestP40ShouldApply:
    def test_disabled_by_default(self, monkeypatch):
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(
            guards, "is_sm_at_least", lambda major, minor=0: True,
        )
        # No env set → OFF by default
        k = _reload_kernel_module()
        assert k.should_apply() is False

    def test_enabled_with_env_and_nvidia(self, monkeypatch):
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(
            guards, "is_sm_at_least", lambda major, minor=0: True,
        )
        monkeypatch.setenv("GENESIS_ENABLE_P40", "1")
        k = _reload_kernel_module()
        assert k.should_apply() is True

    def test_disabled_on_non_nvidia_even_with_env(self, monkeypatch):
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: False)
        monkeypatch.setenv("GENESIS_ENABLE_P40", "1")
        k = _reload_kernel_module()
        assert k.should_apply() is False

    def test_disabled_on_pre_ampere(self, monkeypatch):
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(
            guards, "is_sm_at_least", lambda major, minor=0: False,
        )
        monkeypatch.setenv("GENESIS_ENABLE_P40", "1")
        k = _reload_kernel_module()
        assert k.should_apply() is False


class TestP40DispatcherDecision:
    """Correctness of `should_use_grouped_kernel` routing logic."""

    @pytest.fixture
    def enabled(self, monkeypatch):
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(
            guards, "is_sm_at_least", lambda major, minor=0: True,
        )
        monkeypatch.setenv("GENESIS_ENABLE_P40", "1")
        return _reload_kernel_module()

    def test_k8v4_gqa_routes_to_grouped(self, enabled):
        assert enabled.should_use_grouped_kernel(
            kv_group_size=8, key_fp8=True, value_quant_bits=4,
        ) is True

    def test_mha_does_not_route(self, enabled):
        # kv_group_size == 1 → MHA, no gain from grouping
        assert enabled.should_use_grouped_kernel(
            kv_group_size=1, key_fp8=True, value_quant_bits=4,
        ) is False

    def test_mse_key_preset_does_not_route(self, enabled):
        # Grouped kernel hard-codes key_fp8; MSE keys stay on scalar
        assert enabled.should_use_grouped_kernel(
            kv_group_size=8, key_fp8=False, value_quant_bits=4,
        ) is False

    def test_non_4bit_values_does_not_route(self, enabled):
        # tl.static_assert(VQB == 4) would fire; dispatch gate filters
        assert enabled.should_use_grouped_kernel(
            kv_group_size=8, key_fp8=True, value_quant_bits=3,
        ) is False

    def test_disabled_when_env_off(self, monkeypatch):
        # Even with all-correct shape, env-off forces scalar path
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(
            guards, "is_sm_at_least", lambda major, minor=0: True,
        )
        monkeypatch.delenv("GENESIS_ENABLE_P40", raising=False)
        k = _reload_kernel_module()
        assert k.should_use_grouped_kernel(
            kv_group_size=8, key_fp8=True, value_quant_bits=4,
        ) is False


class TestP40WiringSurface:
    def test_wiring_public_surface(self):
        from vllm._genesis.wiring.legacy import patch_40_tq_grouped_decode as p40
        assert callable(p40.apply)
        assert callable(p40.is_applied)
        assert callable(p40.revert)
        assert callable(p40.should_apply)

    def test_apply_skips_when_env_off(self, monkeypatch):
        from vllm._genesis.wiring.legacy import patch_40_tq_grouped_decode as p40
        monkeypatch.setattr(p40, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(
            p40, "is_sm_at_least", lambda major, minor=0: True,
        )
        monkeypatch.delenv("GENESIS_ENABLE_P40", raising=False)
        _reload_kernel_module()
        status, reason = p40.apply()
        assert status == "skipped"
        assert "opt-in" in reason.lower() or "40792" in reason

    def test_apply_skips_on_non_nvidia(self, monkeypatch):
        from vllm._genesis.wiring.legacy import patch_40_tq_grouped_decode as p40
        monkeypatch.setattr(p40, "is_nvidia_cuda", lambda: False)
        status, reason = p40.apply()
        assert status == "skipped"
        assert "NVIDIA" in reason

    def test_apply_skips_when_target_missing(self, monkeypatch):
        """On CPU unit-test env the target vLLM module isn't installed;
        apply() must gracefully skip without raising."""
        from vllm._genesis.wiring.legacy import patch_40_tq_grouped_decode as p40
        monkeypatch.setattr(p40, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(
            p40, "is_sm_at_least", lambda major, minor=0: True,
        )
        monkeypatch.setenv("GENESIS_ENABLE_P40", "1")
        _reload_kernel_module()
        status, _reason = p40.apply()
        # Either "skipped" (target not importable) or "applied" if
        # vllm is fully installed in the unit env
        assert status in ("skipped", "applied")


class TestP40UpstreamSelfRetirement:
    """Verify apply() auto-skips once upstream #40792 merges (detected
    by presence of `_tq_grouped_decode_stage1` on the target module)."""

    def test_self_retires_when_upstream_symbol_present(self, monkeypatch):
        from vllm._genesis.wiring.legacy import patch_40_tq_grouped_decode as p40
        from vllm._genesis import guards as _guards

        # Patch BOTH the wiring module's named imports AND the guards
        # module itself — the wiring imports by name
        # (`from vllm._genesis.guards import is_nvidia_cuda`), but the
        # kernel's `should_apply` imports again locally, so both need
        # to be green for `should_apply()` to return True and let
        # apply() reach the upstream-drift check.
        monkeypatch.setattr(p40, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(
            p40, "is_sm_at_least", lambda major, minor=0: True,
        )
        monkeypatch.setattr(_guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(
            _guards, "is_sm_at_least", lambda major, minor=0: True,
        )
        monkeypatch.setenv("GENESIS_ENABLE_P40", "1")
        _reload_kernel_module()

        # Inject a fake upstream module that already has the symbol.
        import sys, types
        fake_mod = types.ModuleType(p40._MODULE_PATH)

        def _fake_orig(*args, **kwargs):
            return None

        fake_mod.triton_turboquant_decode_attention = _fake_orig
        # Upstream drift symbol PRESENT → our wiring should bail
        fake_mod._tq_grouped_decode_stage1 = lambda: None
        monkeypatch.setitem(sys.modules, p40._MODULE_PATH, fake_mod)

        status, reason = p40.apply()
        assert status == "skipped"
        assert "upstream" in reason.lower() or "40792" in reason
        assert "retired" in reason.lower() or "merged" in reason.lower()
