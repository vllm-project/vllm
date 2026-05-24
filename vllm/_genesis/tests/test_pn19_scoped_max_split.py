# SPDX-License-Identifier: Apache-2.0
"""Tests for PN19 — Scoped max_split_size_mb during model load.

PN19 backports vllm#41268 (MatthewBonanni). PyTorch 2.10+ introduced
load-time allocator fragmentation; PN19 mitigates by setting
`max_split_size_mb=20` (PyTorch minimum) during model load and
restoring on exit.

These tests pin:
- Module imports cleanly
- Both anchor + replacement strings preserve the existing context
  managers (memory_pool_context + set_current_vllm_config)
- Replacement injects the helper method WITH torch fallback for
  versions lacking _accelerator_setAllocatorSettings
- Replacement uses 20 MiB (PyTorch minimum), not arbitrary value
- apply() respects the env-flag gate (default skip)
- PN19 is registered in PATCH_REGISTRY + apply_all + PATCHES.md
"""
from __future__ import annotations



class TestPN19ModuleStructure:
    def test_module_importable(self):
        from vllm._genesis.wiring.perf_hotfix import patch_N19_scoped_max_split  # noqa: F401

    def test_helper_anchor_targets_init_device(self):
        """The helper-injection sub-patch anchors on the @instrument
        decorator + def init_device line — that's the unique 2-line
        block immediately after _maybe_get_memory_pool_context."""
        from vllm._genesis.wiring.perf_hotfix.patch_N19_scoped_max_split import (
            PN19_HELPER_OLD,
            PN19_HELPER_NEW,
        )
        assert "@instrument(span_name=\"Init device\")" in PN19_HELPER_OLD
        assert "def init_device(self):" in PN19_HELPER_OLD
        # Replacement must KEEP these two lines (they survive after
        # injecting our helper above them)
        assert "@instrument(span_name=\"Init device\")" in PN19_HELPER_NEW
        assert "def init_device(self):" in PN19_HELPER_NEW

    def test_load_model_anchor_preserves_existing_context(self):
        """Wrapping load_model must KEEP the two existing context
        managers; if we accidentally drop one, model loading breaks."""
        from vllm._genesis.wiring.perf_hotfix.patch_N19_scoped_max_split import (
            LOAD_MODEL_NEW,
        )
        # Must keep both upstream contexts
        assert "self._maybe_get_memory_pool_context(tag=\"weights\")" in LOAD_MODEL_NEW
        assert "set_current_vllm_config(self.vllm_config)" in LOAD_MODEL_NEW
        # Must add our new context
        assert "self._genesis_pn19_scoped_allocator_max_split" in LOAD_MODEL_NEW
        # Must keep the actual load call unchanged
        assert "self.model_runner.load_model(load_dummy_weights=load_dummy_weights)" in LOAD_MODEL_NEW


class TestPN19MaxSplitValue:
    def test_uses_pytorch_minimum_20mb(self):
        """PyTorch's allowed minimum for max_split_size_mb is 20 MiB.
        PR #41268 chose this value deliberately. Don't bikeshed it
        upward without measurement."""
        from vllm._genesis.wiring.perf_hotfix.patch_N19_scoped_max_split import (
            LOAD_MODEL_NEW,
        )
        assert "max_split_size_mb=20" in LOAD_MODEL_NEW, (
            "PN19 must use 20 MiB (PyTorch minimum) like upstream PR #41268. "
            "Higher values reduce cudaMalloc count but allow more "
            "fragmentation."
        )


class TestPN19TorchVersionFallback:
    """torch < 2.11 lacks `_accelerator_setAllocatorSettings`. PN19
    must detect this and fall through unchanged — must NOT crash on
    older torch."""

    def test_helper_includes_torch_fallback(self):
        from vllm._genesis.wiring.perf_hotfix.patch_N19_scoped_max_split import (
            PN19_HELPER_NEW,
        )
        assert "torch._C._accelerator_setAllocatorSettings" in PN19_HELPER_NEW
        # The setter call must be wrapped in try/except so older torch
        # falls through cleanly
        assert "except Exception" in PN19_HELPER_NEW
        assert "yield" in PN19_HELPER_NEW

    def test_helper_includes_cuda_check(self):
        """Non-CUDA platforms (CPU, ROCm with different API) must yield
        early."""
        from vllm._genesis.wiring.perf_hotfix.patch_N19_scoped_max_split import (
            PN19_HELPER_NEW,
        )
        assert "current_platform.is_cuda()" in PN19_HELPER_NEW

    def test_helper_restores_prior_value(self):
        """On exit, prior PYTORCH_CUDA_ALLOC_CONF max_split_size_mb
        value MUST be restored. Otherwise we leak the 20 MiB setting
        into post-load runtime."""
        from vllm._genesis.wiring.perf_hotfix.patch_N19_scoped_max_split import (
            PN19_HELPER_NEW,
        )
        assert "original_value" in PN19_HELPER_NEW
        assert "_SIZE_MAX_MB" in PN19_HELPER_NEW or "SIZE_MAX" in PN19_HELPER_NEW.upper()
        assert "finally:" in PN19_HELPER_NEW


class TestPN19EnvGate:
    def test_apply_skipped_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("GENESIS_ENABLE_PN19_SCOPED_MAX_SPLIT", raising=False)
        from vllm._genesis.wiring.perf_hotfix.patch_N19_scoped_max_split import (
            apply,
        )
        status, reason = apply()
        assert status == "skipped"
        assert "GENESIS_ENABLE_PN19_SCOPED_MAX_SPLIT" in reason

    def test_apply_skipped_explains_pr_and_acceptance_bar(self, monkeypatch):
        """The skip reason must reference PR #41268 + acknowledge the
        unverified-on-Ampere status so operators see the institutional
        rationale."""
        monkeypatch.delenv("GENESIS_ENABLE_PN19_SCOPED_MAX_SPLIT", raising=False)
        from vllm._genesis.wiring.perf_hotfix.patch_N19_scoped_max_split import (
            apply,
        )
        _, reason = apply()
        assert "vllm#41268" in reason or "#41268" in reason
        assert "200-500" in reason or "fragmentation" in reason

    def test_env_flag_truthy_values_recognized(self, monkeypatch):
        from vllm._genesis.wiring.perf_hotfix.patch_N19_scoped_max_split import (
            _is_enabled,
        )
        for val in ("1", "true", "TRUE", "Yes", "on", "ON"):
            monkeypatch.setenv("GENESIS_ENABLE_PN19_SCOPED_MAX_SPLIT", val)
            assert _is_enabled() is True
        for val in ("", "0", "false", "off", "garbage"):
            monkeypatch.setenv("GENESIS_ENABLE_PN19_SCOPED_MAX_SPLIT", val)
            assert _is_enabled() is False


class TestPN19DispatcherIntegration:
    def test_pn19_in_patch_registry(self):
        from vllm._genesis.dispatcher import PATCH_REGISTRY
        assert "PN19" in PATCH_REGISTRY
        meta = PATCH_REGISTRY["PN19"]
        assert meta["env_flag"] == "GENESIS_ENABLE_PN19_SCOPED_MAX_SPLIT"
        assert meta["default_on"] is False
        assert meta["upstream_pr"] == 41268

    def test_pn19_category_is_memory_savings(self):
        from vllm._genesis.dispatcher import PATCH_REGISTRY
        assert PATCH_REGISTRY["PN19"]["category"] == "memory_savings"

    def test_pn19_in_apply_all(self):
        from vllm._genesis.patches import apply_all
        assert hasattr(apply_all, "apply_patch_N19_scoped_max_split")

    def test_pn19_in_patches_md(self):
        from pathlib import Path
        repo_root = Path(__file__).resolve().parents[3]
        patches_md = (repo_root / "docs" / "PATCHES.md").read_text()
        assert "PN19" in patches_md
        assert "GENESIS_ENABLE_PN19_SCOPED_MAX_SPLIT" in patches_md
        assert "#41268" in patches_md


class TestPN19DriftMarkers:
    def test_drift_markers_watch_upstream_landing(self):
        from vllm._genesis.wiring.perf_hotfix.patch_N19_scoped_max_split import (
            UPSTREAM_DRIFT_MARKERS,
        )
        markers = "\n".join(UPSTREAM_DRIFT_MARKERS)
        # Watching upstream's helper method name
        assert "_scoped_allocator_max_split" in markers
        # Watching upstream's specific value choice
        assert "max_split_size_mb=20" in markers
