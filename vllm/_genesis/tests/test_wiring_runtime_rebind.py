# SPDX-License-Identifier: Apache-2.0
"""TDD for runtime rebind wiring (P22 + P31).

These tests use synthetic mocked target classes/modules — they DO NOT
import real vLLM. Real-engine validation happens in integration tests.

IMPORTANT: these tests rely on sys.modules INJECTION of fake target
modules. If the REAL vLLM is already importable in the environment
(integration container), the injection fails to override a previously
imported module and the tests fail misleadingly. Skip those tests in
real-vllm environments — the integration diagnostic probes (Probe J/K
in validate_integration.sh) verify the rebinds on the REAL engine.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import importlib.util
import sys
import types
import pytest


# Skip entire file when running in an environment with REAL vLLM —
# the sys.modules-injection approach assumes vLLM is not already loaded.
# find_spec itself can raise ImportError on broken parent packages, so
# wrap it defensively (CPU-only VM 103 has no `vllm` top-level at all).
def _probe_real_vllm() -> bool:
    try:
        return importlib.util.find_spec(
            "vllm.model_executor.layers.fused_moe.router.grouped_topk_router"
        ) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


_real_vllm_available = _probe_real_vllm()
pytestmark = pytest.mark.skipif(
    _real_vllm_available,
    reason=(
        "Real vLLM present — sys.modules injection cannot override already-"
        "imported target module. Integration probes (J/K) verify these "
        "rebinds on the real engine instead."
    ),
)


@pytest.fixture
def mock_tq_impl_module(monkeypatch):
    """Inject a fake `vllm.v1.attention.backends.turboquant_attn` module
    into sys.modules so the wiring's import-by-string succeeds."""
    mod_name = "vllm.v1.attention.backends.turboquant_attn"

    class FakeTQAttentionImpl:
        def _ensure_on_device(self, layer, device):
            # Original behavior: just stamp the layer
            layer._tq_cached = True
            layer._call_log = layer.__dict__.get("_call_log", []) + ["original"]
            return None

        def _init_turboquant_buffers(self, *a, **kw):
            # Sentinel: presence of this method tells P22's drift-check
            # that upstream PR #40655 has NOT moved buffer init out yet
            # (the patch is still applicable).
            return None

    fake_mod = types.ModuleType(mod_name)
    fake_mod.TurboQuantAttentionImpl = FakeTQAttentionImpl

    # Ensure parent path exists in sys.modules (no real vllm needed)
    parent = "vllm.v1.attention.backends"
    for p in ["vllm", "vllm.v1", "vllm.v1.attention", parent]:
        if p not in sys.modules:
            sys.modules[p] = types.ModuleType(p)

    monkeypatch.setitem(sys.modules, mod_name, fake_mod)
    yield FakeTQAttentionImpl
    sys.modules.pop(mod_name, None)


@pytest.fixture
def mock_grouped_router_module(monkeypatch):
    """Inject a fake grouped_topk_router module."""
    mod_name = "vllm.model_executor.layers.fused_moe.router.grouped_topk_router"

    fake_mod = types.ModuleType(mod_name)

    def fake_grouped_topk(hidden_states, gating_output, *args, **kwargs):
        # Record the dtype we received — proves wrapper upcast happened
        return ("ok", gating_output.dtype)

    fake_mod.grouped_topk = fake_grouped_topk

    # Parent path
    parent_chain = [
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.fused_moe",
        "vllm.model_executor.layers.fused_moe.router",
    ]
    for p in parent_chain:
        if p not in sys.modules:
            sys.modules[p] = types.ModuleType(p)

    monkeypatch.setitem(sys.modules, mod_name, fake_mod)
    yield fake_mod
    sys.modules.pop(mod_name, None)


# ──────────────────────────────────────────────────────────────────────────
#                              P22 wiring tests
# ──────────────────────────────────────────────────────────────────────────

class TestPatch22Wiring:
    def test_apply_rebinds_class_method(self, mock_tq_impl_module, monkeypatch):
        """The wired wrapper takes the place of _ensure_on_device."""
        from vllm._genesis.wiring.legacy import patch_22_tq_prealloc

        # Force-enable platform guards
        monkeypatch.setattr(patch_22_tq_prealloc, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(patch_22_tq_prealloc, "is_sm_at_least",
                            lambda *a, **kw: True)

        # Stub the helper so we don't need real torch/cuda
        from vllm._genesis.kernels import dequant_buffer
        call_log = []
        monkeypatch.setattr(
            dequant_buffer, "ensure_turboquant_buffers",
            lambda impl, layer, device: call_log.append(("genesis", layer)),
        )

        status, reason = patch_22_tq_prealloc.apply()
        assert status == "applied", f"{status}: {reason}"
        assert patch_22_tq_prealloc.is_applied()

        # Live invocation: original + genesis both run, original first
        instance = mock_tq_impl_module()

        class FakeLayer:
            pass
        layer = FakeLayer()
        instance._ensure_on_device(layer, device="cpu")

        # Original should have set _tq_cached and called genesis after
        assert layer._tq_cached is True
        assert "original" in layer._call_log
        assert any(c[0] == "genesis" for c in call_log)

        # Cleanup
        patch_22_tq_prealloc.revert()

    def test_idempotent_reapply(self, mock_tq_impl_module, monkeypatch):
        from vllm._genesis.wiring.legacy import patch_22_tq_prealloc

        monkeypatch.setattr(patch_22_tq_prealloc, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(patch_22_tq_prealloc, "is_sm_at_least",
                            lambda *a, **kw: True)
        from vllm._genesis.kernels import dequant_buffer
        monkeypatch.setattr(
            dequant_buffer, "ensure_turboquant_buffers",
            lambda *a, **kw: None,
        )

        s1, _ = patch_22_tq_prealloc.apply()
        s2, _ = patch_22_tq_prealloc.apply()
        assert s1 == "applied"
        assert s2 == "applied"  # idempotent reason in r2
        # Wrapper applied only once — original stash points to the REAL
        # original method, not another wrapper.
        method = mock_tq_impl_module._ensure_on_device
        original_ref = getattr(method, "_genesis_p22_original", None)
        assert original_ref is not None
        assert not getattr(original_ref, "_genesis_p22_wrapped", False)

        patch_22_tq_prealloc.revert()

    def test_skip_on_non_nvidia(self, monkeypatch):
        from vllm._genesis.wiring.legacy import patch_22_tq_prealloc
        monkeypatch.setattr(patch_22_tq_prealloc, "is_nvidia_cuda", lambda: False)

        status, reason = patch_22_tq_prealloc.apply()
        assert status == "skipped"
        assert "NVIDIA" in reason

    def test_skip_when_tq_module_missing(self, monkeypatch):
        from vllm._genesis.wiring.legacy import patch_22_tq_prealloc

        monkeypatch.setattr(patch_22_tq_prealloc, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(patch_22_tq_prealloc, "is_sm_at_least",
                            lambda *a, **kw: True)
        # Force the import to fail by monkey-patching _import_tq_impl to return None.
        # Can't rely on sys.modules manipulation alone because real vLLM re-imports
        # the module on access (integration-container scenario).
        monkeypatch.setattr(
            patch_22_tq_prealloc, "_import_tq_impl", lambda: None,
        )

        status, reason = patch_22_tq_prealloc.apply()
        assert status == "skipped"
        assert "TurboQuant" in reason or "turboquant_attn" in reason

    def test_revert_restores_original(self, mock_tq_impl_module, monkeypatch):
        from vllm._genesis.wiring.legacy import patch_22_tq_prealloc

        monkeypatch.setattr(patch_22_tq_prealloc, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(patch_22_tq_prealloc, "is_sm_at_least",
                            lambda *a, **kw: True)
        from vllm._genesis.kernels import dequant_buffer
        monkeypatch.setattr(
            dequant_buffer, "ensure_turboquant_buffers", lambda *a, **kw: None,
        )

        original = mock_tq_impl_module._ensure_on_device
        patch_22_tq_prealloc.apply()
        assert mock_tq_impl_module._ensure_on_device is not original

        assert patch_22_tq_prealloc.revert() is True
        assert mock_tq_impl_module._ensure_on_device is original


# ──────────────────────────────────────────────────────────────────────────
#                              P31 wiring tests
# ──────────────────────────────────────────────────────────────────────────

class TestPatch31Wiring:
    def test_apply_wraps_grouped_topk(self, mock_grouped_router_module, monkeypatch):
        from vllm._genesis.wiring.legacy import patch_31_router_softmax
        # Force not-cpu-only
        monkeypatch.setattr(patch_31_router_softmax, "is_cpu_only", lambda: False)

        status, reason = patch_31_router_softmax.apply()
        assert status == "applied", f"{status}: {reason}"
        assert patch_31_router_softmax.is_applied()

        # Live test: pass bf16 logits → wrapper should upcast to fp32
        import torch
        gating = torch.randn(4, 8, dtype=torch.bfloat16)
        result, observed_dtype = mock_grouped_router_module.grouped_topk(
            None, gating,
        )
        assert result == "ok"
        # Wrapper upcasted: original receives fp32
        assert observed_dtype == torch.float32

        patch_31_router_softmax.revert()

    def test_skip_on_cpu(self, monkeypatch):
        from vllm._genesis.wiring.legacy import patch_31_router_softmax
        monkeypatch.setattr(patch_31_router_softmax, "is_cpu_only", lambda: True)

        status, reason = patch_31_router_softmax.apply()
        assert status == "skipped"
        assert "CPU" in reason

    def test_idempotent(self, mock_grouped_router_module, monkeypatch):
        from vllm._genesis.wiring.legacy import patch_31_router_softmax
        monkeypatch.setattr(patch_31_router_softmax, "is_cpu_only", lambda: False)

        s1, _ = patch_31_router_softmax.apply()
        s2, _ = patch_31_router_softmax.apply()
        assert s1 == "applied"
        assert s2 == "applied"

        # Single layer of wrap — stashed original is NOT another wrapper
        fn = mock_grouped_router_module.grouped_topk
        original_ref = getattr(fn, "_genesis_p31_original", None)
        assert original_ref is not None
        assert not getattr(original_ref, "_genesis_p31_wrapped", False)

        patch_31_router_softmax.revert()

    def test_revert_restores_original(self, mock_grouped_router_module, monkeypatch):
        from vllm._genesis.wiring.legacy import patch_31_router_softmax
        monkeypatch.setattr(patch_31_router_softmax, "is_cpu_only", lambda: False)

        original = mock_grouped_router_module.grouped_topk
        patch_31_router_softmax.apply()
        assert mock_grouped_router_module.grouped_topk is not original

        assert patch_31_router_softmax.revert() is True
        assert mock_grouped_router_module.grouped_topk is original

    def test_fp32_input_not_double_upcast(self, mock_grouped_router_module, monkeypatch):
        """Already-fp32 input passes through unchanged (no extra .float() copy)."""
        from vllm._genesis.wiring.legacy import patch_31_router_softmax
        monkeypatch.setattr(patch_31_router_softmax, "is_cpu_only", lambda: False)

        patch_31_router_softmax.apply()

        import torch
        gating = torch.randn(4, 8, dtype=torch.float32)
        _result, observed_dtype = mock_grouped_router_module.grouped_topk(
            None, gating,
        )
        assert observed_dtype == torch.float32

        patch_31_router_softmax.revert()
