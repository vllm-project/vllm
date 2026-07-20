# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Integrated lifecycle tests for GraphStorageRegistry.

Tests the full layerwise reload lifecycle:
  snapshot → PWAL (process_weights_after_loading) → copy_back_registered
  → walk exclusion → metadata preservation

Tests three first-scope backends:
  - W4A8 MoE: quant_method.b_strides1, quant_method.b_strides2
  - FlashInfer CUTLASS MoE: quant_method.moe_kernel.fused_experts.gemm1_alpha,
    gemm1_beta, gemm1_clamp_limit
  - MLA attention: W_UV, W_UK_T

Also tests backward compatibility:
  - Empty registry = walk-only behavior (no regressions)
"""
import sys
import pathlib

import pytest
import torch
import torch.nn as nn

# Ensure repo root is on path for production imports
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vllm.model_executor.model_loader.reload.graph_storage_registry import (
    DriftError,
    GraphStorageRegistry,
    PathResolutionError,
    get_by_path,
    set_by_path,
    walk_audit,
)
from vllm.model_executor.model_loader.reload.tensor_collector import (
    collect_extra_tensors,
    copy_back_extra_tensors,
)
from vllm.model_executor.model_loader.reload import tensor_collector as _tc_mod


# ---------------------------------------------------------------------------
# CPU-compatible collect_extra_tensors (patches out CUDA filter)
# ---------------------------------------------------------------------------

def _collect_extra_tensors_cpu(layer, exclude_paths=None):
    """collect_extra_tensors with CUDA filter removed for CPU testing."""
    import functools
    import types

    _MAX_DEPTH = 10

    def _walk_cpu(path, obj, managed_storages, visited, results, depth):
        if depth > _MAX_DEPTH:
            return
        if isinstance(obj, torch.Tensor):
            if obj.numel() == 0:
                return
            if obj.untyped_storage().data_ptr() in managed_storages:
                return
            results.append((path, obj))
            return
        if isinstance(obj, nn.Module):
            return
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)
        if isinstance(obj, dict):
            for k, v in obj.items():
                if v is not None:
                    _walk_cpu(f"{path}[{k!r}]", v, managed_storages, visited,
                              results, depth + 1)
            return
        if isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                if v is not None:
                    _walk_cpu(f"{path}[{i}]", v, managed_storages, visited,
                              results, depth + 1)
            return
        if isinstance(obj, functools.partial):
            for i, arg in enumerate(obj.args):
                _walk_cpu(f"{path}.args[{i}]", arg, managed_storages, visited,
                          results, depth + 1)
            for k, v in obj.keywords.items():
                _walk_cpu(f"{path}.keywords[{k!r}]", v, managed_storages,
                          visited, results, depth + 1)
            return
        if isinstance(obj, types.FunctionType) and obj.__closure__:
            for i, cell in enumerate(obj.__closure__):
                try:
                    cell_val = cell.cell_contents
                except ValueError:
                    continue
                _walk_cpu(f"{path}.__closure__[{i}]", cell_val,
                          managed_storages, visited, results, depth + 1)
            return
        # Generic Python object (quant methods, kernel instances, etc.)
        obj_dict = getattr(obj, "__dict__", None)
        if obj_dict is None or isinstance(obj, type):
            return
        for attr_name in list(obj_dict):
            if attr_name.startswith("__"):
                continue
            val = obj_dict.get(attr_name)
            if val is not None:
                _walk_cpu(f"{path}.{attr_name}", val, managed_storages,
                          visited, results, depth + 1)

    from unittest.mock import patch
    with patch.object(_tc_mod, '_walk', _walk_cpu):
        return collect_extra_tensors(layer, exclude_paths)


# ---------------------------------------------------------------------------
# Helper: simulate PWAL (creates new tensors at registered paths)
# ---------------------------------------------------------------------------

def _simulate_pwal_replace(layer, paths):
    """Simulate process_weights_after_loading by replacing tensors at paths
    with new tensors that have different storage but same metadata."""
    for path in paths:
        old = get_by_path(layer, path)
        # Create new tensor with same shape/dtype/device but different storage
        new_tensor = torch.randn_like(old)
        # Set it at the path
        set_by_path(layer, path, new_tensor)


# ---------------------------------------------------------------------------
# Mock layers mimicking the three backend patterns
# ---------------------------------------------------------------------------

class MockQuantMethodW4A8:
    """Mimics CompressedTensorsW4A8Fp8MoEMethod with b_strides tensors."""
    def __init__(self):
        self.b_strides1 = torch.randn(4, device="cpu")
        self.b_strides2 = torch.randn(4, device="cpu")


class MockW4A8Layer(nn.Module):
    """Layer with quant_method.b_strides1/b_strides2."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 8))
        self.quant_method = MockQuantMethodW4A8()


class MockFlashInferExperts:
    """Mimics FlashInferExperts with gemm1_alpha/beta/clamp_limit."""
    def __init__(self):
        self.gemm1_alpha = torch.tensor(1.0)
        self.gemm1_beta = torch.tensor(0.0)
        self.gemm1_clamp_limit = torch.tensor(127.0)


class MockMoEKernel:
    def __init__(self):
        self.fused_experts = MockFlashInferExperts()


class MockFlashInferQuantMethod:
    def __init__(self):
        self.moe_kernel = MockMoEKernel()


class MockFlashInferLayer(nn.Module):
    """Layer with quant_method.moe_kernel.fused_experts.gemm1_*."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 8))
        self.quant_method = MockFlashInferQuantMethod()


class MockMLALayer(nn.Module):
    """Layer with W_UV and W_UK_T attributes."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 8))
        self.W_UV = torch.randn(16, 8)
        self.W_UK_T = torch.randn(8, 16)


# ---------------------------------------------------------------------------
# AC-4: Lifecycle Tests
# ---------------------------------------------------------------------------

class TestW4A8Lifecycle:
    """Full lifecycle for W4A8 MoE backend: b_strides1, b_strides2."""

    def setup_method(self):
        self.registry = GraphStorageRegistry()
        self.layer = MockW4A8Layer()

    def test_register_and_snapshot(self):
        """Registration captures correct metadata; snapshot resolves tensors."""
        self.registry.register_graph_storage(
            self.layer, "quant_method.b_strides1",
            self.layer.quant_method.b_strides1)
        self.registry.register_graph_storage(
            self.layer, "quant_method.b_strides2",
            self.layer.quant_method.b_strides2)

        snapshot = self.registry.snapshot(self.layer)
        assert "quant_method.b_strides1" in snapshot
        assert "quant_method.b_strides2" in snapshot
        assert snapshot["quant_method.b_strides1"].tensor is self.layer.quant_method.b_strides1
        assert snapshot["quant_method.b_strides2"].tensor is self.layer.quant_method.b_strides2

    def test_copy_back_restores_identity(self):
        """After PWAL, copy_back restores old tensor storage identity."""
        self.registry.register_graph_storage(
            self.layer, "quant_method.b_strides1",
            self.layer.quant_method.b_strides1)
        self.registry.register_graph_storage(
            self.layer, "quant_method.b_strides2",
            self.layer.quant_method.b_strides2)

        # Capture old pointers
        old_ptr1 = self.layer.quant_method.b_strides1.data_ptr()
        old_ptr2 = self.layer.quant_method.b_strides2.data_ptr()

        # Snapshot before PWAL
        snapshot = self.registry.snapshot(self.layer)

        # Simulate PWAL: replace tensors at paths
        _simulate_pwal_replace(
            self.layer, ["quant_method.b_strides1", "quant_method.b_strides2"])

        # Verify PWAL changed pointers
        assert self.layer.quant_method.b_strides1.data_ptr() != old_ptr1
        assert self.layer.quant_method.b_strides2.data_ptr() != old_ptr2

        # Copy back
        count = self.registry.copy_back_registered(self.layer, snapshot)
        assert count == 2

        # Old storage identity restored
        assert self.layer.quant_method.b_strides1.data_ptr() == old_ptr1
        assert self.layer.quant_method.b_strides2.data_ptr() == old_ptr2

    def test_metadata_preserved_after_copy_back(self):
        """dtype/shape/stride/storage_offset unchanged after copy-back."""
        t = self.layer.quant_method.b_strides1
        self.registry.register_graph_storage(
            self.layer, "quant_method.b_strides1", t)

        old_meta = (t.dtype, tuple(t.shape), tuple(t.stride()),
                    t.storage_offset())

        snapshot = self.registry.snapshot(self.layer)
        _simulate_pwal_replace(self.layer, ["quant_method.b_strides1"])
        self.registry.copy_back_registered(self.layer, snapshot)

        restored = self.layer.quant_method.b_strides1
        new_meta = (restored.dtype, tuple(restored.shape),
                    tuple(restored.stride()), restored.storage_offset())
        assert old_meta == new_meta

    def test_registered_paths_excluded_from_walk(self):
        """Registered paths are excluded from walk_audit."""
        self.registry.register_graph_storage(
            self.layer, "quant_method.b_strides1",
            self.layer.quant_method.b_strides1)
        self.registry.register_graph_storage(
            self.layer, "quant_method.b_strides2",
            self.layer.quant_method.b_strides2)

        registered = self.registry.get_registered_paths(self.layer)
        discovered = walk_audit(self.layer, registered_paths=registered)
        extra_paths = {p for p, _ in discovered}

        assert "quant_method.b_strides1" not in extra_paths
        assert "quant_method.b_strides2" not in extra_paths


class TestFlashInferLifecycle:
    """Full lifecycle for FlashInfer CUTLASS MoE backend."""

    PATHS = [
        "quant_method.moe_kernel.fused_experts.gemm1_alpha",
        "quant_method.moe_kernel.fused_experts.gemm1_beta",
        "quant_method.moe_kernel.fused_experts.gemm1_clamp_limit",
    ]

    def setup_method(self):
        self.registry = GraphStorageRegistry()
        self.layer = MockFlashInferLayer()

    def test_deep_nesting_register_and_snapshot(self):
        """Registration works through deeply nested path."""
        fe = self.layer.quant_method.moe_kernel.fused_experts
        for path, tensor in zip(self.PATHS,
                                [fe.gemm1_alpha, fe.gemm1_beta,
                                 fe.gemm1_clamp_limit]):
            self.registry.register_graph_storage(self.layer, path, tensor)

        snapshot = self.registry.snapshot(self.layer)
        assert len(snapshot) == 3
        for path in self.PATHS:
            assert path in snapshot

    def test_copy_back_restores_deep_identity(self):
        """Copy-back works for deeply nested paths."""
        fe = self.layer.quant_method.moe_kernel.fused_experts
        for path, tensor in zip(self.PATHS,
                                [fe.gemm1_alpha, fe.gemm1_beta,
                                 fe.gemm1_clamp_limit]):
            self.registry.register_graph_storage(self.layer, path, tensor)

        old_ptrs = {p: get_by_path(self.layer, p).data_ptr()
                    for p in self.PATHS}

        snapshot = self.registry.snapshot(self.layer)
        _simulate_pwal_replace(self.layer, self.PATHS)

        # Verify PWAL changed pointers
        for p in self.PATHS:
            assert get_by_path(self.layer, p).data_ptr() != old_ptrs[p]

        count = self.registry.copy_back_registered(self.layer, snapshot)
        assert count == 3

        # Identity restored
        for p in self.PATHS:
            assert get_by_path(self.layer, p).data_ptr() == old_ptrs[p]

    def test_registered_paths_excluded_from_walk(self):
        """Deep registered paths excluded from walk_audit."""
        fe = self.layer.quant_method.moe_kernel.fused_experts
        for path, tensor in zip(self.PATHS,
                                [fe.gemm1_alpha, fe.gemm1_beta,
                                 fe.gemm1_clamp_limit]):
            self.registry.register_graph_storage(self.layer, path, tensor)

        registered = self.registry.get_registered_paths(self.layer)
        discovered = walk_audit(
            self.layer, registered_paths=registered)
        extra_paths = {p for p, _ in discovered}

        for path in self.PATHS:
            assert path not in extra_paths


class TestMLALifecycle:
    """Full lifecycle for MLA attention backend: W_UV, W_UK_T."""

    PATHS = ["W_UV", "W_UK_T"]

    def setup_method(self):
        self.registry = GraphStorageRegistry()
        self.layer = MockMLALayer()

    def test_register_and_copy_back(self):
        """Full lifecycle for MLA top-level attrs."""
        for path in self.PATHS:
            tensor = getattr(self.layer, path)
            self.registry.register_graph_storage(self.layer, path, tensor)

        old_ptrs = {p: getattr(self.layer, p).data_ptr() for p in self.PATHS}
        snapshot = self.registry.snapshot(self.layer)

        # Simulate PWAL: replace with new tensors
        for path in self.PATHS:
            setattr(self.layer, path, torch.randn_like(getattr(self.layer, path)))

        # Verify PWAL changed pointers
        for p in self.PATHS:
            assert getattr(self.layer, p).data_ptr() != old_ptrs[p]

        count = self.registry.copy_back_registered(self.layer, snapshot)
        assert count == 2

        for p in self.PATHS:
            assert getattr(self.layer, p).data_ptr() == old_ptrs[p]

    def test_metadata_preserved(self):
        """dtype/shape/stride/storage_offset preserved for MLA tensors."""
        for path in self.PATHS:
            tensor = getattr(self.layer, path)
            self.registry.register_graph_storage(self.layer, path, tensor)

        old_metas = {}
        for p in self.PATHS:
            t = getattr(self.layer, p)
            old_metas[p] = (t.dtype, tuple(t.shape), tuple(t.stride()),
                            t.storage_offset())

        snapshot = self.registry.snapshot(self.layer)
        for path in self.PATHS:
            setattr(self.layer, path, torch.randn_like(getattr(self.layer, path)))
        self.registry.copy_back_registered(self.layer, snapshot)

        for p in self.PATHS:
            t = getattr(self.layer, p)
            meta = (t.dtype, tuple(t.shape), tuple(t.stride()),
                    t.storage_offset())
            assert meta == old_metas[p]

    def test_registered_paths_excluded_from_walk(self):
        """MLA registered paths excluded from walk_audit."""
        for path in self.PATHS:
            tensor = getattr(self.layer, path)
            self.registry.register_graph_storage(self.layer, path, tensor)

        registered = self.registry.get_registered_paths(self.layer)
        discovered = walk_audit(
            self.layer, registered_paths=registered)
        extra_paths = {p for p, _ in discovered}

        for path in self.PATHS:
            assert path not in extra_paths


# ---------------------------------------------------------------------------
# Negative tests: wrong-path registration fails closed
# ---------------------------------------------------------------------------

class TestNegativeRegistration:
    """Wrong-path and drift detection (fail-closed behavior)."""

    def test_wrong_path_raises_resolution_error(self):
        """Registering a non-existent path raises PathResolutionError."""
        layer = MockW4A8Layer()
        registry = GraphStorageRegistry()
        with pytest.raises(PathResolutionError):
            registry.register_graph_storage(
                layer, "quant_method.nonexistent_tensor",
                torch.randn(4))

    def test_wrong_tensor_raises_drift_error(self):
        """Registering path that resolves to different tensor raises DriftError."""
        layer = MockW4A8Layer()
        registry = GraphStorageRegistry()
        wrong_tensor = torch.randn(4)  # Different storage
        with pytest.raises(DriftError):
            registry.register_graph_storage(
                layer, "quant_method.b_strides1", wrong_tensor)

    def test_snapshot_after_drift_raises(self):
        """If tensor drifts between register and snapshot, raises DriftError."""
        layer = MockW4A8Layer()
        registry = GraphStorageRegistry()
        registry.register_graph_storage(
            layer, "quant_method.b_strides1",
            layer.quant_method.b_strides1)

        # Externally replace the tensor (simulating uncontrolled drift)
        layer.quant_method.b_strides1 = torch.randn(4)

        with pytest.raises(DriftError):
            registry.snapshot(layer)

    def test_copy_back_metadata_mismatch_raises(self):
        """If PWAL produces tensor with different shape, raises DriftError."""
        layer = MockW4A8Layer()
        registry = GraphStorageRegistry()
        registry.register_graph_storage(
            layer, "quant_method.b_strides1",
            layer.quant_method.b_strides1)

        snapshot = registry.snapshot(layer)

        # Replace with tensor of different shape (metadata mismatch)
        layer.quant_method.b_strides1 = torch.randn(8)  # Was shape (4,)

        with pytest.raises(DriftError):
            registry.copy_back_registered(layer, snapshot)


# ---------------------------------------------------------------------------
# AC-7: Backward Compatibility - Empty Registry Equivalence
# ---------------------------------------------------------------------------

class TestEmptyRegistryEquivalence:
    """When no registrations exist, system behaves like walk-only path.

    Uses both walk_audit (device-agnostic) and production collect_extra_tensors
    (with CUDA filter patched for CPU testing) to prove equivalence.
    """

    def test_empty_registry_walk_discovers_all(self):
        """With no registrations, walk_audit finds all unmanaged tensors."""
        layer = MockW4A8Layer()
        # No registered paths — walk finds everything
        discovered = walk_audit(layer, registered_paths=set())
        paths = {p for p, _ in discovered}
        assert "quant_method.b_strides1" in paths
        assert "quant_method.b_strides2" in paths

    def test_empty_registry_snapshot_is_empty(self):
        """Empty registry produces empty snapshot."""
        layer = MockW4A8Layer()
        registry = GraphStorageRegistry()
        snapshot = registry.snapshot(layer)
        assert snapshot == {}

    def test_empty_registry_copy_back_is_noop(self):
        """copy_back_registered with empty snapshot returns 0."""
        layer = MockW4A8Layer()
        registry = GraphStorageRegistry()
        count = registry.copy_back_registered(layer, {})
        assert count == 0

    def test_walk_audit_copy_back_without_registry(self):
        """Walk-only copy-back works independently of registry.

        Uses walk_audit to discover tensors, then manually copies back
        (simulating what copy_back_extra_tensors does for CUDA tensors).
        """
        layer = MockW4A8Layer()
        old_tensor = layer.quant_method.b_strides1
        old_ptr = old_tensor.data_ptr()

        # Walk discovers the tensor
        discovered = walk_audit(layer, registered_paths=set())
        extras = [(p, t) for p, t in discovered
                  if p == "quant_method.b_strides1"]
        assert len(extras) == 1

        # Simulate PWAL replacing the tensor
        layer.quant_method.b_strides1 = torch.randn(4)
        assert layer.quant_method.b_strides1.data_ptr() != old_ptr

        # Manual copy-back: copy new data into old storage, restore reference
        old_tensor.data.copy_(layer.quant_method.b_strides1.data)
        layer.quant_method.b_strides1 = old_tensor
        assert layer.quant_method.b_strides1.data_ptr() == old_ptr

    def test_production_collect_and_copy_back_without_registry(self):
        """Production collect_extra_tensors + copy_back_extra_tensors works
        independently of registry (empty-registry = walk-only behavior)."""
        layer = MockW4A8Layer()
        old_ptr1 = layer.quant_method.b_strides1.data_ptr()
        old_ptr2 = layer.quant_method.b_strides2.data_ptr()

        # Production collect (no registry, no exclude_paths)
        extras = _collect_extra_tensors_cpu(layer, exclude_paths=None)
        paths = {p for p, _ in extras}
        assert "quant_method.b_strides1" in paths
        assert "quant_method.b_strides2" in paths

        # PWAL replaces tensors
        _simulate_pwal_replace(
            layer, ["quant_method.b_strides1", "quant_method.b_strides2"])
        assert layer.quant_method.b_strides1.data_ptr() != old_ptr1
        assert layer.quant_method.b_strides2.data_ptr() != old_ptr2

        # Production copy_back_extra_tensors restores identity
        copy_back_extra_tensors(layer, extras)
        assert layer.quant_method.b_strides1.data_ptr() == old_ptr1
        assert layer.quant_method.b_strides2.data_ptr() == old_ptr2

    def test_registered_subset_walk_still_finds_unregistered(self):
        """Registering some paths allows walk_audit to still find others."""
        layer = MockW4A8Layer()
        registry = GraphStorageRegistry()

        # Register only b_strides1
        registry.register_graph_storage(
            layer, "quant_method.b_strides1",
            layer.quant_method.b_strides1)

        registered = registry.get_registered_paths(layer)
        # walk_audit excludes registered paths
        discovered = walk_audit(layer, registered_paths=registered)
        paths = {p for p, _ in discovered}

        # b_strides1 excluded, b_strides2 still found
        assert "quant_method.b_strides1" not in paths
        assert "quant_method.b_strides2" in paths


# ---------------------------------------------------------------------------
# Multi-reload lifecycle (simulates 3 consecutive reloads)
# ---------------------------------------------------------------------------

class TestConsecutiveReloads:
    """Verify identity preservation across 3 consecutive same-checkpoint reloads."""

    def test_three_consecutive_reloads_w4a8(self):
        """W4A8 tensors maintain identity across 3 reloads.

        Simulates backend re-registration during PWAL (the real lifecycle
        re-registers tensors after process_weights_after_loading). The snapshot
        captured BEFORE PWAL must still be the source of truth for copy-back.
        """
        layer = MockW4A8Layer()
        registry = GraphStorageRegistry()
        paths = ["quant_method.b_strides1", "quant_method.b_strides2"]

        # Initial registration
        for p in paths:
            registry.register_graph_storage(layer, p, get_by_path(layer, p))

        # Capture original identity
        original_ptrs = {p: get_by_path(layer, p).data_ptr() for p in paths}

        for reload_idx in range(3):
            snapshot = registry.snapshot(layer)
            _simulate_pwal_replace(layer, paths)

            # Simulate backend re-registration during PWAL
            # (this overwrites registry metadata with post-PWAL tensors)
            for p in paths:
                registry.register_graph_storage(layer, p, get_by_path(layer, p))

            # After PWAL + re-registration, pointers changed
            for p in paths:
                assert get_by_path(layer, p).data_ptr() != original_ptrs[p], \
                    f"Reload {reload_idx}: PWAL did not replace {p}"

            # Copy-back uses snapshot (not registry) — restores identity
            registry.copy_back_registered(layer, snapshot)
            for p in paths:
                assert get_by_path(layer, p).data_ptr() == original_ptrs[p], \
                    f"Reload {reload_idx}: copy_back did not restore {p}"

    def test_three_consecutive_reloads_mla(self):
        """MLA tensors maintain identity across 3 reloads."""
        layer = MockMLALayer()
        registry = GraphStorageRegistry()
        paths = ["W_UV", "W_UK_T"]

        for p in paths:
            registry.register_graph_storage(layer, p, getattr(layer, p))

        original_ptrs = {p: getattr(layer, p).data_ptr() for p in paths}

        for reload_idx in range(3):
            snapshot = registry.snapshot(layer)
            for p in paths:
                setattr(layer, p, torch.randn_like(getattr(layer, p)))

            for p in paths:
                assert getattr(layer, p).data_ptr() != original_ptrs[p]

            registry.copy_back_registered(layer, snapshot)
            for p in paths:
                assert getattr(layer, p).data_ptr() == original_ptrs[p], \
                    f"Reload {reload_idx}: failed for {p}"


# ---------------------------------------------------------------------------
# Integration-order test: mimics layerwise.py's exact sequence
# ---------------------------------------------------------------------------

class TestLayerwiseIntegrationOrder:
    """Tests that follow the exact sequence used by layerwise.py:

    1. get_registered_paths(layer)
    2. registry.snapshot(layer)          → captures old tensors
    3. walk_audit(layer, registered)     → extra_tensor_slots (excludes registered)
    4. [PWAL occurs, replacing tensors]
    5. registry.copy_back_registered()   → restores registered identity
    6. manual walk copy-back             → restores unregistered identity

    This proves registry and walk cooperate in the correct order.
    """

    def test_full_integration_sequence_w4a8(self):
        """W4A8: registered paths get registry copy-back, unregistered get
        production copy_back_extra_tensors."""
        layer = MockW4A8Layer()
        registry = GraphStorageRegistry()

        # Only register b_strides1 (leave b_strides2 for walk fallback)
        registry.register_graph_storage(
            layer, "quant_method.b_strides1",
            layer.quant_method.b_strides1)

        # Step 1: get registered paths
        registered_paths = registry.get_registered_paths(layer)
        assert "quant_method.b_strides1" in registered_paths

        # Step 2: snapshot registered tensors
        snapshot = registry.snapshot(layer)
        assert "quant_method.b_strides1" in snapshot

        # Step 3: collect_extra_tensors with exclude_paths (production API)
        extra_slots = _collect_extra_tensors_cpu(
            layer, exclude_paths=registered_paths)
        extra_paths = {p for p, _ in extra_slots}
        assert "quant_method.b_strides1" not in extra_paths
        assert "quant_method.b_strides2" in extra_paths

        # Capture original pointers
        old_ptr1 = layer.quant_method.b_strides1.data_ptr()
        old_ptr2 = layer.quant_method.b_strides2.data_ptr()

        # Step 4: PWAL replaces both tensors
        _simulate_pwal_replace(
            layer, ["quant_method.b_strides1", "quant_method.b_strides2"])
        assert layer.quant_method.b_strides1.data_ptr() != old_ptr1
        assert layer.quant_method.b_strides2.data_ptr() != old_ptr2

        # Step 5: registry copy-back (registered path only)
        count = registry.copy_back_registered(layer, snapshot)
        assert count == 1
        assert layer.quant_method.b_strides1.data_ptr() == old_ptr1
        # b_strides2 still has new pointer (not in registry)
        assert layer.quant_method.b_strides2.data_ptr() != old_ptr2

        # Step 6: production copy_back_extra_tensors for walk-discovered paths
        copy_back_extra_tensors(layer, extra_slots)

        # Both restored
        assert layer.quant_method.b_strides1.data_ptr() == old_ptr1
        assert layer.quant_method.b_strides2.data_ptr() == old_ptr2

    def test_registry_before_walk_ordering(self):
        """Registry copy-back first, then production copy_back_extra_tensors."""
        layer = MockMLALayer()
        registry = GraphStorageRegistry()

        # Register W_UV only
        registry.register_graph_storage(layer, "W_UV", layer.W_UV)

        registered = registry.get_registered_paths(layer)
        snapshot = registry.snapshot(layer)
        # Use production collect_extra_tensors with exclude_paths
        extra_slots = _collect_extra_tensors_cpu(
            layer, exclude_paths=registered)

        old_uv_ptr = layer.W_UV.data_ptr()
        old_ukt_ptr = layer.W_UK_T.data_ptr()

        # PWAL
        layer.W_UV = torch.randn_like(layer.W_UV)
        layer.W_UK_T = torch.randn_like(layer.W_UK_T)

        # Registry copy-back first (fail-closed)
        registry.copy_back_registered(layer, snapshot)
        assert layer.W_UV.data_ptr() == old_uv_ptr

        # Production walk copy-back second (best-effort)
        copy_back_extra_tensors(layer, extra_slots)

        assert layer.W_UK_T.data_ptr() == old_ukt_ptr


# ---------------------------------------------------------------------------
# AC-5: Pointer-Identity Oracle
# ---------------------------------------------------------------------------

# Backend configurations for parameterized oracle tests
_BACKEND_CONFIGS = {
    "w4a8_moe": {
        "layer_factory": MockW4A8Layer,
        "paths": ["quant_method.b_strides1", "quant_method.b_strides2"],
    },
    "flashinfer_cutlass_moe": {
        "layer_factory": MockFlashInferLayer,
        "paths": [
            "quant_method.moe_kernel.fused_experts.gemm1_alpha",
            "quant_method.moe_kernel.fused_experts.gemm1_beta",
            "quant_method.moe_kernel.fused_experts.gemm1_clamp_limit",
        ],
    },
    "mla": {
        "layer_factory": MockMLALayer,
        "paths": ["W_UV", "W_UK_T"],
    },
}


def _validate_tensor_identity(tensor, expected):
    """Validate all pointer-identity properties of a tensor.

    Returns dict with property name -> (actual, expected, pass/fail).
    """
    report = {}
    report["data_ptr"] = (
        tensor.data_ptr(), expected["data_ptr"],
        tensor.data_ptr() == expected["data_ptr"])
    report["dtype"] = (
        tensor.dtype, expected["dtype"],
        tensor.dtype == expected["dtype"])
    report["shape"] = (
        tuple(tensor.shape), expected["shape"],
        tuple(tensor.shape) == expected["shape"])
    report["stride"] = (
        tuple(tensor.stride()), expected["stride"],
        tuple(tensor.stride()) == expected["stride"])
    report["storage_offset"] = (
        tensor.storage_offset(), expected["storage_offset"],
        tensor.storage_offset() == expected["storage_offset"])
    return report


def _capture_identity(tensor):
    """Capture all identity properties of a tensor."""
    return {
        "data_ptr": tensor.data_ptr(),
        "dtype": tensor.dtype,
        "shape": tuple(tensor.shape),
        "stride": tuple(tensor.stride()),
        "storage_offset": tensor.storage_offset(),
    }


@pytest.mark.parametrize("backend", list(_BACKEND_CONFIGS.keys()))
class TestPointerIdentityOracle:
    """Parameterized pointer-identity oracle validating storage pointer,
    dtype, shape, stride, and storage_offset across 3 consecutive reloads
    for all 3 backends."""

    NUM_RELOADS = 3

    def test_identity_preserved_across_reloads(self, backend):
        """Full oracle: validate all 5 properties across 3 reloads."""
        config = _BACKEND_CONFIGS[backend]
        layer = config["layer_factory"]()
        paths = config["paths"]
        registry = GraphStorageRegistry()

        # Register all paths
        for p in paths:
            registry.register_graph_storage(layer, p, get_by_path(layer, p))

        # Capture original identity for all tensors
        original_ids = {p: _capture_identity(get_by_path(layer, p))
                        for p in paths}

        # Run 3 consecutive reloads
        for reload_idx in range(self.NUM_RELOADS):
            snapshot = registry.snapshot(layer)
            _simulate_pwal_replace(layer, paths)

            # Re-register (simulates backend re-registration during PWAL)
            for p in paths:
                registry.register_graph_storage(
                    layer, p, get_by_path(layer, p))

            # Copy-back restores identity
            registry.copy_back_registered(layer, snapshot)

            # Validate all properties for each tensor
            for p in paths:
                tensor = get_by_path(layer, p)
                report = _validate_tensor_identity(tensor, original_ids[p])
                for prop, (actual, expected, passed) in report.items():
                    assert passed, (
                        f"Reload {reload_idx}, path '{p}', "
                        f"property '{prop}': "
                        f"expected {expected}, got {actual}")

    def test_per_tensor_validation_report(self, backend):
        """Generate per-tensor validation report after reload."""
        config = _BACKEND_CONFIGS[backend]
        layer = config["layer_factory"]()
        paths = config["paths"]
        registry = GraphStorageRegistry()

        for p in paths:
            registry.register_graph_storage(layer, p, get_by_path(layer, p))

        original_ids = {p: _capture_identity(get_by_path(layer, p))
                        for p in paths}

        snapshot = registry.snapshot(layer)
        _simulate_pwal_replace(layer, paths)
        for p in paths:
            registry.register_graph_storage(layer, p, get_by_path(layer, p))
        registry.copy_back_registered(layer, snapshot)

        # Build full report
        full_report = {}
        for p in paths:
            tensor = get_by_path(layer, p)
            full_report[p] = _validate_tensor_identity(tensor, original_ids[p])

        # All properties should pass
        for p, report in full_report.items():
            for prop, (_, _, passed) in report.items():
                assert passed, f"Report: {p}.{prop} failed"

        # Report is non-empty and covers all paths
        assert set(full_report.keys()) == set(paths)


# ---------------------------------------------------------------------------
# AC-5: Negative Injection Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", list(_BACKEND_CONFIGS.keys()))
class TestOracleNegativeInjection:
    """Negative injection tests: detect pointer drift, shape mismatch,
    dtype mismatch, and storage_offset mismatch."""

    def _setup_and_reload(self, backend):
        """Setup layer, register, reload once, return (layer, paths, originals)."""
        config = _BACKEND_CONFIGS[backend]
        layer = config["layer_factory"]()
        paths = config["paths"]
        registry = GraphStorageRegistry()

        for p in paths:
            registry.register_graph_storage(layer, p, get_by_path(layer, p))

        original_ids = {p: _capture_identity(get_by_path(layer, p))
                        for p in paths}

        snapshot = registry.snapshot(layer)
        _simulate_pwal_replace(layer, paths)
        for p in paths:
            registry.register_graph_storage(layer, p, get_by_path(layer, p))
        registry.copy_back_registered(layer, snapshot)

        return layer, paths, original_ids

    def test_pointer_drift_detected(self, backend):
        """Injecting a new tensor (different data_ptr) is detected."""
        layer, paths, original_ids = self._setup_and_reload(backend)
        target_path = paths[0]

        # Inject tensor with different storage
        injected = torch.randn_like(get_by_path(layer, target_path))
        set_by_path(layer, target_path, injected)

        report = _validate_tensor_identity(
            get_by_path(layer, target_path), original_ids[target_path])
        assert not report["data_ptr"][2], "Pointer drift not detected"

    def test_shape_mismatch_detected(self, backend):
        """Injecting a tensor with different shape is detected."""
        layer, paths, original_ids = self._setup_and_reload(backend)
        target_path = paths[0]

        # Inject tensor with different shape
        original_t = get_by_path(layer, target_path)
        wrong_shape = torch.randn(original_t.numel() * 2,
                                  dtype=original_t.dtype)
        set_by_path(layer, target_path, wrong_shape)

        report = _validate_tensor_identity(
            get_by_path(layer, target_path), original_ids[target_path])
        assert not report["shape"][2], "Shape mismatch not detected"

    def test_dtype_mismatch_detected(self, backend):
        """Injecting a tensor with different dtype is detected."""
        layer, paths, original_ids = self._setup_and_reload(backend)
        target_path = paths[0]

        # Inject tensor with different dtype
        original_t = get_by_path(layer, target_path)
        wrong_dtype = original_t.to(torch.float16)
        set_by_path(layer, target_path, wrong_dtype)

        report = _validate_tensor_identity(
            get_by_path(layer, target_path), original_ids[target_path])
        assert not report["dtype"][2], "Dtype mismatch not detected"

    def test_storage_offset_mismatch_detected(self, backend):
        """Injecting a tensor with non-zero storage_offset is detected."""
        layer, paths, original_ids = self._setup_and_reload(backend)
        target_path = paths[0]

        # Create tensor with non-zero storage_offset via slicing
        original_t = get_by_path(layer, target_path)
        base = torch.randn(original_t.numel() + 4, dtype=original_t.dtype)
        offset_tensor = base[4:]  # storage_offset = 4
        assert offset_tensor.storage_offset() > 0
        set_by_path(layer, target_path, offset_tensor)

        report = _validate_tensor_identity(
            get_by_path(layer, target_path), original_ids[target_path])
        assert not report["storage_offset"][2], \
            "Storage offset mismatch not detected"
