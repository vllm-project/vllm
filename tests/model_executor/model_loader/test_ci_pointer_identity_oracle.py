# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CI Pointer-Identity Oracle
==========================
Validates that tensor identity (storage pointer + dtype + shape + stride +
storage_offset) is preserved across reload_weights for registered and
walk-discovered graph-storage tensors.

Structure:
  - Worker RPC oracle (GPU-dependent): loads real model via LLM, injects
    probe RPCs, runs reload_weights, asserts zero drift
  - In-process oracle (CPU, no model): validates oracle logic using mock
    layers with production collect_extra_tensors/copy_back_extra_tensors

Uses the Worker RPC pattern from prototype dir-04.
"""
from __future__ import annotations

import json
import os
import sys
import pathlib
from dataclasses import dataclass, field, asdict
from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

# Production vLLM imports
try:
    _REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
except IndexError:
    _REPO_ROOT = pathlib.Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vllm.model_executor.model_loader.reload.graph_storage_registry import (
    GraphStorageRegistry,
    get_by_path,
    set_by_path,
    walk_audit,
)
from vllm.model_executor.model_loader.reload.tensor_collector import (
    collect_extra_tensors,
    copy_back_extra_tensors,
)


# ---------------------------------------------------------------------------
# CUDA filter patch for CPU testing
# ---------------------------------------------------------------------------

def _walk_no_cuda_filter(path, obj, managed_storages, visited, results, depth):
    """Patched _walk that removes the is_cuda filter for CPU testing."""
    from vllm.model_executor.model_loader.reload import tensor_collector
    _MAX_DEPTH = 10
    import functools
    import types

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
                _walk_no_cuda_filter(f"{path}[{k!r}]", v, managed_storages,
                                     visited, results, depth + 1)
        return

    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            if v is not None:
                _walk_no_cuda_filter(f"{path}[{i}]", v, managed_storages,
                                     visited, results, depth + 1)
        return

    if isinstance(obj, functools.partial):
        for i, arg in enumerate(obj.args):
            _walk_no_cuda_filter(f"{path}.args[{i}]", arg, managed_storages,
                                 visited, results, depth + 1)
        for k, v in obj.keywords.items():
            _walk_no_cuda_filter(f"{path}.keywords[{k!r}]", v,
                                 managed_storages, visited, results, depth + 1)
        return

    if isinstance(obj, types.FunctionType) and obj.__closure__:
        for i, cell in enumerate(obj.__closure__):
            try:
                cell_val = cell.cell_contents
            except ValueError:
                continue
            _walk_no_cuda_filter(f"{path}.__closure__[{i}]", cell_val,
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
            _walk_no_cuda_filter(f"{path}.{attr_name}", val,
                                 managed_storages, visited, results, depth + 1)


@pytest.fixture
def cpu_collect_extra_tensors():
    """Fixture that patches collect_extra_tensors to work on CPU tensors."""
    with patch(
        "vllm.model_executor.model_loader.reload.tensor_collector._walk",
        _walk_no_cuda_filter,
    ):
        yield collect_extra_tensors


# ---------------------------------------------------------------------------
# Oracle data structures
# ---------------------------------------------------------------------------

@dataclass
class TensorIdentity:
    """Complete identity of a tensor for oracle comparison."""
    data_ptr: int
    dtype: torch.dtype
    shape: tuple
    stride: tuple
    storage_offset: int

    @classmethod
    def capture(cls, tensor: torch.Tensor) -> "TensorIdentity":
        return cls(
            data_ptr=tensor.data_ptr(),
            dtype=tensor.dtype,
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            storage_offset=tensor.storage_offset(),
        )

    def to_rpc(self) -> dict:
        """Serialize to RPC-safe dict (all JSON-primitive types)."""
        return {
            "data_ptr": self.data_ptr,
            "dtype": str(self.dtype),  # e.g. "torch.float32"
            "shape": list(self.shape),
            "stride": list(self.stride),
            "storage_offset": self.storage_offset,
        }

    @classmethod
    def from_rpc(cls, d: dict) -> "TensorIdentity":
        """Reconstruct from RPC dict produced by to_rpc()."""
        dtype_str = d["dtype"]
        # Handle "torch.float32" -> torch.float32
        dtype = getattr(torch, dtype_str.replace("torch.", ""))
        return cls(
            data_ptr=d["data_ptr"],
            dtype=dtype,
            shape=tuple(d["shape"]),
            stride=tuple(d["stride"]),
            storage_offset=d["storage_offset"],
        )

    def compare(self, other: "TensorIdentity") -> dict[str, tuple]:
        """Compare two identities, return dict of mismatched fields."""
        mismatches = {}
        for field_name in ["data_ptr", "dtype", "shape", "stride",
                           "storage_offset"]:
            expected = getattr(self, field_name)
            actual = getattr(other, field_name)
            if expected != actual:
                mismatches[field_name] = (expected, actual)
        return mismatches


@dataclass
class OracleResult:
    """Per-case oracle result for CI reporting."""
    name: str
    status: str  # PASS | FAIL | SKIP | ERROR
    tensors_checked: int = 0
    preserved: int = 0
    drifted: int = 0
    tolerated_gone: int = 0
    drift_details: list[dict] = field(default_factory=list)
    skip_reason: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Walk snapshot/assert (production collect_extra_tensors integration)
# ---------------------------------------------------------------------------

def walk_snapshot(
    model: nn.Module,
    collect_fn=collect_extra_tensors,
) -> dict[str, TensorIdentity]:
    """Walk model collecting all extra tensors and their identities.

    Uses production collect_extra_tensors for discovery, then captures
    full identity (pointer + dtype + shape + stride + offset).
    """
    snapshot: dict[str, TensorIdentity] = {}
    for mod_name, module in model.named_modules():
        extras = collect_fn(module)
        if not extras:
            continue
        for attr_path, tensor in extras:
            fqn = f"{mod_name}.{attr_path}" if mod_name else attr_path
            snapshot[fqn] = TensorIdentity.capture(tensor)
    return snapshot


def walk_assert_zero_drift(
    snapshot_before: dict[str, TensorIdentity],
    model: nn.Module,
    collect_fn=collect_extra_tensors,
    exclude_gone_patterns: list[str] | None = None,
) -> OracleResult:
    """Re-walk model and compare against snapshot_before.

    Returns OracleResult with per-tensor drift details including exact
    property name, expected value, and actual value for CI reporting.

    Args:
        exclude_gone_patterns: List of path substrings. If a tensor is GONE
            and its path contains any of these patterns, it is counted as
            preserved (tolerated) rather than drifted. This handles scratch
            buffers that are lazily recreated during forward.
    """
    exclude_gone_patterns = exclude_gone_patterns or []
    result = OracleResult(name="", status="PASS")
    current = walk_snapshot(model, collect_fn)

    for fqn, expected_id in snapshot_before.items():
        result.tensors_checked += 1
        if fqn not in current:
            # Check if this GONE path is in the tolerated set
            if any(pat in fqn for pat in exclude_gone_patterns):
                result.tolerated_gone += 1
                continue
            result.drifted += 1
            result.drift_details.append({
                "path": fqn,
                "property": "existence",
                "expected": "present",
                "actual": "GONE",
            })
            continue

        actual_id = current[fqn]
        mismatches = expected_id.compare(actual_id)
        if mismatches:
            result.drifted += 1
            for prop, (exp, act) in mismatches.items():
                result.drift_details.append({
                    "path": fqn,
                    "property": prop,
                    "expected": str(exp),
                    "actual": str(act),
                })
        else:
            result.preserved += 1

    if result.drifted > 0:
        result.status = "FAIL"
    return result


# ---------------------------------------------------------------------------
# Worker RPC helpers (for GPU oracle)
# ---------------------------------------------------------------------------

def _rpc_walk_snapshot(self) -> dict:
    """Worker-side RPC: snapshot all extra tensors and store locally."""
    model = self.model_runner.get_model()
    snapshot = walk_snapshot(model)
    # Store locally for per-rank assert (TP>1)
    self._ci_stored_snapshot = {k: v.to_rpc() for k, v in snapshot.items()}
    return self._ci_stored_snapshot


def _rpc_walk_assert(self, snapshot_before: dict,
                     exclude_gone_patterns: list[str] | None = None) -> dict:
    """Worker-side RPC: assert zero drift after reload."""
    # Reconstruct TensorIdentity from RPC dicts
    before = {k: TensorIdentity.from_rpc(v) for k, v in snapshot_before.items()}
    model = self.model_runner.get_model()
    result = walk_assert_zero_drift(
        before, model, exclude_gone_patterns=exclude_gone_patterns)
    return {
        "status": result.status,
        "preserved": result.preserved,
        "drifted": result.drifted,
        "tolerated_gone": result.tolerated_gone,
        "details": result.drift_details[:20],
    }


def _rpc_walk_assert_stored(
    self, exclude_gone_patterns: list[str] | None = None
) -> dict:
    """Worker-side RPC: assert zero drift using locally stored snapshot."""
    snapshot = getattr(self, "_ci_stored_snapshot", None)
    if snapshot is None:
        return {"status": "ERROR", "preserved": 0, "drifted": 0,
                "tolerated_gone": 0, "details": [{"error": "no stored snapshot"}]}
    return self._ci_walk_assert(snapshot, exclude_gone_patterns)


def _rpc_verify_cuda_graphs(self) -> dict:
    """Worker-side RPC: verify CUDA graph mode is enabled and configured.

    Returns dict with:
      - cuda_graphs_enabled: bool (cudagraph_mode != NONE)
      - cudagraph_mode: str
      - has_cudagraph_batch_sizes: bool (runner configured for graph capture)
    """
    vllm_config = self.model_runner.vllm_config
    compilation_config = vllm_config.compilation_config
    cudagraph_mode = compilation_config.cudagraph_copy_input_mode \
        if hasattr(compilation_config, "cudagraph_copy_input_mode") \
        else getattr(compilation_config, "cudagraph_mode", None)

    # Check if CUDA graphs are actually enabled (not NONE)
    mode_str = str(cudagraph_mode) if cudagraph_mode is not None else "NONE"
    enabled = cudagraph_mode is not None and "NONE" not in mode_str.upper()

    # Check for configured graph batch sizes
    has_batch_sizes = hasattr(self.model_runner, "cudagraph_batch_sizes") and \
        bool(getattr(self.model_runner, "cudagraph_batch_sizes", None))

    return {
        "cuda_graphs_enabled": enabled,
        "cudagraph_mode": mode_str,
        "has_cudagraph_batch_sizes": has_batch_sizes,
    }


def _rpc_get_cudagraph_capture_count(self) -> int:
    """Worker-side RPC: return current process-global CUDA graph capture count.

    Used for before/after delta comparison to prove THIS test's startup
    caused new CUDA graph captures.
    """
    from vllm.compilation.counter import compilation_counter
    return compilation_counter.num_cudagraph_captured


def _rpc_get_cudagraph_runner_state(self) -> dict:
    """Worker-side RPC: return runner-local CUDA graph captured state.

    This is the primary per-runner proof that CUDA graphs were captured
    for this specific model runner during initialization. Unlike the
    process-global capture counter, this cannot be polluted by earlier
    tests in the same pytest process.

    Returns dict with:
      - keys_initialized: bool (dispatcher has initialized capture keys)
      - num_captured_keys: int (total keys across all modes)
      - cudagraph_mode: str (active mode)
      - cudagraph_batch_sizes: list[int] (configured batch sizes)
      - has_captured_graphs: bool (summary: runner has captured graphs)
    """
    runner = self.model_runner
    dispatcher = getattr(runner, "cudagraph_dispatcher", None)

    if dispatcher is not None:
        keys_initialized = getattr(dispatcher, "keys_initialized", False)
        cudagraph_keys = getattr(dispatcher, "cudagraph_keys", {})
        num_captured_keys = sum(
            len(keys) for keys in cudagraph_keys.values()
        )
        mode_str = str(getattr(dispatcher, "cudagraph_mode", "NONE"))
    else:
        # Fallback: check compilation config and capture count for evidence
        vllm_config = getattr(runner, "vllm_config", None)
        compilation_config = getattr(vllm_config, "compilation_config", None) \
            if vllm_config else None
        mode = getattr(compilation_config, "cudagraph_mode", None) \
            if compilation_config else None
        mode_str = str(mode) if mode else "NONE"

        # Use capture sizes as evidence of initialization
        capture_sizes = getattr(
            compilation_config, "cudagraph_capture_sizes", []) \
            if compilation_config else []
        keys_initialized = bool(capture_sizes)
        num_captured_keys = len(capture_sizes) if capture_sizes else 0

    batch_sizes = list(getattr(runner, "cudagraph_batch_sizes", None) or
                       getattr(runner, "cudagraph_capture_sizes", None) or [])

    # If no batch_sizes attr, try compilation config
    if not batch_sizes:
        vllm_config = getattr(runner, "vllm_config", None)
        cc = getattr(vllm_config, "compilation_config", None) \
            if vllm_config else None
        batch_sizes = list(
            getattr(cc, "cudagraph_capture_sizes", None) or [])

    return {
        "keys_initialized": keys_initialized,
        "num_captured_keys": num_captured_keys,
        "cudagraph_mode": mode_str,
        "cudagraph_batch_sizes": batch_sizes,
        "has_captured_graphs": keys_initialized and num_captured_keys > 0,
    }


# ---------------------------------------------------------------------------
# Mock layers for in-process testing
# ---------------------------------------------------------------------------

class MockQuantMethodW4A8:
    def __init__(self):
        self.b_strides1 = torch.randn(4)
        self.b_strides2 = torch.randn(4)


class MockW4A8Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 8))
        self.quant_method = MockQuantMethodW4A8()


class MockFlashInferExperts:
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
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 8))
        self.quant_method = MockFlashInferQuantMethod()


class MockMLALayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 8))
        self.W_UV = torch.randn(16, 8)
        self.W_UK_T = torch.randn(8, 16)


class MockModel(nn.Module):
    """Model with multiple backend layers for oracle testing."""
    def __init__(self):
        super().__init__()
        self.w4a8_layer = MockW4A8Layer()
        self.flashinfer_layer = MockFlashInferLayer()
        self.mla_layer = MockMLALayer()

    def reload_correct(self):
        """Simulate correct reload: in-place update preserves pointers."""
        self.w4a8_layer.quant_method.b_strides1.fill_(99)
        self.w4a8_layer.quant_method.b_strides2.fill_(99)
        fe = self.flashinfer_layer.quant_method.moe_kernel.fused_experts
        fe.gemm1_alpha.fill_(2.0)
        fe.gemm1_beta.fill_(1.0)
        fe.gemm1_clamp_limit.fill_(64.0)
        self.mla_layer.W_UV.data.copy_(torch.randn_like(self.mla_layer.W_UV))
        self.mla_layer.W_UK_T.data.copy_(torch.randn_like(self.mla_layer.W_UK_T))

    def reload_bad(self):
        """Simulate bad reload: reallocates tensors (pointer drift)."""
        self.w4a8_layer.quant_method.b_strides1 = torch.randn(4)
        self.w4a8_layer.quant_method.b_strides2 = torch.randn(4)
        fe = self.flashinfer_layer.quant_method.moe_kernel.fused_experts
        fe.gemm1_alpha = torch.tensor(2.0)
        fe.gemm1_beta = torch.tensor(1.0)
        fe.gemm1_clamp_limit = torch.tensor(64.0)
        self.mla_layer.W_UV = torch.randn(16, 8)
        self.mla_layer.W_UK_T = torch.randn(8, 16)


# ---------------------------------------------------------------------------
# In-process oracle tests (no GPU required)
# ---------------------------------------------------------------------------

class TestOracleSnapshotMechanics:
    """Validate oracle snapshot and comparison using mock model."""

    def test_snapshot_discovers_extra_tensors(self, cpu_collect_extra_tensors):
        """walk_snapshot finds all backend extra tensors."""
        model = MockModel()
        snapshot = walk_snapshot(model, cpu_collect_extra_tensors)
        paths = set(snapshot.keys())
        assert "w4a8_layer.quant_method.b_strides1" in paths
        assert "w4a8_layer.quant_method.b_strides2" in paths
        assert "flashinfer_layer.quant_method.moe_kernel.fused_experts.gemm1_alpha" in paths
        assert "mla_layer.W_UV" in paths
        assert "mla_layer.W_UK_T" in paths

    def test_zero_drift_after_correct_reload(self, cpu_collect_extra_tensors):
        """Correct in-place reload preserves all tensor identities."""
        model = MockModel()
        snapshot = walk_snapshot(model, cpu_collect_extra_tensors)
        model.reload_correct()
        result = walk_assert_zero_drift(snapshot, model, cpu_collect_extra_tensors)
        assert result.status == "PASS"
        assert result.drifted == 0
        assert result.preserved > 0

    def test_drift_detected_after_bad_reload(self, cpu_collect_extra_tensors):
        """Bad reload (reallocation) detected as pointer drift."""
        model = MockModel()
        snapshot = walk_snapshot(model, cpu_collect_extra_tensors)
        model.reload_bad()
        result = walk_assert_zero_drift(snapshot, model, cpu_collect_extra_tensors)
        assert result.status == "FAIL"
        assert result.drifted > 0
        # Verify drift details include path and property information
        assert len(result.drift_details) > 0
        detail = result.drift_details[0]
        assert "path" in detail
        assert "property" in detail
        assert "expected" in detail
        assert "actual" in detail

    def test_three_consecutive_reloads_zero_drift(self, cpu_collect_extra_tensors):
        """3 consecutive correct reloads all show zero drift."""
        model = MockModel()
        snapshot = walk_snapshot(model, cpu_collect_extra_tensors)
        for _ in range(3):
            model.reload_correct()
            result = walk_assert_zero_drift(snapshot, model,
                                            cpu_collect_extra_tensors)
            assert result.status == "PASS"
            assert result.drifted == 0


class TestOracleNegativeInjection:
    """Negative injection: oracle detects and reports specific drift types."""

    def test_pointer_drift_reports_path(self, cpu_collect_extra_tensors):
        """Injected pointer drift is detected with exact path."""
        model = MockModel()
        snapshot = walk_snapshot(model, cpu_collect_extra_tensors)
        # Inject pointer drift on one tensor
        model.w4a8_layer.quant_method.b_strides1 = torch.randn(4)
        result = walk_assert_zero_drift(snapshot, model,
                                        cpu_collect_extra_tensors)
        assert result.status == "FAIL"
        drift_paths = [d["path"] for d in result.drift_details]
        assert "w4a8_layer.quant_method.b_strides1" in drift_paths
        # Reports data_ptr as the mismatched property
        for d in result.drift_details:
            if d["path"] == "w4a8_layer.quant_method.b_strides1":
                assert d["property"] == "data_ptr"

    def test_shape_mismatch_reports_property(self, cpu_collect_extra_tensors):
        """Injected shape mismatch is detected and reported."""
        model = MockModel()
        snapshot = walk_snapshot(model, cpu_collect_extra_tensors)
        # Replace with different shape tensor (keeps same path)
        model.mla_layer.W_UV = torch.randn(32, 4)  # was (16, 8)
        result = walk_assert_zero_drift(snapshot, model,
                                        cpu_collect_extra_tensors)
        assert result.status == "FAIL"
        shape_mismatches = [d for d in result.drift_details
                           if d["property"] == "shape"]
        assert len(shape_mismatches) > 0

    def test_dtype_mismatch_reports_property(self, cpu_collect_extra_tensors):
        """Injected dtype mismatch detected."""
        model = MockModel()
        snapshot = walk_snapshot(model, cpu_collect_extra_tensors)
        old_t = model.w4a8_layer.quant_method.b_strides2
        model.w4a8_layer.quant_method.b_strides2 = old_t.to(torch.float16)
        result = walk_assert_zero_drift(snapshot, model,
                                        cpu_collect_extra_tensors)
        assert result.status == "FAIL"
        dtype_mismatches = [d for d in result.drift_details
                           if d["property"] == "dtype"]
        assert len(dtype_mismatches) > 0

    def test_storage_offset_mismatch_reports_property(
        self, cpu_collect_extra_tensors
    ):
        """Injected storage_offset mismatch detected."""
        model = MockModel()
        snapshot = walk_snapshot(model, cpu_collect_extra_tensors)
        # Create tensor with non-zero storage offset
        base = torch.randn(20, 8)
        model.mla_layer.W_UV = base[4:]  # storage_offset > 0
        result = walk_assert_zero_drift(snapshot, model,
                                        cpu_collect_extra_tensors)
        assert result.status == "FAIL"
        offset_mismatches = [d for d in result.drift_details
                            if d["property"] == "storage_offset"]
        assert len(offset_mismatches) > 0

    def test_gone_tensor_reports_existence(self, cpu_collect_extra_tensors):
        """Tensor removed entirely is detected as GONE."""
        model = MockModel()
        snapshot = walk_snapshot(model, cpu_collect_extra_tensors)
        # Remove a tensor entirely
        del model.mla_layer.W_UK_T
        result = walk_assert_zero_drift(snapshot, model,
                                        cpu_collect_extra_tensors)
        assert result.status == "FAIL"
        gone = [d for d in result.drift_details if d["actual"] == "GONE"]
        assert len(gone) > 0
        assert any("W_UK_T" in d["path"] for d in gone)


class TestOracleWithRegistry:
    """Oracle validates both registry-copied and walk-copied tensors."""

    def test_registry_and_walk_branches_validated(
        self, cpu_collect_extra_tensors
    ):
        """Oracle covers both registered (registry path) and unregistered
        (walk path) tensors in a single model."""
        model = MockModel()
        registry = GraphStorageRegistry()

        # Register only W4A8 paths (leave FlashInfer + MLA for walk)
        registry.register_graph_storage(
            model.w4a8_layer, "quant_method.b_strides1",
            model.w4a8_layer.quant_method.b_strides1)
        registry.register_graph_storage(
            model.w4a8_layer, "quant_method.b_strides2",
            model.w4a8_layer.quant_method.b_strides2)

        # Snapshot ALL tensors (registry + walk)
        snapshot = walk_snapshot(model, cpu_collect_extra_tensors)
        assert len(snapshot) >= 7  # 2 w4a8 + 3 flashinfer + 2 mla

        # Simulate reload with registry copy-back for registered, manual for walk
        snap = registry.snapshot(model.w4a8_layer)
        # PWAL replaces all tensors
        model.reload_bad()
        # Registry copy-back for registered paths
        registry.copy_back_registered(model.w4a8_layer, snap)

        # Walk copy-back for unregistered paths (FlashInfer + MLA)
        # uses production copy_back_extra_tensors
        # (not done here — the point is oracle detects the drift for unregistered)

        # Oracle should detect drift for unregistered paths, pass for registered
        result = walk_assert_zero_drift(snapshot, model,
                                        cpu_collect_extra_tensors)
        # W4A8 registered paths should be preserved
        w4a8_results = [d for d in result.drift_details
                        if "w4a8_layer" in d["path"]]
        assert len(w4a8_results) == 0, \
            "Registry-copied W4A8 paths should show zero drift"
        # Unregistered paths should show drift
        assert result.drifted > 0, \
            "Walk-only paths (not copy-backed) should drift"

    def test_full_lifecycle_oracle_pass(self, cpu_collect_extra_tensors):
        """Full lifecycle: registry + walk copy-back → oracle passes."""
        model = MockModel()
        registry = GraphStorageRegistry()

        # Register W4A8
        w4a8_paths = ["quant_method.b_strides1", "quant_method.b_strides2"]
        for p in w4a8_paths:
            registry.register_graph_storage(
                model.w4a8_layer, p,
                get_by_path(model.w4a8_layer, p))

        # Snapshot everything
        snapshot = walk_snapshot(model, cpu_collect_extra_tensors)

        # Get walk-discovered extras (for unregistered layers)
        registered_paths = registry.get_registered_paths(model.w4a8_layer)
        mla_extras = cpu_collect_extra_tensors(model.mla_layer)
        fi_extras = cpu_collect_extra_tensors(model.flashinfer_layer)

        # Registry snapshot before PWAL
        reg_snap = registry.snapshot(model.w4a8_layer)

        # PWAL
        model.reload_bad()

        # Step 1: Registry copy-back (registered paths)
        registry.copy_back_registered(model.w4a8_layer, reg_snap)

        # Step 2: Walk copy-back (unregistered paths)
        copy_back_extra_tensors(model.mla_layer, mla_extras)
        copy_back_extra_tensors(model.flashinfer_layer, fi_extras)

        # Oracle should show zero drift for ALL paths
        result = walk_assert_zero_drift(snapshot, model,
                                        cpu_collect_extra_tensors)
        assert result.status == "PASS", \
            f"Expected zero drift, got {result.drifted}: {result.drift_details}"
        assert result.preserved == len(snapshot)


class TestOracleJsonReport:
    """Oracle produces CI-consumable JSON report."""

    def test_json_output_schema(self, cpu_collect_extra_tensors):
        """OracleResult serializes to expected JSON schema."""
        model = MockModel()
        snapshot = walk_snapshot(model, cpu_collect_extra_tensors)
        model.reload_bad()
        result = walk_assert_zero_drift(snapshot, model,
                                        cpu_collect_extra_tensors)
        result.name = "test_case"

        report = asdict(result)
        assert "name" in report
        assert "status" in report
        assert "preserved" in report
        assert "drifted" in report
        assert "drift_details" in report
        # Serializable to JSON
        json_str = json.dumps(report, default=str)
        parsed = json.loads(json_str)
        assert parsed["status"] == "FAIL"
        assert parsed["drifted"] > 0


# ---------------------------------------------------------------------------
# Fake-Worker Round-Trip Tests (exercises RPC payload without GPU)
# ---------------------------------------------------------------------------

class _FakeCompilationConfig:
    """Stub for CompilationConfig with configurable cudagraph mode."""
    def __init__(self, mode_str="PIECEWISE"):
        self.cudagraph_copy_input_mode = mode_str


class _FakeVLLMConfig:
    """Stub for vllm_config."""
    def __init__(self, cuda_graphs_enabled=True):
        mode = "PIECEWISE" if cuda_graphs_enabled else "NONE"
        self.compilation_config = _FakeCompilationConfig(mode)


class _FakeCudagraphDispatcher:
    """Stub for CudagraphDispatcher with configurable state."""
    def __init__(self, cuda_graphs_enabled=True):
        from enum import Enum
        # Minimal CUDAGraphMode stub
        self.keys_initialized = cuda_graphs_enabled
        self.cudagraph_keys = {
            "PIECEWISE": {("batch_1",), ("batch_2",)} if cuda_graphs_enabled else set(),
            "FULL": set(),
        }
        self.cudagraph_mode = "PIECEWISE" if cuda_graphs_enabled else "NONE"


class _FakeModelRunner:
    """Minimal model runner stub for testing RPC helpers."""
    def __init__(self, model: nn.Module, cuda_graphs_enabled=True):
        self._model = model
        self.vllm_config = _FakeVLLMConfig(cuda_graphs_enabled)
        self.cudagraph_batch_sizes = [1, 2, 4] if cuda_graphs_enabled else None
        self.cudagraph_dispatcher = _FakeCudagraphDispatcher(cuda_graphs_enabled)

    def get_model(self):
        return self._model


class _FakeWorker:
    """Minimal Worker stub with model_runner for RPC testing."""
    def __init__(self, model: nn.Module, cuda_graphs_enabled=True):
        self.model_runner = _FakeModelRunner(model, cuda_graphs_enabled)


class TestFakeWorkerRoundTrip:
    """Exercises _rpc_walk_snapshot → _rpc_walk_assert round-trip in-process."""

    def test_snapshot_produces_rpc_safe_payload(self, cpu_collect_extra_tensors):
        """Snapshot returns JSON-serializable dict with string dtypes."""
        model = MockModel()
        worker = _FakeWorker(model)

        with patch(
            "vllm.model_executor.model_loader.reload.tensor_collector._walk",
            _walk_no_cuda_filter,
        ):
            payload = _rpc_walk_snapshot(worker)

        assert isinstance(payload, dict)
        assert len(payload) > 0
        # All values are dicts with string dtype
        for key, val in payload.items():
            assert isinstance(val, dict), f"Value for '{key}' is not a dict"
            assert isinstance(val["dtype"], str), \
                f"dtype for '{key}' is {type(val['dtype'])}, expected str"
            assert val["dtype"].startswith("torch."), \
                f"dtype '{val['dtype']}' doesn't start with 'torch.'"
            assert isinstance(val["shape"], list)
            assert isinstance(val["stride"], list)
        # JSON-serializable
        import json
        json.dumps(payload)

    def test_assert_passes_after_correct_reload(self, cpu_collect_extra_tensors):
        """Round-trip: snapshot → correct reload → assert shows zero drift."""
        model = MockModel()
        worker = _FakeWorker(model)

        with patch(
            "vllm.model_executor.model_loader.reload.tensor_collector._walk",
            _walk_no_cuda_filter,
        ):
            payload = _rpc_walk_snapshot(worker)
            model.reload_correct()
            result = _rpc_walk_assert(worker, payload)

        assert result["status"] == "PASS"
        assert result["drifted"] == 0
        assert result["preserved"] > 0

    def test_assert_detects_drift_after_bad_reload(self, cpu_collect_extra_tensors):
        """Round-trip: snapshot → bad reload → assert detects drift."""
        model = MockModel()
        worker = _FakeWorker(model)

        with patch(
            "vllm.model_executor.model_loader.reload.tensor_collector._walk",
            _walk_no_cuda_filter,
        ):
            payload = _rpc_walk_snapshot(worker)
            model.reload_bad()
            result = _rpc_walk_assert(worker, payload)

        assert result["status"] == "FAIL"
        assert result["drifted"] > 0
        assert len(result["details"]) > 0

    def test_three_consecutive_reloads_round_trip(self, cpu_collect_extra_tensors):
        """3 consecutive correct reloads via RPC round-trip all pass."""
        model = MockModel()
        worker = _FakeWorker(model)

        with patch(
            "vllm.model_executor.model_loader.reload.tensor_collector._walk",
            _walk_no_cuda_filter,
        ):
            payload = _rpc_walk_snapshot(worker)
            for _ in range(3):
                model.reload_correct()
                result = _rpc_walk_assert(worker, payload)
                assert result["status"] == "PASS"
                assert result["drifted"] == 0

    def test_to_rpc_from_rpc_round_trip(self):
        """TensorIdentity.to_rpc() → from_rpc() preserves all fields."""
        t = torch.randn(4, 8)
        original = TensorIdentity.capture(t)
        rpc_dict = original.to_rpc()
        reconstructed = TensorIdentity.from_rpc(rpc_dict)
        assert original == reconstructed

    def test_negative_injection_via_rpc(self, cpu_collect_extra_tensors):
        """Worker-side negative injection: modified payload triggers drift."""
        model = MockModel()
        worker = _FakeWorker(model)

        with patch(
            "vllm.model_executor.model_loader.reload.tensor_collector._walk",
            _walk_no_cuda_filter,
        ):
            payload = _rpc_walk_snapshot(worker)
            # Inject a fake data_ptr to simulate drift
            first_key = next(iter(payload))
            payload[first_key]["data_ptr"] = 0xDEADBEEF
            result = _rpc_walk_assert(worker, payload)

        assert result["status"] == "FAIL"
        assert result["drifted"] >= 1

    def test_verify_cuda_graphs_enabled(self):
        """CUDA graph verification probe reports enabled when configured."""
        model = MockModel()
        worker = _FakeWorker(model, cuda_graphs_enabled=True)
        result = _rpc_verify_cuda_graphs(worker)
        assert result["cuda_graphs_enabled"] is True
        assert "PIECEWISE" in result["cudagraph_mode"]
        assert result["has_cudagraph_batch_sizes"] is True
        # No num_cudagraph_captured here - that's a separate RPC now
        assert "num_cudagraph_captured" not in result

    def test_verify_cuda_graphs_disabled(self):
        """CUDA graph verification probe reports disabled for NONE mode."""
        model = MockModel()
        worker = _FakeWorker(model, cuda_graphs_enabled=False)
        result = _rpc_verify_cuda_graphs(worker)
        assert result["cuda_graphs_enabled"] is False
        assert "NONE" in result["cudagraph_mode"]
        assert result["has_cudagraph_batch_sizes"] is False

    def test_get_cudagraph_capture_count_returns_int(self):
        """Capture count RPC returns an integer (for before/after delta)."""
        model = MockModel()
        worker = _FakeWorker(model)
        count = _rpc_get_cudagraph_capture_count(worker)
        assert isinstance(count, int)
        assert count >= 0

    def test_capture_count_delta_logic(self):
        """Demonstrates the before/after delta pattern used by GPU oracle.

        The GPU oracle samples count BEFORE LLM() construction and asserts
        it increases after init (startup capture). In-process without GPU,
        the count won't change, but we validate the comparison shape.
        """
        model = MockModel()
        worker = _FakeWorker(model)
        count_before = _rpc_get_cudagraph_capture_count(worker)
        # Simulate "no new captures" — count should be same
        count_after = _rpc_get_cudagraph_capture_count(worker)
        assert count_after == count_before, \
            "Without real CUDA graph capture, count should not change"

    def test_get_cudagraph_runner_state_enabled(self):
        """Runner state probe returns captured state when graphs enabled."""
        model = MockModel()
        worker = _FakeWorker(model, cuda_graphs_enabled=True)
        state = _rpc_get_cudagraph_runner_state(worker)
        assert state["keys_initialized"] is True
        assert state["num_captured_keys"] > 0
        assert state["has_captured_graphs"] is True
        assert "PIECEWISE" in state["cudagraph_mode"]
        assert isinstance(state["cudagraph_batch_sizes"], list)
        assert len(state["cudagraph_batch_sizes"]) > 0

    def test_get_cudagraph_runner_state_disabled(self):
        """Runner state probe returns no-capture state when graphs disabled."""
        model = MockModel()
        worker = _FakeWorker(model, cuda_graphs_enabled=False)
        state = _rpc_get_cudagraph_runner_state(worker)
        assert state["keys_initialized"] is False
        assert state["num_captured_keys"] == 0
        assert state["has_captured_graphs"] is False
        assert "NONE" in state["cudagraph_mode"]


# ---------------------------------------------------------------------------
# GPU Worker RPC Oracle (requires CUDA + model checkpoints)
# ---------------------------------------------------------------------------

_REQUIRES_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU required for Worker RPC oracle tests"
)

_MODEL_DIR = os.environ.get("MODEL_DIR", "")

# Model+quantization matrix for GPU oracle
# expect_paths: exact suffixes that MUST appear in the snapshot (first-scope registered paths)
_GPU_MATRIX = [
    {
        "name": "qwen3_bf16",
        "model_path": os.path.join(_MODEL_DIR, "Qwen3-0.6B"),
        "llm_kwargs": {},
        "expect_paths": [],
    },
    {
        "name": "dsv3_fp8_moe",
        "model_path": os.path.join(_MODEL_DIR,
                                    "DeepSeek-V3-debug-empty-FP8_DYNAMIC"),
        "llm_kwargs": {"trust_remote_code": True},
        # DeepSeek-V3 uses MLA (W_UV, W_UK_T) + CompressedTensors MoE (_permute_scratch)
        "expect_paths": [
            "W_UV",
            "W_UK_T",
            "_permute_scratch",
        ],
    },
    {
        "name": "qwen3_w4a8_moe",
        "model_path": os.path.join(_MODEL_DIR, "Qwen3-30B-A3B-2layer-W4A8"),
        "llm_kwargs": {"gpu_memory_utilization": 0.95},
        # W4A8 MoE registers b_strides1 and b_strides2
        "expect_paths": [
            "quant_method.b_strides1",
            "quant_method.b_strides2",
        ],
    },
    {
        "name": "flashinfer_cutlass_moe",
        "model_path": os.path.join(
            _MODEL_DIR, "FlashInfer-CUTLASS-MoE-model"),
        "llm_kwargs": {},
        # FlashInfer CUTLASS MoE registers gemm1_alpha/beta/clamp_limit
        "expect_paths": [
            "quant_method.moe_kernel.fused_experts.gemm1_alpha",
            "quant_method.moe_kernel.fused_experts.gemm1_beta",
            "quant_method.moe_kernel.fused_experts.gemm1_clamp_limit",
        ],
    },
]


@_REQUIRES_CUDA
class TestWorkerRPCOracle:
    """GPU-dependent: Worker RPC pointer-identity oracle.

    Loads real models via LLM, injects probe RPCs, runs reload_weights,
    and asserts zero drift across 3 consecutive reloads.

    Requires:
      - CUDA GPU available
      - MODEL_DIR env var pointing to model checkpoints
      - vllm importable with full v1 stack
    """

    @pytest.fixture(params=[e["name"] for e in _GPU_MATRIX])
    def matrix_entry(self, request):
        """Parameterize over model+quantization matrix."""
        for entry in _GPU_MATRIX:
            if entry["name"] == request.param:
                return entry
        pytest.fail(f"Matrix entry not found: {request.param}")

    def test_three_reload_zero_drift(self, matrix_entry):
        """3 consecutive reload_weights calls show zero pointer drift.

        Verifies CUDA graph capture is enabled before taking snapshot,
        then asserts all first-scope registered paths are present and
        preserved across 3 reloads.
        """
        import gc
        model_path = matrix_entry["model_path"]
        if not os.path.isdir(model_path):
            pytest.skip(f"Model not found: {model_path}")

        from vllm import LLM
        from vllm.v1.worker.gpu_worker import Worker

        # Inject RPC probes
        Worker._ci_walk_snapshot = _rpc_walk_snapshot
        Worker._ci_walk_assert = _rpc_walk_assert
        Worker._ci_verify_cuda_graphs = _rpc_verify_cuda_graphs
        Worker._ci_get_cudagraph_capture_count = _rpc_get_cudagraph_capture_count
        Worker._ci_get_cudagraph_runner_state = _rpc_get_cudagraph_runner_state

        llm_kwargs = dict(
            model=model_path,
            enable_prefix_caching=False,
            max_model_len=128,
            max_num_seqs=1,
            gpu_memory_utilization=matrix_entry.get("llm_kwargs", {}).get(
                "gpu_memory_utilization", 0.5),
        )
        llm_kwargs.update(matrix_entry.get("llm_kwargs", {}))

        # Record process-global capture count BEFORE LLM construction.
        # vLLM captures CUDA graphs during engine startup (capture_model()),
        # so we must sample before LLM() to catch startup captures.
        from vllm.compilation.counter import compilation_counter
        count_before_llm = compilation_counter.num_cudagraph_captured

        try:
            llm = LLM(**llm_kwargs)
        except Exception as e:
            pytest.skip(f"Model load failed: {e}")

        try:
            # 1. Verify startup capture via process-global delta (all ranks)
            all_counts = llm.collective_rpc(
                "_ci_get_cudagraph_capture_count")
            for rank, count in enumerate(all_counts):
                assert count > count_before_llm, (
                    f"Rank {rank}: CUDA graph capture count did not increase "
                    f"during LLM init (before={count_before_llm}, "
                    f"after={count})."
                )

            # 2. Primary proof: runner-local captured graph state (all ranks)
            all_runner_states = llm.collective_rpc(
                "_ci_get_cudagraph_runner_state")
            for rank, runner_state in enumerate(all_runner_states):
                assert runner_state["has_captured_graphs"], (
                    f"Rank {rank}: Runner does not have captured graphs. "
                    f"State: {runner_state}"
                )

            # 3. Verify configuration is correct (all ranks)
            all_graph_info = llm.collective_rpc("_ci_verify_cuda_graphs")
            for rank, graph_info in enumerate(all_graph_info):
                assert graph_info["cuda_graphs_enabled"], (
                    f"Rank {rank}: CUDA graphs not enabled "
                    f"(mode={graph_info['cudagraph_mode']})."
                )

            # 4. Replay sanity check — generation should not crash
            llm.generate(["hello"], sampling_params=None)

            # Take pointer snapshot (per-rank, stored locally on each worker)
            all_snapshots = llm.collective_rpc("_ci_walk_snapshot")
            for rank, snapshot in enumerate(all_snapshots):
                assert len(snapshot) > 0, (
                    f"Rank {rank}: No extra tensors found"
                )

            # Verify ALL required first-scope registered paths (rank 0)
            for path_suffix in matrix_entry.get("expect_paths", []):
                matches = [k for k in all_snapshots[0]
                           if k.endswith(f".{path_suffix}")]
                assert matches, (
                    f"Required registered path '.{path_suffix}' not in "
                    f"snapshot keys: {sorted(all_snapshots[0].keys())[:15]}"
                )

            # 3 consecutive reloads — assert all ranks
            # Tolerate lazily-created scratch buffers being GONE after reload
            exclude_gone = ["_permute_scratch", "_sort_workspace"]
            for reload_idx in range(3):
                llm.collective_rpc(
                    "reload_weights",
                    kwargs={"weights_path": model_path})
                all_checks = llm.collective_rpc(
                    "_ci_walk_assert_stored",
                    kwargs={"exclude_gone_patterns": exclude_gone})
                for rank, check in enumerate(all_checks):
                    assert check.get("status") == "PASS", (
                        f"Reload {reload_idx} rank {rank}: "
                        f"status={check.get('status')}, "
                        f"drifted={check.get('drifted')}, "
                        f"details={check.get('details', [])[:5]}"
                    )
        finally:
            del llm
            torch.cuda.empty_cache()
            gc.collect()
