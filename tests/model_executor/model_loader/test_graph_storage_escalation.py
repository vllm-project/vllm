# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for graph_storage_escalation: strict mode, production warnings,
rate limiting, and address-set filtering.
"""
import logging
import os
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.model_loader.reload.graph_storage_escalation import (
    GraphStorageStrictError,
    _is_graph_relevant,
    escalate_walk_discoveries,
    is_strict_mode,
    reset_warned_paths,
)


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset module state before each test."""
    import vllm.model_executor.model_loader.reload.graph_storage_escalation as mod
    mod._STRICT_MODE = None
    reset_warned_paths()
    # Enable propagation on 'vllm' logger so pytest caplog captures output
    vllm_logger = logging.getLogger("vllm")
    old_propagate = vllm_logger.propagate
    vllm_logger.propagate = True
    yield
    vllm_logger.propagate = old_propagate
    mod._STRICT_MODE = None
    reset_warned_paths()


# ---------------------------------------------------------------------------
# Address-set filtering (graph-captured layer + address in set)
# ---------------------------------------------------------------------------

class TestAddressSetFiltering:
    def test_graph_captured_layer_with_addr_in_set_is_relevant(self):
        """Graph-captured layer + addr in set → relevant."""
        t = torch.randn(3)
        addr = t.untyped_storage().data_ptr()
        address_set = {addr}
        assert _is_graph_relevant(
            tensor=t, graph_address_set=address_set,
            layer_is_graph_captured=True)

    def test_non_graph_captured_layer_with_addr_in_set_not_relevant(self):
        """Non-captured layer → not relevant."""
        t = torch.randn(3)
        addr = t.untyped_storage().data_ptr()
        address_set = {addr}
        assert not _is_graph_relevant(
            tensor=t, graph_address_set=address_set,
            layer_is_graph_captured=False)

    def test_graph_captured_layer_addr_not_in_set_not_relevant(self):
        """Requires addr in set."""
        t = torch.randn(3)
        address_set = {99999999}  # bogus address
        assert not _is_graph_relevant(
            tensor=t, graph_address_set=address_set,
            layer_is_graph_captured=True)

    def test_none_address_set_skips_check(self):
        """When graph_address_set is None, not relevant."""
        t = torch.randn(3)
        assert not _is_graph_relevant(
            tensor=t, graph_address_set=None,
            layer_is_graph_captured=True)

    def test_none_tensor_not_relevant(self):
        """None tensor is never relevant."""
        assert not _is_graph_relevant(
            tensor=None, graph_address_set={123},
            layer_is_graph_captured=True)

    def test_address_set_strict_raises_graph_captured(self):
        """Strict raises for graph-captured layer with tensor in addr set."""
        layer = nn.Linear(4, 4)
        t = torch.randn(3)
        addr = t.untyped_storage().data_ptr()
        address_set = {addr}
        captured_layers = {layer}
        slots = [("some_tensor", t)]

        with patch.dict(os.environ,
                        {"VLLM_GRAPH_STORAGE_STRICT": "1"}):
            with pytest.raises(GraphStorageStrictError):
                escalate_walk_discoveries(
                    layer, slots,
                    graph_address_set=address_set,
                    graph_captured_layers=captured_layers)

    def test_address_set_strict_no_raise_non_captured(self, caplog):
        """Non-graph-captured layer does NOT raise in strict, only warns."""
        layer = nn.Linear(4, 4)
        t = torch.randn(3)
        addr = t.untyped_storage().data_ptr()
        address_set = {addr}
        captured_layers: set[nn.Module] = set()
        slots = [("some_tensor", t)]

        with patch.dict(os.environ,
                        {"VLLM_GRAPH_STORAGE_STRICT": "1"}):
            with caplog.at_level(logging.WARNING, logger="vllm.model_executor.model_loader.reload.graph_storage_escalation"):
                escalate_walk_discoveries(
                    layer, slots,
                    graph_address_set=address_set,
                    graph_captured_layers=captured_layers)
        assert "Unregistered" in caplog.text


# ---------------------------------------------------------------------------
# Production mode (default)
# ---------------------------------------------------------------------------

class TestProductionMode:
    def test_warns_for_unregistered_tensors(self, caplog):
        layer = nn.Linear(4, 4)
        t = torch.randn(3)
        slots = [("some_tensor", t)]
        with caplog.at_level(logging.WARNING, logger="vllm.model_executor.model_loader.reload.graph_storage_escalation"):
            escalate_walk_discoveries(layer, slots)
        assert "Unregistered graph-storage tensor" in caplog.text
        assert "some_tensor" in caplog.text
        assert "register_graph_storage" in caplog.text

    def test_empty_slots_no_warning(self, caplog):
        layer = nn.Linear(4, 4)
        with caplog.at_level(logging.WARNING, logger="vllm.model_executor.model_loader.reload.graph_storage_escalation"):
            escalate_walk_discoveries(layer, [])
        assert caplog.text == ""


# ---------------------------------------------------------------------------
# Strict mode
# ---------------------------------------------------------------------------

class TestStrictMode:
    def test_no_raise_without_graph_relevance(self, caplog):
        """Strict mode warns but does not raise for non-graph-relevant tensors."""
        layer = nn.Linear(4, 4)
        t = torch.randn(3)
        slots = [("some_tensor", t)]

        with patch.dict(os.environ,
                        {"VLLM_GRAPH_STORAGE_STRICT": "1"}):
            with caplog.at_level(logging.WARNING, logger="vllm.model_executor.model_loader.reload.graph_storage_escalation"):
                escalate_walk_discoveries(layer, slots)
        assert "Unregistered graph-storage tensor" in caplog.text

    def test_strict_raises_for_graph_relevant(self):
        """Strict mode raises for graph-relevant (addr-set match) tensors."""
        layer = nn.Linear(4, 4)
        t = torch.randn(3)
        addr = t.untyped_storage().data_ptr()
        slots = [("some_tensor", t)]

        with patch.dict(os.environ,
                        {"VLLM_GRAPH_STORAGE_STRICT": "1"}):
            with pytest.raises(GraphStorageStrictError,
                               match="graph-relevant"):
                escalate_walk_discoveries(
                    layer, slots,
                    graph_address_set={addr},
                    graph_captured_layers={layer})

    def test_strict_error_includes_all_violating_paths(self):
        """All graph-relevant paths are listed in the error."""
        layer = nn.Linear(4, 4)
        t1 = torch.randn(3)
        t2 = torch.randn(3)
        addr1 = t1.untyped_storage().data_ptr()
        addr2 = t2.untyped_storage().data_ptr()
        slots = [("tensor_a", t1), ("tensor_b", t2)]

        with patch.dict(os.environ,
                        {"VLLM_GRAPH_STORAGE_STRICT": "1"}):
            with pytest.raises(GraphStorageStrictError) as exc_info:
                escalate_walk_discoveries(
                    layer, slots,
                    graph_address_set={addr1, addr2},
                    graph_captured_layers={layer})
            assert "tensor_a" in str(exc_info.value)
            assert "tensor_b" in str(exc_info.value)
            assert "2 graph-relevant" in str(exc_info.value)

    def test_strict_not_rate_limited(self):
        """Strict violations always raise even after prior warning."""
        import vllm.model_executor.model_loader.reload.graph_storage_escalation as mod

        layer = nn.Linear(4, 4)
        t = torch.randn(3)
        addr = t.untyped_storage().data_ptr()
        slots = [("some_tensor", t)]

        # First call in production mode (no graph info → just warns)
        escalate_walk_discoveries(layer, slots)

        mod._STRICT_MODE = None

        # Now enable strict mode with graph info - should raise
        with patch.dict(os.environ,
                        {"VLLM_GRAPH_STORAGE_STRICT": "1"}):
            with pytest.raises(GraphStorageStrictError):
                escalate_walk_discoveries(
                    layer, slots,
                    graph_address_set={addr},
                    graph_captured_layers={layer})


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class TestRateLimiting:
    def test_same_path_warned_once(self, caplog):
        layer = nn.Linear(4, 4)
        t = torch.randn(3)
        slots = [("some_tensor", t)]

        with caplog.at_level(logging.WARNING, logger="vllm.model_executor.model_loader.reload.graph_storage_escalation"):
            escalate_walk_discoveries(layer, slots)
            first_count = len(caplog.records)
            escalate_walk_discoveries(layer, slots)
            assert len(caplog.records) == first_count

    def test_different_path_warned_separately(self, caplog):
        layer = nn.Linear(4, 4)
        slots1 = [("tensor_a", torch.randn(3))]
        slots2 = [("tensor_b", torch.randn(3))]

        with caplog.at_level(logging.WARNING, logger="vllm.model_executor.model_loader.reload.graph_storage_escalation"):
            escalate_walk_discoveries(layer, slots1)
            first_count = len(caplog.records)
            escalate_walk_discoveries(layer, slots2)
            assert len(caplog.records) > first_count

    def test_reset_allows_re_warning(self, caplog):
        layer = nn.Linear(4, 4)
        slots = [("some_tensor", torch.randn(3))]

        with caplog.at_level(logging.WARNING, logger="vllm.model_executor.model_loader.reload.graph_storage_escalation"):
            escalate_walk_discoveries(layer, slots)
            first_count = len(caplog.records)
            reset_warned_paths()
            escalate_walk_discoveries(layer, slots)
            assert len(caplog.records) > first_count


# ---------------------------------------------------------------------------
# Address-set collection coverage (verifies nested object walk semantics)
# ---------------------------------------------------------------------------

class TestAddressSetCollectionCoverage:
    """Verify that the recursive walk pattern finds tensors on nested objects."""

    def _collect_ptrs_recursive(self, module: nn.Module) -> set[int]:
        """Simplified version of GPUModelRunner._collect_graph_address_set."""
        import functools
        import types as _types

        ptrs: set[int] = set()
        visited: set[int] = set()

        def _walk_obj(obj, depth=0):
            if depth > 8:
                return
            if isinstance(obj, torch.Tensor):
                if obj.is_cuda or True:  # CPU for test
                    ptrs.add(obj.untyped_storage().data_ptr())
                return
            if isinstance(obj, nn.Module):
                return
            obj_id = id(obj)
            if obj_id in visited:
                return
            visited.add(obj_id)
            if isinstance(obj, dict):
                for v in obj.values():
                    if v is not None:
                        _walk_obj(v, depth + 1)
                return
            if isinstance(obj, (list, tuple)):
                for v in obj:
                    if v is not None:
                        _walk_obj(v, depth + 1)
                return
            if isinstance(obj, functools.partial):
                for arg in obj.args:
                    _walk_obj(arg, depth + 1)
                for v in obj.keywords.values():
                    _walk_obj(v, depth + 1)
                return
            if isinstance(obj, _types.FunctionType) and obj.__closure__:
                for cell in obj.__closure__:
                    try:
                        _walk_obj(cell.cell_contents, depth + 1)
                    except ValueError:
                        pass
                return
            obj_dict = getattr(obj, "__dict__", None)
            if obj_dict is None or isinstance(obj, type):
                return
            for val in obj_dict.values():
                if val is not None:
                    _walk_obj(val, depth + 1)

        for val in vars(module).values():
            if val is not None and not isinstance(val, nn.Module):
                _walk_obj(val)
        return ptrs

    def test_finds_nested_object_tensor(self):
        """Tensor on layer.quant_method.b_strides1 is found."""
        layer = nn.Linear(4, 4)

        class QuantMethod:
            pass

        qm = QuantMethod()
        qm.b_strides1 = torch.randn(3)
        layer.quant_method = qm

        ptrs = self._collect_ptrs_recursive(layer)
        expected_ptr = qm.b_strides1.untyped_storage().data_ptr()
        assert expected_ptr in ptrs

    def test_finds_deeply_nested_tensor(self):
        """Tensor on layer.quant_method.moe_kernel.fused_experts.gemm1_alpha."""
        layer = nn.Linear(4, 4)

        class Obj:
            pass

        qm = Obj()
        mk = Obj()
        fe = Obj()
        fe.gemm1_alpha = torch.randn(2)
        mk.fused_experts = fe
        qm.moe_kernel = mk
        layer.quant_method = qm

        ptrs = self._collect_ptrs_recursive(layer)
        expected_ptr = fe.gemm1_alpha.untyped_storage().data_ptr()
        assert expected_ptr in ptrs

    def test_finds_tensor_in_dict_attr(self):
        """Tensor stored in a dict attribute."""
        layer = nn.Linear(4, 4)
        layer.extra_data = {"workspace": torch.randn(5)}

        ptrs = self._collect_ptrs_recursive(layer)
        expected_ptr = layer.extra_data["workspace"].untyped_storage().data_ptr()
        assert expected_ptr in ptrs

    def test_finds_tensor_in_list_attr(self):
        """Tensor stored in a list attribute."""
        layer = nn.Linear(4, 4)
        t = torch.randn(3)
        layer.cached_tensors = [t]

        ptrs = self._collect_ptrs_recursive(layer)
        assert t.untyped_storage().data_ptr() in ptrs

    def test_skips_child_modules(self):
        """Child nn.Module attrs are not recursed into."""
        parent = nn.Linear(4, 4)
        child = nn.Linear(4, 4)
        child.secret_tensor = torch.randn(3)
        parent.child_module = child

        ptrs = self._collect_ptrs_recursive(parent)
        secret_ptr = child.secret_tensor.untyped_storage().data_ptr()
        assert secret_ptr not in ptrs
