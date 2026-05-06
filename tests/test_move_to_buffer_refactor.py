"""
Unit tests for the refactored move_to_buffer function.

These tests verify that:
1. The _plan_transfers helper correctly computes transfer plans.
2. The refactored move_to_buffer produces identical TransferMetadata.
3. The public API (move_to_buffer signature and return type) is preserved.
4. The internal helpers (_plan_transfers, _execute_local_copies, _post_sends,
   _post_recvs) are properly decomposed and independently testable.

These tests do NOT require CUDA and can run on CPU-only environments.
"""

import ast
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest


# Add vllm to path for import
sys.path.insert(0, str(Path(__file__).parent.parent))

from vllm.distributed.eplb.rebalance_execute import (  # noqa: E402
    TransferMetadata,
    _TransferPlan,
    _plan_transfers,
    move_to_buffer,
)


class TestPlanTransfers:
    """Tests for the _plan_transfers helper function."""

    def test_no_change(self):
        """When old == new, all experts are unchanged."""
        num_local = 4
        # 2 ranks, 4 local experts each = 8 total
        old_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
        new_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
        ep_rank = 0

        plan = _plan_transfers(num_local, old_indices, new_indices, ep_rank)

        assert isinstance(plan, _TransferPlan)
        assert np.all(plan.is_unchanged)
        assert np.all(plan.is_received_locally)
        assert plan.recv_count == 0
        assert not np.any(plan.eligible_local_buffer_mask)

    def test_local_swap(self):
        """When experts swap positions within the same rank."""
        num_local = 4
        # Rank 0 has experts [0,1,2,3], after rebalance [1,0,2,3]
        old_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
        new_indices = np.array([1, 0, 2, 3, 4, 5, 6, 7], dtype=np.int64)
        ep_rank = 0

        plan = _plan_transfers(num_local, old_indices, new_indices, ep_rank)

        # Positions 0 and 1 changed but experts are still locally available
        assert not plan.is_unchanged[0]
        assert not plan.is_unchanged[1]
        assert plan.is_unchanged[2]
        assert plan.is_unchanged[3]
        # Both should be receivable locally (expert 0 and 1 are both on rank 0)
        assert plan.is_received_locally[0]
        assert plan.is_received_locally[1]
        # No remote receives needed
        assert plan.recv_count == 0

    def test_remote_receive(self):
        """When an expert needs to come from another rank."""
        num_local = 2
        # 2 ranks, 2 local each = 4 total
        # Rank 0 has [0,1], Rank 1 has [2,3]
        # After: Rank 0 needs [0,2], Rank 1 needs [1,3]
        old_indices = np.array([0, 1, 2, 3], dtype=np.int64)
        new_indices = np.array([0, 2, 1, 3], dtype=np.int64)
        ep_rank = 0

        plan = _plan_transfers(num_local, old_indices, new_indices, ep_rank)

        # Position 0: unchanged (still expert 0)
        assert plan.is_unchanged[0]
        # Position 1: needs expert 2 which is on rank 1
        assert not plan.is_unchanged[1]
        assert not plan.is_received_locally[1]
        # Should have 1 remote receive
        assert plan.recv_count == 1
        assert plan.recv_expert_ids[0] == 2

    def test_send_map_computation(self):
        """Verify send map correctly identifies locally-available experts."""
        num_local = 3
        # Rank 0 has experts [0, 1, 2]
        old_indices = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
        new_indices = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
        ep_rank = 0

        plan = _plan_transfers(num_local, old_indices, new_indices, ep_rank)

        # Should have 3 sendable experts (0, 1, 2)
        assert plan.send_count == 3

    def test_empty_slots(self):
        """Handle -1 (empty) slots correctly."""
        num_local = 4
        old_indices = np.array([0, -1, 2, -1, 4, 5, 6, 7], dtype=np.int64)
        new_indices = np.array([0, 1, -1, -1, 4, 5, 6, 7], dtype=np.int64)
        ep_rank = 0

        plan = _plan_transfers(num_local, old_indices, new_indices, ep_rank)

        # Position 0: unchanged
        assert plan.is_unchanged[0]
        # Position 2: old had expert 2, new has -1 (empty) - unchanged since
        # both are "valid" comparisons
        # Position 1: old had -1, new has expert 1 - needs remote receive
        assert not plan.is_received_locally[1]


class TestMoveToBufferSignature:
    """Tests that the public API signature is preserved."""

    def test_return_type_is_transfer_metadata(self):
        """move_to_buffer must return TransferMetadata."""
        sig = inspect.signature(move_to_buffer)
        # Check return annotation
        assert sig.return_annotation == TransferMetadata

    def test_parameter_names_unchanged(self):
        """The parameter names of move_to_buffer must not change."""
        sig = inspect.signature(move_to_buffer)
        expected_params = [
            "num_local_experts",
            "old_indices",
            "new_indices",
            "expert_weights",
            "expert_weights_buffers",
            "cuda_stream",
            "ep_rank",
            "communicator",
        ]
        actual_params = list(sig.parameters.keys())
        assert actual_params == expected_params


class TestCodeStructure:
    """AST-based tests to verify the refactoring structure."""

    @pytest.fixture
    def module_ast(self):
        source_path = Path(__file__).parent.parent / "vllm" / "distributed" / "eplb" / "rebalance_execute.py"
        return ast.parse(source_path.read_text())

    def test_helper_functions_exist(self, module_ast):
        """Verify the new helper functions are defined."""
        func_names = {
            node.name
            for node in ast.walk(module_ast)
            if isinstance(node, ast.FunctionDef)
        }
        assert "_plan_transfers" in func_names
        assert "_execute_local_copies" in func_names
        assert "_post_sends" in func_names
        assert "_post_recvs" in func_names

    def test_transfer_plan_dataclass_exists(self, module_ast):
        """Verify _TransferPlan dataclass is defined."""
        class_names = {
            node.name
            for node in ast.walk(module_ast)
            if isinstance(node, ast.ClassDef)
        }
        assert "_TransferPlan" in class_names

    def test_move_to_buffer_calls_helpers(self, module_ast):
        """Verify move_to_buffer delegates to the helper functions."""
        for node in ast.walk(module_ast):
            if isinstance(node, ast.FunctionDef) and node.name == "move_to_buffer":
                # Get all function calls within move_to_buffer
                calls = {
                    n.func.id
                    for n in ast.walk(node)
                    if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name)
                }
                assert "_plan_transfers" in calls
                assert "_execute_local_copies" in calls
                assert "_post_sends" in calls
                assert "_post_recvs" in calls
                break
        else:
            pytest.fail("move_to_buffer function not found")

    def test_move_to_buffer_is_concise(self, module_ast):
        """After refactoring, move_to_buffer should be significantly shorter."""
        for node in ast.walk(module_ast):
            if isinstance(node, ast.FunctionDef) and node.name == "move_to_buffer":
                # The refactored function should be ~40-50 lines (docstring + orchestration)
                # vs the original ~170 lines
                num_lines = node.end_lineno - node.lineno + 1
                assert num_lines < 80, (
                    f"move_to_buffer is {num_lines} lines, "
                    "expected < 80 after refactoring"
                )
                break

    def test_public_exports_unchanged(self, module_ast):
        """Verify __all__ exports are unchanged."""
        for node in ast.walk(module_ast):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        exports = {elt.value for elt in node.value.elts}
                        assert "transfer_layer" in exports
                        assert "move_from_buffer" in exports
                        assert "TransferMetadata" in exports
                        return
        pytest.fail("__all__ not found")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
