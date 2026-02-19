# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for EncoderCudaGraphManager.

Test organization (all pure Python, no GPU required):
  - TestFindBudgetGraph      — greedy budget selection logic
  - TestCountInputPatches    — T*H*W patch counting
  - TestCountOutputTokens    — T*(H//m)*(W//m) output token counting
  - TestGenerateGridConfig   — dummy grid generation for graph capture
  - TestGetCumulativeStats   — hit/miss rate statistics
"""

import pytest

from vllm.v1.worker.gpu.mm.encoder_cudagraph import (
    EncoderCudaGraphManager,
    _count_input_patches,
    _count_output_tokens,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager_with_budgets(budgets: list[int]) -> EncoderCudaGraphManager:
    """Create a minimal EncoderCudaGraphManager with only token_budgets set.

    Skips the parts of __init__ that require a real VllmConfig / vision model
    by patching the attributes directly after construction.
    """
    mgr = object.__new__(EncoderCudaGraphManager)
    mgr.token_budgets = sorted(budgets)
    mgr.max_batch_size = 16
    mgr.use_dp = False
    mgr.budget_graphs = {}
    mgr.graph_hits = 0
    mgr.graph_misses = 0
    mgr.log_stats_interval = 100
    return mgr


# ---------------------------------------------------------------------------
# _find_budget_graph
# ---------------------------------------------------------------------------

class TestFindBudgetGraph:
    """Budget greedy selection: smallest budget >= total_tokens."""

    @pytest.mark.parametrize("total_tokens,budgets,expected", [
        # Exact match
        (2048, [2048, 4096, 8192], 2048),
        # Below smallest budget — picks smallest
        (100,  [2048, 4096, 8192], 2048),
        # Zero tokens — picks smallest
        (0,    [2048, 4096, 8192], 2048),
        # Between budgets — picks next one up
        (2049, [2048, 4096, 8192], 4096),
        (4097, [2048, 4096, 8192], 8192),
        # Exceeds all budgets — returns None (eager fallback)
        (9000, [2048, 4096, 8192], None),
        # Single budget, fits
        (1000, [2048], 2048),
        # Single budget, does not fit
        (3000, [2048], None),
    ])
    def test_find_budget(self, total_tokens, budgets, expected):
        mgr = _make_manager_with_budgets(budgets)
        result = mgr._find_budget_graph(total_tokens)
        assert result == expected

    def test_budgets_are_sorted(self):
        """Manager always sorts budgets ascending at init."""
        mgr = _make_manager_with_budgets([8192, 2048, 4096])
        assert mgr.token_budgets == [2048, 4096, 8192]
        # Budget selection still works correctly after sorting
        assert mgr._find_budget_graph(3000) == 4096


# ---------------------------------------------------------------------------
# Token counting helpers
# ---------------------------------------------------------------------------

class TestCountInputPatches:
    """_count_input_patches: T*H*W per image, no spatial merge."""

    @pytest.mark.parametrize("grid_thw_list,expected", [
        # Single image
        ([[1, 14, 14]], 196),
        # Two images
        ([[1, 14, 14], [1, 28, 28]], 196 + 784),
        # Video: T>1
        ([[2, 14, 14]], 2 * 14 * 14),
        # Mixed video and image
        ([[2, 8, 8], [1, 4, 4]], 2 * 64 + 16),
        # Empty batch
        ([], 0),
    ])
    def test_count_input_patches(self, grid_thw_list, expected):
        assert _count_input_patches(grid_thw_list) == expected


class TestCountOutputTokens:
    """_count_output_tokens: T*(H//m)*(W//m) per image, after spatial merge."""

    @pytest.mark.parametrize("grid_thw_list,spatial_merge_size,expected", [
        # Single image, merge=2: 1 * (14//2) * (14//2) = 49
        ([[1, 14, 14]], 2, 49),
        # Two images: 49 + 196
        ([[1, 14, 14], [1, 28, 28]], 2, 49 + 196),
        # No merge (merge=1): same as input patches
        ([[1, 14, 14]], 1, 196),
        # Video T=2: 2 * 7 * 7 = 98
        ([[2, 14, 14]], 2, 98),
        # Larger merge=4: 1 * 3 * 3 = 9
        ([[1, 12, 12]], 4, 9),
        # Empty batch
        ([], 2, 0),
    ])
    def test_count_output_tokens(self, grid_thw_list, spatial_merge_size, expected):
        assert _count_output_tokens(grid_thw_list, spatial_merge_size) == expected

    @pytest.mark.parametrize("spatial_merge_size", [1, 2, 4])
    def test_spatial_merge_reduces_by_exact_factor(self, spatial_merge_size):
        """When H and W are divisible by m, output_tokens * m^2 == input_patches."""
        m = spatial_merge_size
        # Use dimensions exactly divisible by m for each test
        grid_thw_list = [[1, 4 * m, 4 * m], [2, 6 * m, 6 * m]]
        output = _count_output_tokens(grid_thw_list, m)
        patches = _count_input_patches(grid_thw_list)
        assert output * (m * m) == patches

    def test_merge_1_equals_input_patches(self):
        """With merge=1, output tokens == input patches exactly."""
        grid_thw_list = [[1, 14, 14], [1, 28, 28]]
        assert _count_output_tokens(grid_thw_list, 1) == _count_input_patches(grid_thw_list)


# ---------------------------------------------------------------------------
# _generate_grid_config_for_budget
# ---------------------------------------------------------------------------

class TestGenerateGridConfig:
    """Grid config generation for CUDA graph capture."""

    @pytest.mark.parametrize("token_budget,max_batch_size,spatial_merge_size", [
        (2048, 16, 2),
        (4096, 8,  2),
        (8192, 16, 2),
        (13824, 16, 2),
        (1024, 4,  4),
    ])
    def test_grid_produces_exact_budget(
        self, token_budget, max_batch_size, spatial_merge_size, monkeypatch
    ):
        """The generated grid config produces exactly token_budget output tokens."""
        mgr = _make_manager_with_budgets([token_budget])
        mgr.max_batch_size = max_batch_size
        # Inject spatial_merge_size via a mock vision_model attribute
        mgr.vision_model = type("M", (), {"spatial_merge_size": spatial_merge_size})()

        grid_config = mgr._generate_grid_config_for_budget(token_budget, max_batch_size)

        assert len(grid_config) == max_batch_size
        # Each entry is [T, H, W]; output tokens = T * (H//m) * (W//m)
        m = spatial_merge_size
        total_output_tokens = sum(
            t * (h // m) * (w // m) for t, h, w in grid_config
        )
        assert total_output_tokens == token_budget

    def test_grid_entries_are_rectangular(self):
        """Grid uses the pattern [1, merge, per_image * merge] for each image."""
        mgr = _make_manager_with_budgets([2048])
        mgr.max_batch_size = 16
        mgr.vision_model = type("M", (), {"spatial_merge_size": 2})()

        grid_config = mgr._generate_grid_config_for_budget(2048, 16)

        # All images have T=1
        assert all(t == 1 for t, h, w in grid_config)
        # H equals spatial_merge_size
        assert all(h == 2 for t, h, w in grid_config)
        # All images are identical
        assert len(set(map(tuple, grid_config))) == 1


# ---------------------------------------------------------------------------
# get_cumulative_stats
# ---------------------------------------------------------------------------

class TestGetCumulativeStats:
    """Statistics tracking and reporting."""

    def test_initial_stats_are_zero(self):
        mgr = _make_manager_with_budgets([2048])
        stats = mgr.get_cumulative_stats()
        assert stats["graph_hits"] == 0
        assert stats["graph_misses"] == 0
        assert stats["hit_rate"] == 0.0

    def test_hit_rate_calculation(self):
        mgr = _make_manager_with_budgets([2048])
        mgr.graph_hits = 75
        mgr.graph_misses = 25
        stats = mgr.get_cumulative_stats()
        assert stats["graph_hits"] == 75
        assert stats["graph_misses"] == 25
        assert stats["hit_rate"] == pytest.approx(0.75)

    def test_all_hits(self):
        mgr = _make_manager_with_budgets([2048])
        mgr.graph_hits = 100
        mgr.graph_misses = 0
        assert mgr.get_cumulative_stats()["hit_rate"] == pytest.approx(1.0)

    def test_all_misses(self):
        mgr = _make_manager_with_budgets([2048])
        mgr.graph_hits = 0
        mgr.graph_misses = 50
        assert mgr.get_cumulative_stats()["hit_rate"] == pytest.approx(0.0)

    def test_stats_report_budget_info(self):
        budgets = [2048, 4096, 8192]
        mgr = _make_manager_with_budgets(budgets)
        stats = mgr.get_cumulative_stats()
        assert stats["num_budgets"] == 0  # no graphs captured yet
        assert stats["token_budgets"] == budgets
