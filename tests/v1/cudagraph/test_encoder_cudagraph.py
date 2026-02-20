# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for EncoderCudaGraphManager.

Test organization:
  No GPU required:
    - TestFindBudgetGraph      — greedy budget selection logic
    - TestCountInputPatches    — T*H*W patch counting
    - TestCountOutputTokens    — T*(H//m)*(W//m) output token counting
    - TestGenerateGridConfig   — dummy grid generation for graph capture
    - TestGetCumulativeStats   — hit/miss rate statistics
  GPU required:
    - TestEncoderCudaGraphCaptureReplay — capture, replay, fallback, counters, chunking
"""

import pytest
import torch

from vllm.platforms import current_platform
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
# _find_smallest_fitting_budget_given_tokens
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
        result = mgr._find_smallest_fitting_budget_given_tokens(total_tokens)
        assert result == expected

    def test_budgets_are_sorted(self):
        """Manager always sorts budgets ascending at init."""
        mgr = _make_manager_with_budgets([8192, 2048, 4096])
        assert mgr.token_budgets == [2048, 4096, 8192]
        # Budget selection still works correctly after sorting
        assert mgr._find_smallest_fitting_budget_given_tokens(3000) == 4096


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


# ---------------------------------------------------------------------------
# GPU fixtures and helpers
# ---------------------------------------------------------------------------

# Mock encoder parameters (kept small for fast capture)
_SPATIAL_MERGE = 2
_HIDDEN = 32
_PATCH_SIZE = 4           # H/W per patch in grid_thw units
_TEMPORAL_PATCH = 1
_IN_CHANNELS = 3
# flattened_patch_size = in_channels * temporal_patch * patch_size^2
_FLAT = _IN_CHANNELS * _TEMPORAL_PATCH * _PATCH_SIZE * _PATCH_SIZE  # 48

# Test budgets: small to keep capture fast
_BUDGETS = [16, 64]
_MAX_BATCH = 4


class SimpleMockViTEncoder(torch.nn.Module):
    """Minimal ViT encoder for CUDA graph tests.

    Implements the interface expected by EncoderCudaGraphManager:
      - spatial_merge_size, out_hidden_size        (attributes)
      - patch_embed.proj.in_channels, .patch_size, .temporal_patch_size
      - fast_pos_embed_interpolate(grid_thw_list)  → [n_out, hidden]
      - rot_pos_emb(grid_thw_list)                 → ([n_out, d], [n_out, d])
      - forward(pixel_values, grid_thw, encoder_metadata=None) → [n_out, hidden]

    Forward: project all input patches, then simulate spatial merge by
    averaging groups of m² patches → exactly token_budget output tokens.
    """

    spatial_merge_size = _SPATIAL_MERGE
    out_hidden_size = _HIDDEN

    def __init__(self):
        super().__init__()
        # Fake patch_embed namespace used by _prepare_dummy_inputs
        PE = type("PE", (), {
            "proj": type("P", (), {"in_channels": _IN_CHANNELS})(),
            "patch_size": _PATCH_SIZE,
            "temporal_patch_size": _TEMPORAL_PATCH,
        })()
        self.patch_embed = PE
        self.proj = torch.nn.Linear(_FLAT, _HIDDEN)

    def _n_out(self, grid_thw_list: list[list[int]]) -> int:
        m = self.spatial_merge_size
        return sum(t * (h // m) * (w // m) for t, h, w in grid_thw_list)

    def fast_pos_embed_interpolate(
        self, grid_thw_list: list[list[int]]
    ) -> torch.Tensor:
        p = next(self.parameters())
        return torch.zeros(self._n_out(grid_thw_list), _HIDDEN,
                           device=p.device, dtype=p.dtype)

    def rot_pos_emb(
        self, grid_thw_list: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        p = next(self.parameters())
        n = self._n_out(grid_thw_list)
        cos = torch.zeros(n, 16, device=p.device, dtype=p.dtype)
        sin = torch.zeros(n, 16, device=p.device, dtype=p.dtype)
        return cos, sin

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor | list[list[int]],
        encoder_metadata: dict | None = None,
    ) -> torch.Tensor:
        # pixel_values: [n_patches, _FLAT]
        # Simulate spatial merge: every m² input patches → 1 output token
        m2 = self.spatial_merge_size ** 2
        out = self.proj(pixel_values)       # [n_patches, hidden]
        n_out = out.shape[0] // m2
        return out[:n_out * m2].view(n_out, m2, _HIDDEN).mean(dim=1)


def _make_manager_for_gpu(
    encoder: torch.nn.Module,
    token_budgets: list[int],
    max_batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> EncoderCudaGraphManager:
    """Create EncoderCudaGraphManager bypassing VllmConfig for GPU tests."""
    mgr = object.__new__(EncoderCudaGraphManager)
    mgr.token_budgets = sorted(token_budgets)
    mgr.max_batch_size = max_batch_size
    mgr.use_dp = False
    mgr.budget_graphs = {}
    mgr.graph_hits = 0
    mgr.graph_misses = 0
    mgr.log_stats_interval = 100
    mgr.vision_model = encoder
    mgr.device = device
    mgr.dtype = dtype
    return mgr


def _make_pixel_values(
    grid_thw_list: list[list[int]],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Random pixel_values matching the total input patch count."""
    n = _count_input_patches(grid_thw_list)
    return torch.randn(n, _FLAT, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# GPU tests — capture, replay, fallback, counters, chunking
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not current_platform.is_cuda(), reason="Skip if not cuda")
class TestEncoderCudaGraphCaptureReplay:

    def setup_method(self):
        self.device = torch.device("cuda:0")
        self.dtype = torch.float16
        self.encoder = SimpleMockViTEncoder().to(self.device).half()
        self.mgr = _make_manager_for_gpu(
            self.encoder, _BUDGETS, _MAX_BATCH, self.device, self.dtype
        )
        self.mgr.capture()

    # --- capture ---

    def test_capture_creates_one_graph_per_budget(self):
        assert len(self.mgr.budget_graphs) == len(_BUDGETS)
        assert set(self.mgr.budget_graphs.keys()) == set(_BUDGETS)

    # --- output shape ---

    def test_execute_returns_one_tensor_per_image(self):
        grid_thw = [[1, 4, 4], [1, 4, 4]]
        pv = _make_pixel_values(grid_thw, self.device, self.dtype)
        result = self.mgr.execute(pv, grid_thw)
        assert result is not None
        assert len(result) == 2

    def test_execute_output_tokens_per_image(self):
        # [1,4,4] → 1*(4//2)*(4//2) = 4 tokens; [1,8,8] → 16 tokens
        grid_thw = [[1, 4, 4], [1, 8, 8]]
        pv = _make_pixel_values(grid_thw, self.device, self.dtype)
        result = self.mgr.execute(pv, grid_thw)
        assert result is not None
        assert result[0].shape == (4, _HIDDEN)
        assert result[1].shape == (16, _HIDDEN)

    # --- budget fallback ---

    def test_eager_fallback_when_tokens_exceed_all_budgets(self):
        # [1,18,18] → 1*(18//2)*(18//2) = 81 tokens > max budget 64.
        # Greedy packing handles the fallback internally: the oversized image
        # gets an eager forward pass and is returned as part of the output list
        # (execute() no longer returns None for individual image misses).
        grid_thw = [[1, 18, 18]]
        pv = _make_pixel_values(grid_thw, self.device, self.dtype)
        result = self.mgr.execute(pv, grid_thw)
        assert result is not None
        assert len(result) == 1
        # Eager output: SimpleMockViTEncoder produces n_out = 81 tokens
        assert result[0].shape == (81, _HIDDEN)
        assert self.mgr.graph_misses == 1

    # --- counters ---

    def test_hit_counter_increments_by_num_images(self):
        grid_thw = [[1, 4, 4], [1, 4, 4]]
        pv = _make_pixel_values(grid_thw, self.device, self.dtype)
        self.mgr.execute(pv, grid_thw)
        assert self.mgr.graph_hits == 2

    def test_miss_counter_increments_by_num_images(self):
        grid_thw = [[1, 18, 18]]   # 81 tokens > 64
        pv = _make_pixel_values(grid_thw, self.device, self.dtype)
        self.mgr.execute(pv, grid_thw)
        assert self.mgr.graph_misses == 1

    # --- chunking ---

    def test_chunking_when_images_exceed_max_batch(self):
        # 8 images > max_batch_size=4 → 2 chunks of 4
        # each chunk: 4 * 4 = 16 tokens → fits budget 16
        n_images = _MAX_BATCH * 2
        grid_thw = [[1, 4, 4]] * n_images
        pv = _make_pixel_values(grid_thw, self.device, self.dtype)
        result = self.mgr.execute(pv, grid_thw)
        assert result is not None
        assert len(result) == n_images
        for out in result:
            assert out.shape == (4, _HIDDEN)
