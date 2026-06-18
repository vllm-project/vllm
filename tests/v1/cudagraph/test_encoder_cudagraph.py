# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for EncoderCudaGraphManager.

Test organization:
  No GPU required:
    - TestFindBudgetGraph      — greedy budget selection logic
    - TestGetCumulativeStats   — hit/miss rate statistics
    - TestGetInputModality     — modality routing from mm_kwargs keys
  GPU required:
    - TestEncoderCudaGraphCaptureReplay — capture, replay, fallback, counters, chunking
    - TestEncoderCudaGraphVideoReplay   — video modality capture, replay
"""

from typing import Any

import pytest
import torch

from vllm.model_executor.models.interfaces import SupportsEncoderCudaGraph
from vllm.platforms import current_platform
from vllm.v1.worker.encoder_cudagraph import (
    EncoderCudaGraphManager,
)
from vllm.v1.worker.encoder_cudagraph_defs import (
    EncoderCudaGraphCaptureInputs,
    EncoderCudaGraphConfig,
    EncoderCudaGraphReplayBuffers,
    EncoderItemSpec,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockCompilationConfig:
    """Minimal mock for VllmConfig.compilation_config."""

    def __init__(
        self,
        token_budgets: list[int] | None = None,
        max_mm_items: int = 0,
    ):
        self.encoder_cudagraph_token_budgets = token_budgets or []
        self.encoder_cudagraph_max_vision_items_per_batch = max_mm_items
        self.encoder_cudagraph_max_frames_per_batch = None


class _MockMultimodalConfig:
    mm_encoder_tp_mode = "replicate"

    def get_limit_per_prompt(self, modality: str) -> int:
        # Image-only mocks — return 0 for "video" to short-circuit the
        # max_frames_per_batch branch, so tests don't need a video-frame mock.
        return 0


class _MockModelConfig:
    multimodal_config = _MockMultimodalConfig()


class _MockParallelConfig:
    tensor_parallel_size = 1


class _MockVllmConfig:
    """Minimal mock for VllmConfig used in __init__ tests."""

    def __init__(
        self,
        token_budgets: list[int] | None = None,
        max_mm_items: int = 0,
    ):
        self.compilation_config = _MockCompilationConfig(token_budgets, max_mm_items)
        self.model_config = _MockModelConfig()
        self.parallel_config = _MockParallelConfig()


class _MockModel(SupportsEncoderCudaGraph):
    """Minimal mock implementing SupportsEncoderCudaGraph for __init__."""

    def __init__(self, min_budget: int = 4, max_budget: int = 128):
        self._min_budget = min_budget
        self._max_budget = max_budget

    def get_encoder_cudagraph_config(self) -> EncoderCudaGraphConfig:
        return EncoderCudaGraphConfig(
            modalities=["image"],
            buffer_keys=[
                "pixel_values",
                "dummy_buf",
            ],
            out_hidden_size=32,
        )

    def get_encoder_cudagraph_budget_range(self, vllm_config):
        return (self._min_budget, self._max_budget)


def _make_manager_with_budgets(budgets: list[int]) -> EncoderCudaGraphManager:
    """Create a minimal EncoderCudaGraphManager with only token_budgets set.

    Skips the parts of __init__ that require a real VllmConfig / model
    by patching the attributes directly after construction.
    """
    mgr = object.__new__(EncoderCudaGraphManager)
    mgr.token_budgets = sorted(budgets)
    mgr.max_batch_size = 16
    mgr.use_dp = False
    mgr.budget_graphs = {"default": {}}
    mgr.graph_pool = None
    mgr.graph_hits = 0
    mgr.graph_misses = 0
    mgr.log_stats_interval = 100
    return mgr


# ---------------------------------------------------------------------------
# _generate_budgets
# ---------------------------------------------------------------------------


class TestGenerateBudgets:
    """Auto-generate power-of-2 budgets from min to max."""

    def test_exact_powers_of_2(self):
        result = EncoderCudaGraphManager._generate_budgets(64, 1024)
        assert result == [64, 128, 256, 512, 1024]

    def test_max_not_power_of_2(self):
        result = EncoderCudaGraphManager._generate_budgets(64, 800)
        assert result == [64, 128, 256, 512, 800]

    def test_min_equals_max(self):
        result = EncoderCudaGraphManager._generate_budgets(64, 64)
        assert result == [64]

    def test_large_range(self):
        result = EncoderCudaGraphManager._generate_budgets(64, 8192)
        assert result == [64, 128, 256, 512, 1024, 2048, 4096, 8192]


# ---------------------------------------------------------------------------
# _find_smallest_fitting_budget_given_tokens
# ---------------------------------------------------------------------------


class TestFindBudgetGraph:
    """Budget greedy selection: smallest budget >= total_tokens."""

    @pytest.mark.parametrize(
        "total_tokens,budgets,expected",
        [
            # Exact match
            (2048, [2048, 4096, 8192], 2048),
            # Below smallest budget — picks smallest
            (100, [2048, 4096, 8192], 2048),
            # Zero tokens — picks smallest
            (0, [2048, 4096, 8192], 2048),
            # Between budgets — picks next one up
            (2049, [2048, 4096, 8192], 4096),
            (4097, [2048, 4096, 8192], 8192),
            # Exceeds all budgets — returns None (eager fallback)
            (9000, [2048, 4096, 8192], None),
            # Single budget, fits
            (1000, [2048], 2048),
            # Single budget, does not fit
            (3000, [2048], None),
        ],
    )
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

    def test_num_graphs_to_capture_tracks_budgets(self):
        mgr = _make_manager_with_budgets([8192, 2048, 4096])
        assert mgr.get_num_graphs_to_capture() == 3


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
_PATCH_SIZE = 4  # H/W per patch in grid_thw units
_TEMPORAL_PATCH = 1
_IN_CHANNELS = 3
# flattened_patch_size = in_channels * temporal_patch * patch_size^2
_FLAT = _IN_CHANNELS * _TEMPORAL_PATCH * _PATCH_SIZE * _PATCH_SIZE  # 48

# Test budgets: small to keep capture fast
_BUDGETS = [16, 64]
_MAX_BATCH = 4


def _count_input_patches(grid_thw_list: list[list[int]]) -> int:
    return sum(t * h * w for t, h, w in grid_thw_list)


def _count_output_tokens(
    grid_thw_list: list[list[int]], spatial_merge_size: int
) -> int:
    m = spatial_merge_size
    return sum(t * (h // m) * (w // m) for t, h, w in grid_thw_list)


class SimpleMockViTModel(torch.nn.Module, SupportsEncoderCudaGraph):
    """Minimal ViT model for CUDA graph tests.

    Implements the SupportsEncoderCudaGraph protocol by providing
    all required methods. The forward pass projects patches and
    simulates spatial merge by averaging groups of m^2 patches.
    """

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(_FLAT, _HIDDEN)
        self.spatial_merge_size = _SPATIAL_MERGE
        self.out_hidden_size = _HIDDEN

    def get_encoder_cudagraph_config(self) -> EncoderCudaGraphConfig:
        return EncoderCudaGraphConfig(
            modalities=["image"],
            buffer_keys=["dummy_buf"],
            out_hidden_size=_HIDDEN,
        )

    def get_input_modality(
        self,
        mm_kwargs: dict[str, Any],
    ) -> str:
        return "image"

    def get_encoder_cudagraph_budget_range(
        self,
        vllm_config,
    ) -> tuple[int, int]:
        # For tests: min=4, max=128 (small values for fast capture)
        return (4, 128)

    def get_encoder_cudagraph_item_specs(
        self,
        mm_kwargs: dict[str, Any],
    ) -> list[EncoderItemSpec]:
        m = _SPATIAL_MERGE
        return [
            EncoderItemSpec(
                input_size=t * h * w,
                output_tokens=t * (h // m) * (w // m),
            )
            for t, h, w in mm_kwargs["image_grid_thw"]
        ]

    def select_encoder_cudagraph_items(
        self,
        mm_kwargs: dict[str, Any],
        indices: list[int],
    ) -> dict[str, Any]:
        grid_thw = mm_kwargs["image_grid_thw"]
        pixel_values = mm_kwargs["pixel_values"]

        if len(indices) == 0:
            return {
                "pixel_values": pixel_values[:0],
                "image_grid_thw": [],
            }

        patches_per_item = [t * h * w for t, h, w in grid_thw]
        cum_patches = [0]
        for p in patches_per_item:
            cum_patches.append(cum_patches[-1] + p)

        selected_pv = torch.cat(
            [pixel_values[cum_patches[i] : cum_patches[i + 1]] for i in indices]
        )
        selected_grid = [grid_thw[i] for i in indices]
        return {
            "pixel_values": selected_pv,
            "image_grid_thw": selected_grid,
        }

    def prepare_encoder_cudagraph_capture_inputs(
        self,
        token_budget: int,
        max_batch_size: int,
        max_frames_per_batch: int,
        device: torch.device,
        dtype: torch.dtype,
        path: str = "default",
    ) -> EncoderCudaGraphCaptureInputs:
        per_image_output = token_budget // max_batch_size
        grid_config = [
            [1, _SPATIAL_MERGE, per_image_output * _SPATIAL_MERGE]
            for _ in range(max_batch_size)
        ]
        total_patches = _count_input_patches(grid_config)
        dummy_pixel_values = torch.randn(
            total_patches, _FLAT, device=device, dtype=dtype
        )
        n_out = _count_output_tokens(grid_config, _SPATIAL_MERGE)
        dummy_buf = torch.zeros(n_out, _HIDDEN, device=device, dtype=dtype)
        return EncoderCudaGraphCaptureInputs(
            values={
                "pixel_values": dummy_pixel_values,
                "dummy_buf": dummy_buf,
            },
        )

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, Any],
        max_batch_size: int,
        max_frames_per_batch: int,
        path: str = "default",
    ) -> EncoderCudaGraphReplayBuffers:
        grid_thw = mm_kwargs["image_grid_thw"]
        n_out = _count_output_tokens(grid_thw, _SPATIAL_MERGE)
        p = next(self.parameters())
        dummy_buf = torch.zeros(n_out, _HIDDEN, device=p.device, dtype=p.dtype)
        return EncoderCudaGraphReplayBuffers(
            values={
                "pixel_values": mm_kwargs["pixel_values"],
                "dummy_buf": dummy_buf,
            }
        )

    def encoder_cudagraph_forward(
        self,
        values: dict[str, torch.Tensor],
        path: str = "default",
    ) -> torch.Tensor:
        return self._forward(values["pixel_values"])

    def encoder_eager_forward(
        self,
        mm_kwargs: dict[str, Any],
        path: str = "default",
    ) -> torch.Tensor:
        return self._forward(mm_kwargs["pixel_values"])

    def _forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        m2 = _SPATIAL_MERGE**2
        out = self.proj(pixel_values)
        n_out = out.shape[0] // m2
        return out[: n_out * m2].view(n_out, m2, _HIDDEN).mean(dim=1)


def _make_manager_for_gpu(
    model: SimpleMockViTModel,
    token_budgets: list[int],
    max_batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    max_frames_per_batch: int | None = None,
) -> EncoderCudaGraphManager:
    """Create EncoderCudaGraphManager bypassing VllmConfig for GPU tests."""
    mgr = object.__new__(EncoderCudaGraphManager)
    mgr.token_budgets = sorted(token_budgets)
    mgr.max_batch_size = max_batch_size
    mgr.max_frames_per_batch = (
        max_frames_per_batch if max_frames_per_batch is not None else max_batch_size * 2
    )
    mgr.use_dp = False
    mgr.budget_graphs = {"default": {}}
    mgr.graph_pool = None
    mgr.graph_hits = 0
    mgr.graph_misses = 0
    mgr.log_stats_interval = 100
    mgr.model = model
    mgr.config = model.get_encoder_cudagraph_config()
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


def _make_mm_kwargs(
    grid_thw_list: list[list[int]],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Create mm_kwargs for testing."""
    return {
        "pixel_values": _make_pixel_values(grid_thw_list, device, dtype),
        "image_grid_thw": grid_thw_list,
    }


def _make_video_mm_kwargs(
    grid_thw_list: list[list[int]],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Create video mm_kwargs (pixel_values_videos / video_grid_thw) for testing."""
    return {
        "pixel_values_videos": _make_pixel_values(grid_thw_list, device, dtype),
        "video_grid_thw": grid_thw_list,
    }


# ---------------------------------------------------------------------------
# GPU tests — capture, replay, fallback, counters, chunking
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Skip if not cuda")
class TestEncoderCudaGraphCaptureReplay:
    def setup_method(self):
        self.device = torch.device("cuda:0")
        self.dtype = torch.float16
        self.model = SimpleMockViTModel().to(self.device).half()
        self.mgr = _make_manager_for_gpu(
            self.model, _BUDGETS, _MAX_BATCH, self.device, self.dtype
        )
        self.graph_pool = current_platform.graph_pool_handle()
        self.mgr.capture(graph_pool=self.graph_pool)

    # --- capture ---

    def test_capture_creates_one_graph_per_budget(self):
        assert len(self.mgr.budget_graphs["default"]) == len(_BUDGETS)
        assert set(self.mgr.budget_graphs["default"].keys()) == set(_BUDGETS)

    def test_capture_uses_supplied_graph_pool(self):
        assert self.mgr.graph_pool is self.graph_pool

    def test_clear_releases_graphs_and_pool(self):
        self.mgr.clear()
        assert self.mgr.budget_graphs == {"default": {}}
        assert self.mgr.graph_pool is None

    # --- output shape ---

    def test_execute_returns_one_tensor_per_image(self):
        grid_thw = [[1, 4, 4], [1, 4, 4]]
        mm_kwargs = _make_mm_kwargs(grid_thw, self.device, self.dtype)
        result = self.mgr.execute(mm_kwargs)
        assert result is not None
        assert len(result) == 2

    def test_execute_output_tokens_per_image(self):
        # [1,4,4] → 1*(4//2)*(4//2) = 4 tokens; [1,8,8] → 16 tokens
        grid_thw = [[1, 4, 4], [1, 8, 8]]
        mm_kwargs = _make_mm_kwargs(grid_thw, self.device, self.dtype)
        result = self.mgr.execute(mm_kwargs)
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
        mm_kwargs = _make_mm_kwargs(grid_thw, self.device, self.dtype)
        result = self.mgr.execute(mm_kwargs)
        assert result is not None
        assert len(result) == 1
        # Eager output: SimpleMockViTModel produces n_out = 81 tokens
        assert result[0].shape == (81, _HIDDEN)
        assert self.mgr.graph_misses == 1

    # --- counters ---

    def test_hit_counter_increments_by_num_images(self):
        grid_thw = [[1, 4, 4], [1, 4, 4]]
        mm_kwargs = _make_mm_kwargs(grid_thw, self.device, self.dtype)
        self.mgr.execute(mm_kwargs)
        assert self.mgr.graph_hits == 2

    def test_miss_counter_increments_by_num_images(self):
        grid_thw = [[1, 18, 18]]  # 81 tokens > 64
        mm_kwargs = _make_mm_kwargs(grid_thw, self.device, self.dtype)
        self.mgr.execute(mm_kwargs)
        assert self.mgr.graph_misses == 1

    # --- chunking ---

    def test_chunking_when_images_exceed_max_batch(self):
        # 8 images > max_batch_size=4 → 2 chunks of 4
        # each chunk: 4 * 4 = 16 tokens → fits budget 16
        n_images = _MAX_BATCH * 2
        grid_thw = [[1, 4, 4]] * n_images
        mm_kwargs = _make_mm_kwargs(grid_thw, self.device, self.dtype)
        result = self.mgr.execute(mm_kwargs)
        assert result is not None
        assert len(result) == n_images
        for out in result:
            assert out.shape == (4, _HIDDEN)


# ---------------------------------------------------------------------------
# SimpleMockViTVideoModel — extends SimpleMockViTModel with video support
# ---------------------------------------------------------------------------


class SimpleMockViTVideoModel(SimpleMockViTModel):
    """ViT mock that supports both image and video modalities.

    Reuses SimpleMockViTModel's NN weights and _forward() logic.
    Only the protocol methods that are key-dependent are overridden.
    """

    def get_encoder_cudagraph_config(self) -> EncoderCudaGraphConfig:
        return EncoderCudaGraphConfig(
            modalities=["image", "video"],
            buffer_keys=["dummy_buf"],
            out_hidden_size=_HIDDEN,
        )

    def get_input_modality(self, mm_kwargs: dict[str, Any]) -> str:
        return "video" if "video_grid_thw" in mm_kwargs else "image"

    # ------------------------------------------------------------------
    # Private helpers — route to the correct mm_kwargs keys
    # ------------------------------------------------------------------

    def _get_grid_thw(self, mm_kwargs: dict[str, Any]) -> list[list[int]]:
        key = (
            "video_grid_thw"
            if self.get_input_modality(mm_kwargs) == "video"
            else "image_grid_thw"
        )
        return mm_kwargs[key]

    def _get_pixel_values(self, mm_kwargs: dict[str, Any]) -> torch.Tensor:
        key = (
            "pixel_values_videos"
            if self.get_input_modality(mm_kwargs) == "video"
            else "pixel_values"
        )
        return mm_kwargs[key]

    # ------------------------------------------------------------------
    # Protocol overrides that depend on modality keys
    # ------------------------------------------------------------------

    def get_encoder_cudagraph_item_specs(
        self,
        mm_kwargs: dict[str, Any],
    ) -> list[EncoderItemSpec]:
        m = _SPATIAL_MERGE
        return [
            EncoderItemSpec(
                input_size=t * h * w,
                output_tokens=t * (h // m) * (w // m),
            )
            for t, h, w in self._get_grid_thw(mm_kwargs)
        ]

    def select_encoder_cudagraph_items(
        self, mm_kwargs: dict[str, Any], indices: list[int]
    ) -> dict[str, Any]:
        modality = self.get_input_modality(mm_kwargs)
        pv_key = "pixel_values_videos" if modality == "video" else "pixel_values"
        grid_key = "video_grid_thw" if modality == "video" else "image_grid_thw"

        grid_thw = self._get_grid_thw(mm_kwargs)
        pixel_values = self._get_pixel_values(mm_kwargs)

        if len(indices) == 0:
            return {pv_key: pixel_values[:0], grid_key: []}

        patches_per_item = [t * h * w for t, h, w in grid_thw]
        cum_patches = [0]
        for p in patches_per_item:
            cum_patches.append(cum_patches[-1] + p)

        selected_pv = torch.cat(
            [pixel_values[cum_patches[i] : cum_patches[i + 1]] for i in indices]
        )
        return {pv_key: selected_pv, grid_key: [grid_thw[i] for i in indices]}

    def prepare_encoder_cudagraph_capture_inputs(
        self,
        token_budget: int,
        max_batch_size: int,
        max_frames_per_batch: int,
        device: torch.device,
        dtype: torch.dtype,
        path: str = "default",
    ) -> EncoderCudaGraphCaptureInputs:
        per_item_output = token_budget // max_batch_size
        frames_per_item = max_frames_per_batch // max_batch_size
        if frames_per_item > 1:
            # Video-format capture: size cu_seqlens for T frames per item.
            tokens_per_frame = (
                per_item_output + frames_per_item - 1
            ) // frames_per_item
            grid_config = [
                [frames_per_item, _SPATIAL_MERGE, tokens_per_frame * _SPATIAL_MERGE]
                for _ in range(max_batch_size)
            ]
        else:
            grid_config = [
                [1, _SPATIAL_MERGE, per_item_output * _SPATIAL_MERGE]
                for _ in range(max_batch_size)
            ]
        total_patches = _count_input_patches(grid_config)
        # Use pixel_values (image key) for capture — same patch shape as video.
        dummy_pixel_values = torch.randn(
            total_patches, _FLAT, device=device, dtype=dtype
        )
        n_out = _count_output_tokens(grid_config, _SPATIAL_MERGE)
        dummy_buf = torch.zeros(n_out, _HIDDEN, device=device, dtype=dtype)
        return EncoderCudaGraphCaptureInputs(
            values={
                "pixel_values": dummy_pixel_values,
                "dummy_buf": dummy_buf,
            },
        )

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, Any],
        max_batch_size: int,
        max_frames_per_batch: int,
        path: str = "default",
    ) -> EncoderCudaGraphReplayBuffers:
        n_out = _count_output_tokens(self._get_grid_thw(mm_kwargs), _SPATIAL_MERGE)
        p = next(self.parameters())
        dummy_buf = torch.zeros(n_out, _HIDDEN, device=p.device, dtype=p.dtype)
        return EncoderCudaGraphReplayBuffers(
            values={
                "pixel_values": self._get_pixel_values(mm_kwargs),
                "dummy_buf": dummy_buf,
            }
        )

    def encoder_cudagraph_forward(
        self,
        values: dict[str, torch.Tensor],
        path: str = "default",
    ) -> torch.Tensor:
        return self._forward(values["pixel_values"])

    def encoder_eager_forward(
        self,
        mm_kwargs: dict[str, Any],
        path: str = "default",
    ) -> torch.Tensor:
        return self._forward(self._get_pixel_values(mm_kwargs))


# ---------------------------------------------------------------------------
# No-GPU tests — get_input_modality routing
# ---------------------------------------------------------------------------


class TestGetInputModality:
    """get_input_modality returns correct modality based on mm_kwargs keys."""

    def test_image_only_model_always_returns_image(self):
        model = SimpleMockViTModel()
        mm_kwargs = {
            "pixel_values": torch.zeros(1, _FLAT),
            "image_grid_thw": [[1, 4, 4]],
        }
        assert model.get_input_modality(mm_kwargs) == "image"

    def test_video_model_returns_image_for_image_kwargs(self):
        model = SimpleMockViTVideoModel()
        mm_kwargs = {
            "pixel_values": torch.zeros(1, _FLAT),
            "image_grid_thw": [[1, 4, 4]],
        }
        assert model.get_input_modality(mm_kwargs) == "image"

    def test_video_model_returns_video_for_video_kwargs(self):
        model = SimpleMockViTVideoModel()
        mm_kwargs = {
            "pixel_values_videos": torch.zeros(8, _FLAT),
            "video_grid_thw": [[2, 4, 4]],
        }
        assert model.get_input_modality(mm_kwargs) == "video"


# ---------------------------------------------------------------------------
# GPU tests — DeepSeek-VL2 batched-tile encoder CUDA graph
# ---------------------------------------------------------------------------

# DeepSeek-VL2 mock constants (kept small for fast capture)
_DSV2_H = 2  # projector output side per tile (h = w)
_DSV2_DIM = 16  # hidden size
_DSV2_IMG_SIZE = 4  # each tile is 4×4 pixels
_DSV2_BUDGETS = [16, 64]
_DSV2_MAX_BATCH = 4


def _dsv2_output_tokens(tw: int, th: int) -> int:
    """Expected output tokens for an image with (tw, th) local tiles."""
    h = w = _DSV2_H
    return h * (w + 1) + th * h * (tw * w + 1) + 1


def _make_deepseek_vl2_mm_kwargs(
    spatial_crops: list[list[int]],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """spatial_crops: [[tw, th], ...] per image."""
    total_tiles = sum(1 + int(tw) * int(th) for tw, th in spatial_crops)
    pixel_values = torch.randn(
        total_tiles, 3, _DSV2_IMG_SIZE, _DSV2_IMG_SIZE, device=device, dtype=dtype
    )
    return {
        "pixel_values": pixel_values,
        "images_spatial_crop": torch.tensor(
            spatial_crops, dtype=torch.long, device=device
        ),
    }


class MockDeepseekVL2Model(torch.nn.Module, SupportsEncoderCudaGraph):
    """Mock of DeepseekVLV2ForCausalLM for CUDA graph tests.

    Simulates the batched-tile ViT pattern: pixel_values is
    [N_tiles, 3, H, W], output is assembled with newlines and
    view-separator in postprocess_encoder_output.
    """

    def __init__(self):
        super().__init__()
        hw = _DSV2_H * _DSV2_H
        in_features = 3 * _DSV2_IMG_SIZE * _DSV2_IMG_SIZE
        self.proj = torch.nn.Linear(in_features, hw * _DSV2_DIM)
        self.image_newline = torch.nn.Parameter(torch.randn(_DSV2_DIM))
        self.view_seperator = torch.nn.Parameter(torch.randn(_DSV2_DIM))
        self.global_view_pos = "head"

    def get_max_frames_per_video(self) -> int:
        return 1

    def get_encoder_cudagraph_config(self) -> EncoderCudaGraphConfig:
        return EncoderCudaGraphConfig(
            modalities=["image"],
            buffer_keys=["pixel_values"],
            out_hidden_size=_DSV2_DIM,
        )

    def get_encoder_cudagraph_budget_range(self, vllm_config) -> tuple[int, int]:
        return (_dsv2_output_tokens(1, 1), 128)

    def get_encoder_cudagraph_item_specs(
        self, mm_kwargs: dict[str, Any]
    ) -> list[EncoderItemSpec]:
        h = w = _DSV2_H
        specs = []
        for row in mm_kwargs["images_spatial_crop"].tolist():
            tw, th = int(row[0]), int(row[1])
            if tw == 0 or th == 0:
                break
            specs.append(
                EncoderItemSpec(
                    input_size=1 + tw * th,
                    output_tokens=h * (w + 1) + th * h * (tw * w + 1) + 1,
                )
            )
        return specs

    def select_encoder_cudagraph_items(
        self, mm_kwargs: dict[str, Any], indices: list[int]
    ) -> dict[str, Any]:
        pixel_values = mm_kwargs["pixel_values"]
        images_spatial_crop = mm_kwargs["images_spatial_crop"]

        if not indices:
            return {
                "pixel_values": pixel_values[:0],
                "images_spatial_crop": images_spatial_crop[:0],
            }

        tiles_per_image = [
            1 + int(row[0]) * int(row[1]) for row in images_spatial_crop.tolist()
        ]
        cum = [0]
        for t in tiles_per_image:
            cum.append(cum[-1] + t)

        selected_pv = torch.cat([pixel_values[cum[i] : cum[i + 1]] for i in indices])
        return {
            "pixel_values": selected_pv,
            "images_spatial_crop": images_spatial_crop[list(indices)],
        }

    def prepare_encoder_cudagraph_capture_inputs(
        self,
        token_budget: int,
        max_batch_size: int,
        max_frames_per_batch: int,
        device: torch.device,
        dtype: torch.dtype,
        path: str = "default",
    ) -> EncoderCudaGraphCaptureInputs:
        hw = _DSV2_H * _DSV2_H
        max_tiles = (token_budget + hw - 1) // hw
        dummy = torch.randn(
            max_tiles, 3, _DSV2_IMG_SIZE, _DSV2_IMG_SIZE, device=device, dtype=dtype
        )
        return EncoderCudaGraphCaptureInputs(values={"pixel_values": dummy})

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, Any],
        max_batch_size: int,
        max_frames_per_batch: int,
        path: str = "default",
    ) -> EncoderCudaGraphReplayBuffers:
        return EncoderCudaGraphReplayBuffers(
            values={"pixel_values": mm_kwargs["pixel_values"]}
        )

    def _tile_forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Project tiles: [N, 3, H, W] → [N, hw, DIM]."""
        n = pixel_values.shape[0]
        flat = pixel_values.view(n, -1)
        hw = _DSV2_H * _DSV2_H
        return self.proj(flat).view(n, hw, _DSV2_DIM)

    def encoder_cudagraph_forward(
        self, values: dict[str, torch.Tensor], path: str = "default"
    ) -> torch.Tensor:
        return self._tile_forward(values["pixel_values"])

    def encoder_eager_forward(
        self, mm_kwargs: dict[str, Any], path: str = "default"
    ) -> torch.Tensor:
        # Eager path uses scatter_output_slices (not postprocess_encoder_output),
        # so we must return a flat [total_tokens, dim] assembled tensor.
        tile_feats = self._tile_forward(mm_kwargs["pixel_values"])
        specs = self.get_encoder_cudagraph_item_specs(mm_kwargs)
        dest: dict[int, torch.Tensor] = {}
        self.postprocess_encoder_output(
            tile_feats,
            list(range(len(specs))),
            [s.output_tokens for s in specs],
            dest,
            batch_mm_kwargs=mm_kwargs,
        )
        return torch.cat([dest[i] for i in range(len(specs))], dim=0)

    def postprocess_encoder_output(
        self,
        output: torch.Tensor,
        indices: list[int],
        per_item_out_tokens: list[int],
        dest: dict[int, torch.Tensor] | list[torch.Tensor | None],
        clone: bool = False,
        batch_mm_kwargs: dict[str, Any] | None = None,
        local_output: torch.Tensor | None = None,
    ) -> None:
        assert batch_mm_kwargs is not None
        images_spatial_crop = batch_mm_kwargs["images_spatial_crop"]
        _, hw, n_dim = output.shape
        h = w = int(hw**0.5)

        tile_offset = 0
        for rank, img_idx in enumerate(indices):
            tw = int(images_spatial_crop[rank][0])
            th = int(images_spatial_crop[rank][1])
            n_tiles = 1 + tw * th
            tiles = output[tile_offset : tile_offset + n_tiles]
            tile_offset += n_tiles

            global_f = tiles[0].view(h, w, n_dim)
            newline_g = self.image_newline.view(1, 1, n_dim).expand(h, 1, n_dim)
            global_f = torch.cat([global_f, newline_g], dim=1).view(-1, n_dim)

            local_f = tiles[1:].view(th, tw, h, w, n_dim)
            local_f = local_f.permute(0, 2, 1, 3, 4).reshape(th * h, tw * w, n_dim)
            newline_l = self.image_newline.view(1, 1, n_dim).expand(th * h, 1, n_dim)
            local_f = torch.cat([local_f, newline_l], dim=1).view(-1, n_dim)

            if self.global_view_pos == "head":
                emb = torch.cat([global_f, self.view_seperator[None], local_f])
            else:
                emb = torch.cat([local_f, self.view_seperator[None], global_f])

            if isinstance(dest, dict):
                dest[img_idx] = emb.clone() if clone else emb
            else:
                dest[rank] = emb.clone() if clone else emb


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Skip if not cuda")
class TestDeepseekVL2CudaGraph:
    """CUDA graph tests for the batched-tile encoder pattern (DeepSeek-VL2)."""

    def setup_method(self):
        self.device = torch.device("cuda:0")
        self.dtype = torch.float16
        self.model = MockDeepseekVL2Model().to(self.device).half()
        self.mgr = _make_manager_for_gpu(
            self.model, _DSV2_BUDGETS, _DSV2_MAX_BATCH, self.device, self.dtype
        )
        self.graph_pool = current_platform.graph_pool_handle()
        self.mgr.capture(graph_pool=self.graph_pool)

    def test_capture_creates_one_graph_per_budget(self):
        assert set(self.mgr.budget_graphs["default"].keys()) == set(_DSV2_BUDGETS)

    def test_execute_returns_one_tensor_per_image(self):
        mm_kwargs = _make_deepseek_vl2_mm_kwargs(
            [[1, 1], [1, 1]], self.device, self.dtype
        )
        result = self.mgr.execute(mm_kwargs)
        assert result is not None
        assert len(result) == 2

    def test_output_tokens_1x1_tile(self):
        # global=h*(w+1)=2*3=6, local=1*2*(1*2+1)=6, sep=1 → 13
        expected = _dsv2_output_tokens(1, 1)
        mm_kwargs = _make_deepseek_vl2_mm_kwargs([[1, 1]], self.device, self.dtype)
        result = self.mgr.execute(mm_kwargs)
        assert result is not None
        assert result[0].shape == (expected, _DSV2_DIM)

    def test_output_tokens_2x1_tile(self):
        # global=6, local=1*2*(2*2+1)=10, sep=1 → 17
        expected = _dsv2_output_tokens(2, 1)
        mm_kwargs = _make_deepseek_vl2_mm_kwargs([[2, 1]], self.device, self.dtype)
        result = self.mgr.execute(mm_kwargs)
        assert result is not None
        assert result[0].shape == (expected, _DSV2_DIM)

    def test_output_tokens_multi_image(self):
        # Image 0: tw=1,th=1 → 13;  Image 1: tw=2,th=1 → 17
        e0 = _dsv2_output_tokens(1, 1)
        e1 = _dsv2_output_tokens(2, 1)
        mm_kwargs = _make_deepseek_vl2_mm_kwargs(
            [[1, 1], [2, 1]], self.device, self.dtype
        )
        result = self.mgr.execute(mm_kwargs)
        assert result is not None
        assert result[0].shape == (e0, _DSV2_DIM)
        assert result[1].shape == (e1, _DSV2_DIM)

    def test_eager_fallback_when_tokens_exceed_all_budgets(self):
        # tw=4,th=4: global=6, local=4*2*(4*2+1)=72, sep=1 → 79 > max budget 64
        tw, th = 4, 4
        expected = _dsv2_output_tokens(tw, th)
        assert expected > _DSV2_BUDGETS[-1], "test precondition: must exceed max budget"
        mm_kwargs = _make_deepseek_vl2_mm_kwargs([[tw, th]], self.device, self.dtype)
        result = self.mgr.execute(mm_kwargs)
        assert result is not None
        assert len(result) == 1
        assert result[0].shape == (expected, _DSV2_DIM)
        assert self.mgr.graph_misses >= 1

    def test_graph_hit_counter(self):
        mm_kwargs = _make_deepseek_vl2_mm_kwargs(
            [[1, 1], [1, 1]], self.device, self.dtype
        )
        self.mgr.execute(mm_kwargs)
        assert self.mgr.graph_hits == 2


# ---------------------------------------------------------------------------
# GPU tests — video capture, replay, fallback, and mixed image+video
# ---------------------------------------------------------------------------

_VIDEO_MAX_BATCH = 4
_VIDEO_MAX_FRAMES = 8  # 2 frames per item at max_batch_size=4


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Skip if not cuda")
class TestEncoderCudaGraphVideoReplay:
    def setup_method(self):
        self.device = torch.device("cuda:0")
        self.dtype = torch.float16
        self.model = SimpleMockViTVideoModel().to(self.device).half()
        self.mgr = _make_manager_for_gpu(
            self.model,
            _BUDGETS,
            _VIDEO_MAX_BATCH,
            self.device,
            self.dtype,
            max_frames_per_batch=_VIDEO_MAX_FRAMES,
        )
        self.graph_pool = current_platform.graph_pool_handle()
        self.mgr.capture(graph_pool=self.graph_pool)

    # --- capture ---

    def test_capture_creates_one_graph_per_budget(self):
        assert len(self.mgr.budget_graphs["default"]) == len(_BUDGETS)
        assert set(self.mgr.budget_graphs["default"].keys()) == set(_BUDGETS)

    # --- output shape ---

    def test_video_execute_returns_one_tensor_per_video(self):
        # T=2, 4x4 → 2*(4//2)*(4//2) = 8 tokens per video
        grid_thw = [[2, 4, 4], [2, 4, 4]]
        mm_kwargs = _make_video_mm_kwargs(grid_thw, self.device, self.dtype)
        result = self.mgr.execute(mm_kwargs)
        assert result is not None
        assert len(result) == 2

    def test_video_output_tokens_per_item(self):
        # T=2,4x4 → 8 tokens; T=1,4x4 → 4 tokens
        grid_thw = [[2, 4, 4], [1, 4, 4]]
        mm_kwargs = _make_video_mm_kwargs(grid_thw, self.device, self.dtype)
        result = self.mgr.execute(mm_kwargs)
        assert result is not None
        assert result[0].shape == (8, _HIDDEN)
        assert result[1].shape == (4, _HIDDEN)

    # --- budget fallback ---

    def test_video_eager_fallback_when_tokens_exceed_all_budgets(self):
        # T=2, 18x18 → 2*(18//2)*(18//2) = 162 tokens > max budget 64
        grid_thw = [[2, 18, 18]]
        mm_kwargs = _make_video_mm_kwargs(grid_thw, self.device, self.dtype)
        result = self.mgr.execute(mm_kwargs)
        assert result is not None
        assert len(result) == 1
        assert result[0].shape == (162, _HIDDEN)
        assert self.mgr.graph_misses == 1

    # --- counters ---

    def test_video_hit_counter_increments_by_num_videos(self):
        grid_thw = [[2, 4, 4], [1, 4, 4]]
        mm_kwargs = _make_video_mm_kwargs(grid_thw, self.device, self.dtype)
        self.mgr.execute(mm_kwargs)
        assert self.mgr.graph_hits == 2

    def test_video_miss_counter_increments_for_oversized_video(self):
        grid_thw = [[2, 18, 18]]  # 162 tokens > 64
        mm_kwargs = _make_video_mm_kwargs(grid_thw, self.device, self.dtype)
        self.mgr.execute(mm_kwargs)
        assert self.mgr.graph_misses == 1

    # --- image and video sharing the same manager ---

    def test_image_and_video_share_manager(self):
        """Image and video inputs can both be executed through the same manager."""
        img_grid = [[1, 4, 4], [1, 4, 4]]
        img_result = self.mgr.execute(
            _make_mm_kwargs(img_grid, self.device, self.dtype)
        )

        vid_grid = [[2, 4, 4]]
        vid_result = self.mgr.execute(
            _make_video_mm_kwargs(vid_grid, self.device, self.dtype)
        )

        assert len(img_result) == 2
        assert len(vid_result) == 1
        assert img_result[0].shape == (4, _HIDDEN)
        assert vid_result[0].shape == (8, _HIDDEN)


# ---------------------------------------------------------------------------
# __init__ invariant validation tests (no GPU required)
# ---------------------------------------------------------------------------


class TestInitInvariantValidation:
    """Ensure max_batch_size <= min(token_budgets) for all config paths."""

    def _make_mgr(
        self,
        token_budgets=None,
        max_mm_items=0,
        min_budget=4,
        max_budget=128,
    ):
        vllm_config = _MockVllmConfig(token_budgets, max_mm_items)
        model = _MockModel(min_budget, max_budget)
        return EncoderCudaGraphManager(
            vllm_config=vllm_config,
            device=torch.device("cpu"),
            dtype=torch.float32,
            model=model,
        )

    # --- Finding 1: fully auto-inferred ---

    def test_auto_inferred_invariant_holds(self):
        mgr = self._make_mgr(min_budget=64, max_budget=16384)
        assert mgr.max_batch_size <= min(mgr.token_budgets)

    def test_auto_inferred_small_range(self):
        mgr = self._make_mgr(min_budget=4, max_budget=128)
        assert mgr.max_batch_size <= min(mgr.token_budgets)

    # --- Finding 2: fully user-specified, bad combo ---

    def test_user_specified_bad_combo_raises(self):
        with pytest.raises(ValueError, match="must be <= smallest token budget"):
            self._make_mgr(token_budgets=[64], max_mm_items=256)

    def test_user_specified_valid_combo(self):
        mgr = self._make_mgr(token_budgets=[64, 128], max_mm_items=32)
        assert mgr.max_batch_size == 32
        assert mgr.token_budgets == [64, 128]

    def test_user_specified_exact_boundary(self):
        # max_mm_items == min(budgets) is OK (per_image_output = 1)
        mgr = self._make_mgr(token_budgets=[64, 128], max_mm_items=64)
        assert mgr.max_batch_size == 64

    # --- Finding 3: user provides only max_mm_items ---

    def test_user_max_mm_items_only_adjusts_budgets(self):
        # model min_budget=64, user max_mm_items=128 → budgets start at 128
        mgr = self._make_mgr(max_mm_items=128, min_budget=64, max_budget=16384)
        assert mgr.max_batch_size == 128
        assert min(mgr.token_budgets) >= 128

    def test_user_max_mm_items_smaller_than_min_budget(self):
        # max_mm_items=2, model min=4 → budgets start at 4 (>= 2), OK
        mgr = self._make_mgr(max_mm_items=2, min_budget=4, max_budget=128)
        assert mgr.max_batch_size == 2
        assert min(mgr.token_budgets) >= 2

    # --- Finding 4: user provides only budgets ---

    def test_user_budgets_only_caps_max_batch_size(self):
        # user budgets start at 32, model min_budget=64
        # without fix: max_batch_size = min(128//64, 64) = 2 → OK
        # but if user budgets=[16, 64]:
        # without fix: max_batch_size = min(128//4, 4) = 4 > 16? No.
        # Let's use a case that triggers it:
        # model min=64, max=16384 → max_budget//min_budget = 256
        # user budgets=[32, 64] → min = 32
        # without fix: max_batch_size = min(256, 64) = 64 > 32 → BUG
        # with fix: max_batch_size = min(256, 32) = 32 → OK
        mgr = self._make_mgr(token_budgets=[32, 64], min_budget=64, max_budget=16384)
        assert mgr.max_batch_size <= min(mgr.token_budgets)
        assert mgr.max_batch_size == 32

    # --- Finding 5/6: bad model budget range ---

    def test_zero_min_budget_raises(self):
        with pytest.raises(ValueError, match="Both must be positive"):
            self._make_mgr(min_budget=0, max_budget=128)

    def test_negative_max_budget_raises(self):
        with pytest.raises(ValueError, match="Both must be positive"):
            self._make_mgr(min_budget=4, max_budget=-1)

    def test_min_greater_than_max_raises(self):
        with pytest.raises(ValueError, match="min_budget=200 > max_budget=100"):
            self._make_mgr(min_budget=200, max_budget=100)

    # --- Finding 7: user-provided budgets with non-positive values ---

    def test_user_budgets_zero_raises(self):
        """Non-positive budgets should be caught at config validation."""
        from vllm.config.compilation import CompilationConfig

        with pytest.raises(ValueError, match="must be positive"):
            CompilationConfig(encoder_cudagraph_token_budgets=[0, 128])

    def test_user_budgets_negative_raises(self):
        from vllm.config.compilation import CompilationConfig

        with pytest.raises(ValueError, match="must be positive"):
            CompilationConfig(encoder_cudagraph_token_budgets=[-1, 64])
