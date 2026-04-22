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

from vllm.platforms import current_platform
from vllm.v1.worker.encoder_cudagraph import (
    EncoderCudaGraphManager,
)
from vllm.v1.worker.encoder_cudagraph_defs import (
    EncoderCudaGraphCaptureInputs,
    EncoderCudaGraphConfig,
    EncoderCudaGraphReplayBuffers,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager_with_budgets(budgets: list[int]) -> EncoderCudaGraphManager:
    """Create a minimal EncoderCudaGraphManager with only token_budgets set.

    Skips the parts of __init__ that require a real VllmConfig / model
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


class SimpleMockViTModel(torch.nn.Module):
    """Minimal ViT model for CUDA graph tests.

    Implements the SupportsEncoderCudaGraph protocol by providing
    all required methods. The forward pass projects patches and
    simulates spatial merge by averaging groups of m^2 patches.
    """

    supports_encoder_cudagraph = True

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(_FLAT, _HIDDEN)
        self.spatial_merge_size = _SPATIAL_MERGE
        self.out_hidden_size = _HIDDEN

    def get_encoder_cudagraph_config(self) -> EncoderCudaGraphConfig:
        return EncoderCudaGraphConfig(
            modalities=["image"],
            input_key_by_modality={
                "image": "pixel_values",
            },
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

    def get_encoder_cudagraph_num_items(
        self,
        mm_kwargs: dict[str, Any],
    ) -> int:
        return len(mm_kwargs["image_grid_thw"])

    def get_encoder_cudagraph_per_item_output_tokens(
        self,
        mm_kwargs: dict[str, Any],
    ) -> list[int]:
        m = _SPATIAL_MERGE
        return [t * (h // m) * (w // m) for t, h, w in mm_kwargs["image_grid_thw"]]

    def get_encoder_cudagraph_per_item_input_sizes(
        self,
        mm_kwargs: dict[str, Any],
    ) -> list[int]:
        return [t * h * w for t, h, w in mm_kwargs["image_grid_thw"]]

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
            mm_kwargs={
                "pixel_values": dummy_pixel_values,
                "image_grid_thw": grid_config,
            },
            buffers={"dummy_buf": dummy_buf},
        )

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, Any],
        max_batch_size: int,
        max_frames_per_batch: int,
    ) -> EncoderCudaGraphReplayBuffers:
        grid_thw = mm_kwargs["image_grid_thw"]
        n_out = _count_output_tokens(grid_thw, _SPATIAL_MERGE)
        p = next(self.parameters())
        dummy_buf = torch.zeros(n_out, _HIDDEN, device=p.device, dtype=p.dtype)
        return EncoderCudaGraphReplayBuffers(buffers={"dummy_buf": dummy_buf})

    def encoder_cudagraph_forward(
        self,
        mm_kwargs: dict[str, Any],
        buffers: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self._forward(mm_kwargs["pixel_values"])

    def encoder_eager_forward(
        self,
        mm_kwargs: dict[str, Any],
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
    mgr.budget_graphs = {}
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
        self.mgr.capture()

    # --- capture ---

    def test_capture_creates_one_graph_per_budget(self):
        assert len(self.mgr.budget_graphs) == len(_BUDGETS)
        assert set(self.mgr.budget_graphs.keys()) == set(_BUDGETS)

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
            input_key_by_modality={
                "image": "pixel_values",
                "video": "pixel_values_videos",
            },
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

    def get_encoder_cudagraph_num_items(self, mm_kwargs: dict[str, Any]) -> int:
        return len(self._get_grid_thw(mm_kwargs))

    def get_encoder_cudagraph_per_item_output_tokens(
        self, mm_kwargs: dict[str, Any]
    ) -> list[int]:
        m = _SPATIAL_MERGE
        return [t * (h // m) * (w // m) for t, h, w in self._get_grid_thw(mm_kwargs)]

    def get_encoder_cudagraph_per_item_input_sizes(
        self, mm_kwargs: dict[str, Any]
    ) -> list[int]:
        return [t * h * w for t, h, w in self._get_grid_thw(mm_kwargs)]

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
            mm_kwargs={
                "pixel_values": dummy_pixel_values,
                "image_grid_thw": grid_config,
            },
            buffers={"dummy_buf": dummy_buf},
        )

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, Any],
        max_batch_size: int,
        max_frames_per_batch: int,
    ) -> EncoderCudaGraphReplayBuffers:
        n_out = _count_output_tokens(self._get_grid_thw(mm_kwargs), _SPATIAL_MERGE)
        p = next(self.parameters())
        dummy_buf = torch.zeros(n_out, _HIDDEN, device=p.device, dtype=p.dtype)
        return EncoderCudaGraphReplayBuffers(buffers={"dummy_buf": dummy_buf})

    def encoder_cudagraph_forward(
        self, mm_kwargs: dict[str, Any], buffers: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return self._forward(self._get_pixel_values(mm_kwargs))

    def encoder_eager_forward(self, mm_kwargs: dict[str, Any]) -> torch.Tensor:
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

    def test_video_model_config_has_both_modalities(self):
        model = SimpleMockViTVideoModel()
        cfg = model.get_encoder_cudagraph_config()
        assert "image" in cfg.modalities
        assert "video" in cfg.modalities
        assert cfg.input_key_by_modality["image"] == "pixel_values"
        assert cfg.input_key_by_modality["video"] == "pixel_values_videos"


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
        self.mgr.capture()

    # --- capture ---

    def test_capture_creates_one_graph_per_budget(self):
        assert len(self.mgr.budget_graphs) == len(_BUDGETS)
        assert set(self.mgr.budget_graphs.keys()) == set(_BUDGETS)

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
