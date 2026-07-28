# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for pre-computed HF processor output inputs
(`--enable-mm-processor-outputs`).

These tests are CPU-only: they exercise the data parser routing,
construction-time shape validation, and the flag gating logic,
without loading any model weights.
"""

import pytest
import torch

from vllm.model_executor.models.qwen2_vl import (
    Qwen2VLMultiModalDataParser,
    _create_qwen2vl_field_factory,
)
from vllm.multimodal.parse import (
    DictEmbeddingItems,
    DictProcessorOutputItems,
)

SPATIAL_MERGE_SIZE = 2
HIDDEN_SIZE = 32


def _make_parser() -> Qwen2VLMultiModalDataParser:
    return Qwen2VLMultiModalDataParser(SPATIAL_MERGE_SIZE)


def _grid_thw(*grids: tuple[int, int, int]) -> torch.Tensor:
    return torch.tensor(list(grids), dtype=torch.int64)


def _pixel_values_for(grid_thw: torch.Tensor, feat_dim: int = 1176) -> torch.Tensor:
    total = int(grid_thw.prod(-1).sum().item())
    return torch.randn(total, feat_dim)


class TestParserRouting:
    def test_pixel_values_dict_routes_to_processor_outputs(self):
        parser = _make_parser()
        grid = _grid_thw((1, 4, 4))
        items = parser._parse_image_data(
            {
                "pixel_values": _pixel_values_for(grid),
                "image_grid_thw": grid,
            }
        )
        assert isinstance(items, DictProcessorOutputItems)
        assert items.get_count() == 1

    def test_image_embeds_dict_routes_to_embeddings(self):
        parser = _make_parser()
        grid = _grid_thw((1, 4, 4))
        merge_sq = SPATIAL_MERGE_SIZE**2
        n_embeds = int(grid.prod(-1).sum().item()) // merge_sq
        items = parser._parse_image_data(
            {
                "image_embeds": torch.randn(n_embeds, HIDDEN_SIZE),
                "image_grid_thw": grid,
            }
        )
        assert isinstance(items, DictEmbeddingItems)
        assert not isinstance(items, DictProcessorOutputItems)

    def test_video_pixel_values_dict_routes_to_processor_outputs(self):
        parser = _make_parser()
        grid = _grid_thw((2, 4, 4))
        items = parser._parse_video_data(
            {
                "pixel_values_videos": _pixel_values_for(grid),
                "video_grid_thw": grid,
            }
        )
        assert isinstance(items, DictProcessorOutputItems)
        assert items.get_count() == 1

    def test_video_embeds_dict_still_routes_to_embeddings(self):
        parser = _make_parser()
        grid = _grid_thw((2, 4, 4))
        merge_sq = SPATIAL_MERGE_SIZE**2
        n_embeds = int(grid.prod(-1).sum().item()) // merge_sq
        items = parser._parse_video_data(
            {
                "video_embeds": torch.randn(n_embeds, HIDDEN_SIZE),
                "video_grid_thw": grid,
            }
        )
        assert isinstance(items, DictEmbeddingItems)
        assert not isinstance(items, DictProcessorOutputItems)

    def test_qwen3vl_parser_construction_routes_processor_outputs(self):
        # Qwen3-VL / Qwen3.5 construct the shared parser with
        # `video_needs_metadata=True`; routing must behave identically.
        parser = Qwen2VLMultiModalDataParser(
            SPATIAL_MERGE_SIZE,
            video_needs_metadata=True,
            expected_hidden_size=HIDDEN_SIZE,
        )
        grid = _grid_thw((1, 4, 4))
        items = parser._parse_image_data(
            {
                "pixel_values": _pixel_values_for(grid),
                "image_grid_thw": grid,
            }
        )
        assert isinstance(items, DictProcessorOutputItems)

        grid_v = _grid_thw((2, 4, 4))
        items_v = parser._parse_video_data(
            {
                "pixel_values_videos": _pixel_values_for(grid_v),
                "video_grid_thw": grid_v,
            }
        )
        assert isinstance(items_v, DictProcessorOutputItems)

    def test_multi_image_item_split(self):
        parser = _make_parser()
        grid = _grid_thw((1, 4, 4), (1, 8, 8))
        items = parser._parse_image_data(
            {
                "pixel_values": _pixel_values_for(grid),
                "image_grid_thw": grid,
            }
        )
        assert isinstance(items, DictProcessorOutputItems)
        assert items.get_count() == 2

        item0 = items.get(0)
        item1 = items.get(1)
        assert item0["pixel_values"].shape[0] == 16
        assert item1["pixel_values"].shape[0] == 64

    def test_missing_required_field_raises(self):
        parser = _make_parser()
        grid = _grid_thw((1, 4, 4))
        # pixel_values present routes to the processor-output branch,
        # which then rejects the missing image_grid_thw.
        with pytest.raises(ValueError, match="image_grid_thw"):
            parser._parse_image_data(
                {
                    "pixel_values": _pixel_values_for(grid),
                }
            )

    def test_dict_without_known_keys_falls_back_to_embeds_error(self):
        parser = _make_parser()
        grid = _grid_thw((1, 4, 4))
        # No pixel_values key: routed to the embeddings branch, which
        # requires image_embeds.
        with pytest.raises(ValueError, match="image_embeds"):
            parser._parse_image_data(
                {
                    "pixel_value": _pixel_values_for(grid),  # typo key
                    "image_grid_thw": grid,
                }
            )


class TestShapeValidation:
    def test_truncated_pixel_values_raises(self):
        parser = _make_parser()
        grid = _grid_thw((1, 4, 4))
        good = _pixel_values_for(grid)
        with pytest.raises(ValueError, match="implies"):
            parser._parse_image_data(
                {
                    "pixel_values": good[:-1],  # one row short
                    "image_grid_thw": grid,
                }
            )

    def test_oversized_pixel_values_raises(self):
        parser = _make_parser()
        grid = _grid_thw((1, 4, 4))
        good = _pixel_values_for(grid)
        extra = torch.cat([good, good[:1]], dim=0)
        with pytest.raises(ValueError, match="implies"):
            parser._parse_image_data(
                {
                    "pixel_values": extra,
                    "image_grid_thw": grid,
                }
            )

    def test_exact_pixel_values_ok(self):
        parser = _make_parser()
        grid = _grid_thw((1, 4, 4), (1, 6, 8))
        items = parser._parse_image_data(
            {
                "pixel_values": _pixel_values_for(grid),
                "image_grid_thw": grid,
            }
        )
        assert items.get_count() == 2

    def test_truncated_video_pixel_values_raises(self):
        parser = _make_parser()
        grid = _grid_thw((2, 4, 4))
        good = _pixel_values_for(grid)
        with pytest.raises(ValueError, match="implies"):
            parser._parse_video_data(
                {
                    "pixel_values_videos": good[:-1],
                    "video_grid_thw": grid,
                }
            )

    def test_non_tensor_pixel_values_raises(self):
        parser = _make_parser()
        grid = _grid_thw((1, 4, 4))
        import numpy as np
        with pytest.raises(ValueError, match="must be a torch.Tensor"):
            parser._parse_image_data(
                {
                    "pixel_values": np.random.randn(16, 1176),
                    "image_grid_thw": grid,
                }
            )

    def test_non_tensor_grid_raises(self):
        parser = _make_parser()
        grid = _grid_thw((1, 4, 4))
        good = _pixel_values_for(grid)
        with pytest.raises(ValueError, match="must be a torch.Tensor"):
            parser._parse_image_data(
                {
                    "pixel_values": good,
                    "image_grid_thw": [[1, 4, 4]],  # list, not tensor
                }
            )


class TestDirectConstruction:
    """Constructing DictProcessorOutputItems directly (as the generic
    entry point that other model adapters would use)."""

    def test_passthrough_data_returns_original_dict(self):
        grid = _grid_thw((1, 4, 4))
        data = {
            "pixel_values": _pixel_values_for(grid),
            "image_grid_thw": grid,
        }
        items = DictProcessorOutputItems(
            data,
            modality="image",
            required_fields={"pixel_values", "image_grid_thw"},
            fields_factory=_create_qwen2vl_field_factory(SPATIAL_MERGE_SIZE),
        )
        # Processor must be skipped: no processor data, everything
        # passes straight through to the model.
        assert items.get_processor_data() == {}
        assert items.get_passthrough_data() is data

    def test_is_subclass_of_dict_embedding_items(self):
        # Gating in context.parse_mm_data relies on this hierarchy:
        # the DictProcessorOutputItems check must come first.
        assert issubclass(DictProcessorOutputItems, DictEmbeddingItems)


class TestFlagGating:
    """The `--enable-mm-processor-outputs` gate in
    MultiModalProcessingContext-based validation."""

    def _gate(self, items, enable_embeds: bool, enable_proc_outputs: bool):
        """Replicates the gating branch in context.parse_mm_data."""
        from vllm.multimodal.parse import EmbeddingItems

        if isinstance(items, DictProcessorOutputItems):
            if not enable_proc_outputs:
                raise ValueError("enable-mm-processor-outputs required")
        elif isinstance(items, (EmbeddingItems, DictEmbeddingItems)):
            if not enable_embeds:
                raise ValueError("enable-mm-embeds required")

    def test_processor_outputs_blocked_without_flag(self):
        parser = _make_parser()
        grid = _grid_thw((1, 4, 4))
        items = parser._parse_image_data(
            {"pixel_values": _pixel_values_for(grid), "image_grid_thw": grid}
        )
        with pytest.raises(ValueError, match="processor-outputs"):
            self._gate(items, enable_embeds=True, enable_proc_outputs=False)

    def test_processor_outputs_allowed_with_flag(self):
        parser = _make_parser()
        grid = _grid_thw((1, 4, 4))
        items = parser._parse_image_data(
            {"pixel_values": _pixel_values_for(grid), "image_grid_thw": grid}
        )
        self._gate(items, enable_embeds=False, enable_proc_outputs=True)

    def test_embeds_still_gated_independently(self):
        parser = _make_parser()
        grid = _grid_thw((1, 4, 4))
        merge_sq = SPATIAL_MERGE_SIZE**2
        n_embeds = int(grid.prod(-1).sum().item()) // merge_sq
        items = parser._parse_image_data(
            {
                "image_embeds": torch.randn(n_embeds, HIDDEN_SIZE),
                "image_grid_thw": grid,
            }
        )
        with pytest.raises(ValueError, match="embeds"):
            self._gate(items, enable_embeds=False, enable_proc_outputs=True)
