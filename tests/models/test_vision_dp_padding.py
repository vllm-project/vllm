# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for DP ViT padding used by --mm-encoder-tp-mode data (#52654)."""

import math

import pytest
import torch

from vllm.model_executor.models.vision import (
    get_dummy_mrope_grid_thw,
    get_load_balance_assignment,
    pad_local_mrope_vision_inputs,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


class _SimpleMRopeVisionModel(torch.nn.Module):
    def __init__(self, spatial_merge_size: int = 2, out_hidden_size: int = 64):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.out_hidden_size = out_hidden_size
        self.linear = torch.nn.Linear(768, out_hidden_size)

    def forward(self, pixel_values: torch.Tensor, grid_thw_list: list[list[int]]):
        embeddings = self.linear(pixel_values)
        merge_factor = self.spatial_merge_size * self.spatial_merge_size
        merged = []
        start_idx = 0
        for grid_thw in grid_thw_list:
            num_patches = math.prod(grid_thw)
            end_idx = start_idx + num_patches
            image_patches = embeddings[start_idx:end_idx]
            merged_patches = num_patches // merge_factor
            if merged_patches > 0:
                reshaped = image_patches[: merged_patches * merge_factor].view(
                    merged_patches, merge_factor, -1
                )
                merged.append(reshaped.mean(dim=1))
            start_idx = end_idx
        if merged:
            return torch.cat(merged, dim=0)
        return torch.empty(
            (0, self.out_hidden_size),
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        )


def test_five_images_four_gpus_assignment():
    """Greedy balancer gives 2/1/1/1 for five equal images on TP=4."""
    sizes = [100, 100, 100, 100, 100]
    shuffle, counts, grouped = get_load_balance_assignment(sizes, num_gpus=4)
    assert counts == [2, 1, 1, 1]
    assert grouped == [200, 100, 100, 100]
    assert sorted(shuffle) == list(range(5))
    assert sum(counts) == 5


def test_pad_local_mrope_vision_inputs_strips_cleanly():
    """Dummy-padded local batches must not change real-image embeddings."""
    vision_model = _SimpleMRopeVisionModel()
    dummy_grid = get_dummy_mrope_grid_thw(vision_model, "rope_3d")
    assert dummy_grid == [1, 2, 2]

    grid_thw_list = [[1, 4, 4], [1, 2, 2]]
    pixel_values = torch.cat(
        [torch.randn(math.prod(grid), 768) for grid in grid_thw_list], dim=0
    )

    padded_pixels, padded_grids, n_pad = pad_local_mrope_vision_inputs(
        pixel_values,
        grid_thw_list,
        target_num_images=4,
        dummy_grid=dummy_grid,
    )
    assert n_pad == 2
    assert len(padded_grids) == 4
    assert padded_pixels.shape[0] == pixel_values.shape[0] + n_pad * math.prod(
        dummy_grid
    )

    merge_factor = vision_model.spatial_merge_size**2
    dummy_out_tokens = n_pad * (math.prod(dummy_grid) // merge_factor)
    with torch.inference_mode():
        direct = vision_model(pixel_values, grid_thw_list)
        padded_out = vision_model(padded_pixels, padded_grids)
        stripped = padded_out[:-dummy_out_tokens]
    assert torch.allclose(direct, stripped, rtol=1e-5, atol=1e-5)


def test_pad_local_mrope_vision_inputs_empty_rank():
    """Empty ranks still get a dummy image so the encoder is not skipped."""
    empty = torch.empty((0, 768))
    dummy_grid = [1, 2, 2]
    padded_pixels, padded_grids, n_pad = pad_local_mrope_vision_inputs(
        empty,
        [],
        target_num_images=2,
        dummy_grid=dummy_grid,
    )
    assert n_pad == 2
    assert padded_grids == [dummy_grid, dummy_grid]
    assert padded_pixels.shape == (2 * math.prod(dummy_grid), 768)


def test_pad_local_mrope_vision_inputs_noop_when_full():
    pixels = torch.randn(8, 768)
    grids = [[1, 2, 2], [1, 2, 2]]
    out_pixels, out_grids, n_pad = pad_local_mrope_vision_inputs(
        pixels,
        grids,
        target_num_images=2,
        dummy_grid=[1, 2, 2],
    )
    assert n_pad == 0
    assert out_grids == grids
    assert out_pixels is pixels


def test_flashinfer_wrapper_skips_empty_query():
    """0-token queries must not reach FlashInfer's stable-ABI empty_like."""
    from vllm.v1.attention.ops.vit_attn_wrappers import flashinfer_wrapper

    q = torch.empty(0, 2, 4, 8)
    workspace = torch.zeros(8, dtype=torch.uint8)
    out = flashinfer_wrapper(q, q, q, 1.0, workspace, o_data_type=torch.float32)
    assert out.shape == q.shape
    assert out.dtype == torch.float32
