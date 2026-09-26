# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.cosmos3_edge import (
    Cosmos3EdgeForConditionalGeneration,
    _build_merge_gather_idx,
    patch_merging_by_param,
)


def test_cosmos3_edge_merge_gather_idx_matches_eager_layout():
    grids = [[1, 4, 6], [1, 6, 4], [2, 4, 4]]
    grid_thw = torch.tensor(grids, dtype=torch.int64)
    hidden_size = 3
    num_patches = sum(t * h * w for t, h, w in grids)
    image_embeds = torch.arange(num_patches * hidden_size).view(
        num_patches, hidden_size
    )

    expected = patch_merging_by_param(image_embeds, grid_thw, merge_size=2)
    gather_idx = _build_merge_gather_idx(
        grids, merge_size=2, device=torch.device("cpu")
    )
    actual = image_embeds.index_select(0, gather_idx).view(-1, 4 * hidden_size)

    assert torch.equal(actual, expected)


def _make_encoder_cudagraph_config_model(
    max_items: int,
    *,
    enable_mm_embeds: bool = False,
):
    model = object.__new__(Cosmos3EdgeForConditionalGeneration)
    model.multimodal_config = SimpleNamespace(
        enable_mm_embeds=enable_mm_embeds,
        get_limit_per_prompt=lambda modality: 1 if modality == "image" else 0,
    )
    model.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            encoder_cudagraph_max_vision_items_per_batch=max_items
        )
    )
    model.visual = SimpleNamespace(out_hidden_size=2048)
    return model


def test_cosmos3_edge_encoder_cudagraph_uses_exact_single_image_budgets():
    model = _make_encoder_cudagraph_config_model(max_items=0)

    config = model.get_encoder_cudagraph_config()

    assert model.get_encoder_cudagraph_budget_range(model.vllm_config) == (64, 64)
    assert config.modalities == ["image"]
    assert config.paths["default"].require_exact_token_budget_match


def test_cosmos3_edge_encoder_cudagraph_rejects_multiple_images_per_replay():
    model = _make_encoder_cudagraph_config_model(max_items=2)

    with pytest.raises(ValueError, match="at most one image"):
        model.get_encoder_cudagraph_config()


def test_cosmos3_edge_encoder_cudagraph_disables_precomputed_embeddings():
    model = _make_encoder_cudagraph_config_model(
        max_items=0,
        enable_mm_embeds=True,
    )

    assert model.get_encoder_cudagraph_config().modalities == []
