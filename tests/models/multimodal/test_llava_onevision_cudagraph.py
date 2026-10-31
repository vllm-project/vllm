# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.llava_onevision import (
    LlavaOnevisionForConditionalGeneration,
)
from vllm.v1.worker.encoder_cudagraph_defs import (
    ENCODER_CUDAGRAPH_AXIS_KEYS_KWARG,
)


def _model_stub():
    model = object.__new__(LlavaOnevisionForConditionalGeneration)
    config = SimpleNamespace(
        image_grid_pinpoints=[
            [384, 384],
            [384, 768],
            [768, 384],
            [768, 768],
        ],
        spatial_pool_stride=2,
        vision_feature_select_strategy="full",
        text_config=SimpleNamespace(hidden_size=64),
        vision_config=SimpleNamespace(image_size=384, patch_size=14),
    )
    object.__setattr__(model, "config", config)
    return model


def _patch_encoder_tokens(monkeypatch):
    monkeypatch.setattr(
        LlavaOnevisionForConditionalGeneration,
        "_encoder_tokens_per_tile",
        lambda self: 729,
        raising=False,
    )


def test_llava_onevision_image_item_specs(monkeypatch):
    _patch_encoder_tokens(monkeypatch)
    model = _model_stub()
    pixels = [
        torch.empty(2, 3, 384, 384),
        torch.empty(2, 3, 384, 384),
    ]
    sizes = torch.tensor([[384, 384], [384, 384]])
    specs = model.get_encoder_cudagraph_item_specs(
        {"pixel_values": pixels, "image_sizes": sizes}
    )
    assert [(s.input_size, s.output_tokens) for s in specs] == [
        (2, 1485),
        (2, 1485),
    ]
    assert [s.path_output_tokens for s in specs] == [
        {"default": 1458},
        {"default": 1458},
    ]


def test_llava_onevision_video_item_specs(monkeypatch):
    _patch_encoder_tokens(monkeypatch)
    model = _model_stub()
    pixels = [torch.empty(4, 3, 384, 384), torch.empty(2, 3, 384, 384)]
    specs = model.get_encoder_cudagraph_item_specs({"pixel_values_videos": pixels})
    assert [(s.input_size, s.output_tokens) for s in specs] == [
        (4, 785),
        (2, 393),
    ]
    assert [s.path_output_tokens for s in specs] == [
        {"default": 2916},
        {"default": 1458},
    ]


def test_llava_onevision_graph_forward_flattens_projected_tokens(monkeypatch):
    model = _model_stub()
    object.__setattr__(model, "vision_tower", object())
    object.__setattr__(
        model,
        "_image_pixels_to_features",
        lambda tower, pixels: pixels.mean(dim=(-2, -1)).unsqueeze(1),
    )
    object.__setattr__(model, "multi_modal_projector", lambda features: features + 1)
    pixels = torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2)
    output = model.encoder_cudagraph_forward({"pixel_values": pixels})
    expected = pixels.mean(dim=(-2, -1)) + 1
    assert output.shape == (2, 3)
    assert torch.allclose(output, expected)


def test_llava_onevision_capture_buffer_rounds_up_to_budget():
    model = _model_stub()
    object.__setattr__(
        model,
        "_encoder_tokens_per_tile",
        lambda: 729,
    )
    capture = model.prepare_encoder_cudagraph_capture_inputs(
        token_budget=4096,
        max_batch_size=8,
        max_frames_per_batch=16,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    # ceil(4096 / 729) = 6. Using floor would under-allocate the graph
    # buffer and make a valid 6-tile replay impossible.
    assert capture.values["pixel_values"].shape[0] == 6


def test_llava_onevision_image_postprocess_matches_eager_merge():
    model = _model_stub()
    hidden_size = 2
    image_newline = torch.tensor([1000.0, 2000.0])
    object.__setattr__(model, "image_newline", image_newline)
    object.__setattr__(model, "_encoder_tokens_per_tile", lambda: 729)
    raw_tiles = torch.arange(2 * 729 * hidden_size, dtype=torch.float32)
    raw_tiles = raw_tiles.reshape(2, 729, hidden_size)
    image_size = torch.tensor([384, 384])
    expected = model._merge_image_patch_embeddings(
        image_size,
        raw_tiles,
        image_newline=image_newline,
        strategy="spatial_unpad",
    )
    dest = [None]
    model.postprocess_encoder_output(
        {"default": raw_tiles.reshape(-1, hidden_size)},
        indices=[0],
        per_item_out_tokens=[expected.shape[0]],
        dest=dest,
        batch_mm_kwargs={
            "pixel_values": [torch.empty(2, 3, 384, 384)],
            "image_sizes": image_size[None],
        },
    )
    torch.testing.assert_close(dest[0], expected, rtol=0, atol=0)


def test_llava_onevision_video_postprocess_matches_eager_path():
    model = _model_stub()
    hidden_size = 2
    image_newline = torch.tensor([1000.0, 2000.0])
    object.__setattr__(model, "image_newline", image_newline)
    object.__setattr__(model, "_encoder_tokens_per_tile", lambda: 729)
    raw_frames = torch.arange(4 * 729 * hidden_size, dtype=torch.float32)
    raw_frames = raw_frames.reshape(4, 729, hidden_size)
    pooled = model.apply_pooling(raw_frames)
    expected = torch.cat(
        (pooled.reshape(-1, hidden_size), image_newline[None]),
        dim=0,
    )
    dest = [None]
    model.postprocess_encoder_output(
        {"default": raw_frames.reshape(-1, hidden_size)},
        indices=[0],
        per_item_out_tokens=[expected.shape[0]],
        dest=dest,
        batch_mm_kwargs={
            "pixel_values_videos": [torch.empty(4, 3, 384, 384)],
        },
    )
    torch.testing.assert_close(dest[0], expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "height,width",
    [(384, 384), (384, 768), (768, 384), (700, 500), (500, 700)],
)
def test_llava_onevision_image_token_count_matches_eager_merge(height, width):
    model = _model_stub()
    hidden_size = 2
    image_newline = torch.zeros(hidden_size)
    object.__setattr__(model, "image_newline", image_newline)
    image_size = torch.tensor([height, width])
    padded = [torch.empty(8, 3, 384, 384)]
    num_tiles = model._get_image_tile_counts(padded, image_size[None])[0]
    raw_tiles = torch.zeros(num_tiles, 729, hidden_size)
    expected = model._merge_image_patch_embeddings(
        image_size,
        raw_tiles,
        image_newline=image_newline,
        strategy="spatial_unpad",
    )
    assert model._get_image_output_tokens(image_size) == expected.shape[0]


def _set_explicit_encoder_budgets(model, budgets):
    object.__setattr__(
        model,
        "vllm_config",
        SimpleNamespace(
            compilation_config=SimpleNamespace(
                encoder_cudagraph_token_budgets=list(budgets),
            ),
        ),
    )


def test_llava_onevision_auto_budgets_respect_user_max_vision_items(monkeypatch):
    _patch_encoder_tokens(monkeypatch)
    model = _model_stub()
    object.__setattr__(
        model,
        "vllm_config",
        SimpleNamespace(
            compilation_config=SimpleNamespace(
                encoder_cudagraph_token_budgets=[],
                encoder_cudagraph_max_vision_items_per_batch=2000,
            ),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=10000),
            model_config=SimpleNamespace(max_model_len=10000),
        ),
    )
    assert model._encoder_cudagraph_token_budgets() == [2000, 4000, 8000, 10000]


def test_llava_onevision_capture_axis_covers_explicit_budget_padding(monkeypatch):
    _patch_encoder_tokens(monkeypatch)
    model = _model_stub()
    _set_explicit_encoder_budgets(model, [8192])
    config = model.get_encoder_cudagraph_config()
    assert config.capture_axes == (tuple(range(12)),)


def test_llava_onevision_select_items_returns_padding_axis(monkeypatch):
    _patch_encoder_tokens(monkeypatch)
    model = _model_stub()
    _set_explicit_encoder_budgets(model, [8192])
    pixels = [torch.empty(3, 3, 384, 384)]
    selected = model.select_encoder_cudagraph_items(
        {
            "pixel_values": pixels,
            "image_sizes": torch.tensor([[384, 768]]),
        },
        [0],
    )
    assert selected[ENCODER_CUDAGRAPH_AXIS_KEYS_KWARG] == (9,)


def test_llava_onevision_capture_axis_restores_exact_tile_count(monkeypatch):
    _patch_encoder_tokens(monkeypatch)
    model = _model_stub()
    _set_explicit_encoder_budgets(model, [8192])
    capture = model.prepare_encoder_cudagraph_capture_inputs(
        token_budget=8192,
        max_batch_size=1,
        max_frames_per_batch=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        axis_keys=(9,),
    )
    assert capture.values["pixel_values"].shape[0] == 3


def test_llava_onevision_video_selection_uses_total_frames_axis(monkeypatch):
    _patch_encoder_tokens(monkeypatch)
    model = _model_stub()
    _set_explicit_encoder_budgets(model, [8192])
    selected = model.select_encoder_cudagraph_items(
        {"pixel_values_videos": [torch.empty(4, 3, 384, 384)]}, [0]
    )
    assert selected[ENCODER_CUDAGRAPH_AXIS_KEYS_KWARG] == (8,)


def test_llava_onevision_missing_image_sizes_matches_eager_merge(monkeypatch):
    _patch_encoder_tokens(monkeypatch)
    model = _model_stub()
    hidden_size = 2
    image_newline = torch.zeros(hidden_size)
    object.__setattr__(model, "image_newline", image_newline)
    raw_tiles = torch.zeros(2, 729, hidden_size)
    default_size = torch.tensor([384, 384])
    expected = model._merge_image_patch_embeddings(
        default_size,
        raw_tiles,
        image_newline=image_newline,
        strategy="spatial_unpad",
    )
    specs = model.get_encoder_cudagraph_item_specs(
        {"pixel_values": [torch.empty(2, 3, 384, 384)], "image_sizes": None}
    )
    assert specs[0].output_tokens == expected.shape[0]
