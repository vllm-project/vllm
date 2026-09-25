# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.model_executor.models.minicpmv4_6 import MiniCPMV4_6ForConditionalGeneration
from vllm.model_executor.models.minicpmv4_7 import (
    MiniCPMV4_7ForConditionalGeneration,
    MiniCPMV4_7MultiModalProcessor,
    MiniCPMV4_7ProcessingInfo,
    _compute_canvas_single,
    _stack_vit_merger_qkv,
    build_image_bounds,
    canvas_rope_delta,
)

pytestmark = pytest.mark.skip_global_cleanup


def test_4_7_is_separate_from_4_6():
    assert MiniCPMV4_7ForConditionalGeneration is not (
        MiniCPMV4_6ForConditionalGeneration
    )
    assert issubclass(
        MiniCPMV4_7ForConditionalGeneration, MiniCPMV4_6ForConditionalGeneration
    )
    assert issubclass(MiniCPMV4_7ProcessingInfo, object)
    assert issubclass(MiniCPMV4_7MultiModalProcessor, object)


def test_vit_merger_qkv_is_stacked_before_load():
    mapped = list(
        _stack_vit_merger_qkv(
            [
                (
                    "model.vision_tower.vit_merger.self_attn.q_proj.weight",
                    torch.zeros(1),
                ),
                (
                    "model.vision_tower.vit_merger.self_attn.k_proj.weight",
                    torch.zeros(1),
                ),
                (
                    "model.vision_tower.vit_merger.self_attn.v_proj.weight",
                    torch.zeros(1),
                ),
                ("vit_merger.self_attn.qkv_proj.weight", torch.zeros(1)),
            ]
        )
    )
    names = [name for name, _ in mapped]
    shards = [getattr(tensor, "shard_id", None) for _, tensor in mapped]
    assert names == [
        "model.vision_tower.vit_merger.self_attn.qkv_proj.weight",
        "model.vision_tower.vit_merger.self_attn.qkv_proj.weight",
        "model.vision_tower.vit_merger.self_attn.qkv_proj.weight",
        "vit_merger.self_attn.qkv_proj.weight",
    ]
    assert shards == ["q", "k", "v", None]


# Canvas M-RoPE goldens, cross-checked against the reference implementation in
# transformers PR 48979 (`modular_minicpmv4_7.py`). Synthetic ids keep the token
# sequences readable: the canvas math only compares ids for equality.
_IM_START = 10
_IM_END = 11
_SLICE_START = 12
_SLICE_END = 13
_NEWLINE = 14
_IMAGE_PAD = 20
_VIDEO_PAD = 21
_TEXT = 100

_CANVAS_IDS = {
    "im_start_id": _IM_START,
    "im_end_id": _IM_END,
    "slice_start_id": _SLICE_START,
    "slice_end_id": _SLICE_END,
    "newline_id": _NEWLINE,
}


def _canvas_positions(tokens, target_sizes):
    input_ids = torch.tensor([tokens], dtype=torch.long)
    targets = torch.tensor(target_sizes, dtype=torch.long)
    bounds = build_image_bounds(input_ids[0], _CANVAS_IDS)
    positions = _compute_canvas_single(
        input_ids[0],
        torch.arange(len(tokens), dtype=torch.long),
        bounds,
        targets,
        _CANVAS_IDS,
    )
    return positions.tolist(), canvas_rope_delta(positions, len(tokens))


def test_canvas_mrope_thumbnail_matches_reference():
    # <text> <image> PAD </image> <text>; 16x turns a 4x4 patch grid into 1 token.
    tokens = [_TEXT, _IM_START, _IMAGE_PAD, _IM_END, _TEXT]
    positions, delta = _canvas_positions(tokens, [[4, 4]])

    assert positions == [
        [0, 1, 1, 1, 3],
        [0, 0, 1, 2, 3],
        [0, 0, 1, 2, 3],
    ]
    assert delta == -1


def test_canvas_mrope_slice_row_matches_reference():
    # Thumbnail then a 1x2 slice row; the slices tile the thumbnail's canvas.
    tokens = [
        _TEXT,
        _IM_START,
        _IMAGE_PAD,
        _IM_END,
        _SLICE_START,
        _IMAGE_PAD,
        _SLICE_END,
        _SLICE_START,
        _IMAGE_PAD,
        _SLICE_END,
        _TEXT,
    ]
    positions, delta = _canvas_positions(tokens, [[4, 4], [4, 4], [4, 4]])

    assert positions == [
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
        [0, 0, 1, 2, 1, 1, 1, 1, 1, 1, 4],
        [0, 0, 1, 3, 1, 1, 1, 2, 2, 2, 4],
    ]
    assert delta == -6


def test_canvas_mrope_resamples_thumbnail_over_canvas():
    # A 8x8 patch grid is 2x2 LLM tokens, so the canvas is resampled (not arange).
    tokens = [_TEXT, _IM_START] + [_IMAGE_PAD] * 4 + [_IM_END, _TEXT]
    positions, delta = _canvas_positions(tokens, [[8, 8]])

    assert positions == [
        [0, 1, 1, 1, 1, 1, 1, 4],
        [0, 0, 1, 1, 2, 2, 3, 4],
        [0, 0, 1, 2, 1, 2, 3, 4],
    ]
    assert delta == -3


def test_canvas_mrope_video_frames_share_one_canvas():
    tokens = [
        _TEXT,
        _IM_START,
        _VIDEO_PAD,
        _IM_END,
        _IM_START,
        _VIDEO_PAD,
        _IM_END,
        _TEXT,
    ]
    positions, delta = _canvas_positions(tokens, [[4, 4], [4, 4]])

    assert positions == [
        [0, 1, 1, 1, 3, 3, 3, 5],
        [0, 0, 1, 2, 2, 3, 4, 5],
        [0, 0, 1, 2, 2, 3, 4, 5],
    ]
    assert delta == -2


def test_canvas_mrope_interleaved_image_and_video():
    tokens = (
        [_TEXT, _IM_START]
        + [_IMAGE_PAD] * 4
        + [_IM_END, _IM_START]
        + [_VIDEO_PAD] * 4
        + [_IM_END, _TEXT]
    )
    positions, delta = _canvas_positions(tokens, [[8, 8], [8, 8]])

    assert positions == [
        [0, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 7],
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7],
        [0, 0, 1, 2, 1, 2, 3, 3, 4, 5, 4, 5, 6, 7],
    ]
    assert delta == -6


def test_canvas_mrope_delta_leaves_room_for_the_first_decoded_token():
    # First decoded token must land on max_prefill_position + 1, matching
    # transformers and the other native M-RoPE models.
    tokens = [_TEXT, _IM_START, _IMAGE_PAD, _IM_END, _TEXT]
    positions, delta = _canvas_positions(tokens, [[4, 4]])

    max_position = max(max(channel) for channel in positions)
    assert delta == max_position + 1 - len(tokens)
