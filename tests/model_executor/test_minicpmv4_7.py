# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.model_executor.models.minicpmv4_6 import _stack_vit_merger_qkv
from vllm.model_executor.models.minicpmv4_7 import (
    MiniCPMV4_7MultiModalProcessor,
    _compute_canvas_single,
    build_image_bounds,
    canvas_rope_delta,
)

pytestmark = pytest.mark.skip_global_cleanup


def test_canvas_mrope_sizes_survive_prefix_cache_stripping():
    """A cached image must keep `tgt_sizes`, or canvas M-RoPE falls back.

    `strip_covered_mm_data` drops the payload of a prefix-cache-covered item and
    keeps only CPU-side metadata. Canvas M-RoPE places slices from `tgt_sizes`,
    so 4.7 marks it `keep_on_cpu` while the shared 4.6 config stays as it was.
    """
    from vllm.model_executor.models.minicpmv4_6 import _minicpmv4_6_field_config

    shared = _minicpmv4_6_field_config({})
    assert shared["tgt_sizes"].field.keep_on_cpu is False
    assert shared["video_tgt_sizes"].field.keep_on_cpu is False

    processor = MiniCPMV4_7MultiModalProcessor.__new__(MiniCPMV4_7MultiModalProcessor)
    fields = processor._get_mm_fields_config({}, {})

    assert fields["tgt_sizes"].field.keep_on_cpu is True
    assert fields["video_tgt_sizes"].field.keep_on_cpu is True
    # Other fields keep their previous setting.
    assert fields["pixel_values"].field.keep_on_cpu is False
    assert fields["video_image_sizes"].field.keep_on_cpu is True


def test_modality_mm_kwargs_resolution():
    """Flat keys reach both modalities; scopes stay scoped, and win on conflict.

    `_merge_and_resolve_mm_processor_kwargs` routes a flat key into every scope
    whose HF processor declares it, so reading is a plain scope lookup. The
    routing itself is upstream's, tested here against the MiniCPM-V schema to
    pin the precedence rule (`scoped > flat`).
    """
    from vllm.model_executor.models.minicpmv4_6 import (
        _flat_processor_kwargs,
        _scoped_mm_kwarg,
    )
    from vllm.multimodal.processing.context import _resolve_mm_processor_kwargs

    # What MiniCPMV4_6{Image,Video}ProcessorKwargs declare.
    processor_keys = {"downsample_mode", "max_slice_nums"}
    schema = {
        "text_kwargs": set(),
        "images_kwargs": set(processor_keys),
        "videos_kwargs": set(processor_keys),
        "audio_kwargs": set(),
    }

    def read(mm_kwargs, modality, key):
        merged = _resolve_mm_processor_kwargs(dict(mm_kwargs), schema)
        return _scoped_mm_kwarg(merged, modality, key)

    assert read({}, "image", "downsample_mode") is None
    # A flat key is routed into both scopes.
    assert read({"downsample_mode": "4x"}, "image", "downsample_mode") == "4x"
    assert read({"downsample_mode": "4x"}, "video", "downsample_mode") == "4x"
    # A scoped key stays in its scope.
    assert (
        read({"images_kwargs": {"downsample_mode": "16x"}}, "image", "downsample_mode")
        == "16x"
    )
    assert (
        read({"images_kwargs": {"downsample_mode": "16x"}}, "video", "downsample_mode")
        is None
    )
    videos_only = {"videos_kwargs": {"max_slice_nums": 1}}
    assert read(videos_only, "video", "max_slice_nums") == 1
    # Scoped to videos, so image must not see it.
    assert read(videos_only, "image", "max_slice_nums") is None
    # An explicit scope wins over the flat key it is merged with.
    assert (
        read(
            {"downsample_mode": "16x", "images_kwargs": {"downsample_mode": "4x"}},
            "image",
            "downsample_mode",
        )
        == "4x"
    )
    # A non-mapping scope is ignored rather than raising.
    assert read({"images_kwargs": None}, "image", "downsample_mode") is None

    # `_flat_processor_kwargs` drops the scoped keys and pins the mode.
    flat = _flat_processor_kwargs(
        {
            "downsample_mode": "4x",
            "images_kwargs": {"a": 1},
            "videos_kwargs": {"b": 2},
            "audio_kwargs": {"c": 3},
        },
        "16x",
    )
    assert flat == {"downsample_mode": "16x"}


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


def test_canvas_mrope_trailing_newline_stays_inside_the_canvas():
    """A newline right after `</image>` belongs to the canvas, not to the text.

    The chat template joins content parts with ``\\n``, so an image followed by
    text always leaves one there. The transformers reference closes a frame over
    every marker including newlines; leaving it out pushes every later position
    one step forward and makes `rope_delta` differ by one.
    """
    tokens = [_TEXT, _IM_START, _IMAGE_PAD, _IM_END, _NEWLINE, _TEXT, _TEXT]
    positions, delta = _canvas_positions(tokens, [[4, 4]])

    assert positions == [
        [0, 1, 1, 1, 1, 3, 4],
        [0, 0, 1, 2, 1, 3, 4],
        [0, 0, 1, 2, 1, 3, 4],
    ]
    assert delta == -2


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


def test_canvas_span_is_shared_by_image_and_video():
    """A video frame and a same-geometry image must place identical offsets.

    Both paths go through `_assign_canvas_span`; this pins that they cannot
    drift apart now that the duplicated geometry is gone.
    """
    thumb = [_IM_START] + [_IMAGE_PAD] * 4 + [_IM_END]

    img_pos, _ = _canvas_positions([_TEXT, *thumb, _TEXT], [[8, 8]])
    vid_pos, _ = _canvas_positions([_TEXT, *thumb, *thumb, _TEXT], [[8, 8], [8, 8]])

    def offsets(pos, start):
        # The <im_start> halo is one below the canvas base.
        base = pos[1][start] + 1
        return [[c[start + k] - base for k in range(3)] for c in pos]

    assert offsets(img_pos, 0) == offsets(vid_pos, 0)


def test_canvas_mrope_delta_leaves_room_for_the_first_decoded_token():
    # First decoded token must land on max_prefill_position + 1, matching
    # transformers and the other native M-RoPE models.
    tokens = [_TEXT, _IM_START, _IMAGE_PAD, _IM_END, _TEXT]
    positions, delta = _canvas_positions(tokens, [[4, 4]])

    max_position = max(max(channel) for channel in positions)
    assert delta == max_position + 1 - len(tokens)
