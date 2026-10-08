# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from torch import nn

from vllm.model_executor.models.interfaces import EagleModelMixin, supports_eagle3
from vllm.model_executor.models.minicpmv4_6 import (
    MiniCPMV4_6ForConditionalGeneration,
    _stack_vit_merger_qkv,
)
from vllm.model_executor.models.minicpmv4_7 import (
    MiniCPMV4_7ForConditionalGeneration,
    MiniCPMV4_7MultiModalProcessor,
    MiniCPMV4_7ProcessingInfo,
    _compute_canvas_single,
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


@pytest.mark.parametrize(
    "model_cls",
    [MiniCPMV4_6ForConditionalGeneration, MiniCPMV4_7ForConditionalGeneration],
)
def test_exposes_eagle3_aux_hidden_states(model_cls):
    """dspark/EAGLE-3 drafting needs the target to hand out aux hidden states.

    The GPU model runner enables auxiliary hidden state outputs for `dspark`
    and then calls ``set_eagle3_aux_hidden_state_layers``, which raises unless
    the target declares ``SupportsEagle3``. MiniCPM-V has no layers of its own
    to hook: it delegates to the Qwen3.5 text backbone, which already
    implements the interface.
    """

    class DummyBackbone(nn.Module, EagleModelMixin):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Identity(), nn.Identity()])

    class DummyLanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = DummyBackbone()

        def embed_input_ids(self, input_ids):
            return input_ids

    model = model_cls.__new__(model_cls)
    nn.Module.__init__(model)
    model.language_model = DummyLanguageModel()

    assert supports_eagle3(model)
    model.set_aux_hidden_state_layers((1, 2))
    assert model.language_model.model.aux_hidden_state_layers == (1, 2)


def test_v47_does_not_read_drop_vision_last_layer():
    """MiniCPMV4_7Config drops the field the 4.6 constructor reads.

    The shared 4.6 ``__init__`` builds the vision tower, so 4.7 overrides the
    hook instead of the base class tolerating a missing field. 4.6 keeps
    reading the config directly.
    """
    from types import SimpleNamespace

    def make(cls, config):
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.config = config
        return model

    class NoSuchField:
        """Stands in for MiniCPMV4_7Config, which has no such attribute."""

    v47 = make(MiniCPMV4_7ForConditionalGeneration, NoSuchField())
    assert v47._drop_vision_last_layer() is False

    for value in (True, False):
        v46 = make(
            MiniCPMV4_6ForConditionalGeneration,
            SimpleNamespace(drop_vision_last_layer=value),
        )
        assert v46._drop_vision_last_layer() is value


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
    """Flat keys apply to both modalities; nested scopes apply to one.

    This is what `--mm-processor-kwargs` and per-request overrides are read
    through, so `images_kwargs` must not leak into video and vice versa.
    """
    from vllm.model_executor.models.minicpmv4_6 import (
        _flat_processor_kwargs,
        _resolve_modality_mm_kwarg,
    )

    resolve = _resolve_modality_mm_kwarg

    assert resolve({}, "image", "downsample_mode") is None
    assert resolve({"downsample_mode": "4x"}, "image", "downsample_mode") == "4x"
    assert resolve({"downsample_mode": "4x"}, "video", "downsample_mode") == "4x"
    assert (
        resolve(
            {"images_kwargs": {"downsample_mode": "16x"}}, "image", "downsample_mode"
        )
        == "16x"
    )
    # Scoped to images, so video must not see it.
    assert (
        resolve(
            {"images_kwargs": {"downsample_mode": "16x"}}, "video", "downsample_mode"
        )
        is None
    )
    assert (
        resolve({"videos_kwargs": {"max_slice_nums": 1}}, "video", "max_slice_nums")
        == 1
    )
    # A non-mapping scope is ignored rather than raising.
    assert resolve({"images_kwargs": None}, "image", "downsample_mode") is None

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
