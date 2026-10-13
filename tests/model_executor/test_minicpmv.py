# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.model_executor.models.minicpmv import MiniCPMV2_6
from vllm.model_executor.models.minicpmv4_6 import (
    MiniCPMV4_6ForConditionalGeneration,
    _expand_flags,
    _select_vision_items,
    _vision_downsample_mode,
)

pytestmark = pytest.mark.skip_global_cleanup


def make_model() -> MiniCPMV2_6:
    return object.__new__(MiniCPMV2_6)


def make_model_4_6() -> MiniCPMV4_6ForConditionalGeneration:
    model = object.__new__(MiniCPMV4_6ForConditionalGeneration)
    model._process_vision_input = lambda vision_input, use_vit_merger=None: [
        vision_input["image_embeds"]
    ]
    return model


def test_video_embeds_reach_the_vision_parser():
    embeds = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)

    modalities = make_model()._parse_and_validate_multimodal_inputs(video_embeds=embeds)

    video_input = modalities["videos"]
    assert video_input is not None
    assert video_input["type"] == "image_embeds"
    assert torch.equal(video_input["image_embeds"], embeds)


def test_video_embeds_are_embedded_by_4_6():
    embeds = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)

    embeddings = make_model_4_6().embed_multimodal(video_embeds=embeds)

    assert len(embeddings) == 1
    assert torch.equal(embeddings[0], embeds)


def test_image_and_video_embeds_stay_in_their_own_modality():
    image_embeds = torch.zeros(1, 4, 8)
    video_embeds = torch.ones(1, 4, 8)

    modalities = make_model()._parse_and_validate_multimodal_inputs(
        image_embeds=image_embeds,
        video_embeds=video_embeds,
    )

    assert torch.equal(modalities["images"]["image_embeds"], image_embeds)
    assert torch.equal(modalities["videos"]["image_embeds"], video_embeds)


class TestPerItemMergerFlags:
    """The merger flag is per request but items from several requests can be
    batched together, so it has to be applied per item."""

    def test_expand_flags_broadcasts_a_single_value(self):
        assert _expand_flags(None, 3) == [None, None, None]
        assert _expand_flags(torch.tensor([True]), 3) == [True, True, True]
        assert _expand_flags(True, 2) == [True, True]

    def test_expand_flags_keeps_per_item_values(self):
        flags = [torch.tensor([True]), torch.tensor([False])]

        assert _expand_flags(flags, 2) == [True, False]

    def test_expand_flags_falls_back_on_a_shape_mismatch(self):
        # Too many values to map onto the items: keep one decision for the
        # batch rather than failing the request.
        assert _expand_flags([True, False, True], 2) == [True, True]

    def test_vision_downsample_mode(self):
        assert _vision_downsample_mode(None) is None
        assert _vision_downsample_mode(True) == "16x"
        assert _vision_downsample_mode(False) == "4x"

    def test_select_vision_items_slices_by_item(self):
        # 3 items holding 2, 1 and 2 slices; one channel, one patch, width 4.
        pixel_values = [torch.full((1, 1, 4), float(i)) for i in range(5)]
        tgt_sizes = torch.arange(10, dtype=torch.long).reshape(5, 2)
        image_input = {
            "pixel_values": pixel_values,
            "tgt_sizes": tgt_sizes,
            "num_slices": torch.tensor([2, 1, 2]),
        }

        selected = _select_vision_items(image_input, [2], [2, 1, 2])

        assert len(selected["pixel_values"]) == 2
        assert bool(selected["pixel_values"][0][0, 0, 0] == 3.0)
        assert bool(selected["pixel_values"][1][0, 0, 0] == 4.0)
        assert selected["num_slices"].tolist() == [2]
        assert torch.equal(selected["tgt_sizes"], tgt_sizes[3:5])

    def test_select_vision_items_preserves_order(self):
        pixel_values = [torch.full((1, 1, 4), float(i)) for i in range(3)]
        image_input = {
            "pixel_values": pixel_values,
            "tgt_sizes": torch.zeros(3, 2, dtype=torch.long),
            "num_slices": torch.tensor([1, 1, 1]),
        }

        selected = _select_vision_items(image_input, [2, 0], [1, 1, 1])

        assert selected["num_slices"].tolist() == [1, 1]
        assert bool(selected["pixel_values"][0][0, 0, 0] == 2.0)
        assert bool(selected["pixel_values"][1][0, 0, 0] == 0.0)

    def test_process_vision_input_applies_flags_per_item(self):
        calls: list[str | None] = []

        def fake_get_vision_hidden_states(data, downsample_mode=None):
            calls.append(downsample_mode)
            tag = 1.0 if downsample_mode == "16x" else 0.0
            return [torch.full((1,), tag) for _ in data["pixel_values"]]

        model = object.__new__(MiniCPMV4_6ForConditionalGeneration)
        model.get_vision_hidden_states = fake_get_vision_hidden_states

        image_input = {
            "type": "pixel_values",
            "pixel_values": [torch.zeros(1, 1, 4) for _ in range(3)],
            "tgt_sizes": torch.zeros(3, 2, dtype=torch.long),
            "num_slices": torch.tensor([1, 1, 1]),
        }

        result = model._process_vision_input(
            image_input,
            use_vit_merger=[
                torch.tensor([True]),
                torch.tensor([False]),
                torch.tensor([True]),
            ],
        )

        # Each item keeps its own mode even though they share one call.
        assert [float(r[0]) for r in result] == [1.0, 0.0, 1.0]
        # Mixed flags are encoded as separate groups, not one collapsed batch.
        assert calls == ["16x", "4x"]

    def test_process_vision_input_uses_one_group_for_uniform_flags(self):
        calls: list[str | None] = []

        def fake_get_vision_hidden_states(data, downsample_mode=None):
            calls.append(downsample_mode)
            return [torch.zeros(1) for _ in data["pixel_values"]]

        model = object.__new__(MiniCPMV4_6ForConditionalGeneration)
        model.get_vision_hidden_states = fake_get_vision_hidden_states

        image_input = {
            "type": "pixel_values",
            "pixel_values": [torch.zeros(1, 1, 4) for _ in range(2)],
            "tgt_sizes": torch.zeros(2, 2, dtype=torch.long),
            "num_slices": torch.tensor([1, 1]),
        }

        model._process_vision_input(image_input, use_vit_merger=torch.tensor([True]))

        # The common case still makes a single encoder call.
        assert calls == ["16x"]
