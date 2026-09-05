# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Roundtrip tests for multimodal serde used by the
token_in_token_out generate endpoint.
"""

import torch

from vllm.entrypoints.scale_out.render.serving import ServingRender
from vllm.entrypoints.scale_out.token_in_token_out.mm_serde import (
    decode_mm_kwargs_item,
    encode_mm_kwargs_item,
)
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    MultiModalFeatures,
    PlaceholderRangeInfo,
)
from vllm.inputs import mm_input
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFieldElem,
    MultiModalFlatField,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    MultiModalSharedField,
    PlaceholderRange,
)


def test_mm_kwargs_item_roundtrip():
    """Full roundtrip test with all three field types and multiple dtypes."""
    e1 = MultiModalFieldElem(
        data=torch.zeros(1000, dtype=torch.bfloat16),
        field=MultiModalBatchedField(),
    )
    e2 = MultiModalFieldElem(
        data=torch.ones(100, dtype=torch.int32),
        field=MultiModalSharedField(batch_size=4),
    )
    e3 = MultiModalFieldElem(
        data=torch.randn(20, dtype=torch.float32),
        field=MultiModalFlatField(slices=[slice(0, 10), slice(10, 20)], dim=0),
    )

    item = MultiModalKwargsItem({"pixel_values": e1, "grid_thw": e2, "embeds": e3})
    encoded = encode_mm_kwargs_item(item)

    # Encoded result is a base64 string
    assert isinstance(encoded, str)

    decoded = decode_mm_kwargs_item(encoded)

    assert set(decoded.keys()) == {"pixel_values", "grid_thw", "embeds"}
    assert torch.equal(item["pixel_values"].data, decoded["pixel_values"].data)
    assert torch.equal(item["grid_thw"].data, decoded["grid_thw"].data)
    assert torch.equal(item["embeds"].data, decoded["embeds"].data)
    assert isinstance(decoded["pixel_values"].field, MultiModalBatchedField)
    assert isinstance(decoded["grid_thw"].field, MultiModalSharedField)
    assert isinstance(decoded["embeds"].field, MultiModalFlatField)


def test_mm_kwargs_item_none_data():
    """Roundtrip with None data field."""
    elem = MultiModalFieldElem(
        data=None,
        field=MultiModalSharedField(batch_size=2),
    )
    item = MultiModalKwargsItem({"empty": elem})
    encoded = encode_mm_kwargs_item(item)
    decoded = decode_mm_kwargs_item(encoded)

    assert decoded["empty"].data is None
    assert isinstance(decoded["empty"].field, MultiModalSharedField)


def test_mm_kwargs_item_nested_tensors():
    """Roundtrip with nested tensor data."""
    nested = [torch.randn(3, 4), torch.randn(5, 4)]
    elem = MultiModalFieldElem(
        data=nested,
        field=MultiModalBatchedField(),
    )
    item = MultiModalKwargsItem({"nested": elem})
    encoded = encode_mm_kwargs_item(item)
    decoded = decode_mm_kwargs_item(encoded)

    decoded_data = decoded["nested"].data
    assert len(decoded_data) == 2
    assert torch.equal(nested[0], decoded_data[0])
    assert torch.equal(nested[1], decoded_data[1])


def test_mm_features_with_kwargs_data():
    """Test that MultiModalFeatures can carry serialized tensor data."""
    elem = MultiModalFieldElem(
        data=torch.randn(5, 3, dtype=torch.float32),
        field=MultiModalBatchedField(),
    )
    item = MultiModalKwargsItem({"pixel_values": elem})
    encoded = encode_mm_kwargs_item(item)

    features = MultiModalFeatures(
        mm_hashes={"image": ["abc123"]},
        mm_placeholders={"image": [PlaceholderRangeInfo(offset=0, length=10)]},
        kwargs_data={"image": [encoded]},
    )

    # JSON roundtrip
    json_str = features.model_dump_json()
    features2 = MultiModalFeatures.model_validate_json(json_str)

    assert features2.mm_hashes == {"image": ["abc123"]}
    assert features2.kwargs_data is not None
    assert len(features2.kwargs_data["image"]) == 1

    decoded = decode_mm_kwargs_item(features2.kwargs_data["image"][0])
    assert torch.equal(elem.data, decoded["pixel_values"].data)


def test_render_features_preserve_is_embed():
    """A sparse `is_embed` mask must reach the generate side intact.

    The model runner branches on `PlaceholderRange.is_embed`: with a mask a
    span consumes only `is_embed.sum()` rows of encoder output and only those
    positions are marked as embeddings; without one it consumes the whole span
    and every position in it is overwritten. Dropping the mask on the wire
    therefore corrupts the prompt for models that use sparse placeholders
    (Gemma 3, Phi-3-V, the Qwen omni thinkers).
    """
    is_embed = torch.tensor([True, False, True, True], dtype=torch.bool)
    engine_input = mm_input(
        prompt_token_ids=[1, 2, 3, 4, 5],
        mm_kwargs=MultiModalKwargsItems({}),
        mm_hashes={"image": ["hash-0"]},
        mm_placeholders={
            "image": [PlaceholderRange(offset=1, length=4, is_embed=is_embed)]
        },
    )

    features = ServingRender._extract_mm_features(engine_input)
    assert features is not None

    # Assert on the serialized form: this is what actually crosses the wire
    # to the generate service.
    dumped = features.model_dump()["mm_placeholders"]["image"][0]
    assert dumped.get("is_embed") == [True, False, True, True]


def test_rebuild_mm_placeholders_restores_is_embed():
    from vllm.entrypoints.scale_out.token_in_token_out.serving import (
        rebuild_mm_placeholders,
    )

    is_embed = [True, False, True, True]
    features = MultiModalFeatures(
        mm_hashes={"image": ["hash-0"]},
        mm_placeholders={
            "image": [PlaceholderRangeInfo(offset=1, length=4, is_embed=is_embed)]
        },
    )
    features = MultiModalFeatures.model_validate(features.model_dump())

    (placeholder,) = rebuild_mm_placeholders(features.mm_placeholders)["image"]
    assert placeholder.offset == 1
    assert placeholder.length == 4
    assert placeholder.is_embed is not None
    assert torch.equal(placeholder.is_embed, torch.tensor(is_embed, dtype=torch.bool))
    # The row count the model runner slices the encoder output by.
    assert placeholder.get_num_embeds() == 3


def test_placeholder_without_is_embed_roundtrips_as_none():
    from vllm.entrypoints.scale_out.token_in_token_out.serving import (
        rebuild_mm_placeholders,
    )

    engine_input = mm_input(
        prompt_token_ids=[1, 2, 3],
        mm_kwargs=MultiModalKwargsItems({}),
        mm_hashes={"image": ["hash-0"]},
        mm_placeholders={"image": [PlaceholderRange(offset=0, length=3)]},
    )

    features = ServingRender._extract_mm_features(engine_input)
    assert features is not None
    assert features.mm_placeholders["image"][0].is_embed is None

    (placeholder,) = rebuild_mm_placeholders(features.mm_placeholders)["image"]
    assert placeholder.is_embed is None
    assert placeholder.get_num_embeds() == 3
