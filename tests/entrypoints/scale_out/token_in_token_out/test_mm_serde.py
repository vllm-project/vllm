# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Roundtrip tests for multimodal serde used by the
token_in_token_out generate endpoint.
"""

import torch
from pydantic import ValidationError

from vllm.entrypoints.scale_out.token_in_token_out.mm_features import (
    extract_mm_features,
    merge_mm_kwargs_items,
    mm_kwargs_from_features,
)
from vllm.entrypoints.scale_out.token_in_token_out.mm_serde import (
    decode_mm_kwargs_item,
    encode_mm_kwargs_item,
)
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    GenerateRequest,
    MultiModalFeatures,
    PlaceholderRangeInfo,
)
from vllm.inputs import mm_input
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFieldElem,
    MultiModalFlatField,
    MultiModalKwargsItem,
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


def _image_engine_input(
    *,
    pixel_keep_on_cpu: bool = False,
    grid_keep_on_cpu: bool = True,
) -> tuple[object, MultiModalFieldElem, MultiModalFieldElem]:
    pixel_values = MultiModalFieldElem(
        data=torch.randn(5, 3, dtype=torch.float32),
        field=MultiModalBatchedField(keep_on_cpu=pixel_keep_on_cpu),
    )
    image_grid_thw = MultiModalFieldElem(
        data=torch.tensor([[1, 24, 24]], dtype=torch.int32),
        field=MultiModalBatchedField(keep_on_cpu=grid_keep_on_cpu),
    )
    item = MultiModalKwargsItem(
        {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}
    )
    engine_input = mm_input(
        prompt_token_ids=[1, 2, 3],
        mm_kwargs={"image": [item]},
        mm_hashes={"image": ["abc123"]},
        mm_placeholders={"image": [PlaceholderRange(offset=0, length=1)]},
    )
    return engine_input, pixel_values, image_grid_thw


def test_render_extracts_metadata_fields_separately():
    """Render metadata contains CPU fields but excludes encoder inputs."""
    engine_input, pixel_values, image_grid_thw = _image_engine_input()

    features = extract_mm_features(engine_input)

    assert features is not None
    assert features.mm_metadata is not None
    features = MultiModalFeatures.model_validate_json(features.model_dump_json())
    assert features.mm_metadata is not None
    metadata = decode_mm_kwargs_item(features.mm_metadata["image"][0])
    assert set(metadata) == {"image_grid_thw"}
    assert torch.equal(metadata["image_grid_thw"].data, image_grid_thw.data)

    assert features.kwargs_data is not None
    full_data = decode_mm_kwargs_item(features.kwargs_data["image"][0])
    assert set(full_data) == {"pixel_values", "image_grid_thw"}
    assert torch.equal(full_data["pixel_values"].data, pixel_values.data)


def test_extract_includes_declared_placeholder_metadata_fields():
    """EC placeholder metadata is kept even when keep_on_cpu is unset."""
    engine_input, _pixel_values, image_grid_thw = _image_engine_input(
        grid_keep_on_cpu=False
    )

    features = extract_mm_features(
        engine_input,
        metadata_fields_for=lambda modality: (
            {"image_grid_thw"} if modality == "image" else set()
        ),
    )

    assert features is not None
    assert features.mm_metadata is not None
    metadata = decode_mm_kwargs_item(features.mm_metadata["image"][0])
    assert set(metadata) == {"image_grid_thw"}
    assert torch.equal(metadata["image_grid_thw"].data, image_grid_thw.data)


def test_legacy_kwargs_only_generate_keeps_full_payload():
    """Old clients that omit mm_metadata still reconstruct full kwargs."""
    engine_input, pixel_values, image_grid_thw = _image_engine_input()
    rendered = extract_mm_features(engine_input)
    assert rendered is not None
    assert rendered.kwargs_data is not None

    features = MultiModalFeatures(
        mm_hashes=rendered.mm_hashes,
        mm_placeholders=rendered.mm_placeholders,
        kwargs_data=rendered.kwargs_data,
    )
    merged = mm_kwargs_from_features(features)
    item = merged["image"][0]
    assert item is not None
    assert set(item) == {"pixel_values", "image_grid_thw"}
    assert torch.equal(item["pixel_values"].data, pixel_values.data)
    assert torch.equal(item["image_grid_thw"].data, image_grid_thw.data)


def test_metadata_only_generate_requires_ec_transfer_params():
    engine_input, _, _ = _image_engine_input()
    rendered = extract_mm_features(engine_input)
    assert rendered is not None
    payload = {
        "token_ids": [1, 2, 3],
        "sampling_params": {"max_tokens": 1},
        "features": {
            "mm_hashes": rendered.mm_hashes,
            "mm_placeholders": {
                modality: [p.model_dump() for p in ranges]
                for modality, ranges in rendered.mm_placeholders.items()
            },
            "mm_metadata": rendered.mm_metadata,
        },
    }

    try:
        GenerateRequest.model_validate(payload)
        raise AssertionError("expected metadata-only request without EC to fail")
    except ValidationError as exc:
        assert "ec_transfer_params" in str(exc)

    payload["ec_transfer_params"] = {"image-hash": {"peer_host": "10.0.0.1"}}
    request = GenerateRequest.model_validate(payload)
    assert request.features is not None
    assert request.features.kwargs_data is None
    merged = mm_kwargs_from_features(request.features)
    item = merged["image"][0]
    assert item is not None
    assert set(item) == {"image_grid_thw"}


def test_kwargs_and_metadata_generate_does_not_require_ec():
    engine_input, pixel_values, image_grid_thw = _image_engine_input()
    rendered = extract_mm_features(engine_input)
    assert rendered is not None
    request = GenerateRequest.model_validate(
        {
            "token_ids": [1, 2, 3],
            "sampling_params": {"max_tokens": 1},
            "features": rendered.model_dump(),
        }
    )
    assert request.features is not None
    merged = mm_kwargs_from_features(request.features)
    item = merged["image"][0]
    assert item is not None
    assert set(item) == {"pixel_values", "image_grid_thw"}
    assert torch.equal(item["pixel_values"].data, pixel_values.data)
    assert torch.equal(item["image_grid_thw"].data, image_grid_thw.data)


def test_cache_hit_null_kwargs_keeps_metadata_slot():
    pixel_values = MultiModalFieldElem(
        data=torch.randn(5, 3, dtype=torch.float32),
        field=MultiModalBatchedField(),
    )
    image_grid_thw = MultiModalFieldElem(
        data=torch.tensor([[1, 24, 24]], dtype=torch.int32),
        field=MultiModalBatchedField(keep_on_cpu=True),
    )
    item = MultiModalKwargsItem(
        {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}
    )
    engine_input = mm_input(
        prompt_token_ids=[1, 2, 3, 4],
        mm_kwargs={"image": [item, None]},
        mm_hashes={"image": ["hit", "cached"]},
        mm_placeholders={
            "image": [
                PlaceholderRange(offset=0, length=1),
                PlaceholderRange(offset=2, length=1),
            ]
        },
    )

    features = extract_mm_features(engine_input)
    assert features is not None
    assert features.kwargs_data is not None
    assert features.mm_metadata is not None
    assert features.kwargs_data["image"][1] is None
    assert features.mm_metadata["image"][1] is None
    assert len(features.mm_metadata["image"]) == 2
    metadata = decode_mm_kwargs_item(features.mm_metadata["image"][0])
    assert torch.equal(metadata["image_grid_thw"].data, image_grid_thw.data)


def test_features_alignment_rejects_length_mismatch():
    try:
        MultiModalFeatures(
            mm_hashes={"image": ["a", "b"]},
            mm_placeholders={"image": [PlaceholderRangeInfo(offset=0, length=1)]},
        )
    except ValidationError:
        return
    raise AssertionError("expected placeholder length mismatch to fail")


def test_metadata_can_replace_or_extend_full_mm_data():
    """Metadata-only requests retain the fields required by the model."""
    metadata_elem = MultiModalFieldElem(
        data=torch.tensor([[1, 24, 24]], dtype=torch.int32),
        field=MultiModalBatchedField(keep_on_cpu=True),
    )
    metadata_item = MultiModalKwargsItem({"image_grid_thw": metadata_elem})
    pixel_elem = MultiModalFieldElem(
        data=torch.randn(5, 3),
        field=MultiModalBatchedField(),
    )
    full_item = MultiModalKwargsItem(
        {"pixel_values": pixel_elem, "image_grid_thw": metadata_elem}
    )

    assert merge_mm_kwargs_items(None, metadata_item) == metadata_item
    merged = merge_mm_kwargs_items(full_item, metadata_item)
    assert merged is not None
    assert set(merged) == {"pixel_values", "image_grid_thw"}
