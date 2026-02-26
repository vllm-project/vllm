# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Roundtrip tests for multimodal serde used by the disagg generate endpoint."""

import torch

from vllm.entrypoints.serve.disagg.mm_serde import (
    decode_field,
    decode_field_elem,
    decode_mm_kwargs_item,
    decode_nested_slices,
    decode_nested_tensors,
    decode_tensor,
    encode_field,
    encode_field_elem,
    encode_mm_kwargs_item,
    encode_nested_slices,
    encode_nested_tensors,
    encode_tensor,
)
from vllm.entrypoints.serve.disagg.protocol import (
    GenerateMultiModalFeature,
    MultiModalKwargsItemData,
)
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFieldElem,
    MultiModalFlatField,
    MultiModalKwargsItem,
    MultiModalSharedField,
    nested_tensors_equal,
)


def test_tensor_roundtrip_float32():
    t = torch.randn(10, 3, dtype=torch.float32)
    td = encode_tensor(t)
    t2 = decode_tensor(td)
    assert torch.equal(t, t2)


def test_tensor_roundtrip_bfloat16():
    t = torch.randn(100, dtype=torch.bfloat16)
    td = encode_tensor(t)
    t2 = decode_tensor(td)
    assert torch.equal(t, t2)


def test_tensor_roundtrip_int32():
    t = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    td = encode_tensor(t)
    t2 = decode_tensor(td)
    assert torch.equal(t, t2)


def test_tensor_roundtrip_int64():
    t = torch.tensor([10, 20, 30], dtype=torch.int64)
    td = encode_tensor(t)
    t2 = decode_tensor(td)
    assert torch.equal(t, t2)


def test_tensor_roundtrip_bool():
    t = torch.tensor([True, False, True], dtype=torch.bool)
    td = encode_tensor(t)
    t2 = decode_tensor(td)
    assert torch.equal(t, t2)


def test_nested_tensors_roundtrip():
    nt = [torch.randn(3, 4), torch.randn(5, 4)]
    encoded = encode_nested_tensors(nt)
    decoded = decode_nested_tensors(encoded)
    assert nested_tensors_equal(nt, decoded)


def test_nested_tensors_scalar():
    assert decode_nested_tensors(42) == 42
    assert decode_nested_tensors(3.14) == 3.14


def test_slice_roundtrip_simple():
    slices = [slice(0, 3, None), slice(3, 7, None)]
    encoded = encode_nested_slices(slices)
    decoded = decode_nested_slices(encoded)
    assert decoded == slices


def test_slice_roundtrip_nested():
    slices = [
        (slice(None, None, None), slice(0, 3)),
        (slice(None, None, None), slice(3, 7)),
    ]
    encoded = encode_nested_slices(slices)
    decoded = decode_nested_slices(encoded)
    assert decoded == [tuple(s) for s in slices]


def test_field_roundtrip_batched():
    field = MultiModalBatchedField(keep_on_cpu=True)
    info = encode_field(field)
    assert info.type == "batched"
    assert info.keep_on_cpu is True
    decoded = decode_field(info)
    assert isinstance(decoded, MultiModalBatchedField)
    assert decoded.keep_on_cpu is True


def test_field_roundtrip_flat():
    field = MultiModalFlatField(
        slices=[slice(0, 3), slice(3, 7)], dim=1, keep_on_cpu=False
    )
    info = encode_field(field)
    assert info.type == "flat"
    decoded = decode_field(info)
    assert isinstance(decoded, MultiModalFlatField)
    assert decoded.dim == 1
    assert decoded.slices == [slice(0, 3), slice(3, 7)]


def test_field_roundtrip_shared():
    field = MultiModalSharedField(batch_size=4)
    info = encode_field(field)
    assert info.type == "shared"
    decoded = decode_field(info)
    assert isinstance(decoded, MultiModalSharedField)
    assert decoded.batch_size == 4


def test_field_elem_roundtrip():
    elem = MultiModalFieldElem(
        data=torch.randn(10, 3, dtype=torch.float32),
        field=MultiModalBatchedField(),
    )
    encoded = encode_field_elem(elem)
    decoded = decode_field_elem(encoded)
    assert torch.equal(elem.data, decoded.data)
    assert isinstance(decoded.field, MultiModalBatchedField)


def test_field_elem_none_data():
    elem = MultiModalFieldElem(
        data=None,
        field=MultiModalSharedField(batch_size=2),
    )
    encoded = encode_field_elem(elem)
    decoded = decode_field_elem(encoded)
    assert decoded.data is None
    assert isinstance(decoded.field, MultiModalSharedField)


def test_mm_kwargs_item_roundtrip():
    """Full roundtrip test with all three field types."""
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

    # Verify it's JSON-serializable via Pydantic
    json_str = encoded.model_dump_json()
    decoded_data = MultiModalKwargsItemData.model_validate_json(json_str)
    decoded = decode_mm_kwargs_item(decoded_data)

    assert set(decoded.keys()) == {"pixel_values", "grid_thw", "embeds"}
    assert torch.equal(item["pixel_values"].data, decoded["pixel_values"].data)
    assert torch.equal(item["grid_thw"].data, decoded["grid_thw"].data)
    assert torch.equal(item["embeds"].data, decoded["embeds"].data)
    assert isinstance(decoded["pixel_values"].field, MultiModalBatchedField)
    assert isinstance(decoded["grid_thw"].field, MultiModalSharedField)
    assert isinstance(decoded["embeds"].field, MultiModalFlatField)


def test_generate_feature_with_kwargs_data():
    """Test that GenerateMultiModalFeature can carry tensor data."""
    elem = MultiModalFieldElem(
        data=torch.randn(5, 3, dtype=torch.float32),
        field=MultiModalBatchedField(),
    )
    item = MultiModalKwargsItem({"pixel_values": elem})
    kwargs_data = encode_mm_kwargs_item(item)

    feat = GenerateMultiModalFeature(
        modality="image",
        mm_hash="abc123",
        offset=0,
        length=10,
        kwargs_data=kwargs_data,
    )

    # Roundtrip via JSON
    json_str = feat.model_dump_json()
    feat2 = GenerateMultiModalFeature.model_validate_json(json_str)

    assert feat2.modality == "image"
    assert feat2.mm_hash == "abc123"
    assert feat2.kwargs_data is not None

    decoded = decode_mm_kwargs_item(feat2.kwargs_data)
    assert torch.equal(elem.data, decoded["pixel_values"].data)


def test_generate_feature_cache_only():
    """Test that GenerateMultiModalFeature works without tensor data."""
    feat = GenerateMultiModalFeature(
        modality="image",
        mm_hash="abc123",
        offset=0,
        length=10,
    )
    json_str = feat.model_dump_json()
    feat2 = GenerateMultiModalFeature.model_validate_json(json_str)
    assert feat2.kwargs_data is None
