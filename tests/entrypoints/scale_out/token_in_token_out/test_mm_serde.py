# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Roundtrip tests for multimodal serde used by the
token_in_token_out generate endpoint.
"""

import pytest
import torch

from vllm.entrypoints.scale_out.token_in_token_out.mm_serde import (
    decode_mm_kwargs_item,
    encode_mm_kwargs_item,
)
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    MultiModalFeatures,
    PlaceholderRangeInfo,
)
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFieldConfig,
    MultiModalFieldElem,
    MultiModalFlatField,
    MultiModalKwargsItem,
    MultiModalSharedField,
)


class _QwenLikeProcessor:
    def _get_mm_fields_config(self, hf_inputs, _hf_processor_mm_kwargs):
        image_grid_thw = hf_inputs["image_grid_thw"]
        return {
            "pixel_values": MultiModalFieldConfig.flat_from_sizes(
                "image", image_grid_thw.prod(-1)
            ),
            "image_grid_thw": MultiModalFieldConfig.batched("image", keep_on_cpu=True),
        }


class _ConfiguredProcessor:
    def __init__(self, key, config):
        self.key = key
        self.config = config

    def _get_mm_fields_config(self, _hf_inputs, _hf_processor_mm_kwargs):
        return {self.key: self.config}


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


def test_mm_features_ignore_caller_selected_processor_context():
    features = MultiModalFeatures.model_validate(
        {
            "mm_hashes": {"image": ["abc123"]},
            "mm_placeholders": {"image": [{"offset": 0, "length": 4}]},
            "mm_processor_kwargs": {"pixel_layout": "batched"},
        }
    )

    assert "mm_processor_kwargs" not in features.model_dump()


def _qwen_like_item(pixel_field) -> MultiModalKwargsItem:
    return MultiModalKwargsItem(
        {
            "pixel_values": MultiModalFieldElem(
                data=torch.zeros((4, 3), dtype=torch.float32),
                field=pixel_field,
            ),
            "image_grid_thw": MultiModalFieldElem(
                data=torch.tensor([1, 2, 2]),
                field=MultiModalFieldConfig.batched("image", keep_on_cpu=True).field,
            ),
        }
    )


def test_decode_rebinds_valid_fields_to_model_schema():
    item = _qwen_like_item(MultiModalFieldConfig.flat("image", [slice(0, 4)]).field)

    decoded = decode_mm_kwargs_item(
        encode_mm_kwargs_item(item),
        modality="image",
        mm_processor=_QwenLikeProcessor(),
    )

    assert isinstance(decoded["pixel_values"].field, MultiModalFlatField)
    assert isinstance(decoded["image_grid_thw"].field, MultiModalBatchedField)


@pytest.mark.parametrize(
    ("modality", "config"),
    [
        ("audio", MultiModalFieldConfig.batched("audio", keep_on_cpu=True)),
        ("image", MultiModalFieldConfig.flat("image", [slice(0, 4)])),
        ("image", MultiModalFieldConfig.shared("image", batch_size=1)),
    ],
)
def test_decode_preserves_valid_non_qwen_field_layouts(modality, config):
    item = MultiModalKwargsItem(
        {
            "payload": MultiModalFieldElem(
                data=torch.zeros((4, 3), dtype=torch.float32),
                field=config.field,
            )
        }
    )

    decoded = decode_mm_kwargs_item(
        encode_mm_kwargs_item(item),
        modality=modality,
        mm_processor=_ConfiguredProcessor("payload", config),
    )

    assert type(decoded["payload"].field) is type(config.field)
    assert decoded["payload"].field.keep_on_cpu == config.field.keep_on_cpu


@pytest.mark.parametrize(
    "forged_field",
    [
        MultiModalFieldConfig.batched("image").field,
        MultiModalFieldConfig.flat("image", [slice(0, 3)], dim=1).field,
    ],
)
def test_decode_rejects_wire_selected_field_processor(forged_field):
    item = _qwen_like_item(forged_field)

    with pytest.raises(ValueError, match="field processor"):
        decode_mm_kwargs_item(
            encode_mm_kwargs_item(item),
            modality="image",
            mm_processor=_QwenLikeProcessor(),
        )
