# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Roundtrip tests for multimodal serde used by the
token_in_token_out generate endpoint.
"""

import pytest
import torch
from pydantic import ValidationError
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    PreTrainedConfig,
    Qwen2_5_VLConfig,
    Qwen2VLConfig,
    SmolVLMConfig,
    TokenizersBackend,
)

from vllm.config import ModelConfig
from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.scale_out.token_in_token_out.mm_features import (
    extract_mm_features,
    merge_mm_kwargs_items,
    mm_kwargs_from_features,
    rebuild_mm_placeholders,
)
from vllm.entrypoints.scale_out.token_in_token_out.mm_serde import (
    decode_mm_kwargs_item,
    encode_mm_kwargs_item,
)
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    GenerateRequest,
    MultiModalFeatures,
    PlaceholderRangeInfo,
    RenderedGenerateRequest,
)
from vllm.inputs import MultiModalInput, mm_input
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFieldElem,
    MultiModalFlatField,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    MultiModalSharedField,
    PlaceholderRange,
)


def _build_processor(tmp_path, config: PreTrainedConfig):
    config.save_pretrained(tmp_path)
    tokenizer = TokenizersBackend(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
    )
    model_config = ModelConfig(str(tmp_path), dtype="float32", max_model_len=512)
    return MULTIMODAL_REGISTRY.create_processor(model_config, tokenizer=tokenizer)


def test_validate_transport_layout_uses_model_processor(tmp_path):
    processor = _build_processor(
        tmp_path,
        SmolVLMConfig(architectures=["SmolVLMForConditionalGeneration"]),
    )
    embeds = torch.zeros(2, 3)

    item = MultiModalKwargsItem(
        {"image_embeds": MultiModalFieldElem(embeds, MultiModalBatchedField())}
    )
    options = dict(placeholder_length=2, metadata_only=False, allow_metadata_only=False)
    processor.validate_preprocessed_item("image", item, **options)

    item = MultiModalKwargsItem(
        {
            "image_embeds": MultiModalFieldElem(
                embeds, MultiModalSharedField(batch_size=1)
            )
        }
    )
    with pytest.raises(ValueError, match="Invalid batching layout.*image_embeds"):
        processor.validate_preprocessed_item("image", item, **options)


@pytest.fixture(
    scope="module",
    params=[
        pytest.param((Qwen2VLConfig, "Qwen2VLForConditionalGeneration"), id="qwen2-vl"),
        pytest.param(
            (Qwen2_5_VLConfig, "Qwen2_5_VLForConditionalGeneration"), id="qwen2.5-vl"
        ),
    ],
)
def qwen_feature_processor(request, tmp_path_factory):
    config_cls, architecture = request.param
    config = config_cls(architectures=[architecture])
    return _build_processor(tmp_path_factory.mktemp(architecture), config), config


@pytest.mark.parametrize(
    "variant",
    [
        "valid",
        "embeds",
        "grid-override",
        "patch-count",
        "empty",
        "rank",
        "multi-frame-image",
        "ec-metadata",
        "missing-ec",
    ],
)
def test_validate_decoded_qwen_image_features(qwen_feature_processor, variant):
    processor, config = qwen_feature_processor
    grid = MultiModalBatchedField(keep_on_cpu=True)
    pixels = MultiModalFlatField(slices=[slice(0, 256)])
    image = MultiModalKwargsItem(
        {
            "image_grid_thw": MultiModalFieldElem(torch.tensor([1, 16, 16]), grid),
            "pixel_values": MultiModalFieldElem(torch.zeros(256, 1176), pixels),
        }
    )
    metadata = MultiModalKwargsItem({"image_grid_thw": image["image_grid_thw"]})
    if variant in ("grid-override", "patch-count"):
        metadata["image_grid_thw"] = MultiModalFieldElem(torch.tensor([1, 2, 2]), grid)
    elif variant == "embeds":
        image.pop("pixel_values")
        image["image_embeds"] = MultiModalFieldElem(
            torch.zeros(64, config.get_text_config().hidden_size),
            MultiModalFlatField(slices=[slice(0, 64)]),
        )
    elif variant == "empty":
        image = MultiModalKwargsItem({})
    elif variant == "rank":
        image["pixel_values"] = MultiModalFieldElem(torch.zeros(1, 1, 1), pixels)
    elif variant == "multi-frame-image":
        image["image_grid_thw"] = MultiModalFieldElem(torch.tensor([2, 16, 16]), grid)
        image["pixel_values"] = MultiModalFieldElem(
            torch.zeros(512, 1176), MultiModalFlatField(slices=[slice(0, 512)])
        )
        metadata["image_grid_thw"] = image["image_grid_thw"]
    payload = (
        None
        if variant in ("ec-metadata", "missing-ec")
        else encode_mm_kwargs_item(image)
    )
    features = MultiModalFeatures(
        mm_hashes={"image": ["image"]},
        mm_placeholders={
            "image": [
                PlaceholderRangeInfo(
                    offset=0,
                    length=(
                        1
                        if variant == "patch-count"
                        else 128
                        if variant == "multi-frame-image"
                        else 64
                    ),
                )
            ]
        },
        kwargs_data={"image": [payload]},
        mm_metadata={"image": [encode_mm_kwargs_item(metadata)]},
    )
    (item,) = mm_kwargs_from_features(features)["image"]
    assert item is not None
    options = dict(
        placeholder_length=features.mm_placeholders["image"][0].length,
        metadata_only=payload is None,
        allow_metadata_only=variant == "ec-metadata",
    )
    if variant in ("valid", "embeds", "ec-metadata"):
        processor.validate_preprocessed_item("image", item, **options)
    else:
        with pytest.raises(ValueError, match="image|pixel_values"):
            processor.validate_preprocessed_item("image", item, **options)


def test_qwen_derived_processor_keeps_its_own_feature_rules(tmp_path):
    processor = _build_processor(
        tmp_path, Qwen2VLConfig(architectures=["JinaVLForRanking"])
    )
    item = MultiModalKwargsItem(
        {
            "image_grid_thw": MultiModalFieldElem(
                torch.tensor([1, 2, 2]), MultiModalBatchedField(keep_on_cpu=True)
            )
        }
    )
    processor.validate_preprocessed_item(
        "image",
        item,
        placeholder_length=1,
        metadata_only=True,
        allow_metadata_only=False,
    )


@pytest.mark.parametrize("modality", ["image", "video", "audio"])
@pytest.mark.parametrize("payload_field", ["kwargs_data", "mm_metadata"])
def test_mm_features_reject_reserved_model_argument(modality, payload_field):
    """Caller fields must not collide with the bound model method's receiver."""
    item = MultiModalKwargsItem(
        {"self": MultiModalFieldElem(torch.zeros(1), MultiModalBatchedField())}
    )
    features = MultiModalFeatures(
        mm_hashes={modality: ["item"]},
        mm_placeholders={modality: [PlaceholderRangeInfo(offset=0, length=1)]},
        **{payload_field: {modality: [encode_mm_kwargs_item(item)]}},
    )
    with pytest.raises(ValueError, match="reserved.*self"):
        mm_kwargs_from_features(features)


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
) -> tuple[MultiModalInput, MultiModalFieldElem, MultiModalFieldElem]:
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


def test_extract_without_kwargs_keeps_layout():
    """Layout-only callers get hashes and placeholders, no serialized tensors."""
    engine_input, _pixel_values, _image_grid_thw = _image_engine_input()

    full = extract_mm_features(engine_input)
    layout = extract_mm_features(engine_input, mm_payload="none")

    assert full is not None and layout is not None
    assert layout.kwargs_data is None
    assert layout.mm_metadata is None
    assert layout.mm_hashes == full.mm_hashes
    assert layout.mm_placeholders == full.mm_placeholders


def _mixed_engine_input(*, audio_keep_on_cpu: bool = False):
    engine_input, _pixel_values, _image_grid_thw = _image_engine_input()
    audio_item = MultiModalKwargsItem(
        {
            "input_features": MultiModalFieldElem(
                data=torch.randn(5, 3, dtype=torch.float32),
                field=MultiModalBatchedField(keep_on_cpu=audio_keep_on_cpu),
            )
        }
    )
    return mm_input(
        prompt_token_ids=[1, 2, 3],
        mm_kwargs={"image": engine_input["mm_kwargs"]["image"], "audio": [audio_item]},
        mm_hashes={"image": ["abc123"], "audio": ["def456"]},
        mm_placeholders={
            "image": [PlaceholderRange(offset=0, length=1)],
            "audio": [PlaceholderRange(offset=1, length=1)],
        },
    )


@pytest.mark.parametrize("audio_keep_on_cpu", [False, True])
def test_metadata_payload_matches_tensors_without_kwargs(audio_keep_on_cpu):
    """Only `kwargs_data` differs from a `tensors` render of the same input."""
    if audio_keep_on_cpu:
        engine_input = _mixed_engine_input(audio_keep_on_cpu=True)
    else:
        engine_input, _pixel_values, _image_grid_thw = _image_engine_input()

    full = extract_mm_features(engine_input, mm_payload="tensors")
    metadata_only = extract_mm_features(engine_input, mm_payload="metadata")

    assert full is not None and metadata_only is not None
    assert metadata_only.kwargs_data is None
    assert metadata_only.mm_metadata is not None
    assert metadata_only.mm_metadata == full.mm_metadata
    assert metadata_only.mm_hashes == full.mm_hashes
    assert metadata_only.mm_placeholders == full.mm_placeholders


def test_metadata_payload_requires_ec_on_generate_not_render():
    """Render can return metadata-only items; generate needs EC params for them."""
    engine_input, _pixel_values, _image_grid_thw = _image_engine_input()
    features = extract_mm_features(engine_input, mm_payload="metadata")
    body = {"token_ids": [1, 2, 3], "features": features, "sampling_params": {}}

    rendered = RenderedGenerateRequest.model_validate(body)
    forwarded = rendered.model_dump(mode="json")

    with pytest.raises(ValidationError, match="require ec_transfer_params"):
        GenerateRequest.model_validate(forwarded)

    GenerateRequest.model_validate(
        {**forwarded, "ec_transfer_params": {"do_remote": True}}
    )


def test_metadata_payload_rejects_modality_without_metadata():
    with pytest.raises(ValueError, match="requires metadata for every modality"):
        extract_mm_features(_mixed_engine_input(), mm_payload="metadata")


@pytest.mark.parametrize(
    ("mm_payload", "has_kwargs"),
    [("tensors", True), ("none", False)],
)
def test_modality_without_metadata_nulls_mm_metadata(mm_payload, has_kwargs):
    """Metadata is complete or absent, so partial modalities still render."""
    features = extract_mm_features(_mixed_engine_input(), mm_payload=mm_payload)

    assert features is not None
    assert features.mm_metadata is None
    assert set(features.mm_hashes) == {"image", "audio"}
    if has_kwargs:
        assert features.kwargs_data is not None
        assert set(features.kwargs_data) == {"image", "audio"}
    else:
        assert features.kwargs_data is None


_MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize(
    ("request_cls", "body"),
    [
        (ChatCompletionRequest, {"model": "m", "messages": _MESSAGES}),
        (CompletionRequest, {"model": "m", "prompt": "hi"}),
        (ResponsesRequest, {"model": "m", "input": "hi"}),
        (
            AnthropicMessagesRequest,
            {"model": "m", "max_tokens": 8, "messages": _MESSAGES},
        ),
    ],
)
def test_return_mm_kwargs_is_alias_for_mm_payload(request_cls, body):
    """Clients of the deprecated bool keep getting layout-only responses."""
    assert request_cls.model_validate(body).mm_payload == "tensors"

    for return_mm_kwargs, expected in [(False, "none"), (True, "tensors")]:
        request = request_cls.model_validate(
            {**body, "return_mm_kwargs": return_mm_kwargs}
        )
        assert request.mm_payload == expected

    explicit = request_cls.model_validate(
        {**body, "return_mm_kwargs": False, "mm_payload": "tensors"}
    )
    assert explicit.mm_payload == "tensors"

    assert (
        request_cls.model_validate({**body, "mm_payload": "metadata"}).mm_payload
        == "metadata"
    )

    with pytest.raises(ValidationError):
        request_cls.model_validate({**body, "mm_payload": "source"})


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


@pytest.mark.parametrize("modality", ["image", "video"])
@pytest.mark.parametrize(
    ("has_metadata", "has_ec"),
    [(True, False), (True, True), (False, False)],
    ids=["metadata-without-ec", "metadata-with-ec", "cache-hit-without-ec"],
)
def test_mixed_items_require_ec_only_for_metadata_only(
    modality: str, has_metadata: bool, has_ec: bool
):
    """A full item must not bypass the EC requirement for another item."""
    engine_input, _, image_grid_thw = _image_engine_input()
    rendered = extract_mm_features(engine_input)
    assert rendered is not None
    features = rendered.model_dump()
    metadata = (
        encode_mm_kwargs_item(
            MultiModalKwargsItem({f"{modality}_grid_thw": image_grid_thw})
        )
        if has_metadata
        else None
    )
    second_item = {
        "mm_hashes": "second-item",
        "mm_placeholders": {"offset": 2, "length": 1},
        "kwargs_data": None,
        "mm_metadata": metadata,
    }
    for key, value in second_item.items():
        features[key].setdefault(modality, []).append(value)
    payload = {
        "token_ids": [1, 2, 3],
        "sampling_params": {"max_tokens": 1},
        "features": features,
    }
    if has_ec:
        payload["ec_transfer_params"] = {"second-item": {"peer_host": "10.0.0.1"}}

    if has_metadata and not has_ec:
        with pytest.raises(ValidationError, match="ec_transfer_params"):
            GenerateRequest.model_validate(payload)
    else:
        GenerateRequest.model_validate(payload)


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


@pytest.mark.parametrize(
    "is_embed", [None, torch.tensor([True, False, True, True])], ids=["dense", "sparse"]
)
def test_placeholder_roundtrip(is_embed):
    """Placeholders must survive render -> JSON -> generate unchanged."""
    placeholders = {"image": [PlaceholderRange(1, 4, is_embed=is_embed)]}
    engine_input = mm_input(
        prompt_token_ids=[1, 2, 3, 4, 5],
        mm_kwargs=MultiModalKwargsItems({}),
        mm_hashes={"image": ["hash-0"]},
        mm_placeholders=placeholders,
    )
    features = extract_mm_features(engine_input)
    assert features is not None
    wire = MultiModalFeatures.model_validate_json(features.model_dump_json())
    assert rebuild_mm_placeholders(wire.mm_placeholders) == placeholders


def test_placeholder_rejects_mismatched_is_embed():
    with pytest.raises(ValidationError, match="is_embed"):
        PlaceholderRangeInfo(offset=0, length=4, is_embed=[True])
