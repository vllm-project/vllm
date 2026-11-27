# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vllm.config.multimodal import MultiModalConfig
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    GenerateRequest,
    MultiModalFeatures,
    PlaceholderRangeInfo,
)
from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens
from vllm.inputs.engine import MultiModalInput
from vllm.multimodal.cache import (
    LruKeyReplicatedReceiverCache,
    MultiModalCacheMissError,
)
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.mem_constants import GiB_bytes

# =====================================================================
# P0 Helpers: Render-to-Generate Input
# =====================================================================


@dataclass
class _FakeElem:
    """Minimal stand-in for a multimodal feature element inside an item."""

    data: Any
    skip_mm_cache: bool = False


@dataclass
class _FakePlaceholder:
    """Minimal stand-in for placeholder ranges with offset and length."""

    offset: int
    length: int
    is_embed: bool | None = None


def _make_features(
    *,
    mm_hashes: dict[str, list[str]],
    placeholders: dict[str, list[tuple[int, int]]],
    kwargs_data: dict[str, list[str | None]] | None = None,
    kwargs_refs: dict[str, list[dict[str, Any] | None]] | None = None,
    mm_metadata: dict[str, list[str | None]] | None = None,
) -> MultiModalFeatures:
    mm_placeholders = {
        modality: [
            PlaceholderRangeInfo(offset=offset, length=length)
            for offset, length in ranges
        ]
        for modality, ranges in placeholders.items()
    }
    return MultiModalFeatures(
        mm_hashes=mm_hashes,
        mm_placeholders=mm_placeholders,
        kwargs_data=kwargs_data,
        kwargs_refs=kwargs_refs,
        mm_metadata=mm_metadata,
    )


def _make_request(
    *,
    token_ids: list[int] | None = None,
    features: MultiModalFeatures | None = None,
    content_parts: list[dict[str, Any]] | None = None,
    ec_transfer_params: dict[str, Any] | None = None,
    cache_salt: str | None = None,
) -> GenerateRequest:
    return GenerateRequest(
        request_id="req-build-input",
        token_ids=token_ids or list(range(24)),
        features=features,
        content_parts=content_parts,
        ec_transfer_params=ec_transfer_params,
        cache_salt=cache_salt,
        sampling_params=SamplingParams(max_tokens=4, temperature=0.0),
    )


def _make_serving_for_build_input() -> tuple[
    ServingTokens, MagicMock, MagicMock, MagicMock
]:
    serving = cast(Any, ServingTokens.__new__(ServingTokens))
    serving.model_config = MagicMock()
    validate_cache_handles = MagicMock(return_value=None)
    serving._validate_mm_cache_handles = validate_cache_handles

    tracker = MagicMock()
    parser = MagicMock()
    tracker.create_parser.return_value = parser
    tracker.resolve_items = AsyncMock(return_value=({"image": ["resolved-img"]}, None))
    serving._tracker = tracker

    renderer = MagicMock()
    renderer.render_cmpl_async = AsyncMock(
        return_value=[
            {
                "prompt_token_ids": [901, 902, 903, 904],
                "mm_placeholders": {"image": [_FakePlaceholder(offset=1, length=2)]},
                "mm_kwargs": {"image": [{"pixel_values": _FakeElem(data="from-url")}]},
                "mm_hashes": {"image": ["hash-content-part-1"]},
            }
        ]
    )
    online_renderer = MagicMock()
    online_renderer.renderer = renderer
    serving.online_renderer = online_renderer
    return serving, validate_cache_handles, tracker, online_renderer


# =====================================================================
# P0: Render-to-Generate Feature Cache State
# =====================================================================

SERVING_MOD = "vllm.entrypoints.scale_out.token_in_token_out.serving"


class TestBuildEngineInputSkipMmCache:
    """P0 contract: only full tensors from rendered features bypass the P1 LRU."""

    @pytest.mark.asyncio
    async def test_rendered_full_tensor_sets_skip_mm_cache(self):
        serving, validate_cache_handles, _, _ = _make_serving_for_build_input()
        features = _make_features(
            mm_hashes={"image": ["hash-rendered-1"]},
            placeholders={"image": [(2, 8)]},
            kwargs_data={"image": ["encoded-full-tensor"]},
        )
        request = _make_request(features=features)

        rendered_elem = _FakeElem(data={"tensor": [1, 2, 3]}, skip_mm_cache=False)
        fake_mm_kwargs = {"image": [{"pixel_values": rendered_elem}]}

        with (
            patch(
                f"{SERVING_MOD}.rebuild_mm_placeholders",
                side_effect=lambda p: p,
            ),
            patch(
                f"{SERVING_MOD}.mm_kwargs_from_features",
                return_value=fake_mm_kwargs,
            ),
            patch(
                f"{SERVING_MOD}.MultiModalKwargsItems",
                side_effect=lambda x: x,
            ),
            patch(
                f"{SERVING_MOD}.mm_input",
                side_effect=lambda **kwargs: kwargs,
            ),
        ):
            engine_input = cast(
                MultiModalInput, await serving.build_engine_input(request)
            )

        validate_cache_handles.assert_called_once()
        assert rendered_elem.skip_mm_cache is True
        rendered_item = engine_input["mm_kwargs"]["image"][0]
        assert rendered_item is not None
        assert rendered_item["pixel_values"].skip_mm_cache is True

    @pytest.mark.asyncio
    async def test_referenced_full_tensor_sets_skip_mm_cache(self):
        serving, _, _, _ = _make_serving_for_build_input()
        serving._mm_tensor_refs = MagicMock()
        serving._mm_tensor_refs.memory.reserve = AsyncMock(return_value=object())
        features = _make_features(
            mm_hashes={"image": ["hash-rendered-ref"]},
            placeholders={"image": [(2, 8)]},
            kwargs_refs={
                "image": [
                    {
                        "uri": "s3://tensor-bucket/request/image-0",
                        "format": "safetensors-v1",
                        "nbytes": 64,
                        "digest": "a" * 64,
                    }
                ]
            },
        )
        request = _make_request(features=features)
        referenced_elem = _FakeElem(data={"tensor": [1, 2, 3]})
        fake_mm_kwargs = {"image": [{"pixel_values": referenced_elem}]}

        with (
            patch(
                f"{SERVING_MOD}.rebuild_mm_placeholders",
                side_effect=lambda p: p,
            ),
            patch(
                f"{SERVING_MOD}.mm_kwargs_from_features",
                return_value=fake_mm_kwargs,
            ),
            patch(
                f"{SERVING_MOD}.validate_mm_tensor_request_size",
                return_value=64,
            ),
            patch(
                f"{SERVING_MOD}.run_in_read_slot",
                new=AsyncMock(return_value=fake_mm_kwargs),
            ),
            patch(
                f"{SERVING_MOD}.MultiModalKwargsItems",
                side_effect=lambda x: x,
            ),
            patch(
                f"{SERVING_MOD}.mm_input",
                side_effect=lambda **kwargs: kwargs,
            ),
        ):
            engine_input = cast(
                MultiModalInput, await serving.build_engine_input(request)
            )

        serving._mm_tensor_refs.reader.validate.assert_called_once()
        assert engine_input["mm_kwargs"]["image"][0]["pixel_values"].skip_mm_cache

    @pytest.mark.asyncio
    async def test_completion_returns_single_engine_input(self):
        serving, _, _, online_renderer = _make_serving_for_build_input()
        expected = {"type": "token", "prompt_token_ids": [1, 2, 3]}
        preprocess_completion = AsyncMock(return_value=[expected])
        online_renderer.preprocess_completion = preprocess_completion

        engine_input = await serving.build_engine_input(_make_request())

        assert engine_input is expected
        preprocess_completion.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_feature_validation_error_is_raised(self):
        serving, validate_cache_handles, _, _ = _make_serving_for_build_input()
        validate_cache_handles.side_effect = ValueError("bad cache handle")
        request = _make_request(
            features=_make_features(
                mm_hashes={"image": ["hash-1"]},
                placeholders={"image": [(0, 1)]},
                kwargs_data={"image": ["encoded-tensor"]},
            )
        )

        with (
            patch(
                f"{SERVING_MOD}.rebuild_mm_placeholders",
                side_effect=lambda p: p,
            ),
            patch(
                f"{SERVING_MOD}.mm_kwargs_from_features",
                return_value={"image": [{"pixel_values": _FakeElem(data="tensor")}]},
            ),
            pytest.raises(ValueError, match="bad cache handle"),
        ):
            await serving.build_engine_input(request)

    @pytest.mark.asyncio
    async def test_metadata_only_does_not_set_skip_mm_cache(self):
        serving, _, _, _ = _make_serving_for_build_input()
        features = _make_features(
            mm_hashes={"image": ["hash-meta-1"]},
            placeholders={"image": [(2, 8)]},
            kwargs_data={"image": [None]},
            mm_metadata={"image": ["encoded-meta"]},
        )
        request = _make_request(
            features=features,
            ec_transfer_params={"ec_remote_host": "127.0.0.1", "ec_remote_port": 8001},
        )

        meta_elem = _FakeElem(data=None, skip_mm_cache=False)
        fake_mm_kwargs = {"image": [{"meta": meta_elem}]}

        with (
            patch(
                f"{SERVING_MOD}.rebuild_mm_placeholders",
                side_effect=lambda p: p,
            ),
            patch(
                f"{SERVING_MOD}.mm_kwargs_from_features",
                return_value=fake_mm_kwargs,
            ),
            patch(
                f"{SERVING_MOD}.MultiModalKwargsItems",
                side_effect=lambda x: x,
            ),
            patch(
                f"{SERVING_MOD}.mm_input",
                side_effect=lambda **kwargs: kwargs,
            ),
        ):
            engine_input = cast(
                MultiModalInput, await serving.build_engine_input(request)
            )

        meta_item = engine_input["mm_kwargs"]["image"][0]
        assert meta_item is not None
        assert meta_item["meta"] is meta_elem
        assert meta_elem.skip_mm_cache is False

    @pytest.mark.asyncio
    async def test_mixed_rendered_features_and_content_parts_keep_skip_per_feature(
        self,
    ):
        serving, _, tracker, _ = _make_serving_for_build_input()
        # Feature range ends at offset 2 + length 8 = 10 -> start should be 10.
        features = _make_features(
            mm_hashes={"image": ["hash-rendered-1"]},
            placeholders={"image": [(2, 8)]},
            kwargs_data={"image": ["encoded-full-tensor"]},
        )
        request = _make_request(
            token_ids=list(range(20)),
            features=features,
            content_parts=[
                {
                    "type": "image_url",
                    "url": "https://example.com/a.png",
                    "uuid": "img-uuid-1",
                }
            ],
        )

        feature_elem = _FakeElem(data={"tensor": [4, 5, 6]}, skip_mm_cache=False)
        fake_mm_kwargs = {"image": [{"pixel_values": feature_elem}]}
        fake_placeholders = {"image": [_FakePlaceholder(offset=2, length=8)]}

        with (
            patch(
                f"{SERVING_MOD}.AsyncMultiModalItemTracker",
                return_value=tracker,
            ),
            patch(
                f"{SERVING_MOD}.rebuild_mm_placeholders",
                return_value=fake_placeholders,
            ),
            patch(
                f"{SERVING_MOD}.mm_kwargs_from_features",
                return_value=fake_mm_kwargs,
            ),
            patch(
                f"{SERVING_MOD}.MultiModalKwargsItems",
                side_effect=lambda x: x,
            ),
            patch(
                f"{SERVING_MOD}.mm_input",
                side_effect=lambda **kwargs: kwargs,
            ),
        ):
            engine_input = cast(
                MultiModalInput, await serving.build_engine_input(request)
            )

        # Tokens before start=10 preserved, followed by rendered content_parts tokens.
        assert engine_input["prompt_token_ids"] == list(range(10)) + [
            901,
            902,
            903,
            904,
        ]

        # First item comes from rendered features (skip_mm_cache=True),
        # second item comes from content_parts (skip_mm_cache remains False).
        img_items = engine_input["mm_kwargs"]["image"]
        assert len(img_items) == 2
        first_item, second_item = img_items
        assert first_item is not None
        assert second_item is not None
        assert first_item["pixel_values"].skip_mm_cache is True
        assert second_item["pixel_values"].skip_mm_cache is False

        # Content part placeholder offset is shifted by start=10.
        placeholders = engine_input["mm_placeholders"]["image"]
        assert len(placeholders) == 2
        assert placeholders[0].offset == 2
        assert placeholders[1].offset == 11  # 1 + 10

        assert engine_input["mm_hashes"]["image"] == [
            "hash-rendered-1",
            "hash-content-part-1",
        ]


# =====================================================================
# P1: Receiver Cache Skip Behavior
# =====================================================================


class _ModelConfig:
    def __init__(self) -> None:
        self._mm_config = MultiModalConfig(mm_processor_cache_gb=128 / GiB_bytes)

    def get_multimodal_config(self) -> MultiModalConfig:
        return self._mm_config


def _make_receiver_cache() -> LruKeyReplicatedReceiverCache:
    return LruKeyReplicatedReceiverCache(_ModelConfig())


def _make_item() -> MultiModalKwargsItem:
    return MultiModalKwargsItem.dummy(nbytes=64)


def _make_feature(
    mm_hash: str,
    data: MultiModalKwargsItem | None,
    *,
    skip_mm_cache: bool | None = False,
) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=data,
        modality="image",
        identifier=mm_hash,
        mm_position=PlaceholderRange(offset=0, length=1),
        mm_hash=mm_hash,
        skip_mm_cache=skip_mm_cache,
    )


def _call_receiver(
    cache: LruKeyReplicatedReceiverCache,
    *features: MultiModalFeatureSpec,
) -> list[MultiModalFeatureSpec]:
    return cache.get_and_update_features(list(features))


class TestReceiverSkipMmCache:
    """P1 contract: only an explicit true flag skips cache touch and lookup."""

    @pytest.mark.parametrize("skip_mm_cache", [None, False], ids=["none", "false"])
    def test_none_or_false_flag_caches_full_tensor(self, skip_mm_cache):
        cache = _make_receiver_cache()
        item = _make_item()

        _call_receiver(
            cache,
            _make_feature("image-normal", item, skip_mm_cache=skip_mm_cache),
        )

        (resolved,) = _call_receiver(cache, _make_feature("image-normal", None))
        assert resolved.data is item

    def test_true_flag_skips_insert_and_data_none_lookup_raises(self):
        cache = _make_receiver_cache()
        item = _make_item()

        (skipped,) = _call_receiver(
            cache, _make_feature("image-skipped", item, skip_mm_cache=True)
        )
        assert skipped.data is item

        with pytest.raises(MultiModalCacheMissError) as exc_info:
            _call_receiver(cache, _make_feature("image-skipped", None))
        assert exc_info.value.mm_hashes == ["image-skipped"]

    def test_mixed_features_only_cache_unskipped_items(self):
        cache = _make_receiver_cache()
        item_a = _make_item()
        item_b = _make_item()
        _call_receiver(
            cache, _make_feature("image-a", item_a), _make_feature("image-b", item_b)
        )

        _call_receiver(
            cache,
            _make_feature("image-skipped", _make_item(), skip_mm_cache=True),
            _make_feature("image-cached", _make_item(), skip_mm_cache=False),
        )

        (resolved_b,) = _call_receiver(cache, _make_feature("image-b", None))
        assert resolved_b.data is item_b
        with pytest.raises(MultiModalCacheMissError) as exc_info:
            _call_receiver(cache, _make_feature("image-skipped", None))
        assert exc_info.value.mm_hashes == ["image-skipped"]
