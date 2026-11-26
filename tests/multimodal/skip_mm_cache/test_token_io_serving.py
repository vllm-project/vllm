from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    GenerateRequest,
    MultiModalFeatures,
    PlaceholderRangeInfo,
)
from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens
from vllm.sampling_params import SamplingParams


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


def _make_serving_for_build_input() -> ServingTokens:
    serving = ServingTokens.__new__(ServingTokens)
    serving.model_config = MagicMock()
    serving._validate_mm_cache_handles = MagicMock(return_value=None)

    tracker = MagicMock()
    parser = MagicMock()
    tracker.create_parser.return_value = parser
    tracker.resolve_items = AsyncMock(
        return_value=({"image": ["resolved-img"]}, None)
    )
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
    return serving


SERVING_MOD = "vllm.entrypoints.scale_out.token_in_token_out.serving"


class TestBuildEngineInputSkipMmCache:
    """Unit tests for build_engine_input across rendered, metadata, and mixed inputs."""

    @pytest.mark.asyncio
    async def test_rendered_full_tensor_sets_skip_mm_cache(self):
        serving = _make_serving_for_build_input()
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
            engine_input = await serving.build_engine_input(request)

        serving._validate_mm_cache_handles.assert_called_once()
        assert rendered_elem.skip_mm_cache is True
        assert engine_input["mm_kwargs"]["image"][0]["pixel_values"].skip_mm_cache is True

    @pytest.mark.asyncio
    async def test_completion_returns_single_engine_input(self):
        serving = _make_serving_for_build_input()
        expected = {"type": "token", "prompt_token_ids": [1, 2, 3]}
        serving.online_renderer.preprocess_completion = AsyncMock(
            return_value=[expected]
        )

        engine_input = await serving.build_engine_input(_make_request())

        assert engine_input is expected
        serving.online_renderer.preprocess_completion.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_feature_validation_error_is_raised(self):
        serving = _make_serving_for_build_input()
        serving._validate_mm_cache_handles.side_effect = ValueError("bad cache handle")
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
                return_value={"image": [_FakeElem(data="tensor")]},
            ),
            pytest.raises(ValueError, match="bad cache handle"),
        ):
            await serving.build_engine_input(request)

    @pytest.mark.asyncio
    async def test_metadata_only_does_not_set_skip_mm_cache(self):
        serving = _make_serving_for_build_input()
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
        fake_mm_kwargs = {"image": [None, {"meta": meta_elem}]}

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
            engine_input = await serving.build_engine_input(request)

        assert engine_input["mm_kwargs"]["image"][0] is None
        assert meta_elem.skip_mm_cache is False

    @pytest.mark.asyncio
    async def test_mixed_rendered_features_and_content_parts(self):
        serving = _make_serving_for_build_input()
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
                return_value=serving._tracker,
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
            engine_input = await serving.build_engine_input(request)

        # Tokens before start=10 preserved, followed by rendered content_parts tokens.
        assert engine_input["prompt_token_ids"] == list(range(10)) + [901, 902, 903, 904]

        # First item comes from rendered features (skip_mm_cache=True),
        # second item comes from content_parts (skip_mm_cache remains False).
        img_items = engine_input["mm_kwargs"]["image"]
        assert len(img_items) == 2
        assert img_items[0]["pixel_values"].skip_mm_cache is True
        assert img_items[1]["pixel_values"].skip_mm_cache is False

        # Content part placeholder offset is shifted by start=10.
        placeholders = engine_input["mm_placeholders"]["image"]
        assert len(placeholders) == 2
        assert placeholders[0].offset == 2
        assert placeholders[1].offset == 11  # 1 + 10

        assert engine_input["mm_hashes"]["image"] == [
            "hash-rendered-1",
            "hash-content-part-1",
        ]
