# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for P1 receiver-side multimodal cache behavior."""

from unittest.mock import MagicMock, Mock

import pytest

from vllm import SamplingParams
from vllm.config.multimodal import MultiModalConfig
from vllm.inputs.engine import mm_input
from vllm.multimodal.cache import (
    LruKeyReplicatedReceiverCache,
    MultiModalCacheMissError,
)
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.engine.input_processor import InputProcessor

# =====================================================================
# Helpers
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


# =====================================================================
# P1 Cache Skip Behavior
# =====================================================================


class TestReceiverSkipMmCache:
    """Contract: only True skips cache work; False-like flags cache normally."""

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


class TestInputProcessorSkipMmCache:
    def test_skip_flag_is_preserved_per_feature(self, monkeypatch):
        processor = InputProcessor.__new__(InputProcessor)
        processor.vllm_config = MagicMock()
        processor.vllm_config.parallel_config.data_parallel_size = 1
        processor.vllm_config.parallel_config.data_parallel_size_local = 1
        processor.vllm_config.parallel_config.local_engines_only = False
        processor.renderer = MagicMock()
        processor.renderer.get_eos_token_id.return_value = None
        processor.renderer.tokenizer = None
        processor.generation_config_fields = {}
        processor._validate_params = Mock(return_value=True)
        processor._validate_lora = Mock()
        processor._validate_model_inputs = Mock()
        processor.apply_watermarking = Mock()
        processor._get_mm_identifier = lambda mm_hash, lora_request: mm_hash

        monkeypatch.setattr(
            "vllm.v1.engine.input_processor.current_platform.validate_request",
            lambda engine_input, params: None,
        )

        skipped_item = _make_item()
        for elem in skipped_item.values():
            elem.skip_mm_cache = True
        cached_item = _make_item()
        prompt = mm_input(
            prompt_token_ids=[1, 2],
            mm_kwargs={"image": [skipped_item, cached_item]},
            mm_hashes={"image": ["image-skipped", "image-cached"]},
            mm_placeholders={
                "image": [
                    PlaceholderRange(offset=0, length=1),
                    PlaceholderRange(offset=1, length=1),
                ]
            },
        )

        request = processor.process_inputs(
            "req-skip-mm-cache",
            prompt,
            SamplingParams(max_tokens=1, temperature=0.0),
            ("generate",),
        )

        assert [feature.skip_mm_cache for feature in request.mm_features] == [
            True,
            False,
        ]
