# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for P1 receiver-side multimodal cache behavior."""

import pytest

from vllm.config.multimodal import MultiModalConfig
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

    @pytest.mark.parametrize(
        "skip_mm_cache", [None, False], ids=["none", "false"]
    )
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
