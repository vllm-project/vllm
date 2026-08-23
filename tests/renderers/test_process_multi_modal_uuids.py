# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus

import pytest

from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.entrypoints.serve import create_error_response
from vllm.multimodal.parse import parse_mm_uuids
from vllm.renderers.hf import HfRenderer
from vllm.tokenizers.registry import cached_tokenizer_from_config

cherry_pil_image = ImageAsset("cherry_blossom").pil_image
stop_pil_image = ImageAsset("stop_sign").pil_image
baby_reading_np_ndarrays = VideoAsset("baby_reading").np_ndarrays


def _build_renderer(
    *, mm_cache_gb: float = 4.0, enable_prefix_caching: bool = True
) -> HfRenderer:
    model_config = ModelConfig(
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        max_model_len=128,
        mm_processor_cache_gb=mm_cache_gb,
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(enable_prefix_caching=enable_prefix_caching),
    )

    return HfRenderer(
        vllm_config,
        cached_tokenizer_from_config(model_config),
    )


def _build_text_only_renderer() -> HfRenderer:
    model_config = ModelConfig(model="openai-community/gpt2", max_model_len=128)

    return HfRenderer(
        VllmConfig(model_config=model_config),
        cached_tokenizer_from_config(model_config),
    )


def test_text_only_model_mm_data_maps_to_bad_request():
    """Sending multimodal data to a text-only model is a client mistake, so it
    must surface as a ValueError and reach the client as HTTP 400, not 500."""
    renderer = _build_text_only_renderer()

    with pytest.raises(ValueError, match="text-only") as exc_info:
        renderer._process_multimodal(
            prompt=[1],
            mm_data={"image": [cherry_pil_image]},
            mm_uuids=None,
            mm_processor_kwargs=None,
        )

    error_response = create_error_response(exc_info.value)
    assert error_response.error.code == HTTPStatus.BAD_REQUEST


def test_multi_modal_uuids_length_mismatch_raises():
    renderer = _build_renderer()

    mm_data = {"image": [cherry_pil_image, stop_pil_image]}

    # Mismatch: 2 items but only 0 uuids provided
    mm_uuids = {"image": []}  # type: ignore[var-annotated]

    mm_processor = renderer.get_mm_processor()
    mm_data_items = mm_processor.info.parse_mm_data(mm_data)
    mm_uuid_items = parse_mm_uuids(mm_uuids)

    with pytest.raises(ValueError, match="must have same length as"):
        renderer._process_mm_uuids(mm_data, mm_data_items, mm_uuid_items, "req-1a")

    # Mismatch: 2 items but only 1 uuid provided
    mm_uuids = {"image": ["hash_cherry"]}

    mm_processor = renderer.get_mm_processor()
    mm_data_items = mm_processor.info.parse_mm_data(mm_data)
    mm_uuid_items = parse_mm_uuids(mm_uuids)

    with pytest.raises(ValueError, match="must have same length as"):
        renderer._process_mm_uuids(mm_data, mm_data_items, mm_uuid_items, "req-1b")


def test_multi_modal_uuids_missing_modality_raises():
    renderer = _build_renderer()

    mm_data = {
        "image": [cherry_pil_image],
        "video": None,
    }

    # Only image uuids provided; video missing should raise
    mm_uuids = {"image": ["hash_cherry"]}

    mm_processor = renderer.get_mm_processor()
    mm_data_items = mm_processor.info.parse_mm_data(mm_data)
    mm_uuid_items = parse_mm_uuids(mm_uuids)

    with pytest.raises(ValueError, match="is empty but .* is missing"):
        renderer._process_mm_uuids(mm_data, mm_data_items, mm_uuid_items, "req-2")


@pytest.mark.parametrize(
    "mm_cache_gb, enable_prefix_caching",
    [
        (4.0, True),  # default behavior
        (4.0, False),  # prefix caching disabled
        (0.0, True),  # processor cache disabled
    ],
)
def test_multi_modal_uuids_accepts_none_and_passes_through(
    mm_cache_gb: float, enable_prefix_caching: bool
):
    renderer = _build_renderer(
        mm_cache_gb=mm_cache_gb,
        enable_prefix_caching=enable_prefix_caching,
    )

    mm_data = {
        "image": [cherry_pil_image, stop_pil_image],
        "video": baby_reading_np_ndarrays,
    }

    # Use a consistent two-image scenario across all configurations
    mm_uuids = {"image": [None, "hash_stop"], "video": None}

    mm_processor = renderer.get_mm_processor()
    mm_data_items = mm_processor.info.parse_mm_data(mm_data)
    mm_uuid_items = parse_mm_uuids(mm_uuids)

    processed_mm_uuids = renderer._process_mm_uuids(
        mm_data, mm_data_items, mm_uuid_items, "req-3"
    )

    assert processed_mm_uuids == mm_uuids


@pytest.mark.parametrize(
    "mm_cache_gb, enable_prefix_caching",
    [
        (4.0, True),  # default behavior
        (4.0, False),  # prefix caching disabled
        (0.0, True),  # processor cache disabled
    ],
)
def test_multi_modal_uuids_accepts_empty(
    mm_cache_gb: float, enable_prefix_caching: bool
):
    renderer = _build_renderer(
        mm_cache_gb=mm_cache_gb,
        enable_prefix_caching=enable_prefix_caching,
    )

    # While None means cached multi-modal input requiring UUIDs
    # an empty list means no multi-modal input
    mm_data = {"image": [], "video": [], "audio": None}  # type: ignore[var-annotated]
    mm_uuids = {"image": [], "video": None, "audio": []}  # type: ignore[var-annotated]

    mm_processor = renderer.get_mm_processor()
    mm_data_items = mm_processor.info.parse_mm_data(mm_data)
    mm_uuid_items = parse_mm_uuids(mm_uuids)

    processed_mm_uuids = renderer._process_mm_uuids(
        mm_data, mm_data_items, mm_uuid_items, "req-4"
    )

    assert processed_mm_uuids == mm_uuids


class _FakeDataItems:
    """Minimal stub satisfying ModalityDataItems for hash testing."""

    def __init__(self, items: list[object]):
        self._items = items

    def get_all_items_for_hash(self) -> list[object]:
        return self._items

    def get_count(self) -> int:
        return len(self._items)


class _FakeMultiModalDataItems(dict):
    """Minimal stub satisfying MultiModalDataItems for hash testing."""

    def items(self):
        return super().items()


def test_uuid_hash_scoped_by_model_id():
    """Same uuid on different model_ids must produce different hashes."""
    from vllm.multimodal.processing.inputs import ProcessorInputs

    mm_data_items = _FakeMultiModalDataItems(
        {"image": _FakeDataItems([b"placeholder"])}
    )
    mm_uuid_items = {"image": ["shared-uuid-value"]}

    inputs = ProcessorInputs(
        prompt=[1, 2, 3],
        mm_data_items=mm_data_items,
        mm_uuid_items=mm_uuid_items,
    )

    hash_a = inputs.get_mm_hashes("model-A", "blake3")
    hash_b = inputs.get_mm_hashes("model-B", "blake3")

    assert hash_a["image"][0] != hash_b["image"][0]


def test_uuid_hash_domain_separated_from_content_hash():
    """A uuid-derived hash must differ from a content-derived hash,
    even if the uuid string happens to match what content hashing
    would produce."""
    from vllm.multimodal.processing.inputs import ProcessorInputs

    model_id = "test-model"
    image_bytes = b"some image content"

    # Request with uuid
    inputs_with_uuid = ProcessorInputs(
        prompt=[1, 2, 3],
        mm_data_items=_FakeMultiModalDataItems(
            {"image": _FakeDataItems([image_bytes])}
        ),
        mm_uuid_items={"image": ["my-uuid"]},
    )

    # Request without uuid (content-hashed)
    inputs_without_uuid = ProcessorInputs(
        prompt=[1, 2, 3],
        mm_data_items=_FakeMultiModalDataItems(
            {"image": _FakeDataItems([image_bytes])}
        ),
        mm_uuid_items=None,
    )

    hash_uuid = inputs_with_uuid.get_mm_hashes(model_id, "blake3")
    hash_content = inputs_without_uuid.get_mm_hashes(model_id, "blake3")

    assert hash_uuid["image"][0] != hash_content["image"][0]


def test_uuid_hash_deterministic_for_same_inputs():
    """Same uuid + same model_id must produce the same hash (caching works)."""
    from vllm.multimodal.processing.inputs import ProcessorInputs

    model_id = "test-model"
    uuid_value = "stable-uuid"

    inputs1 = ProcessorInputs(
        prompt=[1, 2, 3],
        mm_data_items=_FakeMultiModalDataItems({"image": _FakeDataItems([b"img1"])}),
        mm_uuid_items={"image": [uuid_value]},
    )

    inputs2 = ProcessorInputs(
        prompt=[4, 5, 6],
        mm_data_items=_FakeMultiModalDataItems({"image": _FakeDataItems([b"img2"])}),
        mm_uuid_items={"image": [uuid_value]},
    )

    hash1 = inputs1.get_mm_hashes(model_id, "blake3")
    hash2 = inputs2.get_mm_hashes(model_id, "blake3")

    assert hash1["image"][0] == hash2["image"][0]


def test_uuid_hash_not_raw_uuid_string():
    """The cache key must not be the raw uuid string itself."""
    from vllm.multimodal.processing.inputs import ProcessorInputs

    uuid_value = "user-chosen-uuid"

    inputs = ProcessorInputs(
        prompt=[1, 2, 3],
        mm_data_items=_FakeMultiModalDataItems({"image": _FakeDataItems([b"img"])}),
        mm_uuid_items={"image": [uuid_value]},
    )

    hashes = inputs.get_mm_hashes("test-model", "blake3")

    assert hashes["image"][0] != uuid_value


def test_content_fingerprint_present_when_uuid_and_data():
    """When both uuid and media are provided, get_content_fingerprints
    must return a non-None fingerprint for that item."""
    from vllm.multimodal.processing.inputs import ProcessorInputs

    inputs = ProcessorInputs(
        prompt=[1, 2, 3],
        mm_data_items=_FakeMultiModalDataItems(
            {"image": _FakeDataItems([b"some-image-bytes"])}
        ),
        mm_uuid_items={"image": ["my-uuid"]},
    )

    fps = inputs.get_content_fingerprints("test-model", "blake3")

    assert fps["image"][0] is not None


def test_content_fingerprint_none_for_skip_send():
    """When media is None (skip-send), fingerprint must be None."""
    from vllm.multimodal.processing.inputs import ProcessorInputs

    inputs = ProcessorInputs(
        prompt=[1, 2, 3],
        mm_data_items=_FakeMultiModalDataItems({"image": _FakeDataItems([None])}),
        mm_uuid_items={"image": ["my-uuid"]},
    )

    fps = inputs.get_content_fingerprints("test-model", "blake3")

    assert fps["image"][0] is None


def test_content_fingerprint_none_without_uuid():
    """When no uuid is set, fingerprint must be None (content-hashed path)."""
    from vllm.multimodal.processing.inputs import ProcessorInputs

    inputs = ProcessorInputs(
        prompt=[1, 2, 3],
        mm_data_items=_FakeMultiModalDataItems({"image": _FakeDataItems([b"img"])}),
        mm_uuid_items=None,
    )

    fps = inputs.get_content_fingerprints("test-model", "blake3")

    assert fps["image"][0] is None


def test_content_fingerprint_differs_for_different_content():
    """Same uuid with different media bytes must yield different fingerprints."""
    from vllm.multimodal.processing.inputs import ProcessorInputs

    model_id = "test-model"

    inputs_a = ProcessorInputs(
        prompt=[1],
        mm_data_items=_FakeMultiModalDataItems({"image": _FakeDataItems([b"image-A"])}),
        mm_uuid_items={"image": ["shared-uuid"]},
    )
    inputs_b = ProcessorInputs(
        prompt=[1],
        mm_data_items=_FakeMultiModalDataItems({"image": _FakeDataItems([b"image-B"])}),
        mm_uuid_items={"image": ["shared-uuid"]},
    )

    fp_a = inputs_a.get_content_fingerprints(model_id, "blake3")
    fp_b = inputs_b.get_content_fingerprints(model_id, "blake3")

    assert fp_a["image"][0] != fp_b["image"][0]


def test_content_fingerprint_same_for_same_content():
    """Same content must always produce the same fingerprint."""
    from vllm.multimodal.processing.inputs import ProcessorInputs

    model_id = "test-model"
    content = b"identical-image"

    inputs1 = ProcessorInputs(
        prompt=[1],
        mm_data_items=_FakeMultiModalDataItems({"image": _FakeDataItems([content])}),
        mm_uuid_items={"image": ["uuid-1"]},
    )
    inputs2 = ProcessorInputs(
        prompt=[2],
        mm_data_items=_FakeMultiModalDataItems({"image": _FakeDataItems([content])}),
        mm_uuid_items={"image": ["uuid-2"]},
    )

    fp1 = inputs1.get_content_fingerprints(model_id, "blake3")
    fp2 = inputs2.get_content_fingerprints(model_id, "blake3")

    assert fp1["image"][0] == fp2["image"][0]


def _run_collision_check(cache, mm_hashes, mm_data_items, mm_fingerprints):
    """Run the collision-check portion of _get_cache_missing_items
    without needing a full processor instance."""
    mm_is_cached = {
        modality: cache.is_cached(hashes) for modality, hashes in mm_hashes.items()
    }

    if mm_fingerprints is not None:
        for modality, hashes in mm_hashes.items():
            cached_flags = mm_is_cached[modality]
            fps = mm_fingerprints.get(modality, [])
            for idx, (is_cached, mm_hash) in enumerate(zip(cached_flags, hashes)):
                if not is_cached:
                    continue
                fp = fps[idx] if idx < len(fps) else None
                if fp is None:
                    continue
                stored = cache.get_content_hash(mm_hash)
                if stored is not None and stored != fp:
                    raise ValueError(
                        f"Multimodal uuid collision for {modality} at "
                        f"index {idx}: the same uuid was previously "
                        f"used with different media content."
                    )

    return mm_is_cached


def test_uuid_collision_rejected_by_cache():
    """When a uuid hits the cache but the content fingerprint differs,
    the collision check must raise ValueError."""
    from vllm.multimodal.cache import MultiModalProcessorOnlyCache

    from ..multimodal.test_cache import _StubModelConfig

    model_config = _StubModelConfig(mm_processor_cache_gb=1)
    cache = MultiModalProcessorOnlyCache(model_config)  # type: ignore[arg-type]

    from vllm.multimodal.inputs import MultiModalKwargsItem

    mm_hash = "uuid-key"
    item = MultiModalKwargsItem.dummy(nbytes=64)

    cache.get_and_update_item((item, []), mm_hash)
    cache.store_content_hash(mm_hash, "fp-original")

    mm_hashes = {"image": [mm_hash]}
    mm_fingerprints = {"image": ["fp-different"]}

    mm_data_items = _FakeMultiModalDataItems(
        {"image": _FakeDataItems([b"different-image"])}
    )

    with pytest.raises(ValueError, match="uuid collision"):
        _run_collision_check(cache, mm_hashes, mm_data_items, mm_fingerprints)


def test_uuid_cache_hit_with_matching_fingerprint():
    """When the content fingerprint matches, a uuid cache hit must succeed."""
    from vllm.multimodal.cache import MultiModalProcessorOnlyCache

    from ..multimodal.test_cache import _StubModelConfig

    model_config = _StubModelConfig(mm_processor_cache_gb=1)
    cache = MultiModalProcessorOnlyCache(model_config)  # type: ignore[arg-type]

    from vllm.multimodal.inputs import MultiModalKwargsItem

    mm_hash = "uuid-key"
    item = MultiModalKwargsItem.dummy(nbytes=64)

    cache.get_and_update_item((item, []), mm_hash)
    cache.store_content_hash(mm_hash, "fp-same")

    mm_hashes = {"image": [mm_hash]}
    mm_fingerprints = {"image": ["fp-same"]}

    mm_data_items = _FakeMultiModalDataItems({"image": _FakeDataItems([b"same-image"])})

    mm_is_cached = _run_collision_check(
        cache, mm_hashes, mm_data_items, mm_fingerprints
    )

    assert mm_is_cached["image"][0] is True


def test_uuid_skip_send_no_collision_check():
    """Skip-send (data=None, fingerprint=None) must not trigger collision
    even when a content hash was stored previously."""
    from vllm.multimodal.cache import MultiModalProcessorOnlyCache

    from ..multimodal.test_cache import _StubModelConfig

    model_config = _StubModelConfig(mm_processor_cache_gb=1)
    cache = MultiModalProcessorOnlyCache(model_config)  # type: ignore[arg-type]

    from vllm.multimodal.inputs import MultiModalKwargsItem

    mm_hash = "uuid-key"
    item = MultiModalKwargsItem.dummy(nbytes=64)

    cache.get_and_update_item((item, []), mm_hash)
    cache.store_content_hash(mm_hash, "fp-original")

    mm_hashes = {"image": [mm_hash]}
    mm_fingerprints = {"image": [None]}

    mm_data_items = _FakeMultiModalDataItems({"image": _FakeDataItems([None])})

    mm_is_cached = _run_collision_check(
        cache, mm_hashes, mm_data_items, mm_fingerprints
    )

    assert mm_is_cached["image"][0] is True


def test_different_uuids_do_not_interact():
    """Two different uuids with different content must not collide."""
    from vllm.multimodal.cache import MultiModalProcessorOnlyCache

    from ..multimodal.test_cache import _StubModelConfig

    model_config = _StubModelConfig(mm_processor_cache_gb=1)
    cache = MultiModalProcessorOnlyCache(model_config)  # type: ignore[arg-type]

    from vllm.multimodal.inputs import MultiModalKwargsItem

    item_a = MultiModalKwargsItem.dummy(nbytes=64)
    item_b = MultiModalKwargsItem.dummy(nbytes=64)

    cache.get_and_update_item((item_a, []), "uuid-A")
    cache.store_content_hash("uuid-A", "fp-A")
    cache.get_and_update_item((item_b, []), "uuid-B")
    cache.store_content_hash("uuid-B", "fp-B")

    mm_hashes = {"image": ["uuid-A", "uuid-B"]}
    mm_fingerprints = {"image": ["fp-A", "fp-B"]}

    mm_data_items = _FakeMultiModalDataItems(
        {"image": _FakeDataItems([b"img-A", b"img-B"])}
    )

    mm_is_cached = _run_collision_check(
        cache, mm_hashes, mm_data_items, mm_fingerprints
    )

    assert mm_is_cached["image"] == [True, True]


def test_multi_modal_uuids_ignored_when_caching_disabled():
    # When both processor cache is 0 and prefix caching disabled, the
    # processor builds overrides from request id instead of using user UUIDs.
    renderer = _build_renderer(mm_cache_gb=0.0, enable_prefix_caching=False)

    request_id = "req-42"
    mm_data = {
        "image": [cherry_pil_image, stop_pil_image],
        "video": baby_reading_np_ndarrays,
    }
    mm_uuids = {"image": ["hash_cherry", "hash_stop"], "video": ["hash_video"]}

    mm_processor = renderer.get_mm_processor()
    mm_data_items = mm_processor.info.parse_mm_data(mm_data)
    mm_uuid_items = parse_mm_uuids(mm_uuids)

    processed_mm_uuids = renderer._process_mm_uuids(
        mm_data, mm_data_items, mm_uuid_items, request_id
    )

    # Expect request-id-based overrides are passed through
    assert set(mm_uuids.keys()) == {"image", "video"}
    assert len(mm_uuids["image"]) == 2
    assert len(mm_uuids["video"]) == 1
    assert processed_mm_uuids["image"][0].startswith(
        f"{request_id}-image-"
    ) and processed_mm_uuids["image"][0].endswith("-0")
    assert processed_mm_uuids["image"][1].startswith(
        f"{request_id}-image-"
    ) and processed_mm_uuids["image"][1].endswith("-1")
    assert processed_mm_uuids["video"][0].startswith(
        f"{request_id}-video-"
    ) and processed_mm_uuids["video"][0].endswith("-0")
