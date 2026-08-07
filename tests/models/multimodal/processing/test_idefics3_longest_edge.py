# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.model_executor.models.idefics3 import (
    Idefics3MultiModalProcessor,
    Idefics3ProcessingInfo,
)
from vllm.model_executor.models.smolvlm import SmolVLMProcessingInfo
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import BaseMultiModalProcessor, TimingContext


class _MMConfig:
    def __init__(self, mm_processor_kwargs: dict[str, object]) -> None:
        self.mm_processor_kwargs = mm_processor_kwargs

    def merge_mm_processor_kwargs(
        self, inference_kwargs: dict[str, object]
    ) -> dict[str, object]:
        return self.mm_processor_kwargs | dict(inference_kwargs)


class _ModelConfig:
    def __init__(self, trusted_longest_edge: int) -> None:
        self.model = "test-model"
        mm_processor_kwargs: dict[str, object] = (
            {}
            if trusted_longest_edge == 2048
            else {"size": {"longest_edge": trusted_longest_edge}}
        )
        self.multimodal_config = _MMConfig(mm_processor_kwargs)

    def get_multimodal_config(self) -> _MMConfig:
        return self.multimodal_config


class _ImageProcessor:
    class valid_kwargs:
        size: object
        max_image_size: object
        do_resize: object
        do_image_splitting: object
        return_tensors: object

    do_resize = True
    do_image_splitting = True

    def __init__(
        self,
        *,
        size: dict[str, object],
        max_image_size: dict[str, object] | None = None,
    ) -> None:
        self.size = size
        self.max_image_size = max_image_size

    def get_number_of_image_patches(
        self,
        image_height: int,
        image_width: int,
        images_kwargs: dict[str, object],
    ) -> tuple[int, int, int]:
        del image_height, image_width
        size = images_kwargs.get("size", self.size)
        assert isinstance(size, dict)
        longest_edge = size["longest_edge"]
        assert isinstance(longest_edge, int)
        return max(1, longest_edge // 128), 1, 1


class _Processor:
    image_seq_len = 1

    def __init__(
        self,
        *,
        size: dict[str, object],
        max_image_size: dict[str, object] | None = None,
    ) -> None:
        self.image_processor = _ImageProcessor(
            size=size,
            max_image_size=max_image_size,
        )
        self.tokenizer = SimpleNamespace(init_kwargs={})

    def _merge_kwargs(
        self,
        _processor_kwargs_cls,
        *,
        tokenizer_init_kwargs: dict[str, object],
        **kwargs: object,
    ) -> dict[str, dict[str, object]]:
        del _processor_kwargs_cls, tokenizer_init_kwargs

        images_kwargs: dict[str, object] = {}
        common_kwargs = kwargs.get("common_kwargs")
        if common_kwargs is not None:
            assert isinstance(common_kwargs, dict)
            images_kwargs.update(common_kwargs)

        nested_images_kwargs = kwargs.get("images_kwargs")
        if nested_images_kwargs is not None:
            assert isinstance(nested_images_kwargs, dict)
            images_kwargs.update(nested_images_kwargs)

        for name in ("size", "max_image_size", "do_resize", "do_image_splitting"):
            if name in kwargs:
                if name in images_kwargs:
                    raise ValueError(f"Keyword argument {name} was passed two times")
                images_kwargs[name] = kwargs[name]

        return {"images_kwargs": images_kwargs}


class _ProcessingContext:
    def __init__(self, trusted_longest_edge: int = 2048) -> None:
        self.model_config = _ModelConfig(trusted_longest_edge)
        self.processor_calls: list[dict[str, object]] = []

    def get_tokenizer(self) -> SimpleNamespace:
        return SimpleNamespace(encode=lambda _prompt: [1])

    def get_merged_mm_kwargs(
        self, kwargs: dict[str, object] | None = None
    ) -> dict[str, object]:
        return self.model_config.get_multimodal_config().merge_mm_processor_kwargs(
            kwargs or {}
        )

    def get_hf_processor(self, _processor_cls, **kwargs: object) -> _Processor:
        self.processor_calls.append(dict(kwargs))
        return _Processor(size={"longest_edge": 2048})


@pytest.mark.parametrize(
    "info_cls",
    [Idefics3ProcessingInfo, SmolVLMProcessingInfo],
)
def test_rejects_request_size_longest_edge_above_deployment_limit(info_cls) -> None:
    info = info_cls(_ProcessingContext(trusted_longest_edge=2048))

    with pytest.raises(ValueError, match="size.longest_edge"):
        info.get_hf_processor(size={"longest_edge": 4096})


@pytest.mark.parametrize(
    "info_cls",
    [Idefics3ProcessingInfo, SmolVLMProcessingInfo],
)
@pytest.mark.parametrize("request_longest_edge", [1024, 2048])
def test_allows_request_size_longest_edge_at_or_below_deployment_limit(
    info_cls, request_longest_edge: int
) -> None:
    info = info_cls(_ProcessingContext(trusted_longest_edge=2048))

    processor = info.get_hf_processor(size={"longest_edge": request_longest_edge})

    assert processor.image_processor.size == {"longest_edge": 2048}


@pytest.mark.parametrize(
    "info_cls",
    [Idefics3ProcessingInfo, SmolVLMProcessingInfo],
)
@pytest.mark.parametrize(
    "request_size",
    [
        364,
        [364, 364],
        (364, 364),
        {"longest_edge": 2048, "shortest_edge": 1},
        {"height": 364, "width": 364},
        {"height": 2048, "width": 2048},
    ],
)
def test_allows_request_size_variants_at_or_below_deployment_limit(
    info_cls,
    request_size: object,
) -> None:
    info = info_cls(_ProcessingContext(trusted_longest_edge=2048))

    processor = info.get_hf_processor(
        size=request_size,
        do_image_splitting=False,
    )

    assert processor.image_processor.size == {"longest_edge": 2048}


@pytest.mark.parametrize(
    "info_cls",
    [Idefics3ProcessingInfo, SmolVLMProcessingInfo],
)
@pytest.mark.parametrize(
    "request_size",
    [4096, [2048, 4096], (2048, 4096), {"height": 2048, "width": 4096}],
)
def test_rejects_request_size_variants_above_deployment_limit(
    info_cls,
    request_size: object,
) -> None:
    info = info_cls(_ProcessingContext(trusted_longest_edge=2048))

    with pytest.raises(ValueError, match="exceeds the deployed limit"):
        info.get_hf_processor(
            size=request_size,
            do_image_splitting=False,
        )


def test_rejects_mixed_request_size_shape_before_hf_preprocessing() -> None:
    info = Idefics3ProcessingInfo(_ProcessingContext(trusted_longest_edge=2048))

    with pytest.raises(ValueError, match="Request `size`"):
        info.get_hf_processor(
            size={
                "longest_edge": 1024,
                "height": 4096,
                "width": 4096,
            }
        )


def test_allows_operator_selected_higher_deployment_limit() -> None:
    info = Idefics3ProcessingInfo(_ProcessingContext(trusted_longest_edge=4096))

    processor = info.get_hf_processor(size={"longest_edge": 4096})

    assert processor.image_processor.size == {"longest_edge": 2048}


def test_rejects_nested_images_kwargs_size_longest_edge_above_deployment_limit() -> (
    None
):
    info = Idefics3ProcessingInfo(_ProcessingContext(trusted_longest_edge=2048))

    with pytest.raises(ValueError, match="size.longest_edge"):
        info.get_hf_processor(images_kwargs={"size": {"longest_edge": 4096}})


def test_rejects_nested_common_kwargs_size_longest_edge_above_deployment_limit() -> (
    None
):
    info = Idefics3ProcessingInfo(_ProcessingContext(trusted_longest_edge=2048))

    with pytest.raises(ValueError, match="size.longest_edge"):
        info.get_hf_processor(common_kwargs={"size": {"longest_edge": 4096}})


@pytest.mark.parametrize("invalid_value", [0, -1, True, 2048.0])
def test_rejects_invalid_request_size_longest_edge(invalid_value: object) -> None:
    info = Idefics3ProcessingInfo(_ProcessingContext(trusted_longest_edge=2048))

    with pytest.raises(ValueError, match="size.longest_edge"):
        info.get_hf_processor(size={"longest_edge": invalid_value})


def test_smolvlm_max_image_size_override_remains_supported() -> None:
    info = SmolVLMProcessingInfo(_ProcessingContext(trusted_longest_edge=2048))

    processor = info.get_hf_processor(max_image_size={"longest_edge": 768})

    assert processor.image_processor.max_image_size is None


def test_patch_count_path_reuses_size_validation() -> None:
    info = Idefics3ProcessingInfo(_ProcessingContext(trusted_longest_edge=2048))
    processor = info.get_hf_processor()

    with pytest.raises(ValueError, match="size.longest_edge"):
        info.get_num_patches(
            image_width=4096,
            image_height=4096,
            processor=processor,
            mm_kwargs={"size": {"longest_edge": 4096}},
        )


@pytest.mark.parametrize("do_resize", [False, None])
def test_patch_count_path_rejects_disabled_resize_while_splitting_enabled(
    do_resize: object,
) -> None:
    info = Idefics3ProcessingInfo(_ProcessingContext(trusted_longest_edge=2048))
    processor = info.get_hf_processor()

    with pytest.raises(ValueError, match="do_resize"):
        info.get_num_patches(
            image_width=2048,
            image_height=2048,
            processor=processor,
            mm_kwargs={"do_resize": do_resize},
        )


def test_rejects_oversized_request_before_hf_preprocessing(monkeypatch) -> None:
    info = Idefics3ProcessingInfo(_ProcessingContext(trusted_longest_edge=2048))
    processor = Idefics3MultiModalProcessor.__new__(Idefics3MultiModalProcessor)
    processor.info = info

    def _unexpected_hf_call(*args, **kwargs):
        del args, kwargs
        pytest.fail("HF preprocessing ran before size.longest_edge validation")

    monkeypatch.setattr(
        BaseMultiModalProcessor,
        "_call_hf_processor",
        _unexpected_hf_call,
    )

    with pytest.raises(ValueError, match="size.longest_edge"):
        processor._call_hf_processor(
            "prompt",
            {"images": [SimpleNamespace()]},
            {"size": {"longest_edge": 4096}},
            {},
        )


@pytest.mark.parametrize("do_resize", [False, None])
def test_rejects_disabled_resize_while_splitting_enabled_before_hf_preprocessing(
    monkeypatch,
    do_resize: object,
) -> None:
    info = Idefics3ProcessingInfo(_ProcessingContext(trusted_longest_edge=2048))
    processor = Idefics3MultiModalProcessor.__new__(Idefics3MultiModalProcessor)
    processor.info = info

    def _unexpected_hf_call(*args, **kwargs):
        del args, kwargs
        pytest.fail("HF preprocessing ran before do_resize validation")

    monkeypatch.setattr(
        BaseMultiModalProcessor,
        "_call_hf_processor",
        _unexpected_hf_call,
    )

    with pytest.raises(ValueError, match="do_resize"):
        processor._call_hf_processor(
            "prompt",
            {"images": [SimpleNamespace()]},
            {"do_resize": do_resize},
            {},
        )


def test_allows_disabled_resize_when_image_splitting_is_disabled() -> None:
    info = Idefics3ProcessingInfo(_ProcessingContext(trusted_longest_edge=2048))

    processor = info.get_hf_processor(
        do_resize=False,
        do_image_splitting=False,
    )

    assert processor.image_processor.size == {"longest_edge": 2048}


@pytest.mark.parametrize(
    "tokenization_kwargs",
    [
        {"size": {"longest_edge": 4096}},
        {"do_resize": False},
        {"do_image_splitting": False},
        {"max_image_size": {"longest_edge": 91}},
        {"images_kwargs": {"size": {"longest_edge": 4096}}},
        {"images_kwargs": {"do_image_splitting": False}},
        {"common_kwargs": {"size": {"longest_edge": 4096}}},
        {"common_kwargs": {"max_image_size": {"longest_edge": 91}}},
        {"common_kwargs": [["size", {"longest_edge": 4096}]]},
        {"common_kwargs": [["do_resize", False]]},
        {"common_kwargs": [["max_image_size", {"longest_edge": 91}]]},
    ],
)
def test_rejects_image_processing_kwargs_in_tokenization_kwargs_before_hf_preprocessing(
    monkeypatch,
    tokenization_kwargs: dict[str, object],
) -> None:
    info = Idefics3ProcessingInfo(_ProcessingContext(trusted_longest_edge=2048))
    processor = Idefics3MultiModalProcessor.__new__(Idefics3MultiModalProcessor)
    processor.info = info

    def _unexpected_hf_call(*args, **kwargs):
        del args, kwargs
        pytest.fail("HF preprocessing ran before tokenization_kwargs validation")

    monkeypatch.setattr(
        BaseMultiModalProcessor,
        "_call_hf_processor",
        _unexpected_hf_call,
    )

    with pytest.raises(ValueError, match="tokenization_kwargs"):
        processor._call_hf_processor(
            "prompt",
            {"images": [SimpleNamespace()]},
            {},
            tokenization_kwargs,
        )


def test_rejects_image_processing_tokenization_kwargs_on_text_only_fast_path() -> None:
    info = Idefics3ProcessingInfo(_ProcessingContext(trusted_longest_edge=2048))
    processor = Idefics3MultiModalProcessor.__new__(Idefics3MultiModalProcessor)
    processor.info = info

    with pytest.raises(ValueError, match="tokenization_kwargs"):
        processor._call_hf_processor(
            "prompt",
            {},
            {},
            {"do_image_splitting": False},
        )


@pytest.mark.parametrize(
    "tokenization_kwargs",
    [
        {"add_special_tokens": False, "truncation": False},
        {"image_seq_len": 999},
        {"return_tensors": "pt"},
        {"images_kwargs": {"return_tensors": "pt"}},
        {"common_kwargs": {"return_tensors": "pt"}},
        {"common_kwargs": [["return_tensors", "pt"]]},
    ],
)
def test_allows_non_image_processing_tokenization_kwargs_on_text_only_fast_path(
    tokenization_kwargs: dict[str, object],
) -> None:
    info = Idefics3ProcessingInfo(_ProcessingContext(trusted_longest_edge=2048))
    processor = Idefics3MultiModalProcessor.__new__(Idefics3MultiModalProcessor)
    processor.info = info

    output = processor._call_hf_processor(
        "prompt",
        {},
        {},
        tokenization_kwargs,
    )

    assert output["input_ids"].tolist() == [[1]]


@pytest.mark.parametrize(
    "tokenization_kwargs",
    [
        {"do_image_splitting": False},
        {"max_image_size": {"longest_edge": 91}},
    ],
)
def test_rejects_image_processing_kwargs_in_tokenization_kwargs_on_all_cache_hit_path(
    monkeypatch,
    tokenization_kwargs: dict[str, object],
) -> None:
    info = Idefics3ProcessingInfo(_ProcessingContext(trusted_longest_edge=2048))
    processor = Idefics3MultiModalProcessor.__new__(Idefics3MultiModalProcessor)
    processor.info = info
    processor.dummy_inputs = SimpleNamespace(get_dummy_text=lambda _mm_counts: "prompt")

    class _Cache:
        def touch_sender_cache_item(self, _item_hash: str) -> None:
            return None

        def get_and_update_item(self, _item, _item_hash: str):
            return {}, []

    processor.cache = _Cache()

    class _CachedImageItems:
        def get_processor_data(self) -> dict[str, object]:
            return {}

        def get_passthrough_data(self) -> dict[str, object]:
            return {}

    inputs = SimpleNamespace(
        prompt="prompt",
        mm_data_items=MultiModalDataItems({"image": _CachedImageItems()}),
        hf_processor_mm_kwargs={},
        tokenization_kwargs=tokenization_kwargs,
        get_mm_hashes=lambda _model_id: {"image": ["cached-image"]},
    )

    def _all_cache_hit(**_kwargs):
        return {"image": [True]}, MultiModalDataItems({})

    monkeypatch.setattr(processor, "_get_cache_missing_items", _all_cache_hit)
    monkeypatch.setattr(processor, "_get_mm_prompt_updates", lambda *_args: {})

    with pytest.raises(ValueError, match="tokenization_kwargs"):
        processor._cached_apply_hf_processor(
            inputs,
            TimingContext(enabled=False),
        )
