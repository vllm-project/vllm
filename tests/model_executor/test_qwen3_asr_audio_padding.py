# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for Qwen3-ASR request-owned audio padding kwargs."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.whisper import WhisperFeatureExtractor

from vllm.model_executor.models.qwen3_asr import Qwen3ASRMultiModalProcessor
from vllm.transformers_utils.processors.qwen3_asr import (
    Qwen3ASRProcessor,
    Qwen3ASRProcessorKwargs,
)


class _Tokenizer:
    init_kwargs: dict[str, object] = {}
    audio_token = "<|audio_pad|>"
    audio_bos_token = "<|audio_start|>"
    audio_eos_token = "<|audio_end|>"


class _NonBoolEquality:
    def __eq__(self, other: object) -> Any:
        del other
        return [True]


class _MMConfig:
    def __init__(self, mm_processor_kwargs: dict[str, object] | None) -> None:
        self.mm_processor_kwargs = mm_processor_kwargs

    def merge_mm_processor_kwargs(
        self, inference_kwargs: dict[str, object]
    ) -> dict[str, object]:
        return (self.mm_processor_kwargs or {}) | dict(inference_kwargs)


class _ModelConfig:
    def __init__(self, mm_processor_kwargs: dict[str, object] | None) -> None:
        self.multimodal_config = _MMConfig(mm_processor_kwargs)

    def get_multimodal_config(self) -> _MMConfig:
        return self.multimodal_config


class _Context:
    def __init__(self, mm_processor_kwargs: dict[str, object] | None) -> None:
        self.model_config = _ModelConfig(mm_processor_kwargs)

    def get_merged_mm_kwargs(self, kwargs: dict[str, object]) -> dict[str, object]:
        return self.model_config.get_multimodal_config().merge_mm_processor_kwargs(
            kwargs
        )

    def call_hf_processor(
        self,
        hf_processor: Qwen3ASRProcessor,
        data: dict[str, object],
        kwargs: dict[str, object],
        *,
        num_tries: int = 1,
        max_tries: int = 5,
        merge_mm_kwargs: bool = True,
    ) -> BatchFeature:
        del num_tries, max_tries
        call_kwargs = self.get_merged_mm_kwargs(kwargs) if merge_mm_kwargs else kwargs
        call_kwargs = dict(call_kwargs)
        call_kwargs.setdefault("return_tensors", "pt")
        return hf_processor(**data, **call_kwargs)


class _Info:
    def __init__(self, mm_processor_kwargs: dict[str, object] | None) -> None:
        self.ctx = _Context(mm_processor_kwargs)
        self._hf_processor = object.__new__(Qwen3ASRProcessor)
        self._hf_processor.tokenizer = _Tokenizer()
        self._hf_processor.feature_extractor = WhisperFeatureExtractor()

    def get_hf_processor(self, **kwargs: object) -> Qwen3ASRProcessor:
        del kwargs
        return self._hf_processor

    def get_feature_extractor(self, **kwargs: object) -> WhisperFeatureExtractor:
        del kwargs
        return self._hf_processor.feature_extractor


def _make_processor(
    mm_processor_kwargs: dict[str, object] | None = None,
) -> Qwen3ASRMultiModalProcessor:
    processor = object.__new__(Qwen3ASRMultiModalProcessor)
    processor.info = _Info(mm_processor_kwargs)
    return processor


def _install_effective_audio_padding_call(
    monkeypatch: pytest.MonkeyPatch,
) -> list[object]:
    effective_pads: list[object] = []

    def call_hf_processor(
        self: Qwen3ASRProcessor,
        text: object = None,
        audio: object = None,
        **kwargs: object,
    ) -> BatchFeature:
        del text, audio
        output_kwargs = self._merge_kwargs(
            Qwen3ASRProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        effective_pads.append(output_kwargs["audio_kwargs"].get("pad_to_multiple_of"))
        return BatchFeature({"input_ids": [[1]]})

    monkeypatch.setattr(
        Qwen3ASRProcessor,
        "__call__",
        call_hf_processor,
    )
    return effective_pads


def _install_effective_processor_kwargs_call(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, dict[str, object]]]:
    effective_kwargs: list[dict[str, dict[str, object]]] = []

    def call_hf_processor(
        self: Qwen3ASRProcessor,
        text: object = None,
        audio: object = None,
        **kwargs: object,
    ) -> BatchFeature:
        del text, audio
        output_kwargs = self._merge_kwargs(
            Qwen3ASRProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        effective_kwargs.append(output_kwargs)
        return BatchFeature({"input_ids": [[1]]})

    monkeypatch.setattr(
        Qwen3ASRProcessor,
        "__call__",
        call_hf_processor,
    )
    return effective_kwargs


def _install_transformers_553_audio_padding_merge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_merge = Qwen3ASRProcessor._merge_kwargs

    def merge_kwargs(
        self: Qwen3ASRProcessor,
        *args: object,
        **kwargs: object,
    ) -> dict[str, dict[str, object]]:
        output_kwargs = original_merge(self, *args, **kwargs)
        if "pad_to_multiple_of" in kwargs:
            for modality, output_kwarg in output_kwargs.items():
                modality_kwargs = kwargs.get(modality)
                if (
                    isinstance(modality_kwargs, Mapping)
                    and "pad_to_multiple_of" not in modality_kwargs
                ):
                    output_kwarg.pop("pad_to_multiple_of", None)
        return output_kwargs

    monkeypatch.setattr(Qwen3ASRProcessor, "_merge_kwargs", merge_kwargs)


@pytest.mark.parametrize(
    ("mm_kwargs", "tok_kwargs"),
    [
        ({"audio_kwargs": {"pad_to_multiple_of": 3_200_000}}, {}),
        ({"pad_to_multiple_of": 3_200_000}, {}),
        ({"common_kwargs": {"pad_to_multiple_of": 3_200_000}}, {}),
        ({"common_kwargs": [["pad_to_multiple_of", 3_200_000]]}, {}),
        ({}, {"audio_kwargs": {"pad_to_multiple_of": 3_200_000}}),
        ({}, {"pad_to_multiple_of": 3_200_000}),
        ({}, {"common_kwargs": {"pad_to_multiple_of": 3_200_000}}),
        ({}, {"common_kwargs": [["pad_to_multiple_of", 3_200_000]]}),
    ],
)
def test_rejects_request_owned_audio_padding_before_hf_processor_call(
    monkeypatch: pytest.MonkeyPatch,
    mm_kwargs: dict[str, object],
    tok_kwargs: dict[str, object],
) -> None:
    processor = _make_processor()
    effective_pads = _install_effective_audio_padding_call(monkeypatch)

    with pytest.raises(ValueError, match="pad_to_multiple_of"):
        processor._call_hf_processor(
            prompt="<|audio_pad|>",
            mm_data={"audios": []},
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

    assert effective_pads == []


def test_allows_operator_owned_audio_padding_without_request_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = _make_processor({"audio_kwargs": {"pad_to_multiple_of": 320}})
    effective_pads = _install_effective_audio_padding_call(monkeypatch)

    result = processor._call_hf_processor(
        prompt="<|audio_pad|>",
        mm_data={"audios": []},
        mm_kwargs={},
        tok_kwargs={},
    )

    assert result["input_ids"] == [[1]]
    assert effective_pads == [320]


def test_preserves_operator_owned_audio_padding_through_qwen3_omni_audio_workaround(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = _make_processor({"audio_kwargs": {"pad_to_multiple_of": 320}})
    effective_pads = _install_effective_audio_padding_call(monkeypatch)

    result = processor._call_hf_processor(
        prompt="<|audio_pad|>",
        mm_data={"audios": [np.zeros(1, dtype=np.float32)]},
        mm_kwargs={},
        tok_kwargs={},
    )

    assert result["input_ids"] == [[1]]
    assert effective_pads == [320]


@pytest.mark.parametrize(
    ("mm_kwargs", "tok_kwargs"),
    [
        ({"pad_to_multiple_of": 320}, {}),
        ({"common_kwargs": {"pad_to_multiple_of": 320}}, {}),
        ({"audio_kwargs": {"pad_to_multiple_of": 320}}, {}),
        ({}, {"pad_to_multiple_of": 320}),
        ({}, {"common_kwargs": {"pad_to_multiple_of": 320}}),
    ],
)
def test_allows_same_value_audio_padding_carriers_through_qwen3_omni_audio_workaround(
    monkeypatch: pytest.MonkeyPatch,
    mm_kwargs: dict[str, object],
    tok_kwargs: dict[str, object],
) -> None:
    processor = _make_processor({"audio_kwargs": {"pad_to_multiple_of": 320}})
    effective_pads = _install_effective_audio_padding_call(monkeypatch)

    result = processor._call_hf_processor(
        prompt="<|audio_pad|>",
        mm_data={"audios": [np.zeros(1, dtype=np.float32)]},
        mm_kwargs=mm_kwargs,
        tok_kwargs=tok_kwargs,
    )

    assert result["input_ids"] == [[1]]
    assert effective_pads == [320]


@pytest.mark.parametrize(
    ("deployment_kwargs", "mm_kwargs", "tok_kwargs"),
    [
        ({"pad_to_multiple_of": 320}, {}, {}),
        (
            {"audio_kwargs": {"pad_to_multiple_of": 320}},
            {"pad_to_multiple_of": 320},
            {},
        ),
        (
            {"audio_kwargs": {"pad_to_multiple_of": 320}},
            {},
            {"pad_to_multiple_of": 320},
        ),
        (
            {"pad_to_multiple_of": 320},
            {"audio_kwargs": {"pad_to_multiple_of": 320}},
            {},
        ),
    ],
)
def test_preserves_audio_padding_under_transformers_553_merge_semantics(
    monkeypatch: pytest.MonkeyPatch,
    deployment_kwargs: dict[str, object],
    mm_kwargs: dict[str, object],
    tok_kwargs: dict[str, object],
) -> None:
    processor = _make_processor(deployment_kwargs)
    _install_transformers_553_audio_padding_merge(monkeypatch)
    effective_pads = _install_effective_audio_padding_call(monkeypatch)

    result = processor._call_hf_processor(
        prompt="<|audio_pad|>",
        mm_data={"audios": [np.zeros(1, dtype=np.float32)]},
        mm_kwargs=mm_kwargs,
        tok_kwargs=tok_kwargs,
    )

    assert result["input_ids"] == [[1]]
    assert effective_pads == [320]


def test_allows_same_value_flat_audio_padding_without_audio_under_553_merge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = _make_processor({"audio_kwargs": {"pad_to_multiple_of": 320}})
    _install_transformers_553_audio_padding_merge(monkeypatch)
    effective_pads = _install_effective_audio_padding_call(monkeypatch)

    result = processor._call_hf_processor(
        prompt="<|audio_pad|>",
        mm_data={"audios": []},
        mm_kwargs={"pad_to_multiple_of": 320},
        tok_kwargs={},
    )

    assert result["input_ids"] == [[1]]
    assert effective_pads == [320]


def test_preserves_flat_operator_padding_text_semantics_under_553_merge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = _make_processor({"pad_to_multiple_of": 320})
    _install_transformers_553_audio_padding_merge(monkeypatch)
    effective_kwargs = _install_effective_processor_kwargs_call(monkeypatch)

    result = processor._call_hf_processor(
        prompt="<|audio_pad|>",
        mm_data={"audios": [np.zeros(1, dtype=np.float32)]},
        mm_kwargs={},
        tok_kwargs={},
    )

    assert result["input_ids"] == [[1]]
    assert effective_kwargs[0]["text_kwargs"]["pad_to_multiple_of"] == 320
    assert effective_kwargs[0]["audio_kwargs"]["pad_to_multiple_of"] == 320


@pytest.mark.parametrize(
    ("mm_kwargs", "tok_kwargs"),
    [
        (
            {
                "pad_to_multiple_of": 3_200_000,
                "audio_kwargs": {"sampling_rate": 16000},
            },
            {},
        ),
        (
            {},
            {
                "pad_to_multiple_of": 3_200_000,
                "audio_kwargs": {"sampling_rate": 16000},
            },
        ),
    ],
)
def test_rejects_flat_padding_with_unrelated_audio_kwargs_under_553_merge(
    monkeypatch: pytest.MonkeyPatch,
    mm_kwargs: dict[str, object],
    tok_kwargs: dict[str, object],
) -> None:
    processor = _make_processor()
    _install_transformers_553_audio_padding_merge(monkeypatch)
    effective_pads = _install_effective_audio_padding_call(monkeypatch)

    with pytest.raises(ValueError, match="pad_to_multiple_of"):
        processor._call_hf_processor(
            prompt="<|audio_pad|>",
            mm_data={"audios": [np.zeros(1, dtype=np.float32)]},
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

    assert effective_pads == []


@pytest.mark.parametrize(
    ("deployment_kwargs", "mm_kwargs", "tok_kwargs"),
    [
        (
            {"common_kwargs": {"pad_to_multiple_of": 320}},
            {
                "pad_to_multiple_of": 320,
                "audio_kwargs": {"sampling_rate": 16000},
            },
            {},
        ),
        (
            {"pad_to_multiple_of": 320},
            {
                "audio_kwargs": {
                    "pad_to_multiple_of": 320,
                    "sampling_rate": 16000,
                }
            },
            {},
        ),
        (
            {"audio_kwargs": {"pad_to_multiple_of": 320}},
            {
                "pad_to_multiple_of": 320,
                "audio_kwargs": {"sampling_rate": 16000},
            },
            {},
        ),
        (
            {"audio_kwargs": {"pad_to_multiple_of": 320}},
            {
                "common_kwargs": {"pad_to_multiple_of": 320},
                "audio_kwargs": {"sampling_rate": 16000},
            },
            {},
        ),
        (
            {"common_kwargs": {"pad_to_multiple_of": 320}},
            {},
            {
                "pad_to_multiple_of": 320,
                "audio_kwargs": {"sampling_rate": 16000},
            },
        ),
        (
            {"audio_kwargs": {"pad_to_multiple_of": 320}},
            {},
            {
                "common_kwargs": {"pad_to_multiple_of": 320},
                "audio_kwargs": {"sampling_rate": 16000},
            },
        ),
    ],
)
def test_allows_same_value_padding_with_unrelated_audio_kwargs_under_553_merge(
    monkeypatch: pytest.MonkeyPatch,
    deployment_kwargs: dict[str, object],
    mm_kwargs: dict[str, object],
    tok_kwargs: dict[str, object],
) -> None:
    processor = _make_processor(deployment_kwargs)
    _install_transformers_553_audio_padding_merge(monkeypatch)
    effective_kwargs = _install_effective_processor_kwargs_call(monkeypatch)

    result = processor._call_hf_processor(
        prompt="<|audio_pad|>",
        mm_data={"audios": [np.zeros(1, dtype=np.float32)]},
        mm_kwargs=mm_kwargs,
        tok_kwargs=tok_kwargs,
    )

    assert result["input_ids"] == [[1]]
    assert effective_kwargs[0]["audio_kwargs"]["pad_to_multiple_of"] == 320
    assert effective_kwargs[0]["audio_kwargs"]["sampling_rate"] == 16000


def test_allows_unrelated_request_audio_kwargs_when_padding_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = _make_processor()
    effective_kwargs = _install_effective_processor_kwargs_call(monkeypatch)

    result = processor._call_hf_processor(
        prompt="<|audio_pad|>",
        mm_data={"audios": []},
        mm_kwargs={"audio_kwargs": {"sampling_rate": 16000}},
        tok_kwargs={},
    )

    assert result["input_ids"] == [[1]]
    assert "pad_to_multiple_of" not in effective_kwargs[0]["audio_kwargs"]
    assert effective_kwargs[0]["audio_kwargs"]["sampling_rate"] == 16000


def test_preserves_operator_padding_with_unrelated_request_audio_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = _make_processor({"audio_kwargs": {"pad_to_multiple_of": 320}})
    effective_kwargs = _install_effective_processor_kwargs_call(monkeypatch)

    result = processor._call_hf_processor(
        prompt="<|audio_pad|>",
        mm_data={"audios": []},
        mm_kwargs={"audio_kwargs": {"sampling_rate": 16000}},
        tok_kwargs={},
    )

    assert result["input_ids"] == [[1]]
    assert effective_kwargs[0]["audio_kwargs"]["pad_to_multiple_of"] == 320
    assert effective_kwargs[0]["audio_kwargs"]["sampling_rate"] == 16000


def test_preserves_flat_operator_padding_with_unrelated_audio_kwargs_under_553_merge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = _make_processor({"pad_to_multiple_of": 320})
    _install_transformers_553_audio_padding_merge(monkeypatch)
    effective_kwargs = _install_effective_processor_kwargs_call(monkeypatch)

    result = processor._call_hf_processor(
        prompt="<|audio_pad|>",
        mm_data={"audios": [np.zeros(1, dtype=np.float32)]},
        mm_kwargs={"audio_kwargs": {"sampling_rate": 16000}},
        tok_kwargs={},
    )

    assert result["input_ids"] == [[1]]
    assert effective_kwargs[0]["audio_kwargs"]["pad_to_multiple_of"] == 320
    assert effective_kwargs[0]["audio_kwargs"]["sampling_rate"] == 16000


def test_rejects_non_bool_audio_padding_equality_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = _make_processor(
        {"audio_kwargs": {"pad_to_multiple_of": _NonBoolEquality()}}
    )
    effective_pads = _install_effective_audio_padding_call(monkeypatch)

    with pytest.raises(ValueError, match="pad_to_multiple_of"):
        processor._call_hf_processor(
            prompt="<|audio_pad|>",
            mm_data={"audios": []},
            mm_kwargs={"audio_kwargs": {"pad_to_multiple_of": _NonBoolEquality()}},
            tok_kwargs={},
        )

    assert effective_pads == []


def test_allows_request_that_repeats_operator_owned_audio_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = _make_processor({"audio_kwargs": {"pad_to_multiple_of": 320}})
    effective_pads = _install_effective_audio_padding_call(monkeypatch)

    result = processor._call_hf_processor(
        prompt="<|audio_pad|>",
        mm_data={"audios": []},
        mm_kwargs={"audio_kwargs": {"pad_to_multiple_of": 320}},
        tok_kwargs={},
    )

    assert result["input_ids"] == [[1]]
    assert effective_pads == [320]
