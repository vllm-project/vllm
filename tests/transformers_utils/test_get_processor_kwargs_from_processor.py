# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib

from typing_extensions import TypedDict, Unpack

from vllm.transformers_utils.processor import (
    _collect_dynamic_keys_from_processing_kwargs,
    get_processor_kwargs_from_processor,
)


class _FakeTextKwargs(TypedDict, total=False):
    padding: bool
    truncation: bool
    max_length: int

class _FakeImagesKwargs(TypedDict, total=False):
    do_convert_rgb: bool
    data_format: str

class _FakeVideosKwargs(TypedDict, total=False):
    fps: float
    use_audio_in_video: bool
    seconds_per_chunk: float
    position_id_per_seconds: float

class _FakeAudioKwargs(TypedDict, total=False):
    sampling_rate: int
    return_attention_mask: bool

class _FakeProcessingKwargs(TypedDict, total=False):
    text_kwargs: _FakeTextKwargs
    images_kwargs: _FakeImagesKwargs
    videos_kwargs: _FakeVideosKwargs
    audio_kwargs: _FakeAudioKwargs


_FakeProcessingKwargs.__annotations__.update(_FakeTextKwargs.__annotations__)
_FakeProcessingKwargs.__annotations__.update(_FakeImagesKwargs.__annotations__)
_FakeProcessingKwargs.__annotations__.update(_FakeVideosKwargs.__annotations__)
_FakeProcessingKwargs.__annotations__.update(_FakeAudioKwargs.__annotations__)


def _assert_has_all_expected(keys: set[str]) -> None:
    for wrapper in ("text_kwargs", "images_kwargs", "videos_kwargs", "audio_kwargs"):
        assert wrapper in keys

    for k in ("padding", "truncation", "max_length"):
        assert k in keys
    for k in ("do_convert_rgb", "data_format"):
        assert k in keys
    for k in ("fps", "use_audio_in_video", "seconds_per_chunk",
              "position_id_per_seconds"):
        assert k in keys
    for k in ("sampling_rate", "return_attention_mask"):
        assert k in keys


def test_collect_dynamic_keys_from_processing_kwargs_union():
    keys = _collect_dynamic_keys_from_processing_kwargs(_FakeProcessingKwargs)
    _assert_has_all_expected(keys)


# Path 1: __call__ method has kwargs: Unpack[*ProcessingKwargs]
class _ProcWithUnpack:
    def __call__(self, *args, **kwargs: Unpack[_FakeProcessingKwargs]):
        return None


def test_get_processor_kwargs_from_processor_unpack_path_returns_full_union():
    proc = _ProcWithUnpack()
    keys = get_processor_kwargs_from_processor(proc)
    _assert_has_all_expected(keys)


# ---- Path 2: No Unpack, fallback to scanning *ProcessingKwargs in module ----

class MyTestProcessingKwargs(_FakeProcessingKwargs):
    pass


class _ProcWithoutUnpack:
    def __call__(self, *args, **kwargs):
        return None


def test_get_processor_kwargs_from_processor_module_scan_returns_full_union():
    # ensure the module scanned by fallback is this test module
    module_name = _ProcWithoutUnpack.__module__
    mod = importlib.import_module(module_name)
    assert hasattr(mod, "MyTestProcessingKwargs")

    proc = _ProcWithoutUnpack()
    keys = get_processor_kwargs_from_processor(proc)
    _assert_has_all_expected(keys)