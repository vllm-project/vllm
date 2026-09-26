# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from functools import lru_cache

from transformers.processing_utils import ProcessingKwargs
from typing_extensions import Unpack

from vllm.transformers_utils.processor import (
    HashableDict,
    HashableList,
    get_processor_kwargs_keys,
    get_processor_kwargs_type,
)


class _FakeProcessorKwargs(ProcessingKwargs, total=False):  # type: ignore
    pass


def _assert_has_all_expected(keys: set[str]) -> None:
    # text
    for k in ("text_pair", "text_target", "text_pair_target"):
        assert k in keys
    # image
    for k in ("do_convert_rgb", "do_resize"):
        assert k in keys
    # audio
    for k in (
        "fps",
        "do_sample_frames",
        "input_data_format",
        "default_to_square",
    ):
        assert k in keys
    # audio
    for k in ("padding", "return_attention_mask"):
        assert k in keys


# Path 1: __call__ method has kwargs: Unpack[*ProcessorKwargs]
class _ProcWithUnpack:
    def __call__(self, *args, **kwargs: Unpack[_FakeProcessorKwargs]):  # type: ignore
        return None


def test_get_processor_kwargs_from_processor_unpack_path_returns_full_union():
    proc = _ProcWithUnpack()
    keys = get_processor_kwargs_keys(get_processor_kwargs_type(proc))  # type: ignore[arg-type]
    _assert_has_all_expected(keys)


# ---- Path 2: No Unpack, fallback to scanning *ProcessorKwargs in module ----


class _ProcWithoutUnpack:
    def __call__(self, *args, **kwargs):
        return None


def test_get_processor_kwargs_from_processor_module_scan_returns_full_union():
    # ensure the module scanned by fallback is this test module
    module_name = _ProcWithoutUnpack.__module__
    mod = importlib.import_module(module_name)
    assert hasattr(mod, "_FakeProcessorKwargs")

    proc = _ProcWithoutUnpack()
    keys = get_processor_kwargs_keys(get_processor_kwargs_type(proc))  # type: ignore[arg-type]
    _assert_has_all_expected(keys)


def test_hashable_dict_handles_nested_values():
    # Regression for unhashable nested dict/list values fed to lru_cache.
    nested = HashableDict({"images_kwargs": {"num_buckets": 4}, "tags": ["a", "b"]})
    assert hash(nested) == hash(
        HashableDict({"images_kwargs": {"num_buckets": 4}, "tags": ["a", "b"]})
    )


def test_hashable_list_handles_dicts():
    nested = HashableList([{"a": 1}, {"b": [2, 3]}])
    assert hash(nested) == hash(HashableList([{"a": 1}, {"b": [2, 3]}]))


@lru_cache
def _identity(**kwargs: object) -> int:
    return len(kwargs)


def test_nested_kwargs_usable_in_lru_cache():
    # Mirrors _merge_mm_kwargs: each top-level value is wrapped as a
    # HashableDict/HashableList whose own nested values stay native dicts.
    _identity.cache_clear()
    kwargs = {"images_kwargs": HashableDict({"num_buckets": 4})}
    _identity(**kwargs)
    # Second call with equal keys must hit the cache rather than raise.
    _identity(**kwargs)
    assert _identity.cache_info().hits == 1
