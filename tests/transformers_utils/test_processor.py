# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib

from transformers.processing_utils import ProcessingKwargs
from typing_extensions import Unpack

from vllm.transformers_utils.processor import (
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
    keys = get_processor_kwargs_keys(get_processor_kwargs_type(proc))
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
    keys = get_processor_kwargs_keys(get_processor_kwargs_type(proc))
    _assert_has_all_expected(keys)


# ---- _check_special_mm_tokens patch ----


class _FakeMMProcessor:
    image_token = "<image>"
    image_token_id = 7


def test_check_special_mm_tokens_patched_and_equivalent():
    import pytest
    import torch
    from transformers import BatchFeature
    from transformers.processing_utils import ProcessorMixin

    check = ProcessorMixin._check_special_mm_tokens
    assert check._vllm_patched is True

    proc = _FakeMMProcessor()
    text = ["a <image> b <image>"]
    ok_ids = BatchFeature({"input_ids": torch.tensor([[1, 7, 2, 7]])})
    check(proc, text, ok_ids, modalities=["image"])

    bad_ids = BatchFeature({"input_ids": torch.tensor([[1, 7, 2, 3]])})
    with pytest.raises(ValueError, match="Mismatch in `image` token count"):
        check(proc, text, bad_ids, modalities=["image"])

    list_ids = BatchFeature({"input_ids": [[1, 7, 2, 7]]})
    check(proc, text, list_ids, modalities=["image"])
