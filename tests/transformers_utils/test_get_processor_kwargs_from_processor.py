# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib

from typing_extensions import TypedDict, Unpack

from vllm.transformers_utils.processor import (
    _collect_dynamic_keys_from_processing_kwargs,
    get_processor_kwargs_from_processor,
)
from transformers.processing_utils import ProcessingKwargs


class _FakeProcessorKwargs(ProcessingKwargs, total=False):
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

# Path 1: __call__ method has kwargs: Unpack[*ProcessingKwargs]
class _ProcWithUnpack:
    def __call__(self, *args, **kwargs: Unpack[_FakeProcessorKwargs]):
        return None


def test_get_processor_kwargs_from_processor_unpack_path_returns_full_union():
    proc = _ProcWithUnpack()
    keys = get_processor_kwargs_from_processor(proc)
    _assert_has_all_expected(keys)


# ---- Path 2: No Unpack, fallback to scanning *ProcessingKwargs in module ----

class _ProcWithoutUnpack:
    def __call__(self, *args, **kwargs):
        return None


def test_get_processor_kwargs_from_processor_module_scan_returns_full_union():
    # ensure the module scanned by fallback is this test module
    module_name = _ProcWithoutUnpack.__module__
    mod = importlib.import_module(module_name)
    assert hasattr(mod, "_FakeProcessorKwargs")

    proc = _ProcWithoutUnpack()
    keys = get_processor_kwargs_from_processor(proc)
    _assert_has_all_expected(keys)
