# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from pathlib import Path
from unittest.mock import patch

from transformers import processing_utils
from transformers.processing_utils import ProcessingKwargs
from typing_extensions import Unpack

from vllm.transformers_utils.processor import (
    get_processor,
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


def test_hub_processor_loads_from_local_snapshot():
    """Transformers makes a Hub request per file when given a repo ID."""
    with patch.object(
        processing_utils,
        "list_repo_templates",
        wraps=processing_utils.list_repo_templates,
    ) as list_repo_templates:
        processor = get_processor("Qwen/Qwen2-VL-2B-Instruct")

    assert Path(processor.tokenizer.name_or_path).is_dir()
    assert processor.chat_template is not None
    list_repo_templates.assert_not_called()
