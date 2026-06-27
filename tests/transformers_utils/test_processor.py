# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib

import pytest
from transformers.processing_utils import ProcessingKwargs
from typing_extensions import Unpack

from vllm.transformers_utils import processor as processor_mod
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


# ---- disable_type_check: allow non-ProcessorMixin trust_remote_code procs ----


class _BareProcessor:
    """Mimics a trust_remote_code processor not subclassing ProcessorMixin."""

    def __call__(self, *args, **kwargs):
        return None


def test_get_processor_disable_type_check(monkeypatch):
    bare = _BareProcessor()

    class _FakeAutoProcessor:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return bare

    monkeypatch.setattr(processor_mod, "convert_model_repo_to_path", lambda name: name)
    monkeypatch.setattr(
        processor_mod,
        "get_processor_cls_name_from_config",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(processor_mod, "AutoProcessor", _FakeAutoProcessor)

    # Without the flag, a bare (non-ProcessorMixin) processor is rejected.
    with pytest.raises(TypeError):
        get_processor("dummy-model", trust_remote_code=True)

    # With disable_type_check=True the same bare processor loads fine.
    out = get_processor("dummy-model", trust_remote_code=True, disable_type_check=True)
    assert out is bare


def test_get_hf_processor_ignores_injected_disable_type_check(monkeypatch):
    """A user-supplied ``disable_type_check`` (e.g. via per-request
    ``mm_processor_kwargs`` on the OpenAI-compatible API) must NOT weaken the
    ``ProcessorMixin`` type check. Only the trusted, server-side
    ``get_hf_processor_unchecked`` entry point may skip it."""
    from types import SimpleNamespace

    from vllm.multimodal.processing import context as context_mod

    captured: dict[str, object] = {}

    def fake_cached(
        model_config, *, processor_cls, tokenizer, disable_type_check, **kw
    ):
        captured["disable_type_check"] = disable_type_check
        return object()

    monkeypatch.setattr(context_mod, "cached_processor_from_config", fake_cached)
    monkeypatch.setattr(context_mod, "is_mistral_tokenizer", lambda _t: False)

    mm_config = SimpleNamespace(merge_mm_processor_kwargs=lambda kwargs: dict(kwargs))
    model_config = SimpleNamespace(get_multimodal_config=lambda: mm_config)
    ctx = context_mod.InputProcessingContext(model_config=model_config, tokenizer=None)

    # Injected through the public, user-reachable entry point -> ignored.
    ctx.get_hf_processor(disable_type_check=True)
    assert captured["disable_type_check"] is False

    # Selected by trusted model code via the dedicated method -> honored.
    ctx.get_hf_processor_unchecked()
    assert captured["disable_type_check"] is True
