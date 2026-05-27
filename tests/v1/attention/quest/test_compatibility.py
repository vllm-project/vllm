# SPDX-License-Identifier: Apache-2.0
"""Compatibility-check unit tests."""
from __future__ import annotations

from types import SimpleNamespace


def _model_config(architecture: str, **kwargs) -> SimpleNamespace:
    base = dict(
        architecture=architecture,
        is_mla=False,
        has_sliding_window=False,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_llama_is_supported():
    from vllm.v1.attention.backends.quest.compatibility import check_model_compat

    errors = check_model_compat(_model_config("llama"))
    assert errors == []


def test_qwen2_is_supported():
    from vllm.v1.attention.backends.quest.compatibility import check_model_compat

    errors = check_model_compat(_model_config("qwen2"))
    assert errors == []


def test_qwen25_is_supported():
    from vllm.v1.attention.backends.quest.compatibility import check_model_compat

    errors = check_model_compat(_model_config("qwen2.5"))
    assert errors == []


def test_mistral_is_supported():
    from vllm.v1.attention.backends.quest.compatibility import check_model_compat

    errors = check_model_compat(_model_config("mistral"))
    assert errors == []


def test_unknown_architecture_rejected():
    from vllm.v1.attention.backends.quest.compatibility import check_model_compat

    errors = check_model_compat(_model_config("deepseek_v2"))
    assert len(errors) == 1
    assert "deepseek_v2" in errors[0]


def test_mla_model_rejected():
    from vllm.v1.attention.backends.quest.compatibility import check_model_compat

    errors = check_model_compat(_model_config("llama", is_mla=True))
    assert any("MLA" in e for e in errors)


def test_sliding_window_rejected():
    from vllm.v1.attention.backends.quest.compatibility import check_model_compat

    errors = check_model_compat(_model_config("llama", has_sliding_window=True))
    assert any("sliding window" in e.lower() for e in errors)


def test_supported_set_contains_known_aliases():
    from vllm.v1.attention.backends.quest.compatibility import (
        SUPPORTED_MODEL_FAMILIES,
    )

    assert {"llama", "mistral", "qwen2", "qwen2.5", "yi"} <= SUPPORTED_MODEL_FAMILIES
