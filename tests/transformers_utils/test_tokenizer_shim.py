# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the vllm.transformers_utils.tokenizer backward-compat shim (#52614)."""

import pytest


def test_mistral_tokenizer_legacy_import_path():
    """The pre-#35024 import path used by lm-format-enforcer must keep working
    and resolve to the same class object (it is used for isinstance checks)."""
    with pytest.warns(DeprecationWarning):
        from vllm.transformers_utils.tokenizer import MistralTokenizer as legacy

    from vllm.tokenizers.mistral import MistralTokenizer

    assert legacy is MistralTokenizer


def test_shim_rejects_unknown_attributes():
    import vllm.transformers_utils.tokenizer as shim_module

    with pytest.raises(AttributeError):
        _ = shim_module.does_not_exist
