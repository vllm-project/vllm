# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging

import pytest

from vllm.entrypoints import harmony_utils
from vllm.entrypoints.context import ConversationContext
from vllm.entrypoints.harmony_utils import (get_openai_context,
                                            register_context_loader)

def test_register_context_loader_registers_subclass(monkeypatch):
    monkeypatch.setattr(harmony_utils, "_CONTEXT_LOADER", {})

    @register_context_loader("custom")
    class CustomContext(ConversationContext):
        pass

    assert harmony_utils._CONTEXT_LOADER["custom"] is CustomContext
    assert get_openai_context("custom") is CustomContext

def test_register_context_loader_requires_conversation_context(monkeypatch):
    monkeypatch.setattr(harmony_utils, "_CONTEXT_LOADER", {})

    with pytest.raises(ValueError, match="must be a subclass"):

        @register_context_loader("invalid")
        class InvalidContext:  # pragma: no cover - definition for exception path
            pass

def test_get_openai_context_raises_for_unknown_format(monkeypatch):
    monkeypatch.setattr(harmony_utils, "_CONTEXT_LOADER", {})

    with pytest.raises(ValueError, match="is not supported"):
        get_openai_context("missing")
