# SPDX-License-Identifier: Apache-2.0
"""End-to-end backend registration via env."""
from __future__ import annotations

import importlib

import pytest


def test_quest_backend_class_is_importable():
    mod = importlib.import_module("vllm.v1.attention.backends.quest.backend")
    assert hasattr(mod, "QuestSparseOffloadBackend")


def test_register_helper_records_override():
    from vllm.v1.attention.backends.quest.registration import register
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        _ATTN_OVERRIDES,
    )

    AttentionBackendEnum.CUSTOM.clear_override()
    try:
        register()
        assert AttentionBackendEnum.CUSTOM in _ATTN_OVERRIDES
        assert _ATTN_OVERRIDES[AttentionBackendEnum.CUSTOM].endswith(
            "QuestSparseOffloadBackend"
        )
    finally:
        AttentionBackendEnum.CUSTOM.clear_override()


def test_custom_resolves_to_quest_after_register():
    from vllm.v1.attention.backends.quest.backend import (
        QuestSparseOffloadBackend,
    )
    from vllm.v1.attention.backends.quest.registration import register
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    AttentionBackendEnum.CUSTOM.clear_override()
    try:
        register()
        cls = AttentionBackendEnum.CUSTOM.get_class()
        assert cls is QuestSparseOffloadBackend
    finally:
        AttentionBackendEnum.CUSTOM.clear_override()


def test_custom_unregistered_raises():
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    AttentionBackendEnum.CUSTOM.clear_override()
    with pytest.raises(ValueError, match="must be registered"):
        AttentionBackendEnum.CUSTOM.get_class()
