# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Register QuestSparseOffloadBackend as the CUSTOM backend.

Importing this module is the single side-effect entry point. Callers who set
VLLM_ATTENTION_BACKEND to the QuestSparseOffloadBackend qualname don't need
this — the v1 selector imports the class by path. The helper is for
programmatic users (e.g. embedded vLLM tests) who use the
AttentionBackendEnum.CUSTOM symbol directly.
"""
from __future__ import annotations

from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum,
    register_backend,
)

_QUEST_PATH = (
    "vllm.v1.attention.backends.quest.backend.QuestSparseOffloadBackend"
)


def register() -> None:
    """Bind AttentionBackendEnum.CUSTOM to QuestSparseOffloadBackend."""
    register_backend(AttentionBackendEnum.CUSTOM, _QUEST_PATH)
