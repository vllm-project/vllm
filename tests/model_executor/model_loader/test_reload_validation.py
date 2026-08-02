# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn

from vllm.model_executor.model_loader.reload.validation import (
    reload_storage_guard,
)
from vllm.model_executor.reload_arena import get_reload_arena


def _model_with_arena() -> tuple[nn.Module, nn.Module]:
    model = nn.Module()
    model.layer = nn.Module()
    get_reload_arena(model.layer).get_or_alloc(
        "workspace", (4,), torch.float32, "cpu"
    )
    return model, model.layer


def test_reload_storage_guard_accepts_stable_slots(monkeypatch):
    monkeypatch.setenv("VLLM_RELOAD_GLOBAL_MANIFEST", "off")
    model, _ = _model_with_arena()

    with reload_storage_guard(model):
        pass


def test_reload_storage_guard_rejects_moved_slots(monkeypatch):
    monkeypatch.setenv("VLLM_RELOAD_GLOBAL_MANIFEST", "off")
    model, layer = _model_with_arena()

    with pytest.raises(RuntimeError, match="graph-visible storage identity"):
        with reload_storage_guard(model):
            get_reload_arena(layer)._slots["workspace"] = torch.empty(4)
