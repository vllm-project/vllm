# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn

import vllm.model_executor.model_loader.reload.validation as validation
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


def test_reload_storage_guard_verifies_after_body_failure(monkeypatch):
    monkeypatch.setenv("VLLM_RELOAD_GLOBAL_MANIFEST", "off")
    model, layer = _model_with_arena()
    original_verify = validation.verify_model_arenas
    verify_calls = 0

    def tracked_verify(*args, **kwargs):
        nonlocal verify_calls
        verify_calls += 1
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(validation, "verify_model_arenas", tracked_verify)

    with pytest.raises(ValueError, match="load failed") as exc_info:
        with reload_storage_guard(model):
            get_reload_arena(layer)._slots["workspace"] = torch.empty(4)
            raise ValueError("load failed")

    assert verify_calls == 1
    assert any(
        "must not continue serving" in note
        for note in getattr(exc_info.value, "__notes__", [])
    )


def test_reload_failure_is_not_masked_by_validation_failure(monkeypatch):
    monkeypatch.setenv("VLLM_RELOAD_GLOBAL_MANIFEST", "off")
    model, _ = _model_with_arena()

    def fail_validation(*_args, **_kwargs):
        raise RuntimeError("validation failed")

    monkeypatch.setattr(validation, "verify_model_arenas", fail_validation)

    with pytest.raises(ValueError, match="load failed") as exc_info:
        with reload_storage_guard(model):
            raise ValueError("load failed")

    notes = getattr(exc_info.value, "__notes__", [])
    assert any("validation failed" in note for note in notes)
    assert any("must not continue serving" in note for note in notes)
