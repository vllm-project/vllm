# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib

import pytest
import torch

import vllm.model_executor.models.kimi_k25_vit as kimi_k25_vit

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def test_get_rope_shape_does_not_compile_at_import(monkeypatch):
    """The vision helper must use vLLM's compile lifecycle, not bare torch.compile."""

    def fail_if_compiled(*args, **kwargs):
        raise AssertionError("get_rope_shape must not call torch.compile")

    monkeypatch.setattr(torch, "compile", fail_if_compiled)
    importlib.reload(kimi_k25_vit)

    assert kimi_k25_vit.get_rope_shape.__name__ == "get_rope_shape"
