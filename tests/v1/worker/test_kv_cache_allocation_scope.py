# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import AbstractContextManager
from types import SimpleNamespace
from typing import Any

import torch

import vllm.v1.worker.gpu.attn_utils as attn_utils


class _AllocationScope(AbstractContextManager):
    def __init__(self) -> None:
        self.active = False

    def __enter__(self):
        assert not self.active
        self.active = True
        return self

    def __exit__(self, *args: Any) -> None:
        assert self.active
        self.active = False


def test_mrv2_kv_pool_only_wraps_backing_allocation(monkeypatch) -> None:
    scope = _AllocationScope()
    kv_caches = {"layer": torch.empty(0)}

    def allocate(*args, **kwargs):
        assert scope.active
        return kv_caches

    def bind(*args, **kwargs):
        assert not scope.active

    monkeypatch.setattr(attn_utils, "allocate_kv_cache", allocate)
    monkeypatch.setattr(attn_utils, "bind_kv_cache", bind)
    monkeypatch.setattr(attn_utils, "get_shared_kv_cache_layers", lambda config: {})

    config = SimpleNamespace(
        cache_config=SimpleNamespace(get_resolved_kv_cache_layout=lambda: None),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="test")),
    )
    result = attn_utils.init_kv_cache(
        [],
        {},
        object(),
        torch.device("cpu"),
        [],
        config,
        kv_cache_allocation_context=scope,
    )

    assert result is kv_caches
    assert not scope.active
