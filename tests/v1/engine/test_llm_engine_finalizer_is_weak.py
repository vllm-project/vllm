# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`LLMEngine`'s cleanup finalizer must not own the model.

``weakref.finalize`` keeps its ``args`` strongly in a class-level registry
until it fires, and it only fires once the ``LLMEngine`` itself is collected.
Handing it the model directly therefore pins the weights -- and the KV cache
aliased onto the model's layers -- for as long as the engine is reachable,
which in-process is past engine shutdown.
"""

import gc
import weakref

import pytest
import torch.nn as nn

from vllm.v1.engine.llm_engine import LLMEngine

pytestmark = pytest.mark.cpu_test


class _Model(nn.Module):
    pass


def test_finalize_args_are_held_strongly():
    """The premise: this is why the model must be passed weakly."""

    class Owner:
        pass

    owner, model = Owner(), _Model()
    weakref.finalize(owner, lambda _m: None, model)
    ref = weakref.ref(model)

    del model
    gc.collect()

    assert ref() is not None
    del owner
    gc.collect()
    assert ref() is None


def test_cleanup_tolerates_a_dead_model():
    model = _Model()
    ref = weakref.ref(model)
    del model
    gc.collect()
    assert ref() is None

    LLMEngine._cleanup_instance_caches(ref)


def test_cleanup_still_walks_a_live_model(monkeypatch):
    from vllm.compilation import wrapper

    cleaned = []

    class _FakeWrapper(nn.Module):
        def cleanup(self):
            cleaned.append(self)

    monkeypatch.setattr(
        wrapper, "TorchCompileWithNoGuardsWrapper", _FakeWrapper, raising=False
    )

    model = _Model()
    model.inner = _FakeWrapper()

    LLMEngine._cleanup_instance_caches(weakref.ref(model))

    assert cleaned == [model.inner]


class _FakeEngineCore:
    def __init__(self):
        self.shutdown_calls = 0

    def shutdown(self, timeout=None):
        self.shutdown_calls += 1


class _EngineStub:
    """Mimics the attributes LLMEngine.__del__ reads, without constructing a
    real LLMEngine (which needs a full VllmConfig and a loaded model)."""

    def __init__(self, engine_core):
        self.engine_core = engine_core
        self.dp_group = None
        self.external_launcher_dp = False


def test_del_shuts_down_engine_core_eagerly():
    """LLMEngine.__del__ must not rely solely on InprocClient's own
    weakref.finalize firing once engine_core happens to become unreachable:
    an incidental extra reference anywhere in the attribute graph can delay
    that indefinitely, which is exactly the shape of #21073 (in-process
    engine deletion not freeing GPU memory). Calling shutdown() eagerly here
    is safe because it is idempotent (guarded by the finalizer's own
    detach() on both InprocClient and MPClient)."""
    engine_core = _FakeEngineCore()
    stub = _EngineStub(engine_core)

    LLMEngine.__del__(stub)

    assert engine_core.shutdown_calls == 1


def test_del_tolerates_a_missing_engine_core():
    """A partially-constructed engine (e.g. __init__ raised before
    engine_core was assigned) must not crash on deletion."""
    stub = _EngineStub.__new__(_EngineStub)
    stub.dp_group = None
    stub.external_launcher_dp = False

    LLMEngine.__del__(stub)
