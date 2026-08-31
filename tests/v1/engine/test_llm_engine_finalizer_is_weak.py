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
from types import SimpleNamespace

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


def test_shutdown_continues_after_error(monkeypatch):
    from vllm.v1.engine import core as core_module
    from vllm.v1.engine.core import EngineCore

    calls = []

    def step(name, error=False):
        def run(*args, **kwargs):
            calls.append(name)
            if error:
                raise RuntimeError(name)

        return run

    engine_core = EngineCore.__new__(EngineCore)
    engine_core._shutdown = False
    engine_core._freeze_gc_heap_on_init = False
    engine_core.structured_output_manager = SimpleNamespace(
        clear_backend=step("structured")
    )
    engine_core.model_executor = SimpleNamespace(shutdown=step("executor", error=True))
    engine_core.scheduler = SimpleNamespace(shutdown=step("scheduler"))
    monkeypatch.setattr(core_module.gc, "collect", step("gc"))
    monkeypatch.setattr(core_module, "cleanup_dist_env_and_memory", step("distributed"))

    with pytest.raises(RuntimeError, match="executor"):
        engine_core.shutdown()

    assert calls == ["structured", "executor", "scheduler", "gc", "distributed"]
    assert engine_core.structured_output_manager is None
    assert engine_core.model_executor is None
    assert engine_core.scheduler is None

    calls.clear()
    llm_engine = LLMEngine.__new__(LLMEngine)
    llm_engine._shutdown = False
    llm_engine._finalizer = step("finalizer")
    llm_engine.renderer = SimpleNamespace(shutdown=step("renderer", error=True))
    llm_engine.engine_core = SimpleNamespace(shutdown=step("core"))
    llm_engine.model_executor = object()
    llm_engine._shutdown_dp_group = step("dp")

    with pytest.raises(RuntimeError, match="renderer"):
        llm_engine.shutdown()

    assert calls == ["finalizer", "renderer", "core", "dp"]
    assert llm_engine.engine_core is None
    assert llm_engine.model_executor is None
