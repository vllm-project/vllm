# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for the RFC #51212 Layer 2 platform component factory.

Architecture under test:
    Platform.get_v2_runner_components() -> V2RunnerComponents -> GPUModelRunner

The static/AST tests run without a GPU or a full vllm install.
The remaining tests require the vllm package (all deps) but still no GPU.
"""

import ast
import pathlib
from dataclasses import fields

import pytest

from vllm.v1.worker.gpu.runner_components import V2RunnerComponents

_SRC = pathlib.Path(__file__).parents[3] / "vllm/v1/worker/gpu/model_runner.py"
_OWNED_CLS = {"RequestState", "InputBuffers", "InputBatch",
               "ModelCudaGraphManager", "Sampler"}


# ---------------------------------------------------------------------------
# 1. V2RunnerComponents structure
# ---------------------------------------------------------------------------

def test_v2runner_components_fields():
    """All six required class-fields must be present."""
    expected = {
        "request_state_cls", "input_buffers_cls", "input_batch_cls",
        "cudagraph_manager_cls", "sampler_cls", "pcp_manager_cls",
    }
    assert {f.name for f in fields(V2RunnerComponents)} == expected


def test_v2runner_components_is_frozen():
    """Bundle must be immutable."""
    from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
    from vllm.v1.worker.gpu.pcp_manager import PCPManager
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.states import RequestState

    bundle = V2RunnerComponents(
        request_state_cls=RequestState,
        input_buffers_cls=InputBuffers,
        input_batch_cls=InputBatch,
        cudagraph_manager_cls=ModelCudaGraphManager,
        sampler_cls=Sampler,
        pcp_manager_cls=PCPManager,
    )
    with pytest.raises(Exception):
        bundle.request_state_cls = RequestState  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. Default Platform factory
# ---------------------------------------------------------------------------

def test_default_factory_returns_v2runner_components():
    """current_platform.get_v2_runner_components() must return V2RunnerComponents."""
    from vllm.platforms import current_platform
    assert isinstance(current_platform.get_v2_runner_components(), V2RunnerComponents)


def test_default_factory_returns_canonical_classes():
    """Default bundle must contain the canonical GPU implementation classes."""
    from vllm.platforms import current_platform
    from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
    from vllm.v1.worker.gpu.pcp_manager import PCPManager
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.states import RequestState

    b = current_platform.get_v2_runner_components()
    assert b.request_state_cls is RequestState
    assert b.input_buffers_cls is InputBuffers
    assert b.input_batch_cls is InputBatch
    assert b.cudagraph_manager_cls is ModelCudaGraphManager
    assert b.sampler_cls is Sampler
    assert b.pcp_manager_cls is PCPManager


# ---------------------------------------------------------------------------
# 3. Custom Platform factory
# ---------------------------------------------------------------------------

def test_custom_factory_bundle_is_respected(monkeypatch):
    """A platform-provided bundle with custom subclasses is stored by the runner."""
    from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
    from vllm.v1.worker.gpu.pcp_manager import PCPManager
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.states import RequestState

    class _RS(RequestState): pass
    class _IB(InputBuffers): pass
    class _Batch(InputBatch): pass
    class _CG(ModelCudaGraphManager): pass
    class _S(Sampler): pass
    class _PCP(PCPManager): pass

    custom = V2RunnerComponents(
        request_state_cls=_RS, input_buffers_cls=_IB, input_batch_cls=_Batch,
        cudagraph_manager_cls=_CG, sampler_cls=_S, pcp_manager_cls=_PCP,
    )

    import vllm.v1.worker.gpu.model_runner as mr_mod
    monkeypatch.setattr(mr_mod, "current_platform",
                        type("_P", (), {"get_v2_runner_components": staticmethod(lambda: custom)})())

    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner._components = mr_mod.current_platform.get_v2_runner_components()

    assert runner._components is custom
    assert runner._components.request_state_cls is _RS
    assert runner._components.input_batch_cls is _Batch
    assert runner._components.sampler_cls is _S


# ---------------------------------------------------------------------------
# 4 & 5. Static enforcement — no direct class construction bypasses the factory
# ---------------------------------------------------------------------------

def _bare_calls(tree: ast.AST, class_name: str, attr: str | None = None) -> list[int]:
    """Return line numbers of bare ClassName(...) or ClassName.attr(...) calls."""
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if attr is None:
            if isinstance(f, ast.Name) and f.id == class_name:
                hits.append(node.lineno)
        else:
            if (isinstance(f, ast.Attribute)
                    and isinstance(f.value, ast.Name)
                    and f.value.id == class_name
                    and f.attr == attr):
                hits.append(node.lineno)
    return hits


@pytest.fixture(scope="module")
def model_runner_ast() -> ast.AST:
    return ast.parse(_SRC.read_text())


@pytest.mark.parametrize("cls_name,attr", [
    ("InputBatch", None),
    ("InputBatch", "make_dummy"),
    ("RequestState", None),
    ("InputBuffers", None),
])
def test_no_bare_construction_in_model_runner(model_runner_ast, cls_name, attr):
    """model_runner.py must use self._components for all owned-class construction."""
    lines = _bare_calls(model_runner_ast, cls_name, attr)
    label = f"{cls_name}.{attr}(" if attr else f"{cls_name}("
    assert not lines, (
        f"Bare {label} at line(s) {lines}; use self._components instead."
    )
