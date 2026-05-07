# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for vllm.compilation.codegen — execution code generation.

Each test runs a real Python function through the same pipeline vLLM uses
in production: ``make_fx`` to obtain an aten-level fx graph, ``split_graph``
to split it into the stitching layer + submodules, and then
``generate_execution_code``/``compile_execution_fn`` for codegen.
"""

from collections.abc import Callable

import pytest
import regex as re
import torch
import torch.fx as fx
from torch.fx.experimental.proxy_tensor import make_fx

from vllm.compilation.backends import split_graph
from vllm.compilation.codegen import (
    _node_ref,
    compile_execution_fn,
    generate_execution_code,
    generate_execution_code_with_name,
)
from vllm.utils.torch_utils import is_torch_equal_or_newer


def _trace_and_split(
    model_fn: Callable[..., torch.Tensor],
    example_inputs: tuple[torch.Tensor, ...],
    split_ops: list[str],
) -> fx.GraphModule:
    """Trace ``model_fn`` with make_fx, then split on the named aten ops."""
    gm = make_fx(model_fn)(*example_inputs)
    split_gm, _ = split_graph(gm, split_ops)
    return split_gm


def _to_copy_model(x: torch.Tensor) -> torch.Tensor:
    """Traces to ``aten._to_copy.default`` with device + dtype kwargs."""
    return x.to(device=torch.device("cpu"), dtype=torch.float16)


def _empty_model(x: torch.Tensor) -> torch.Tensor:
    """Traces to ``aten.empty.memory_format`` with device + dtype kwargs."""
    buf = torch.empty(x.shape, device=torch.device("cpu"), dtype=torch.float16)
    return buf.fill_(0).add(x.to(dtype=torch.float16))


@pytest.fixture
def x() -> torch.Tensor:
    return torch.zeros(2, 3)


@pytest.mark.parametrize(
    "model_fn,split_ops",
    [
        (_to_copy_model, ["aten::_to_copy.default"]),
        (_empty_model, []),
    ],
    ids=["aten::_to_copy.default", "aten::empty.memory_format"],
)
def test_non_primitive_kwargs_lifted_to_consts(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    split_ops: list[str],
    x: torch.Tensor,
) -> None:
    """Regression: arguments whose ``repr()`` is not a valid Python
    expression in the generated function's namespace (notably
    ``torch.device``) used to be inlined via ``repr()``, producing source
    like

        out = torch.ops.aten._to_copy.default(x, device=device(type='cpu'))

    which fails at call time — only ``torch`` and ``operator`` are imported
    into the namespace, so ``device`` is unbound. The fix collects such
    objects into ``__vllm_consts__`` and references them by index. The
    unqualified ``device(type=...)`` form must never appear in the
    generated source."""
    split_gm = _trace_and_split(model_fn, (x,), split_ops)
    code, submod_names, consts = generate_execution_code(split_gm)

    assert "device(type=" not in code, (
        "Generated code contains unqualified `device(type=...)` from repr(); "
        "torch.device should be lifted into __vllm_consts__"
    )
    assert torch.device("cpu") in consts, "torch.device kwarg not lifted to consts"
    assert torch.float16 in consts, "torch.dtype kwarg not lifted to consts"

    fn = compile_execution_fn(code, {}, submod_names, consts)
    out = fn(x)
    expected = model_fn(x)
    assert torch.equal(out, expected), "Compiled output does not match reference"


def test_dtype_singleton_deduped(x: torch.Tensor) -> None:
    """``torch.float16`` is a process-wide singleton, so two ops referring
    to it in the traced graph share a single consts slot via ``id()``-based
    dedup. Distinct expressions (``x.to(...)`` vs ``(x*2).to(...)``) ensure
    the tracer can't CSE the two ops into a single node."""

    def model_fn(x: torch.Tensor) -> torch.Tensor:
        return x.to(dtype=torch.float16) + (x * 2).to(dtype=torch.float16)

    split_gm = _trace_and_split(model_fn, (x,), [])
    code, submod_names, consts = generate_execution_code(split_gm)

    # The traced graph must have two distinct _to_copy nodes (otherwise the
    # dedup assertion below is trivially satisfied).
    n_to_copy = sum(
        1
        for n in split_gm.graph.nodes
        if n.op == "call_module"
        for sn in getattr(split_gm, n.target).graph.nodes
        if sn.op == "call_function" and "to_copy" in sn.name
    )
    assert n_to_copy >= 2, (
        f"Test setup failed: expected ≥2 _to_copy nodes, got {n_to_copy}"
    )

    assert consts.count(torch.float16) == 1, (
        f"torch.float16 should occupy exactly one slot, got consts={consts}"
    )
    assert code.count("__vllm_consts__[0]") >= 2, (
        "Deduped const slot should be referenced from both _to_copy nodes"
    )

    fn = compile_execution_fn(code, {}, submod_names, consts)
    assert torch.equal(fn(x), model_fn(x))


def test_distinct_dtypes_get_distinct_slots(x: torch.Tensor) -> None:
    """Distinct dtype singletons in the traced graph occupy distinct slots."""

    def model_fn(x: torch.Tensor) -> torch.Tensor:
        return x.to(dtype=torch.float16) + x.to(dtype=torch.bfloat16)

    split_gm = _trace_and_split(model_fn, (x,), [])
    _, _, consts = generate_execution_code(split_gm)

    assert torch.float16 in consts
    assert torch.bfloat16 in consts
    assert len(consts) == 2, f"Expected 2 distinct dtype slots, got {consts}"


def test_consts_ordering_deterministic(x: torch.Tensor) -> None:
    """Two independent traces of the same model must produce equal consts
    lists *in the same order*. Cache artifacts identify const slots by
    index, so a non-deterministic order would invalidate cached code."""

    def model_fn(x: torch.Tensor) -> torch.Tensor:
        # Multiple distinct non-primitives encountered in a fixed graph order.
        a = x.to(device=torch.device("cpu"), dtype=torch.float16)
        return a.to(dtype=torch.bfloat16)

    _, _, consts1 = generate_execution_code(_trace_and_split(model_fn, (x,), []))
    _, _, consts2 = generate_execution_code(_trace_and_split(model_fn, (x,), []))

    assert len(consts1) >= 2, "Test setup: model should produce ≥2 const slots"
    assert consts1 == consts2, (
        f"consts ordering must be reproducible across traces; "
        f"got {consts1} vs {consts2}"
    )


def test_primitive_args_inlined(x: torch.Tensor) -> None:
    """Primitive args (int dim, etc.) stay inline as repr — no consts."""

    def model_fn(x: torch.Tensor) -> torch.Tensor:
        return torch.transpose(x, 0, 1).relu()

    split_gm = _trace_and_split(model_fn, (x,), [])
    code, submod_names, consts = generate_execution_code(split_gm)

    assert consts == [], "Primitive-only graph must produce empty consts"

    fn = compile_execution_fn(code, {}, submod_names, consts)
    assert torch.equal(fn(x), model_fn(x))


def test_consts_shared_across_split_submods(x: torch.Tensor) -> None:
    """Dedup must apply across inlined submodules, not just within one.

    The function below splits into three inlined submods, two of which
    independently reference ``torch.float16``. The shared ``const_index``
    threaded through recursive ``generate_execution_code_with_name`` calls
    must collapse the dtype to a single slot used from both submods."""

    def model_fn(x: torch.Tensor) -> torch.Tensor:
        a = x.to(dtype=torch.float16)  # submod_0: _to_copy(fp16)
        b = a.relu()  # submod_1: relu (split point)
        c = b.to(dtype=torch.float32)  # submod_2: _to_copy(fp32)
        return c.to(dtype=torch.float16) + 1  # submod_2: another _to_copy(fp16)

    split_gm = _trace_and_split(model_fn, (x,), ["aten::relu.default"])

    n_submods = sum(1 for _ in split_gm.named_children())
    assert n_submods >= 3, (
        f"Test setup failed: expected ≥3 submods after split, got {n_submods}"
    )

    code, submod_names, consts = generate_execution_code(split_gm)

    assert consts.count(torch.float16) == 1, (
        f"fp16 singleton must dedup across submods, got consts={consts}"
    )

    # Find the consts index for fp16 and confirm at least two distinct
    # inlined submods reference it. This rules out the false-positive where
    # one submod references it twice and the other not at all.
    fp16_idx = consts.index(torch.float16)
    submod_bodies = re.findall(
        r"def __vllm_inlined_submods__(\d+)\([^)]*\):\n((?:    .*\n)+)", code
    )
    assert len(submod_bodies) >= 2
    referencing_submods = [
        name for name, body in submod_bodies if f"__vllm_consts__[{fp16_idx}]" in body
    ]
    assert len(referencing_submods) >= 2, (
        f"fp16 slot should be referenced from ≥2 inlined submods, "
        f"got {referencing_submods}"
    )

    fn = compile_execution_fn(code, {}, submod_names, consts)
    assert torch.equal(fn(x), model_fn(x))


def test_non_graphmodule_submod_uses_indexed_callable(x: torch.Tensor) -> None:
    """When a child of split_gm is *not* a ``torch.fx.GraphModule`` — as
    happens in production once ``PiecewiseBackend`` replaces submods —
    codegen emits ``__vllm_submods__[idx](...)`` instead of inlining, and
    the runtime callable is bound from ``submod_callables``."""

    def model_fn(x: torch.Tensor) -> torch.Tensor:
        return x.relu().sigmoid()

    split_gm = _trace_and_split(model_fn, (x,), ["aten::relu.default"])

    # Find a GraphModule child and wrap it in a non-GraphModule nn.Module
    # that delegates to the original — this is the structural shape vLLM
    # produces after PiecewiseBackend takes over a submod.
    child_names = [name for name, _ in split_gm.named_children()]
    target_name = child_names[0]

    class NonGMWrapper(torch.nn.Module):
        def __init__(self, gm: fx.GraphModule) -> None:
            super().__init__()
            self.gm = gm

        def forward(self, *args, **kwargs):
            return self.gm(*args, **kwargs)

    original = getattr(split_gm, target_name)
    del split_gm._modules[target_name]
    split_gm.add_module(target_name, NonGMWrapper(original))

    code, submod_names, consts = generate_execution_code(split_gm)

    assert "__vllm_submods__[" in code, (
        "Non-GraphModule submod should produce an indexed callable reference"
    )
    assert target_name in submod_names

    submod_callables = {
        name: getattr(split_gm, name)
        for name in submod_names
        if not isinstance(getattr(split_gm, name), fx.GraphModule)
    }
    fn = compile_execution_fn(code, submod_callables, submod_names, consts)
    assert torch.equal(fn(x), model_fn(x))


# split_graph only passes tuple_return=True to split_module on PyTorch >= 2.12,
# so getitem nodes only appear in the stitching graph from that version onward.
@pytest.mark.skipif(
    not is_torch_equal_or_newer("2.12.0.dev"),
    reason="split_module tuple_return requires PyTorch >= 2.12",
)
def test_getitem_in_stitching_graph(x: torch.Tensor) -> None:
    """``operator.getitem`` on submod tuple returns is the ``call_function``
    special case at codegen.py — emitted as ``name = source[index]``
    rather than a function call."""

    def model_fn(x: torch.Tensor) -> torch.Tensor:
        return x.relu().sigmoid()

    split_gm = _trace_and_split(model_fn, (x,), ["aten::relu.default"])
    code, _, _ = generate_execution_code(split_gm)

    # split_module wraps each submod return in a tuple, so the stitching
    # graph unpacks via getitem. The codegen must emit it as indexing.
    assert re.search(r"\b\w+ = \w+\[\d+\]\n", code), (
        "Stitching graph should emit `name = source[N]` for getitem nodes"
    )


def test_del_emitted_for_intermediate_values(x: torch.Tensor) -> None:
    """The codegen schedules ``del`` after a value's last use to free
    memory early. Multi-submod splits naturally have intermediates whose
    last use is not the output node."""

    def model_fn(x: torch.Tensor) -> torch.Tensor:
        return x.relu().sigmoid().tanh()

    split_gm = _trace_and_split(
        model_fn, (x,), ["aten::relu.default", "aten::sigmoid.default"]
    )
    code, _, _ = generate_execution_code(split_gm)

    assert re.search(r"^    del \w+", code, re.MULTILINE), (
        "Liveness analysis should emit `del` for intermediates with "
        "last-use before the output"
    )


def test_with_submod_false_rejects_call_module() -> None:
    """``generate_execution_code_with_name(with_submod=False)`` is the
    recursive entry for inlining a GraphModule into its parent. It must
    refuse a graph that itself contains ``call_module`` nodes — the parent
    is responsible for handling those."""
    g = fx.Graph()
    x_node = g.placeholder("x")
    root = torch.nn.Module()
    root.add_module("inner", torch.nn.Identity())
    call = g.call_module("inner", args=(x_node,))
    g.output(call)
    gm = fx.GraphModule(root, g)

    with pytest.raises(RuntimeError, match="call_module is not allowed"):
        generate_execution_code_with_name(gm, "f", with_submod=False)


def test_node_ref_recurses_through_containers() -> None:
    """``_node_ref`` is the recursive walker that lifts non-primitives
    nested inside list/tuple/dict args. Real aten ops rarely produce such
    structures, but the path is needed for DTensor placement lists and
    other future cases — unit-test the walker directly."""
    consts: list = []
    const_index: dict[int, int] = {}
    cpu = torch.device("cpu")

    # Non-primitive in a list, primitive alongside.
    assert _node_ref([cpu, 1], consts, const_index) == "[__vllm_consts__[0], 1]"
    assert consts == [cpu]

    # Same object in a tuple — id-based dedup reuses the existing slot.
    assert _node_ref((cpu, 2), consts, const_index) == "(__vllm_consts__[0], 2)"
    assert consts == [cpu]

    # Single-element tuple uses the trailing-comma form.
    assert _node_ref((cpu,), consts, const_index) == "(__vllm_consts__[0],)"

    # Dict value lifts the same way.
    ref = _node_ref({"k": cpu}, consts, const_index)
    assert ref == "{'k': __vllm_consts__[0]}"


def test_legacy_code_without_consts() -> None:
    """``compile_execution_fn(consts=None)`` must still load code that has
    no ``__vllm_consts__`` reference, so older serialized cache artifacts
    keep working."""
    # Pre-consts codegen: no __vllm_consts__ reference, only torch/operator.
    legacy_code = (
        "import torch\n"
        "def execution_fn(x, *, __vllm_submods__):\n"
        "    return __vllm_submods__[0](x) + 1\n"
    )

    class AddOne(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + 1

    fn = compile_execution_fn(legacy_code, {"sub": AddOne()}, ["sub"], consts=None)
    out = fn(torch.zeros(3))
    assert torch.equal(out, torch.full((3,), 2.0))
