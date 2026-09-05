# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing-structure guards for the FlashInfer fused GDN decode step.

The FlashInfer fused decode step is homed *inside* vLLM's own fused-norm
packed GDN decode route: it replaces the conv1d + gated delta-rule chain
that route runs for plain (non-speculative) decode, which is exactly the
regime vLLM's own fused GDN decode kernel does not serve. That home is a
structural property of ``forward_cuda``, and structure is what breaks when
the surrounding route is rewritten upstream. There are two ways to break it,
and both are silent at import time:

* **broken** -- ``forward_cuda`` skips the ``in_proj_ba`` GEMV (the fused
  step folds it in) on a path that then reads ``ba``;
* **dead** -- a branch returns before the FlashInfer core op is reached, so
  the route is wired but never runs.

These tests read the module as source and prove neither happens, by
enumerating the feasible paths through ``forward_cuda`` under every truth
assignment of its routing predicates. They are deliberately import-free:
no torch, no vLLM, no GPU, no CUDA build -- so they run in any environment
and they run *fast*, which is the point of a guard that has to survive
every upstream rewrite of this file.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_MODULE = (
    _REPO
    / "vllm"
    / "model_executor"
    / "layers"
    / "mamba"
    / "gdn"
    / "qwen_gdn_linear_attn.py"
)

# The predicate that selects vLLM's fused-norm packed core op, and the one
# that selects the FlashInfer step inside it.
_PACKED = "use_fused_gdn_decode"
_FI = "use_fi_fused_decode"

_FI_OP = "qwen_gdn_attention_core_fi"
_PACKED_OP = "qwen_gdn_attention_core_fused_norm_packed"
_LEGACY_OP = "qwen_gdn_attention_core"

# Locals whose definite assignment matters. ``ba`` is the one the FlashInfer
# route skips, and skipping it is what an upstream routing rewrite can turn
# into a latent NameError on the first decode step.
_TRACKED = frozenset({"ba", "b", "a", "z", "mixed_qkv", _PACKED, _FI})


def _class_def(module: ast.Module, name: str) -> ast.ClassDef:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def _method(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"method {name} not found on {cls.name}")


def _attr_chain(node: ast.AST) -> str:
    """Dotted source text of an attribute/name expression, else ''."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(node.id)
    return ".".join(reversed(parts))


def _calls(node: ast.AST) -> list[str]:
    return [_attr_chain(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)]


def _loads(node: ast.AST) -> set[str]:
    return {
        n.id
        for n in ast.walk(node)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }


def _targets(stmt: ast.stmt) -> list[str]:
    names: list[str] = []
    if isinstance(stmt, ast.Assign):
        for target in stmt.targets:
            if isinstance(target, ast.Name):
                names.append(target.id)
            elif isinstance(target, ast.Tuple):
                names.extend(e.id for e in target.elts if isinstance(e, ast.Name))
    elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        names.append(stmt.target.id)
    return names


def _conjuncts(fn: ast.FunctionDef) -> dict[str, list[str]]:
    """``{target: [names]}`` for every ``target = n1 and n2 and ...``.

    These are the implications between the routing predicates: they are what
    makes ``use_fi_fused_decode`` and ``use_fused_gdn_decode`` a nest rather
    than two independent booleans. Without them the enumeration would
    explore states the code cannot be in and report phantom failures.
    """
    out: dict[str, list[str]] = {}
    for stmt in ast.walk(fn):
        if not isinstance(stmt, ast.Assign):
            continue
        value = stmt.value
        if not (isinstance(value, ast.BoolOp) and isinstance(value.op, ast.And)):
            continue
        names = [v.id for v in value.values if isinstance(v, ast.Name)]
        for target in _targets(stmt):
            out[target] = names
    return out


class _State:
    """Truth assignment for the routing predicates on one path."""

    def __init__(self, implies: dict[str, list[str]], known: dict[str, bool]):
        self.implies = implies
        self.known = dict(known)

    def copy(self) -> _State:
        return _State(self.implies, self.known)

    def get(self, name: str) -> bool | None:
        if name in self.known:
            return self.known[name]
        if any(self.known.get(n) is False for n in self.implies.get(name, ())):
            return False
        return None

    def set(self, name: str, truth: bool) -> None:
        self.known[name] = truth
        if truth:
            # `x = a and b` is True only when every conjunct is.
            for conjunct in self.implies.get(name, ()):
                if conjunct not in self.known:
                    self.known[conjunct] = True

    def eval(self, expr: ast.AST) -> bool | None:
        if isinstance(expr, ast.Name):
            return self.get(expr.id)
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
            inner = self.eval(expr.operand)
            return None if inner is None else not inner
        if isinstance(expr, ast.BoolOp):
            values = [self.eval(v) for v in expr.values]
            if isinstance(expr.op, ast.And):
                if any(v is False for v in values):
                    return False
                return True if all(v is True for v in values) else None
            if any(v is True for v in values):
                return True
            return False if all(v is False for v in values) else None
        return None

    def constrain(self, expr: ast.AST, truth: bool) -> None:
        if isinstance(expr, ast.Name):
            self.set(expr.id, truth)
        elif isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
            self.constrain(expr.operand, not truth)
        elif isinstance(expr, ast.BoolOp) and isinstance(expr.op, ast.And) and truth:
            for value in expr.values:
                self.constrain(value, True)


class _Path:
    """One feasible control-flow path through a function body."""

    def __init__(self, state: _State):
        self.state = state
        self.bound: set[str] = set()
        self.calls: list[str] = []
        self.unbound_reads: list[str] = []
        self.returned = False

    def fork(self) -> _Path:
        other = _Path(self.state.copy())
        other.bound = set(self.bound)
        other.calls = list(self.calls)
        other.unbound_reads = list(self.unbound_reads)
        return other

    def predicates(self) -> dict[str, bool]:
        return {k: v for k, v in self.state.known.items() if k in (_PACKED, _FI)}


def _walk(body: list[ast.stmt], path: _Path, out: list[_Path]) -> None:
    for i, stmt in enumerate(body):
        if isinstance(stmt, ast.If):
            truth = path.state.eval(stmt.test)
            rest = list(body[i + 1 :])
            if truth is not False:
                branch = path.fork()
                branch.state.constrain(stmt.test, True)
                _walk(list(stmt.body) + rest, branch, out)
            if truth is not True:
                branch = path.fork()
                branch.state.constrain(stmt.test, False)
                _walk(list(stmt.orelse) + rest, branch, out)
            return
        # Straight-line statement: reads happen before bindings.
        for name in sorted(_loads(stmt) & _TRACKED):
            if name not in path.bound:
                path.unbound_reads.append(name)
        path.calls.extend(_calls(stmt))
        for target in _targets(stmt):
            if isinstance(stmt, ast.Assign):
                value = path.state.eval(stmt.value)
                if value is not None:
                    path.state.set(target, value)
            path.bound.add(target)
        if isinstance(stmt, ast.Return):
            path.returned = True
            break
    out.append(path)


def _paths(fn: ast.FunctionDef) -> list[_Path]:
    out: list[_Path] = []
    _walk(fn.body, _Path(_State(_conjuncts(fn), {})), out)
    return out


@pytest.fixture(scope="module")
def module_ast() -> ast.Module:
    assert _MODULE.is_file(), f"module not found: {_MODULE}"
    return ast.parse(_MODULE.read_text(), filename=str(_MODULE))


@pytest.fixture(scope="module")
def forward_cuda(module_ast: ast.Module) -> ast.FunctionDef:
    return _method(_class_def(module_ast, "QwenGatedDeltaNetAttention"), "forward_cuda")


def test_no_path_reads_a_projection_it_skipped(forward_cuda):
    """``ba`` is never read on a path that skipped the ``in_proj_ba`` GEMV.

    This is the "broken" failure mode: the FlashInfer route folds the b/a
    projection into the fused step, so ``forward_cuda`` skips it -- and a
    branch inserted above the FlashInfer call site that reads ``ba`` turns
    that skip into a NameError on the first decode step.
    """
    offenders = [
        (p.predicates(), p.unbound_reads)
        for p in _paths(forward_cuda)
        if p.unbound_reads
    ]
    assert not offenders, f"unbound local read on some path: {offenders}"


def test_the_flashinfer_route_is_reachable(forward_cuda):
    """Some feasible path actually reaches the FlashInfer core op.

    This is the "dead" failure mode: a branch that returns above the call
    site leaves the route wired, logged, and never executed -- which reads
    as "the fusion did nothing" in a measurement rather than as a bug.
    """
    reaching = [
        p for p in _paths(forward_cuda) if any(c.endswith(_FI_OP) for c in p.calls)
    ]
    assert reaching, "no feasible path through forward_cuda reaches the op"
    for path in reaching:
        assert path.state.get(_FI) is True
        # The home: the FlashInfer step lives inside vLLM's packed route.
        assert path.state.get(_PACKED) is True


def test_vllms_own_routes_are_still_reachable(forward_cuda):
    """vLLM keeps what it owns: both of its core ops stay reachable.

    The FlashInfer step substitutes for exactly one leaf. If no path with
    the route off reaches the packed op, or none reaches the legacy op, the
    integration has taken over more than it replaces.
    """
    paths = _paths(forward_cuda)
    packed = [p for p in paths if any(c.endswith(_PACKED_OP) for c in p.calls)]
    legacy = [
        p
        for p in paths
        if any(c.endswith(_LEGACY_OP) and not c.endswith(_PACKED_OP) for c in p.calls)
    ]
    assert packed, f"vLLM's {_PACKED_OP} is no longer reachable"
    assert legacy, f"vLLM's {_LEGACY_OP} is no longer reachable"
    for path in packed:
        assert path.state.get(_FI) is not True


def test_the_route_cannot_outlive_the_packed_route(forward_cuda):
    """``use_fi_fused_decode`` implies ``use_fused_gdn_decode``, structurally.

    The FlashInfer step is a CUDA decode kernel homed under vLLM's packed
    decode route, so ``VLLM_GDN_DECODE_KERNEL=triton`` -- which turns that
    route off -- must turn this one off too. That is what keeps "triton" an
    exact reproduction of the pre-fusion decode chain, which is what a
    control arm in an A/B is for.
    """
    assigns = [
        stmt
        for stmt in ast.walk(forward_cuda)
        if isinstance(stmt, ast.Assign) and _FI in _targets(stmt)
    ]
    assert len(assigns) == 1, f"expected one {_FI} assignment, got {len(assigns)}"
    value = assigns[0].value
    assert isinstance(value, ast.BoolOp) and isinstance(value.op, ast.And), (
        f"{_FI} must be an `and` of the packed-route predicate and the probe"
    )
    assert _PACKED in _loads(value), f"{_FI} must be conjoined with {_PACKED}"


def test_the_fused_branch_applies_the_gated_rms_norm(module_ast):
    """The packed route owns the norm, so the substituted leaf applies it.

    ``_forward_core_fi`` replaces the core computation only. Dropping the
    gated RMSNorm would be a silent accuracy bug that no shape check
    catches, so assert both branches: the fused one norms in place, and the
    fallback hands the step back to vLLM's ``_forward_core_fused_norm``
    (which norms itself) after projecting the b/a ``forward_cuda`` skipped.
    """
    fn = _method(
        _class_def(module_ast, "QwenGatedDeltaNetAttention"), "_forward_core_fi"
    )
    paths = _paths(fn)
    fused = [p for p in paths if any("_fi_fused_decode_step" in c for c in p.calls)]
    assert fused, "_forward_core_fi never calls the FlashInfer step"
    for path in fused:
        assert any("_rms_norm_gated_cuda" in c for c in path.calls), (
            "the fused branch must apply the gated RMSNorm the packed route owns"
        )

    handed_back = [
        p for p in paths if any(c.endswith("_forward_core_fused_norm") for c in p.calls)
    ]
    assert handed_back, (
        "the non-fused path must hand the step back to _forward_core_fused_norm"
    )
    for path in handed_back:
        assert not any("_fi_fused_decode_step" in c for c in path.calls)
        assert any(c.endswith("in_proj_ba") for c in path.calls), (
            "the non-fused path must project the b/a forward_cuda skipped"
        )


def test_the_core_op_is_a_cudagraph_splitting_op():
    """The op reads per-step metadata on the host: never inline it."""
    compilation = _REPO / "vllm" / "config" / "compilation.py"
    assert f'"vllm::{_FI_OP}"' in compilation.read_text()
