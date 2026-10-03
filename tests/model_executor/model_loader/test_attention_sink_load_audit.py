# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard against direct checkpoint-backed attention sink parameter writes.

A model ``load_weights`` that writes a sink parameter with a bare ``copy_`` is
invisible to ``online_process_loader``, so a live weight update silently keeps
the old sink value. This source-level audit rejects those direct writes.
It does not prove that the current parameter loader is called; the runtime
reload tests cover that contract.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

# Checkpoint-backed attention sink parameters, and the runtime/derived tensors
# that intentionally mirror them.
SINK_NAMES = (
    "attn_sink",
    "attention_sink",
    "attention_sink_bias",
    "learnable_sink",
    "learnable_sink_param",
    "sinks",
)

# Modules that assign a sink-named attribute. Kept explicit so that a new sink
# path fails the audit until it is reviewed, even when the new module does not
# happen to match the load-site scan in `_discover_sink_load_sites`.
SINK_PARAMETER_FILES = (
    "vllm/models/deepseek_v4/attention.py",
    "vllm/models/deepseek_v41/attention.py",
    "vllm/models/hy_v4/nvidia/attention.py",
    "vllm/model_executor/models/granite.py",
    "vllm/model_executor/models/granitemoe.py",
    "vllm/model_executor/models/mimo_v2.py",
    "vllm/model_executor/models/mimo_v2_omni.py",
    "vllm/model_executor/models/qwen3_dflash.py",
)

# Files that legitimately copy a runtime tensor built from a sink during
# `process_weights_after_loading`. Listed rather than silently skipped: these
# are the only modules where a sink-named local may be a `copy_` destination.
SINK_DERIVED_COPY_FILES = ("vllm/models/hy_v4/nvidia/flashmla_sparse.py",)


def _mentions_sink(node: ast.AST) -> bool:
    """Whether a subtree references an attention sink by name."""
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            if any(sink in child.value for sink in SINK_NAMES):
                return True
        elif isinstance(child, ast.Name) and any(
            sink in child.id for sink in SINK_NAMES
        ):
            return True
    return False


def _copy_destination_names(call: ast.Call) -> set[str]:
    """Names a ``copy_`` call writes into.

    Handles ``param.copy_``, ``param.data.copy_`` and
    ``params_dict[name][:n].copy_``; for a subscript destination the index is
    what identifies the parameter, since ``params_dict[name]`` is only a sink
    when ``name`` is the loop variable of a sink-named checkpoint key.
    """
    names: set[str] = set()
    target = call.func.value  # type: ignore[attr-defined]
    while isinstance(target, (ast.Subscript, ast.Attribute)):
        if isinstance(target, ast.Subscript):
            index = target.slice
            if isinstance(index, ast.Constant) and isinstance(index.value, str):
                names.add(index.value)
        target = target.value
    if isinstance(target, ast.Name):
        names.add(target.id)
    return names


def _is_under_sink_test(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    """Whether ``node`` is nested inside a statement that tests a sink name."""
    return _sink_test_scope(node, parents) is not None


def _sink_test_scope(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> ast.AST | None:
    """The nearest enclosing statement whose test names a sink."""
    current: ast.AST | None = node
    while current is not None:
        parent = parents.get(current)
        if parent is None:
            return None
        if isinstance(parent, (ast.If, ast.While)) and _mentions_sink(parent.test):
            return parent
        current = parent
    return None


def _within(inner: ast.AST, outer: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    """Whether ``inner`` is ``outer`` or lexically inside it."""
    current: ast.AST | None = inner
    while current is not None:
        if current is outer:
            return True
        current = parents.get(current)
    return False


def _collect_sink_locals(
    func: ast.AST, parents: dict[ast.AST, ast.AST]
) -> dict[str, ast.AST]:
    """Locals bound to a sink parameter, each with the branch that binds them.

    The scope is the sink branch itself rather than the enclosing function: the
    same variable name is reused for ordinary weights elsewhere in these
    loaders, and only the binding made under a sink test aliases the sink.
    """
    sink_locals: dict[str, ast.AST] = {}
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        if not (
            isinstance(value, ast.Subscript)
            and isinstance(value.value, ast.Name)
            and value.value.id == "params_dict"
        ):
            continue
        scope = _sink_test_scope(node, parents)
        if scope is None:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                sink_locals[target.id] = scope
    return sink_locals


def _find_direct_sink_copies(source: str) -> list[int]:
    """Return line numbers of ``copy_`` calls that write attention sink data.

    All the historical forms are violations, because each writes a sink
    parameter instead of going through its loader:

        params_dict[name][:n].copy_(narrow_weight)
        param.data.copy_(narrow_weight)        # after param = params_dict[name]
        param[:n].copy_(narrow_weight)         # under ``if "attn_sink" in name``

    A ``copy_`` whose destination is reached through an attribute or subscript
    is a violation whenever a sink name guards it: the destination is then a
    parameter, or a view of one, and no other explanation applies. A bare-name
    destination is only a violation when the name is a sink alias or names the
    sink itself, so ``sinks.copy_(sinks.float())`` in
    ``process_weights_after_loading`` stays out of scope.
    """
    tree = ast.parse(source)
    parents: dict[ast.AST, ast.AST] = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    violations: list[int] = []

    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        # Locals bound to the sink parameter inside a sink test:
        # ``param = params_dict[name]`` under ``if "attn_sink" in name``.
        sink_locals = _collect_sink_locals(func, parents)

        for node in ast.walk(func):
            if not isinstance(node, ast.Call):
                continue
            callee = node.func
            if not isinstance(callee, ast.Attribute) or callee.attr != "copy_":
                continue
            target = callee.value
            root = target
            while isinstance(root, (ast.Subscript, ast.Attribute)):
                root = root.value
            is_sink_alias = isinstance(root, ast.Name) and (
                root.id in sink_locals and _within(node, sink_locals[root.id], parents)
            )
            is_sink_parameter = any(
                any(sink in destination for sink in SINK_NAMES)
                for destination in _copy_destination_names(node)
            )
            is_parameter_view = isinstance(target, (ast.Attribute, ast.Subscript))
            # A view under a sink test is a parameter write; a bare name needs
            # to be tied to the sink to count, and the alias is only the sink
            # inside the branch that bound it.
            if is_sink_alias or (
                (is_parameter_view or is_sink_parameter)
                and _is_under_sink_test(node, parents)
            ):
                violations.append(node.lineno)

    return sorted(set(violations))


def _discover_sink_load_sites() -> set[str]:
    """Find candidate checkpoint sink loaders, including cross-file definitions.

    Text filtering keeps the repository scan cheap: a module that loads weights
    and mentions a sink could route one through the wrong path. The filter
    deliberately does not require a ``copy_`` to be present, or a module would
    drop out of the audit set as soon as its sink load was fixed -- exactly the
    state the audit exists to protect. Each candidate then gets the AST check
    below, so a new sink branch using a bare copy is covered even when its
    Parameter is constructed in another module.
    """
    discovered: set[str] = set()
    for path in (REPO_ROOT / "vllm").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if "load_weights" not in source:
            continue
        if any(sink in source for sink in SINK_NAMES):
            discovered.add(str(path.relative_to(REPO_ROOT)).replace("\\", "/"))
    return discovered


def _discover_sink_parameters() -> set[str]:
    """Find every module that assigns a sink-named attribute.

    The AST is used rather than a text match on ``self.<sink> = `` so that a
    sink declared as an attribute of any name containing a sink name -- and a
    parameter constructed in a call rather than a bare ``Parameter(...)`` -- is
    still discovered.
    """
    discovered: set[str] = set()
    for path in (REPO_ROOT / "vllm").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "Parameter" not in text:
            continue
        for node in ast.walk(ast.parse(text)):
            targets: list[ast.expr] = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            if any(
                isinstance(target, ast.Attribute)
                and any(sink in target.attr for sink in SINK_NAMES)
                for target in targets
            ):
                discovered.add(str(path.relative_to(REPO_ROOT)).replace("\\", "/"))
                break
    return discovered


AUDITED_FILES = sorted(set(SINK_PARAMETER_FILES) | _discover_sink_load_sites())


@pytest.mark.parametrize("relative_path", AUDITED_FILES)
def test_checkpoint_sink_has_no_direct_parameter_copy(relative_path: str):
    """Checkpoint sink loads must not write parameter storage directly."""
    path = REPO_ROOT / relative_path
    assert path.is_file(), f"missing sink load file: {relative_path}"

    violations = _find_direct_sink_copies(path.read_text(encoding="utf-8"))
    assert not violations, (
        f"{relative_path} writes a checkpoint-backed attention sink directly "
        f"with copy_ at line(s) {violations}; route the load through the "
        "parameter loader path so layerwise reload can observe and replay it"
    )


def test_audit_covers_every_sink_parameter_assignment():
    """Every module that assigns a sink parameter must be in the audit list."""
    unaudited = sorted(_discover_sink_parameters() - set(AUDITED_FILES))
    assert not unaudited, (
        "these modules assign an attention sink parameter but are not covered "
        f"by SINK_PARAMETER_FILES: {unaudited}"
    )


def test_audit_actually_scans_the_source_tree():
    """A refactor that moves the audit's root must not turn it into a no-op.

    Both scans below are silent when they find nothing wrong, so a misresolved
    repository root would let every violation through unnoticed.
    """
    assert len(_discover_sink_parameters()) >= 5
    assert len(_discover_sink_load_sites()) >= 10


def test_sink_load_files_are_still_candidate_sink_loaders():
    """The explicit list must not rot into paths the scan no longer matches.

    A file listed here is reviewed as a sink loader; if it stops matching the
    candidate scan, either the sink moved or the scan broke.
    """
    stale = sorted(set(SINK_PARAMETER_FILES) - _discover_sink_parameters())
    assert not stale, (
        f"listed as sink parameters but no sink assignment was found: {stale}"
    )


_DERIVED_SINK_SOURCE = """
def process_weights_after_loading(self, layer):
    self.sinks = torch.empty_like(self.sinks)
    self.sinks.copy_(layer.sinks.float())
"""

_LOADER_SINK_SOURCE = """
def load_weights(self, weights):
    for name, loaded_weight in weights:
        if "attn_sink" in name:
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
        elif "sinks" in name:
            param = params_dict[name]
            param[: loaded_weight.shape[0]].copy_(loaded_weight)
"""

_ALIASED_SINK_SOURCE = """
def load_weights(self, weights):
    for name, loaded_weight in weights:
        param = params_dict[name]
        if "attention_sink_bias" in name:
            loaded_weight = loaded_weight.narrow(0, tp_rank, per_rank)
            param.data.copy_(loaded_weight)
"""

_UNGUARDED_COPY_SOURCE = """
def load_weights(self, weights):
    for name, loaded_weight in weights:
        param = params_dict[name]
        if "attention_sink_bias" in name:
            loaded_weight = loaded_weight.narrow(0, tp_rank, per_rank)
        param.data.copy_(loaded_weight)
"""


def test_detector_flags_param_copy_on_the_sink_parameter():
    """``param[:n].copy_`` under a sink test is the historical violation."""
    assert _find_direct_sink_copies(_LOADER_SINK_SOURCE)


def test_detector_flags_subscript_copy_on_the_sink_parameter():
    """``params_dict[name][:n].copy_`` is the other historical violation."""
    source = """
def load_weights(self, weights):
    for name, loaded_weight in weights:
        if "attn_sink" in name:
            narrow = loaded_weight[head_start:head_end]
            params_dict[name][: narrow.shape[0]].copy_(narrow)
"""
    assert _find_direct_sink_copies(source)


def test_detector_flags_alias_copy_under_a_sink_branch():
    """An alias taken before the branch is still a sink write (MiMo-V2)."""
    assert _find_direct_sink_copies(_ALIASED_SINK_SOURCE)


def test_detector_ignores_a_copy_outside_any_sink_branch():
    """The sink branch only slices here, so the copy is not a sink write.

    Narrowing the detector to guarded destinations is what keeps this audit
    from flagging every ``param.data.copy_`` in a file that mentions a sink.
    """
    assert _find_direct_sink_copies(_UNGUARDED_COPY_SOURCE) == []


def test_detector_ignores_derived_runtime_state():
    """Runtime sink state built after loading is not a checkpoint write."""
    assert _find_direct_sink_copies(_DERIVED_SINK_SOURCE) == []


def test_derived_copy_files_are_free_of_direct_parameter_writes():
    """Files reviewed as runtime copies must not copy into a sink parameter."""
    for relative_path in SINK_DERIVED_COPY_FILES:
        path = REPO_ROOT / relative_path
        assert path.is_file(), f"missing derived sink file: {relative_path}"
        assert _find_direct_sink_copies(path.read_text(encoding="utf-8")) == []
