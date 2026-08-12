# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""curated.py is the ONLY home for curated data: no other module may
reuse one of its names or copy a table's VALUE under a new name."""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "ci_analyzer"


def _module_level_names(tree: ast.Module) -> set[str]:
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _module_level_values(tree: ast.Module) -> dict[str, object]:
    """name -> hashable canonical value for module-level container literals
    with >= 3 elements (small ones are coincidence-prone)."""
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if not isinstance(target, ast.Name):
            continue
        if (
            isinstance(value, ast.Call)
            and getattr(value.func, "id", "") == "frozenset"
            and value.args
        ):
            value = value.args[0]
        try:
            lit = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            continue
        if isinstance(lit, (set, frozenset)) and len(lit) >= 3:
            out[target.id] = frozenset(lit)
        elif isinstance(lit, (list, tuple)) and len(lit) >= 3:
            out[target.id] = tuple(lit)
        elif isinstance(lit, dict) and len(lit) >= 3:
            out[target.id] = tuple(sorted(lit.items(), key=repr))
    return out


def test_curated_names_defined_nowhere_else():
    curated_tree = ast.parse((PACKAGE / "curated.py").read_text())
    curated_names = {n for n in _module_level_names(curated_tree) if n.isupper()}
    assert len(curated_names) > 20, "curated.py lost its tables?"
    violations = []
    for path in PACKAGE.rglob("*.py"):
        if path.name == "curated.py":
            continue
        redefined = curated_names & _module_level_names(ast.parse(path.read_text()))
        for name in sorted(redefined):
            violations.append(
                f"{name} redefined in {path.relative_to(PACKAGE.parent)}"
                " -- curated.py is the only home for curated data"
            )
    assert not violations, "\n".join(violations)


def test_curated_values_not_forked_under_new_names():
    """A pasted copy of a curated table under a different name has the same
    value at fork time; catch it at the snapshot, before it diverges."""
    curated_values = _module_level_values(
        ast.parse((PACKAGE / "curated.py").read_text())
    )
    assert len(curated_values) > 6, "curated.py lost its tables / helper broke?"
    by_value = {v: n for n, v in curated_values.items()}
    violations = []
    for path in PACKAGE.rglob("*.py"):
        if path.name == "curated.py":
            continue
        for name, value in _module_level_values(ast.parse(path.read_text())).items():
            twin = by_value.get(value)
            if twin:
                violations.append(
                    f"{path.relative_to(PACKAGE.parent)}:{name} duplicates "
                    f"curated.{twin}'s value -- import it instead"
                )
    assert not violations, "\n".join(violations)
