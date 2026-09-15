# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""handwritten.py is the ONLY home for curated data: no other module may
reuse one of its names or copy a table's VALUE under a new name."""

import ast
from pathlib import Path

import pytest
from helpers import HW, drift_message

PACKAGE = Path(__file__).resolve().parents[1] / "ci_selector"


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
    curated_tree = ast.parse((PACKAGE / "handwritten.py").read_text())
    curated_names = {n for n in _module_level_names(curated_tree) if n.isupper()}
    assert len(curated_names) > 20, "handwritten.py lost its tables?"
    violations = []
    for path in PACKAGE.rglob("*.py"):
        if path.name == "handwritten.py":
            continue
        redefined = curated_names & _module_level_names(ast.parse(path.read_text()))
        for name in sorted(redefined):
            violations.append(
                f"{name} redefined in {path.relative_to(PACKAGE.parent)}"
                " -- handwritten.py is the only home for it"
            )
    assert not violations, "\n".join(violations)


def test_curated_values_not_forked_under_new_names():
    """A pasted copy of a curated table under a different name has the same
    value at fork time; catch it at the snapshot, before it diverges."""
    curated_values = _module_level_values(
        ast.parse((PACKAGE / "handwritten.py").read_text())
    )
    assert len(curated_values) > 6, "handwritten.py lost its tables / helper broke?"
    by_value = {v: n for n, v in curated_values.items()}
    violations = []
    for path in PACKAGE.rglob("*.py"):
        if path.name == "handwritten.py":
            continue
        for name, value in _module_level_values(ast.parse(path.read_text())).items():
            twin = by_value.get(value)
            if twin:
                violations.append(
                    f"{path.relative_to(PACKAGE.parent)}:{name} duplicates "
                    f"handwritten.{twin}'s value -- import it instead"
                )
    assert not violations, "\n".join(violations)


@pytest.mark.drift
def test_rust_gate_env_vars_exist(vllm_repo):
    """The rust rule's env-key leg searches step text for these gates; a
    renamed gate matches nothing and the e2e steps silently lose the leg."""
    from ci_selector.codemap.pipeline.buildkite import load_pipeline_configs
    from ci_selector.handwritten import RUST_GATE_ENV_VARS

    envs = (vllm_repo / "vllm" / "envs.py").read_text()
    missing = [v for v in RUST_GATE_ENV_VARS if v not in envs]
    # Scoped to live job dirs. Counting the whole tree let .buildkite/test-amd
    # .yaml carry the floor on its own, and LEGACY_CI_FILES says no pipeline
    # reads it, so the gate could leave every live step without a word.
    job_dirs = tuple(
        (d if d.startswith(".buildkite") else f".buildkite/{d}").rstrip("/") + "/"
        for c in load_pipeline_configs(vllm_repo)
        for d in c.job_dirs
    )
    buildkite_hits = sum(
        p.read_text().count("VLLM_USE_RUST_FRONTEND")
        for p in (vllm_repo / ".buildkite").rglob("*.yaml")
        if p.relative_to(vllm_repo).as_posix().startswith(job_dirs)
    )
    # All 5 live hits are in .buildkite/test_areas/rust_frontend.yaml, so the
    # floor is deliberately exact: there is no second file to absorb a loss,
    # and a step quietly dropping the gate is the whole point of the check.
    assert not missing and buildkite_hits >= 5, drift_message(
        f"Gate vars missing from envs.py: {missing}; "
        f"VLLM_USE_RUST_FRONTEND appears {buildkite_hits}x in live job dirs.",
        "Steps that opt into the rust frontend are found by these names; "
        "without them a rust change stops selecting the e2e suites that "
        "exist to catch rust regressions.",
        f"vLLM renamed the gate: update RUST_GATE_ENV_VARS in {HW}",
    )


@pytest.mark.drift
def test_build_rust_still_builds_exactly_two_artifacts(vllm_repo):
    """The two-root discriminator assumes exactly two shipped artifacts; a
    third would need its own closure and routing legs."""
    from ci_selector.handwritten import RUST_ARTIFACT_ROOTS

    text = (vllm_repo / "tools" / "build_rust.py").read_text()
    ext_count = text.count("RustExtension(")
    missing = [r for r in RUST_ARTIFACT_ROOTS if f"{r}/Cargo.toml" not in text]
    assert ext_count == 2 and not missing, drift_message(
        f"build_rust.py declares {ext_count} RustExtension calls; roots "
        f"absent from it: {missing}.",
        "Every rust file is bucketed by which of these artifacts its crate "
        "feeds; a moved or third artifact buckets whole crates wrongly.",
        f"update RUST_ARTIFACT_ROOTS in {HW} and teach "
        "codemap/rust_workspace.py the new closure",
    )


@pytest.mark.drift
def test_pyo3_bridge_file_is_the_only_import_site(vllm_repo):
    """Cdylib-closure files borrow this one file's claim; a second import
    site would leave its consumers unrouted."""
    from ci_selector.handwritten import DYNAMIC_IMPORT_FILES, RUST_PYO3_BRIDGE_FILE

    importers = sorted(
        str(p.relative_to(vllm_repo))
        for p in (vllm_repo / "vllm").rglob("*.py")
        if "_rust_tool_parser" in p.read_text()
    )
    assert importers == [RUST_PYO3_BRIDGE_FILE], drift_message(
        f"vllm/ files naming _rust_tool_parser: {importers}.",
        "The rust rule routes cdylib changes through the single import site; "
        "an unlisted second site means its tests never run on parser changes.",
        f"update RUST_PYO3_BRIDGE_FILE (or widen the bridge to a tuple) in {HW}",
    )
    assert RUST_PYO3_BRIDGE_FILE in DYNAMIC_IMPORT_FILES


@pytest.mark.drift
def test_rust_toolchain_files_exist(vllm_repo):
    from ci_selector.handwritten import RUST_TOOLCHAIN_FILES

    missing = [p for p in RUST_TOOLCHAIN_FILES if not (vllm_repo / p).is_file()]
    assert not missing, drift_message(
        f"Toolchain files gone: {missing}.",
        "These route to the widest rust bucket; a moved entry point would "
        "fall to the generic fail-open instead.",
        f"update RUST_TOOLCHAIN_FILES in {HW}",
    )
