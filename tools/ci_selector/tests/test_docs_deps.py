# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Docs dependency signal + cross-reference audit.

Expectations are derived from the live checkout (extractor, graph index,
mkdocs.yaml), never hardcoded, so they survive checkout drift and fail only on a
real regression. The symbol-checker contract is pinned with synthetic modules
under tmp_path.
"""

import ast

import pytest
import regex as re
from ci_selector.codemap.classify import select
from ci_selector.codemap.docs_deps import (
    MKDOCSTRINGS_RE,
    api_autonav_excludes,
    build_docs_deps,
    extract_refs,
    gen_files_scripts,
    hooks,
    load_mkdocs,
    resolve_autoref,
    symbol_status,
)
from helpers import HW, drift_message

FUSED_MOE_LAYER = "vllm/model_executor/layers/fused_moe/layer.py"


@pytest.fixture(scope="module")
def deps(vllm_repo):
    return build_docs_deps(vllm_repo)


# --- Layer A: the docs_affected signal --------------------------------------


def test_vllm_change_is_floored(deps):
    """Any non-excluded vllm file is a floor dep (api-autonav re-renders it)."""
    affected, reasons = deps.docs_affected([FUSED_MOE_LAYER])
    assert affected
    assert any(FUSED_MOE_LAYER in r and r.startswith("floor:") for r in reasons)


def test_unreferenced_non_floor_file_is_not_affected(deps):
    affected, reasons = deps.docs_affected(["csrc/__ci_selector_nonexistent__.cpp"])
    assert not affected
    assert reasons == []


def test_maintainer_floor_is_covered(deps):
    """Our derived floor must cover every DOCS_PATHS the maintainers hand-list in
    docs/pre_run_check.sh. One representative path per tree."""
    for path in [
        "docs/index.md",
        "examples/offline_inference/basic.py",
        "vllm/config/__init__.py",
        "requirements/test/cuda.txt",
        "requirements/docs.txt",
        "requirements/docs.in",
        "mkdocs.yaml",
        ".readthedocs.yaml",
    ]:
        affected, _ = deps.docs_affected([path])
        assert affected, path


def test_out_of_tree_references_are_precise_deps(deps):
    """EVERY out-of-tree file pulled in by a snippet/link flips the signal with a
    `precise:` reason. Derived from the live extractor, skips if none at HEAD."""
    if not deps.extension:
        pytest.skip("no out-of-tree docs references at HEAD")
    for dep_file in deps.extension:
        assert not dep_file.startswith(("docs/", "examples/", "vllm/")), dep_file
        affected, reasons = deps.docs_affected([dep_file])
        assert affected
        assert any(dep_file in r and r.startswith("precise:") for r in reasons)


def test_non_py_link_is_a_precise_dep(deps):
    """url_schemes.py rewrites ANY relative link, so a non-.py out-of-tree target
    (.sh, .txt, Dockerfile) is a real docs dep; a .py-only match under-selects it."""
    non_py = [f for f in deps.extension if not f.endswith(".py")]
    if not non_py:
        pytest.skip("no non-.py out-of-tree references at HEAD")
    for dep_file in non_py:
        affected, _ = deps.docs_affected([dep_file])
        assert affected, dep_file


def test_collapsed_autoref_shorthand_is_extracted():
    """The [identifier][] shorthand must be captured, and the explicit-form regex
    must not also match it (no double count, no miss)."""
    from ci_selector.codemap.docs_deps import _AUTOREF_COLLAPSED_RE, _AUTOREF_RE

    line = "see [vllm.config.ModelConfig][] for details"
    assert _AUTOREF_COLLAPSED_RE.findall(line) == ["vllm.config.ModelConfig"]
    assert _AUTOREF_RE.findall(line) == []


def test_docs_only_diff_is_affected_through_select(state):
    """A docs-only diff short-circuits selection but must still report the docs
    signal."""
    sel = select(state, ["docs/design/moe_kernel_features.md"])
    assert sel.docs_only
    assert sel.docs_affected


def test_vllm_change_sets_signal_through_select(state):
    sel = select(state, [FUSED_MOE_LAYER])
    assert sel.docs_affected
    assert sel.docs_reasons


def test_degraded_parse_fails_open_to_all_vllm():
    """If mkdocs.yaml can't be parsed, no excludes are known, so the vllm
    floor must cover everything (never under-select)."""
    from ci_selector.codemap.docs_deps import DocsDeps

    degraded = DocsDeps(floor_prefixes=("docs/", "examples/"), degraded=True)
    affected, reasons = degraded.docs_affected(["vllm/anything.py"])
    assert affected
    assert any("fail-open" in r for r in reasons)


def test_build_on_unparsable_mkdocs_is_degraded(tmp_path):
    """End-to-end fail-open: a malformed mkdocs.yaml yields a degraded deps set
    with no excludes, so any vllm change is still flagged."""
    (tmp_path / "mkdocs.yaml").write_text("plugins: [ : broken : :")
    deps = build_docs_deps(tmp_path)
    assert deps.degraded
    assert deps.vllm_exclude_prefixes == () and deps.vllm_exclude_regexes == ()
    affected, reasons = deps.docs_affected(["vllm/anything.py"])
    assert affected
    assert any("fail-open" in r for r in reasons)


# --- Layer B: the symbol-presence checker -----------------------------------


def _module(tmp_path, body: str) -> tuple[str, object]:
    (tmp_path / "m.py").write_text(body)
    return "m.py", tmp_path


def test_symbol_absent_is_broken(tmp_path):
    file, tmp_root = _module(tmp_path, "class Other:\n    pass\n")
    assert symbol_status(file, "Missing", tmp_root) == "BROKEN"


def test_symbol_defined_is_present(tmp_path):
    file, tmp_root = _module(tmp_path, "class Real:\n    pass\n")
    assert symbol_status(file, "Real", tmp_root) == "PRESENT"


def test_reexported_symbol_is_present(tmp_path):
    """A `from .x import Foo` (or __all__ entry) in an __init__ is the symbol's
    documented home even though it is not defined there."""
    file, tmp_root = _module(tmp_path, "from .sub import Foo\n__all__ = ['Foo']\n")
    assert symbol_status(file, "Foo", tmp_root) == "PRESENT"


def test_type_checking_import_is_present(tmp_path):
    body = (
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from .x import Cfg\n"
    )
    file, tmp_root = _module(tmp_path, body)
    assert symbol_status(file, "Cfg", tmp_root) == "PRESENT"


def test_module_getattr_is_uncertain_not_broken(tmp_path):
    """A module-level __getattr__ can synthesize any name, so absence is not
    provable: stay silent rather than false-alarm."""
    file, tmp_root = _module(tmp_path, "def __getattr__(name):\n    return None\n")
    assert symbol_status(file, "Anything", tmp_root) == "UNCERTAIN"


def test_star_import_is_uncertain(tmp_path):
    file, tmp_root = _module(tmp_path, "from .everything import *\n")
    assert symbol_status(file, "Whatever", tmp_root) == "UNCERTAIN"


def test_real_reexports_are_not_broken(vllm_repo, full):
    """Precision on live re-export cases a define-only walk would false-flag."""
    index = full.index
    for target in ["vllm.config.ModelConfig", "vllm.LLM"]:
        module, symbol = resolve_autoref(target, index)
        assert module is not None and symbol is not None, target
        assert symbol_status(module, symbol, vllm_repo) != "BROKEN", target


# --- completeness detectors (self-adapting) ---------------------------------


def test_no_unhandled_mkdocstrings_directives(vllm_repo):
    """A `::: module` directive is a reference channel we don't parse; none exist
    today, so a newly-added one must be handled, not silently ignored."""
    offenders = [
        f"{md.relative_to(vllm_repo)}:{i}"
        for md in (vllm_repo / "docs").rglob("*.md")
        for i, line in enumerate(md.read_text().splitlines(), 1)
        if MKDOCSTRINGS_RE.match(line)
    ]
    assert not offenders, offenders


def test_mkdocs_parses_with_expected_sections(vllm_repo):
    """Guards the tolerant loader and mkdocs drift: the lists the floor derives
    from must be present and non-empty."""
    data = load_mkdocs(vllm_repo)
    assert data is not None
    assert gen_files_scripts(data)
    assert hooks(data)
    assert api_autonav_excludes(data)


def test_extractor_captures_every_snippet(vllm_repo):
    """Every `--8<--` occurrence becomes a snippet Ref (no silent drop), checked
    against an independent raw-text count."""
    snippet_re = re.compile(r'--8<--\s*"[^"]+"')
    raw = sum(
        len(snippet_re.findall(md.read_text()))
        for md in (vllm_repo / "docs").rglob("*.md")
    )
    extracted = sum(1 for r in extract_refs(vllm_repo) if r.kind == "snippet")
    assert extracted == raw


def _toplevel_names(text: str) -> set[str]:
    """Independent, simpler re-derivation of module-top names, to check the
    production symbol checker without sharing its code."""
    names: set[str] = set()
    for node in ast.parse(text).body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.Import | ast.ImportFrom):
            names |= {a.asname or a.name.split(".")[0] for a in node.names}
    return names


def test_audit_catches_the_live_broken_reference(vllm_repo, full):
    """While the #44941 FusedMoE cross-reference stays broken in docs, the audit
    reports it BROKEN, independently confirmed absent. Skips once fixed upstream."""
    target = "vllm.model_executor.layers.fused_moe.layer.FusedMoE"
    ref = next(
        (
            r
            for r in extract_refs(vllm_repo)
            if r.kind == "autoref" and r.target == target
        ),
        None,
    )
    if ref is None:
        pytest.skip("the moe_kernel_features FusedMoE reference is gone (fixed?)")
    index = full.index
    module, symbol = resolve_autoref(target, index)
    assert module == FUSED_MOE_LAYER and symbol == "FusedMoE"
    assert symbol not in _toplevel_names((vllm_repo / module).read_text())
    assert symbol_status(module, symbol, vllm_repo) == "BROKEN"


@pytest.mark.drift
def test_docs_infra_files_still_exist(vllm_repo):
    """Anti-vacuity, the same shape as the release-pipeline guard.

    These are named by hand because an edit to any of them changes every
    rendered page, so the docs job is selected without tracing a reference to
    it. An entry that matches nothing selects nothing, and the file falls back
    to ordinary reference tracing, which is exactly what these exist to bypass.
    """
    from ci_selector.handwritten import DOCS_INFRA_FILES

    missing = sorted(p for p in DOCS_INFRA_FILES if not (vllm_repo / p).is_file())
    assert not missing, drift_message(
        f"DOCS_INFRA_FILES names files that do not exist: {missing}",
        "Editing one of these affects every page, so it force-selects the docs "
        "job with no reference tracing. An entry matching nothing means an edit "
        "to the real file reaches the docs job only if something happens to "
        "reference it, which for build config it usually does not.",
        f"the file was renamed in vLLM: update DOCS_INFRA_FILES in {HW}",
        f"it is gone: delete the entry from {HW}",
    )


@pytest.mark.drift
def test_no_docs_build_config_is_unlisted(vllm_repo):
    """The other direction, which the dead-entry check above cannot see.

    These files are listed precisely because nothing references them, so a new
    one is reached by no rule at all: editing it changes every rendered page
    and selects no docs job. A dead entry over-selects; a missing one is the
    silent under-selection.
    """
    from ci_selector.handwritten import DOCS_INFRA_FILES

    found = {
        p.relative_to(vllm_repo).as_posix()
        for pattern in ("mkdocs*.yaml", "mkdocs*.yml", ".readthedocs*.yaml")
        for p in vllm_repo.glob(pattern)
    } | {
        p.relative_to(vllm_repo).as_posix()
        for p in (vllm_repo / "requirements").glob("docs*")
    }
    assert found, drift_message(
        "No docs-build config was found at HEAD at all.",
        "This guard reads the tree to notice a config we do not list. Finding "
        "none lists none, which reads exactly like listing them all.",
        "the docs build moved: update the patterns in this test",
    )
    unlisted = sorted(found - set(DOCS_INFRA_FILES))
    assert not unlisted, drift_message(
        f"Docs-build config that DOCS_INFRA_FILES does not name: {unlisted}",
        "Nothing in the repo references build config, so an unlisted file "
        "reaches the docs job through no rule. An edit to it re-renders every "
        "page and selects nothing to check that.",
        f"add it to DOCS_INFRA_FILES in {HW}",
        "it does not affect the rendered docs: leave it, and say why here",
    )
