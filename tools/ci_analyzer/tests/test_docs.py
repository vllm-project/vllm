# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Docs dependency signal + cross-reference audit.

Expectations are derived from the live checkout (extractor, graph index,
mkdocs.yaml), never hardcoded, so they survive repo drift and fail only on a
real regression. The symbol-checker contract is pinned with synthetic modules
under tmp_path.
"""

import ast

import pytest
import regex as re
from ci_analyzer.docs import (
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
from ci_analyzer.select import select

FUSED_MOE_LAYER = "vllm/model_executor/layers/fused_moe/layer.py"


@pytest.fixture(scope="module")
def deps(repo):
    return build_docs_deps(repo)


# --- Layer A: the docs_affected signal --------------------------------------


def test_vllm_change_is_floored(deps):
    """Any non-excluded vllm file is a floor dep (api-autonav re-renders it)."""
    affected, reasons = deps.docs_affected([FUSED_MOE_LAYER])
    assert affected
    assert any(FUSED_MOE_LAYER in r and r.startswith("floor:") for r in reasons)


def test_unreferenced_non_floor_file_is_not_affected(deps):
    affected, reasons = deps.docs_affected(["csrc/__ci_analyzer_nonexistent__.cpp"])
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
    from ci_analyzer.docs import _AUTOREF_COLLAPSED_RE, _AUTOREF_RE

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
    from ci_analyzer.docs import DocsDeps

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
    file, repo = _module(tmp_path, "class Other:\n    pass\n")
    assert symbol_status(file, "Missing", repo) == "BROKEN"


def test_symbol_defined_is_present(tmp_path):
    file, repo = _module(tmp_path, "class Real:\n    pass\n")
    assert symbol_status(file, "Real", repo) == "PRESENT"


def test_reexported_symbol_is_present(tmp_path):
    """A `from .x import Foo` (or __all__ entry) in an __init__ is the symbol's
    documented home even though it is not defined there."""
    file, repo = _module(tmp_path, "from .sub import Foo\n__all__ = ['Foo']\n")
    assert symbol_status(file, "Foo", repo) == "PRESENT"


def test_type_checking_import_is_present(tmp_path):
    body = (
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from .x import Cfg\n"
    )
    file, repo = _module(tmp_path, body)
    assert symbol_status(file, "Cfg", repo) == "PRESENT"


def test_module_getattr_is_uncertain_not_broken(tmp_path):
    """A module-level __getattr__ can synthesize any name, so absence is not
    provable: stay silent rather than false-alarm."""
    file, repo = _module(tmp_path, "def __getattr__(name):\n    return None\n")
    assert symbol_status(file, "Anything", repo) == "UNCERTAIN"


def test_star_import_is_uncertain(tmp_path):
    file, repo = _module(tmp_path, "from .everything import *\n")
    assert symbol_status(file, "Whatever", repo) == "UNCERTAIN"


def test_real_reexports_are_not_broken(repo, full):
    """Precision on live re-export cases a define-only walk would false-flag."""
    index = full.index
    for target in ["vllm.config.ModelConfig", "vllm.LLM"]:
        module, symbol = resolve_autoref(target, index)
        assert module is not None and symbol is not None, target
        assert symbol_status(module, symbol, repo) != "BROKEN", target


# --- completeness tripwires (self-adapting) ---------------------------------


def test_no_unhandled_mkdocstrings_directives(repo):
    """A `::: module` directive is a reference channel we don't parse; none exist
    today, so a newly-added one must be handled, not silently ignored."""
    offenders = [
        f"{md.relative_to(repo)}:{i}"
        for md in (repo / "docs").rglob("*.md")
        for i, line in enumerate(md.read_text().splitlines(), 1)
        if MKDOCSTRINGS_RE.match(line)
    ]
    assert not offenders, offenders


def test_mkdocs_parses_with_expected_sections(repo):
    """Guards the tolerant loader and mkdocs drift: the lists the floor derives
    from must be present and non-empty."""
    data = load_mkdocs(repo)
    assert data is not None
    assert gen_files_scripts(data)
    assert hooks(data)
    assert api_autonav_excludes(data)


def test_extractor_captures_every_snippet(repo):
    """Every `--8<--` occurrence becomes a snippet Ref (no silent drop), checked
    against an independent raw-text count."""
    snippet_re = re.compile(r'--8<--\s*"[^"]+"')
    raw = sum(
        len(snippet_re.findall(md.read_text())) for md in (repo / "docs").rglob("*.md")
    )
    extracted = sum(1 for r in extract_refs(repo) if r.kind == "snippet")
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


def test_audit_catches_the_live_broken_reference(repo, full):
    """While the #44941 FusedMoE cross-reference stays broken in docs, the audit
    reports it BROKEN, independently confirmed absent. Skips once fixed upstream."""
    target = "vllm.model_executor.layers.fused_moe.layer.FusedMoE"
    ref = next(
        (r for r in extract_refs(repo) if r.kind == "autoref" and r.target == target),
        None,
    )
    if ref is None:
        pytest.skip("the moe_kernel_features FusedMoE reference is gone (fixed?)")
    index = full.index
    module, symbol = resolve_autoref(target, index)
    assert module == FUSED_MOE_LAYER and symbol == "FusedMoE"
    assert symbol not in _toplevel_names((repo / module).read_text())
    assert symbol_status(module, symbol, repo) == "BROKEN"


def test_audit_snippet_and_pylink_branches_flag_only_missing_targets(repo, full):
    """Both directions, because the live checkout only proves one. Every
    non-autoref ref currently resolves, so the precision loop below runs its
    assert zero times: on its own it cannot tell a working branch from one that
    returned None for everything. The synthetic pair is the positive control."""
    from ci_analyzer.docs import Ref
    from ci_analyzer.validate.docs_refs import _broken_reason

    index = full.index
    # The pylink target must resolve OUTSIDE docs/: mkdocs resolves its own
    # internal links, so _broken_pylink stays silent on them by design.
    for kind, target in (
        ("snippet", "vllm/nonexistent_zzz.py"),
        ("pylink", "../vllm/nonexistent_zzz.py"),
    ):
        ref = Ref(kind, target, "docs/index.md", 1)
        assert _broken_reason(ref, index, repo) is not None, kind

    for ref in extract_refs(repo):
        if ref.kind == "autoref" or _broken_reason(ref, index, repo) is None:
            continue
        target = ref.target.split(":", 1)[0]
        bases = [repo / target, (repo / ref.md_file).parent / target]
        assert not any(b.exists() for b in bases), (ref.kind, ref.md_file, ref.target)
