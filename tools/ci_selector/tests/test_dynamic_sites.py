# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynamic-site classification and ambiguity reporting."""

import pytest
from ci_selector.validate.dynamic_sites import classify_dynamic_sites
from helpers import HW, drift_message


@pytest.mark.drift
def test_no_unclassified_dynamic_sites_at_head(vllm_repo, full):
    """Every non-literal dynamic import at HEAD either leaves the repo or has a
    parser reading the table behind it."""
    fg = full
    classified = classify_dynamic_sites(fg.graph.dynamic_sites, fg.graph.table_files)
    assert classified.unclassified == [], drift_message(
        "These do a dynamic import the import graph cannot follow:\n"
        + "\n".join(
            f"    {s.file}:{s.lineno}  ({s.func})" for s in classified.unclassified
        ),
        "Any test reachable only through that import will not be selected, so a "
        "change to what it loads can ship without its tests running.",
        "it loads something outside the repo (a plugin, an optional package, a "
        "name from user config): add the file to DYNAMIC_IMPORT_FILES in " + HW,
        "it reads a table that lives in the repo: teach a parser in "
        "ci_selector/codemap/graph/factories.py to read that table, which links "
        "the edge properly instead of giving up on it",
    )


def test_bare_import_ambiguities_reported_not_silent(full):
    fg = full
    # tests/v1/determinism `from utils import`: sibling wins, and the clash
    # with the indexed top-level tests/utils.py is surfaced.
    clashes = {(file, name) for file, name, _sib, _other in fg.graph.ambiguities}
    assert any(
        file.startswith("tests/v1/determinism/") and name == "utils"
        for file, name in clashes
    ), clashes


def test_new_site_under_former_blanket_is_unclassified(vllm_repo):
    """A NEW dynamic-import site anywhere (even under a formerly-blanketed prefix
    like tests/models/) must land UNCLASSIFIED, not silently blessed."""
    from ci_selector.codemap.graph.imports import DynamicSite

    site = DynamicSite(
        "tests/models/language/generation/dispatch.py", 1, "import_module"
    )
    classified = classify_dynamic_sites([site], set())
    assert classified.unclassified == [site]


@pytest.mark.drift
def test_hand_listed_entries_exist(vllm_repo):
    """Stale-entry guard: every hand-listed path still exists."""
    from ci_selector.handwritten import DYNAMIC_IMPORT_FILES

    missing = [p for p in DYNAMIC_IMPORT_FILES if not (vllm_repo / p).is_file()]
    assert not missing, drift_message(
        f"DYNAMIC_IMPORT_FILES names files that no longer exist: {missing}",
        "Each entry vouches for one file's dynamic import. Pointing at a deleted "
        "path vouches for nothing, and the next import added to whatever replaced "
        "it is pre-approved without anyone looking.",
        "the file moved: update the path in DYNAMIC_IMPORT_FILES in " + HW,
        "the file is gone for good: delete the entry from ci_selector/handwritten.py",
    )


def test_reverse_gate_flags_entries_with_no_live_import(vllm_repo):
    """Not vacuous: with no live sites every hand-listed entry is flagged. An
    entry that outlives its import pre-approves the next one to land there,
    which checking only for unclassified sites never catches."""
    from ci_selector.handwritten import DYNAMIC_IMPORT_FILES
    from ci_selector.validate.dynamic_sites import unused_external_entries

    assert set(unused_external_entries([])) == set(DYNAMIC_IMPORT_FILES)


@pytest.mark.drift
def test_no_unused_hand_list_entries_at_head(vllm_repo, full):
    from ci_selector.validate.dynamic_sites import unused_external_entries

    dead = unused_external_entries(full.graph.dynamic_sites)
    assert dead == [], drift_message(
        "These are listed in DYNAMIC_IMPORT_FILES but no longer contain a "
        f"dynamic import: {dead}",
        "A listed file is exempt from the unclassified check. One that outlived "
        "its import silently exempts the next import added to it.",
        "delete the entry from DYNAMIC_IMPORT_FILES in ci_selector/handwritten.py",
    )


@pytest.mark.drift
def test_every_recording_parser_contributes_a_table_file(full):
    """The derived half's detection floor. table_files is what now vouches for
    a dynamic import, so a parser that quietly stopped recording would bless
    nothing and read exactly like a clean run."""
    files = full.graph.table_files
    for owner, marker in (
        ("register calls", "vllm/distributed/kv_transfer/kv_connector/factory.py"),
        ("lazy parser tables", "vllm/tool_parsers/__init__.py"),
        ("qualname enums", "vllm/v1/attention/backends/registry.py"),
        ("vllm/__init__ MODULE_ATTRS", "vllm/__init__.py"),
        ("lazy export tables", "vllm/transformers_utils/configs/__init__.py"),
        ("pkgutil enumerators", "vllm/kernels/helion/ops/__init__.py"),
        ("model registry", "vllm/model_executor/models/registry.py"),
        ("quant methods", "vllm/model_executor/layers/quantization/__init__.py"),
        ("platform methods", "vllm/platforms/interface.py"),
    ):
        assert marker in files, drift_message(
            f"The {owner} parser recorded no table this run "
            f"(expected it to read {marker}).",
            "table_files is what now vouches for a dynamic import. A parser that "
            "stopped matching blesses nothing and reads exactly like a clean run, "
            "so its file's import goes unclassified with no explanation.",
            "the table moved or was renamed in vLLM: update the parser in "
            "ci_selector/codemap/graph/ to find it again",
            "the anchor path or table name changed: update it in " + HW,
        )


def test_module_level_constant_reads_like_a_literal(full):
    """`import_module(SOME_CONST)` where SOME_CONST is a module-level string is
    exactly as knowable as the literal form, so it is not a hole. Specimen:
    the modelexpress loader, which used to need a hand-list entry."""
    loader = "vllm/model_executor/model_loader/modelexpress_loader.py"
    assert not [s for s in full.graph.dynamic_sites if s.file == loader]


def test_unresolvable_path_inside_our_own_packages_is_still_a_site(vllm_repo, tmp_path):
    """A constant naming something outside vllm/tests/benchmarks proves the
    import leaves the repo. One naming a path INSIDE them that resolves to
    nothing is a broken or built-elsewhere target, and used to vanish here."""
    import ast

    from ci_selector.codemap.graph.imports import ImportGraph, _resolve_call
    from ci_selector.codemap.repo import build_module_index

    index = build_module_index(vllm_repo)
    for source, expect_site in (
        ('importlib.import_module("scipy.linalg")', False),
        ('importlib.import_module("vllm.no.such.module")', True),
    ):
        graph = ImportGraph()
        call = ast.parse(source).body[0].value
        _resolve_call(call, index, graph, "vllm/x.py", {})
        assert bool(graph.dynamic_sites) is expect_site, source


def test_lazy_loader_targets_get_an_edge(full):
    """LazyLoader is a fourth way to import dynamically, and its module is a
    plain literal in the third argument. It went unread until 2026-08-19, so
    these three edges did not exist and the targets lost an importer."""
    for src, dst in (
        ("vllm/utils/mistral.py", "vllm/tokenizers/mistral.py"),
        (
            "vllm/config/speculative.py",
            "vllm/model_executor/layers/quantization/__init__.py",
        ),
        ("vllm/config/model.py", "vllm/model_executor/layers/quantization/__init__.py"),
    ):
        assert dst in full.graph.imports.get(src, set()), (src, dst)


def test_lazy_loader_external_targets_stay_silent(full):
    """Most LazyLoader targets are heavy third-party packages (torch,
    llguidance). Those resolve to nothing by proof, so they must not turn into
    unclassified sites and fail the check."""
    sites = [s for s in full.graph.dynamic_sites if s.func == "LazyLoader"]
    assert sites == [], [(s.file, s.lineno) for s in sites]
