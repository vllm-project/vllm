# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Table-aware diffing: guard fallbacks (inline fixtures), entry-diff
semantics, and real-history cases from PR #45990 in both directions.

Each real-history case asserts the table-diff RULE fired, never job counts
alone: a silent fallback to file-level must not masquerade as a pass.
"""

import subprocess

import pytest
from ci_selector.codemap.registry_diff import (
    REGISTRY_FILE,
    TEST_REGISTRY_FILE,
    diff_table,
    parse_tests_registry_strict,
    parse_vllm_registry_strict,
)
from helpers import HW, drift_message

BASE_VLLM = """
_TEXT_GENERATION_MODELS = {
    "AForCausalLM": ("a", "AForCausalLM"),
    "BForCausalLM": ("b", "BForCausalLM"),
}
_MULTIMODAL_MODELS = {
    "CForConditionalGeneration": ("vllm.models.c", "CModel"),
}
_VLLM_MODELS = {
    **_TEXT_GENERATION_MODELS,
    **_MULTIMODAL_MODELS,
}
_PREVIOUSLY_SUPPORTED_MODELS = {"OldForCausalLM": "0.1"}

def resolve(name):
    return name
"""


def _changes(base, head, path=REGISTRY_FILE):
    d = diff_table(path, base, head)
    if d is None:
        return None
    return {(c.kind, c.key, c.change) for c in d.changes}


def test_entry_add_remove_modify():
    head = BASE_VLLM.replace(
        '"BForCausalLM": ("b", "BForCausalLM"),',
        '"B2ForCausalLM": ("b2", "B2ForCausalLM"),',
    ).replace('("a", "AForCausalLM")', '("a2", "AForCausalLM")')
    assert _changes(BASE_VLLM, head) == {
        ("models", "BForCausalLM", "removed"),
        ("models", "B2ForCausalLM", "added"),
        ("models", "AForCausalLM", "modified"),
    }


def test_previously_supported_is_a_diffable_kind():
    """Model-removal PRs edit _PREVIOUSLY_SUPPORTED_MODELS in the same
    commit; without this kind, every removal would fall back."""
    head = BASE_VLLM.replace('"BForCausalLM": ("b", "BForCausalLM"),', "").replace(
        '{"OldForCausalLM": "0.1"}',
        '{"OldForCausalLM": "0.1", "BForCausalLM": "0.2"}',
    )
    changes = _changes(BASE_VLLM, head)
    assert ("models", "BForCausalLM", "removed") in changes
    assert ("previously_supported_models", "BForCausalLM", "added") in changes


def test_non_table_edit_falls_back():
    head = BASE_VLLM.replace("return name", "return name.lower()")
    assert _changes(BASE_VLLM, head) is None


def test_value_becoming_call_falls_back():
    head = BASE_VLLM.replace('("b", "BForCausalLM")', "pick()")
    assert _changes(BASE_VLLM, head) is None


def test_missing_root_falls_back():
    head = BASE_VLLM.replace("_VLLM_MODELS", "_ALL_MODELS")
    assert _changes(BASE_VLLM, head) is None


def test_dictcomp_subdict_stays_guarded():
    """Sub-dict with a comprehension spread (live _EMBEDDING_MODELS shape) stays
    in remainder: edits to it fall back, edits to other consumed dicts still diff."""
    comp = BASE_VLLM.replace(
        '_MULTIMODAL_MODELS = {\n    "CForConditionalGeneration": ("vllm.models.c", "CModel"),\n}',  # noqa: E501
        '_MULTIMODAL_MODELS = {\n    **{k: v for k, v in _TEXT_GENERATION_MODELS.items()},\n    "CForConditionalGeneration": ("vllm.models.c", "CModel"),\n}',  # noqa: E501
    )
    head_bad = comp.replace(
        '("vllm.models.c", "CModel")', '("vllm.models.c2", "CModel")'
    )
    assert _changes(comp, head_bad) is None
    head_ok = comp.replace('("a", "AForCausalLM")', '("a2", "AForCausalLM")')
    assert _changes(comp, head_ok) == {("models", "AForCausalLM", "modified")}


def test_subdict_reshuffle_yields_empty_diff_with_texts_differ():
    head = BASE_VLLM.replace(
        '_TEXT_GENERATION_MODELS = {\n    "AForCausalLM": ("a", "AForCausalLM"),\n    "BForCausalLM": ("b", "BForCausalLM"),\n}',  # noqa: E501
        '_TEXT_GENERATION_MODELS = {\n    "BForCausalLM": ("b", "BForCausalLM"),\n    "AForCausalLM": ("a", "AForCausalLM"),\n}',  # noqa: E501
    )
    d = diff_table(REGISTRY_FILE, BASE_VLLM, head)
    assert d is not None and d.texts_differ and not d.changes


def test_subdict_consumed_under_failing_parent_forces_fallback():
    """A sub-dict spread into a dict that LATER fails (a DictComp sibling) must
    stay in the remainder: an edit to it forces a file-level fallback, not a
    silent reshuffle (texts_differ, no changes) that selects no arch tests."""
    base = (
        '_SUB = {\n    "AForCausalLM": ("a", "AForCausalLM"),\n}\n'
        "_BROKEN = {\n    **_SUB,\n    **{k: v for k, v in _SUB.items()},\n}\n"
        '_MULTIMODAL_MODELS = {\n    "CForConditionalGeneration": ("vllm.models.c", "CModel"),\n}\n'  # noqa: E501
        "_VLLM_MODELS = {\n    **_BROKEN,\n    **_MULTIMODAL_MODELS,\n}\n"
        '_PREVIOUSLY_SUPPORTED_MODELS = {"OldForCausalLM": "0.1"}\n'
        "def resolve(name):\n    return name\n"
    )
    head = base.replace('("a", "AForCausalLM")', '("a2", "AForCausalLM")')
    assert diff_table(REGISTRY_FILE, base, head) is None


BASE_TESTS = """
_TEXT_EXAMPLES = {
    "AForCausalLM": _HfExamplesInfo("org/model-a", min_transformers_version="4.0"),
}
_MM_EXAMPLES = {
    "CForConditionalGeneration": _HfExamplesInfo("org/model-c"),
}
"""


def test_hf_kwarg_only_change_is_modified():
    head = BASE_TESTS.replace(
        'min_transformers_version="4.0"', 'min_transformers_version="5.0"'
    )
    d = diff_table(TEST_REGISTRY_FILE, BASE_TESTS, head)
    assert {(c.kind, c.key, c.change) for c in d.changes} == {
        ("hf_examples", "AForCausalLM", "modified")
    }
    assert "org/model-a" in d.base.ids["AForCausalLM"]


SHADOWED_TESTS = """
_EMBEDDING_EXAMPLES = {
    "AForCausalLM": _HfExamplesInfo("org/embed-a"),
}
_MM_EXAMPLES = {
    "AForCausalLM": _HfExamplesInfo("org/mm-a"),
}
"""


def test_shadowed_entry_still_diffs():
    """An arch in two dicts must not collapse to the last one. Overwriting made
    an edit to the earlier copy diff as texts_differ with no changes, which
    routes to registry importers only and drops every literal-only test."""
    head = SHADOWED_TESTS.replace(
        '_HfExamplesInfo("org/embed-a")',
        '_HfExamplesInfo("org/embed-a", max_transformers_version="4.48")',
    )
    d = diff_table(TEST_REGISTRY_FILE, SHADOWED_TESTS, head)
    assert {(c.kind, c.key, c.change) for c in d.changes} == {
        ("hf_examples", "AForCausalLM", "modified")
    }


def test_shadowed_arch_specimen_still_exists(vllm_repo):
    """Pins the specimen the case above generalises: if the real registry stops
    listing an arch twice, this test is measuring nothing."""
    src = (vllm_repo / TEST_REGISTRY_FILE).read_text()
    embedding = src.index("_EMBEDDING_EXAMPLE_MODELS")
    multimodal = src.index("_MULTIMODAL_EXAMPLE_MODELS")
    assert embedding < multimodal, "dict order changed; which copy shadows flipped"
    first, second = src[embedding:multimodal], src[multimodal:]
    shared = [
        arch
        for arch in ("LlavaNextForConditionalGeneration", "Phi3VForCausalLM")
        if f'"{arch}"' in first and f'"{arch}"' in second
    ]
    assert shared, "no arch is listed in both dicts any more: update the specimen"


# ---- real-history cases: PR #45990 (Bamba removal), both directions ----

MERGE_45990 = "d682968aa9fcd7e7a78218b548c52fc198a87a6c"


@pytest.fixture(scope="module")
def bamba_shas(vllm_repo):
    probe = subprocess.run(
        ["git", "-C", str(vllm_repo), "cat-file", "-e", MERGE_45990],
        capture_output=True,
    )
    if probe.returncode != 0:
        pytest.skip("PR #45990 merge commit not present locally (shallow clone)")
    base = subprocess.run(
        ["git", "-C", str(vllm_repo), "rev-parse", f"{MERGE_45990}^"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return base, MERGE_45990


def _run_direction(vllm_repo, state_at, base, head):
    from ci_selector.codemap.classify import select
    from ci_selector.codemap.worktree import state_for
    from ci_selector.gitdiff import changed_paths, diff_files

    state = state_for(vllm_repo, state_at)
    paths = changed_paths(diff_files(vllm_repo, base, head))
    return select(state, paths, base=base, head=head)


def test_bamba_removal_direction(vllm_repo, bamba_shas):
    base, merge = bamba_shas
    sel = _run_direction(vllm_repo, base, base, merge)
    rules = {c.rule for c in sel.claims}
    assert "table-diff" in rules, rules
    assert "vllm_ci" not in sel.run_all
    n = len({s for s in sel.selected if s.startswith("vllm_ci:")})
    assert n < 120, f"expected scoped selection, got {n} (file-level was 170)"


def test_table_claim_carries_specific_declarers(
    vllm_repo, bamba_shas, declared_deps_on
):
    """The table-diff claim is built outside _classify, so it used to skip the
    declarer union: a step naming the registry file specifically, but
    running tests elsewhere, was dropped."""
    from ci_selector.codemap.classify import _source_dep_steps
    from ci_selector.codemap.registry_diff import TABLE_FILES
    from ci_selector.codemap.worktree import state_for
    from ci_selector.gitdiff import changed_paths, diff_files

    base, merge = bamba_shas
    state = state_for(vllm_repo, base)
    tables = [
        p for p in changed_paths(diff_files(vllm_repo, base, merge)) if p in TABLE_FILES
    ]
    assert tables, "specimen drifted: the diff no longer touches a table file"
    sel = _run_direction(vllm_repo, base, base, merge)
    reached = set(sel.selected) | set(sel.manual_hits)
    for table in tables:
        declarers = _source_dep_steps(state, table, specific_only=True)
        assert declarers, f"{table} has no specific declarers; nothing is proved"
        assert not declarers - reached, sorted(declarers - reached)[:5]


def test_bamba_add_direction(vllm_repo, bamba_shas):
    """Reversed diff = a model-ADD PR: the added bamba.py must be covered by
    the head-side table entry instead of failing open to run-all."""
    base, merge = bamba_shas
    sel = _run_direction(vllm_repo, merge, merge, base)
    rules = {c.rule for c in sel.claims}
    assert "table-diff" in rules, rules
    assert "vllm_ci" not in sel.run_all, sel.run_all
    covered = [
        c
        for c in sel.claims
        if c.rule == "table-diff" and "newly registered module" in c.detail
    ]
    assert covered, "added bamba.py was not routed via the table claim"


def test_added_conftest_routes_by_target_coverage_not_added_test(vllm_repo, bamba_shas):
    """An added conftest.py routes by subtree target-coverage (steps running tests
    beneath it), not the added-test rule; a real added test file gets added-test."""
    from ci_selector.codemap.classify import _classify
    from ci_selector.codemap.state import DiffContext
    from ci_selector.codemap.worktree import state_for

    base, merge = bamba_shas
    state = state_for(vllm_repo, base)
    ctx = DiffContext(
        base=base,
        head=merge,
        status={"tests/models/language/conftest.py": "A"},
    )
    claim = _classify(state, "tests/models/language/conftest.py", ctx)
    assert claim.rule == "target-coverage", claim.rule
    assert not claim.run_all
    added_test = "tests/models/language/generation/test_totally_new.py"
    ctx.status[added_test] = "A"
    claim2 = _classify(state, added_test, ctx)
    assert claim2.rule == "added-test", claim2.rule
    assert claim2.step_ids


def test_helper_importers_of_test_registry_pinned(full):
    """Table-diff drops helper-mediated coverage; sound only while non-test
    importers of tests/models/registry.py stay the known literal-parameterized
    set. A new entry means: re-audit before trusting table-diff."""
    fg = full
    non_test = {
        f
        for f in fg.graph.reverse.get("tests/models/registry.py", ())
        if not f.rsplit("/", 1)[-1].startswith("test_")
    }
    # vlm_utils/core.py uses find_hf_info(model) with a caller-supplied literal
    # id; literal-parameterized, safe.
    allowed = {
        "tests/models/utils.py",
        "tests/conftest.py",
        "tests/models/multimodal/generation/vlm_utils/core.py",
    }
    assert non_test <= allowed, non_test


def test_an_abandoned_table_diff_is_reported_not_swallowed(state, monkeypatch):
    """A git-show timeout drops the registry file from entry-level scoping to
    whole-file routing. Both readings are safe, but they look identical in the
    output, so the degraded one has to say so."""
    from ci_selector.codemap import classify, registry_diff
    from ci_selector.codemap.state import DiffContext

    table = registry_diff.TABLE_FILES[0]
    monkeypatch.setattr(registry_diff, "git_show", lambda r, ref, p: None)
    monkeypatch.setattr(
        classify,
        "_diff_context",
        lambda st, b, h: DiffContext(base="b", head="h", status={table: "M"}),
    )
    sel = classify.select(state, [table], base="b", head="h")
    note = [n for n in sel.notes if table in n and "no entry-level diff" in n]
    assert note, sel.notes
    assert "git show failed or timed out" in note[0]
    assert sel.selected, "the fallback must still select, not select nothing"


def test_a_parsed_table_reports_no_such_note(state, monkeypatch):
    """The counterpart: when the diff works, nothing is reported. Without this
    the note above passes just as well if it fires unconditionally."""
    from ci_selector.codemap import classify, registry_diff
    from ci_selector.codemap.state import DiffContext

    table = registry_diff.TABLE_FILES[0]
    text = (state.repo / table).read_text()
    monkeypatch.setattr(registry_diff, "git_show", lambda r, ref, p: text)
    monkeypatch.setattr(
        classify,
        "_diff_context",
        lambda st, b, h: DiffContext(base="b", head="h", status={table: "M"}),
    )
    sel = classify.select(state, [table], base="b", head="h")
    assert not [n for n in sel.notes if "no entry-level diff" in n], sel.notes


@pytest.mark.drift
def test_both_registry_anchors_still_parse_at_head(vllm_repo):
    """Detection floor for the strict diff parsers.

    They answer `None` on anything they do not recognise, which abandons the
    table diff and falls back to the ordinary file rules. That direction is
    safe, so a renamed table reads as a slightly noisier run rather than a
    failure. Nothing else notices, so this does.

    The two parsers fill different fields: the vLLM registry populates
    `modules`, the test registry populates `ids`. Both populate `kinds`.
    """
    for path, parse in (
        (REGISTRY_FILE, parse_vllm_registry_strict),
        (TEST_REGISTRY_FILE, parse_tests_registry_strict),
    ):
        cost = (
            f"Entry-level diffing of {path} is how a one-model change routes to "
            "that model instead of to every model. Without it the whole file "
            "counts as changed and every model test runs."
        )
        table = parse((vllm_repo / path).read_text())
        assert table is not None, drift_message(
            f"The strict parser no longer recognises {path}.",
            cost,
            f"the table was renamed: update MODEL_REGISTRY_DICTS or "
            f"TEST_REGISTRY_CALL in {HW}",
            "the table changed shape: teach the parser in "
            "ci_selector/codemap/registry_diff.py",
        )
        assert table.kinds, drift_message(
            f"{path} parsed but yielded no table kinds.",
            cost,
            f"check the dict and call names in {HW} against the live file",
        )
        assert table.modules or table.ids, drift_message(
            f"{path} parsed but yielded no entries.",
            cost,
            "the entry shape changed: teach the parser in "
            "ci_selector/codemap/registry_diff.py",
        )
