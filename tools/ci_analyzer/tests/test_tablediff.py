# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Table-aware diffing: guard fallbacks (inline fixtures), entry-diff
semantics, and real-history cases from PR #45990 in both directions.

Each real-history case asserts the table-diff RULE fired, never job counts
alone: a silent fallback to file-level must not masquerade as a pass.
"""

import subprocess

import pytest
from ci_analyzer.tablediff import (
    REGISTRY_FILE,
    TEST_REGISTRY_FILE,
    diff_table,
)

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


def test_shadowed_arch_specimen_still_exists(repo):
    """Pins the specimen the case above generalises: if the real registry stops
    listing an arch twice, this test is measuring nothing."""
    src = (repo / TEST_REGISTRY_FILE).read_text()
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
def bamba_shas(repo):
    probe = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-e", MERGE_45990],
        capture_output=True,
    )
    if probe.returncode != 0:
        pytest.skip("PR #45990 merge commit not present locally (shallow clone)")
    base = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", f"{MERGE_45990}^"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return base, MERGE_45990


def _run_direction(repo, state_at, base, head):
    from ci_analyzer.gitdiff import changed_paths, diff_files
    from ci_analyzer.select import select
    from ci_analyzer.worktree import state_for

    state = state_for(repo, state_at)
    paths = changed_paths(diff_files(repo, base, head))
    return select(state, paths, base=base, head=head)


def test_bamba_removal_direction(repo, bamba_shas):
    base, merge = bamba_shas
    sel = _run_direction(repo, base, base, merge)
    rules = {c.rule for c in sel.claims}
    assert "table-diff" in rules, rules
    assert "vllm_ci" not in sel.run_all
    n = len({s for s in sel.selected if s.startswith("vllm_ci:")})
    assert n < 120, f"expected scoped selection, got {n} (file-level was 170)"


def test_table_claim_carries_specific_declarers(repo, bamba_shas):
    """The table-diff claim is built outside _classify, so it used to skip the
    declarer-union seam: a step naming the registry file specifically, but
    running tests elsewhere, was dropped."""
    from ci_analyzer.gitdiff import changed_paths, diff_files
    from ci_analyzer.select import _source_dep_steps
    from ci_analyzer.tablediff import TABLE_FILES
    from ci_analyzer.worktree import state_for

    base, merge = bamba_shas
    state = state_for(repo, base)
    tables = [
        p for p in changed_paths(diff_files(repo, base, merge)) if p in TABLE_FILES
    ]
    assert tables, "specimen drifted: the diff no longer touches a table file"
    sel = _run_direction(repo, base, base, merge)
    reached = set(sel.selected) | set(sel.manual_hits)
    for table in tables:
        declarers = _source_dep_steps(state, table, specific_only=True)
        assert declarers, f"{table} has no specific declarers; the seam proves nothing"
        assert not declarers - reached, sorted(declarers - reached)[:5]


def test_bamba_add_direction(repo, bamba_shas):
    """Reversed diff = a model-ADD PR: the added bamba.py must be covered by
    the head-side table entry instead of failing open to run-all."""
    base, merge = bamba_shas
    sel = _run_direction(repo, merge, merge, base)
    rules = {c.rule for c in sel.claims}
    assert "table-diff" in rules, rules
    assert "vllm_ci" not in sel.run_all, sel.run_all
    covered = [
        c
        for c in sel.claims
        if c.rule == "table-diff" and "newly registered module" in c.detail
    ]
    assert covered, "added bamba.py was not routed via the table claim"


def test_added_conftest_routes_by_target_coverage_not_added_test(repo, bamba_shas):
    """An added conftest.py routes by subtree target-coverage (steps running tests
    beneath it), not the added-test rule; a real added test file gets added-test."""
    from ci_analyzer.select import DiffContext, _classify
    from ci_analyzer.worktree import state_for

    base, merge = bamba_shas
    state = state_for(repo, base)
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
