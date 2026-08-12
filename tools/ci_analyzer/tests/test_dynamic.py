# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynamic-site classification and ambiguity reporting."""

from ci_analyzer.validate.dynamic_sites import classify_dynamic_sites


def test_no_unclassified_dynamic_sites_at_head(repo, full):
    """Every non-literal dynamic import at HEAD is external-by-signature or
    handled by a wall parser."""
    fg = full
    classified = classify_dynamic_sites(repo, fg.graph.dynamic_sites)
    assert classified.unclassified == [], [
        f"{s.file}:{s.lineno}" for s in classified.unclassified
    ]


def test_bare_import_ambiguities_reported_not_silent(full):
    fg = full
    # tests/v1/determinism `from utils import`: sibling wins, and the clash
    # with the indexed top-level tests/utils.py is surfaced.
    clashes = {(file, name) for file, name, _sib, _other in fg.graph.ambiguities}
    assert any(
        file.startswith("tests/v1/determinism/") and name == "utils"
        for file, name in clashes
    ), clashes


def test_new_site_under_former_blanket_is_unclassified(repo):
    """A NEW dynamic-import site anywhere (even under a formerly-blanketed prefix
    like tests/models/) must land UNCLASSIFIED, not silently blessed."""
    from ci_analyzer.graph.imports import DynamicSite

    site = DynamicSite(
        "tests/models/language/generation/dispatch.py", 1, "import_module"
    )
    classified = classify_dynamic_sites(repo, [site])
    assert classified.unclassified == [site]


def test_external_by_signature_entries_exist(repo):
    """Stale-entry guard: every censused path still exists."""
    from ci_analyzer.curated import AUDITED_DYNAMIC_FILES

    missing = [p for p in AUDITED_DYNAMIC_FILES if not (repo / p).is_file()]
    assert not missing, missing


def test_census_reverse_gate_flags_dead_dynamic_entries(repo):
    """Reverse gate isn't vacuous: with no live sites every dynamic-by-nature
    entry is flagged dead (a dead entry pre-blesses a site the forward gate misses)."""
    from ci_analyzer.curated import DYNAMIC_IMPORT_FILES
    from ci_analyzer.validate.dynamic_sites import census_rot

    dead, _missing = census_rot(repo, [])
    assert set(dead) == set(DYNAMIC_IMPORT_FILES)


def test_census_reverse_gate_flags_missing_wall_entry(repo, monkeypatch):
    import ci_analyzer.validate.dynamic_sites as ds

    monkeypatch.setattr(ds, "WALL_PARSER_FILES", ("vllm/__gone__.py",))
    _dead, missing = ds.census_rot(repo, [])
    assert missing == ["vllm/__gone__.py"]


def test_census_clean_at_head(repo, full):
    """Every dynamic-section entry has a live site and every wall entry exists:
    no dead blessing sits in the census."""
    from ci_analyzer.validate.dynamic_sites import census_rot

    fg = full
    dead, missing = census_rot(repo, fg.graph.dynamic_sites)
    assert dead == [], dead
    assert missing == [], missing
