# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The three rules, and the two ways coverage is allowed to fail.

    S has a row, and the row shows it ran F     -> select, map gets no vote
    S has a row, and the row shows it ran none  -> drop, if every gate agrees
    S has no row                                -> the map decides

Freshness is stubbed here, since computing it needs a real state at the
table's commit and has its own tests. What these pin is what `decide` does
with the answer.
"""

from __future__ import annotations

import pytest
from ci_selector.codemap.selection import Selection
from ci_selector.coverage import freshness
from ci_selector.coverage.freshness import Freshness, Surface
from ci_selector.coverage.rules import RowKeys
from ci_selector.decide import decide

from .helpers import MODULE_SOURCE, make_table

# Two steps with rows, one without. `runs-mod` executed vllm/mod.py; `elsewhere`
# executed a different file, so on a vllm/mod.py diff its row is real evidence
# of absence. `no-row` is the third arm of the rule.
JOBS = {
    "runs-mod": [("mod.py", "plain")],
    "elsewhere": [("other.py", "elsewhere")],
}
FRESH_ALL = Freshness(commit="R", surfaces={})


@pytest.fixture
def table(tmp_path, tmp_repo):
    return make_table(tmp_path, tmp_repo, JOBS)


@pytest.fixture
def diff(tmp_repo):
    """A real two-ended range whose diff touches `plain` in vllm/mod.py.

    Real refs, not placeholders: `decide` builds the query with git before it
    reaches any of the logic under test, so a fake ref fails in the wrong place
    and every assertion below would be testing the error path.
    """
    base = tmp_repo.head()
    tmp_repo.write("vllm/mod.py", MODULE_SOURCE.replace("return CONSTANT", "return 99"))
    head = tmp_repo.commit("edit plain")
    return base, head


class FakeStep:
    manual_only = False


ALL_STEPS = [f"vllm_ci:{k}" for k in (*JOBS, "no-row")]


def _keys():
    """`RowKeys` without resolving a real pipeline.

    The live `RowKeys.resolve` walks a `RepoState` at the table's commit to work
    out which pipeline owns the rows. That election has its own tests; here it
    would only mean building a worktree to learn something already known.
    """
    return RowKeys(
        {"vllm_ci"}, {"vllm_ci": 1.0}, steps={s: FakeStep() for s in ALL_STEPS}
    )


def _stub_keys(monkeypatch):
    class Stub:
        resolve = staticmethod(lambda *a, **k: _keys())

    monkeypatch.setattr("ci_selector.decide.RowKeys", Stub)


@pytest.fixture
def unstaled(monkeypatch):
    """Nothing has moved since the table was recorded, so no step is held."""
    monkeypatch.setattr(
        freshness,
        "build",
        lambda state, commit: Freshness(
            commit=commit,
            surfaces={
                s: Surface(tests=frozenset({f"tests/{s}.py"})) for s in ALL_STEPS
            },
        ),
    )
    monkeypatch.setattr(freshness, "changed_between", lambda *a: frozenset())
    monkeypatch.setattr("ci_selector.codemap.worktree.state_for", lambda *a: object())
    _stub_keys(monkeypatch)


def _selection(*step_ids, path="vllm/mod.py"):
    return Selection(
        selected={s: [f"{path}: map said so"] for s in step_ids},
        selected_rules={s: ["graph"] for s in step_ids},
        selected_paths={s: [[path]] for s in step_ids},
    )


def test_a_row_that_ran_the_file_selects_it(table, tmp_repo, diff, unstaled):
    """Rule 1. `runs-mod` is not in the map's selection at all; the row alone
    puts it in."""
    d = decide(None, _selection("vllm_ci:no-row"), tmp_repo.root, *diff, table=table)
    assert "vllm_ci:runs-mod" in d.added_by_coverage
    assert "vllm_ci:runs-mod" in d.steps


def test_a_row_that_ran_none_of_it_drops_it(table, tmp_repo, diff, unstaled):
    """Rule 2. `elsewhere` recorded a different file, so on this diff its
    silence is evidence rather than ignorance."""
    d = decide(None, _selection("vllm_ci:elsewhere"), tmp_repo.root, *diff, table=table)
    assert "vllm_ci:elsewhere" in d.dropped_by_coverage
    assert "vllm_ci:elsewhere" not in d.steps


def test_a_step_with_no_row_is_left_to_the_map(table, tmp_repo, diff, unstaled):
    """Rule 3. The record abstains; the map's answer stands."""
    d = decide(None, _selection("vllm_ci:no-row"), tmp_repo.root, *diff, table=table)
    assert "vllm_ci:no-row" in d.steps
    assert "vllm_ci:no-row" not in d.dropped_by_coverage


def test_the_freshness_gate_blocks_a_drop(table, tmp_repo, diff, monkeypatch):
    """The gate, and the reason it exists: a row whose step has moved since it
    was recorded describes a step that no longer exists, so it may not drop."""
    moved = frozenset({"tests/elsewhere.py"})
    monkeypatch.setattr(
        freshness,
        "build",
        lambda state, commit: Freshness(
            commit=commit,
            surfaces={"vllm_ci:elsewhere": Surface(tests=moved)},
        ),
    )
    monkeypatch.setattr(freshness, "changed_between", lambda *a: moved)
    monkeypatch.setattr("ci_selector.codemap.worktree.state_for", lambda *a: object())
    _stub_keys(monkeypatch)

    d = decide(None, _selection("vllm_ci:elsewhere"), tmp_repo.root, *diff, table=table)
    assert d.dropped_by_coverage == set(), "a stale row authorised a drop"
    assert "vllm_ci:elsewhere" in d.steps
    assert d.stale_steps == 1


def test_no_table_returns_the_map_untouched(tmp_repo, diff):
    from ci_selector.coverage.table import Table

    sel = _selection("vllm_ci:a", "vllm_ci:b")
    d = decide(
        None, sel, tmp_repo.root, *diff, table=Table(None, unavailable="no table here")
    )
    assert d.steps == set(sel.selected)
    assert d.coverage_note == "no table here"
    assert not d.used_coverage


def test_a_failure_anywhere_keeps_the_map_rather_than_dropping(
    table, tmp_repo, diff, monkeypatch
):
    """The inverted-failure trap, pinned.

    An empty `stale` set does not mean "gate off", it means "nothing is
    disqualified" -- so a failure that fell through to one would make the
    record MORE willing to drop. The only safe degradation is to skip the
    subtractive half entirely, which is what this asserts.
    """

    def boom(*a, **k):
        raise RuntimeError("commit not in this checkout")

    monkeypatch.setattr(freshness, "build", boom)
    monkeypatch.setattr("ci_selector.codemap.worktree.state_for", lambda *a: object())
    _stub_keys(monkeypatch)

    sel = _selection("vllm_ci:elsewhere")
    d = decide(None, sel, tmp_repo.root, *diff, table=table)
    assert d.steps == set(sel.selected), "a coverage failure changed the answer"
    assert d.dropped_by_coverage == set()
    assert d.added_by_coverage == set()
    assert "commit not in this checkout" in d.coverage_note
