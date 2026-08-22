# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The two halves, joined, against the real checkout.

Every other test here feeds the rule a hand-built `Selection`, which is fast
and cannot catch the one thing that matters: whether the shape the code map
really produces is the shape the rule really reads. Hand-written fixtures agree
with each other and prove nothing about the seam.

`selected_paths` is the specific hazard, being POSITIONAL: one entry per
reason, ordered with `selected_rules`, `None` meaning the reason cannot be
argued with. Off by one and steps are weighed against the wrong file's
evidence, with every unit test still green.
"""

from __future__ import annotations

import pytest
from ci_selector.codemap.classify import select
from ci_selector.coverage.rules import RowKeys, newest_commit, read_pr
from ci_selector.coverage.source import fetch_table


@pytest.fixture(scope="module")
def live_table():
    table = fetch_table()
    if not table.available:
        pytest.skip(f"no coverage table available: {table.unavailable}")
    return table


def _narrow_range(repo, state, limit=25):
    """A recent commit whose diff the code map answers NARROWLY.

    Searched rather than hardcoded, for two reasons. A pinned commit pair rots
    as history moves. And most ranges are useless here: anything that trips
    run-all makes every reason non-droppable and every step already selected,
    so the rule has nothing it is permitted to do and the test would pass while
    proving nothing.
    """
    import subprocess

    from ci_selector.gitdiff import changed_paths, diff_files

    revs = subprocess.run(
        ["git", "-C", str(repo), "rev-list", f"-{limit}", "HEAD", "--", "vllm/"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    for head in revs:
        base = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", f"{head}~1"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if not base:
            continue
        paths = changed_paths(diff_files(repo, base, head))
        if not paths or len(paths) > 6:
            continue
        selection = select(state, paths, base=base, head=head)
        if selection.run_all or not selection.selected:
            continue
        return base, head, selection
    return None, None, None


def test_a_real_selection_survives_the_rule(state, vllm_repo, live_table):
    """A real diff, selected by the real code map, read by the real rule.

    The range is two-ended and deliberately so. This test used to pass the same
    ref for both ends, which built an EMPTY query -- every assertion below
    still passed while the rule was handed nothing to reason about.
    """
    from ci_selector.coverage.changed_funcs import build

    base, head, selection = _narrow_range(vllm_repo, state)
    if base is None:
        pytest.skip("no narrow non-run-all range in recent history")

    query = build(vllm_repo, base, head)
    # Detection floor: the whole point of a two-ended range is that the rule
    # gets something to read. An empty query makes everything below vacuous.
    assert query.files, "the query is empty; this is the bug the test exists for"

    keys = RowKeys.resolve(live_table, vllm_repo, newest_commit(live_table, vllm_repo))
    reading = read_pr(live_table, selection, query, {}, frozenset(), keys)

    # Every selected step is accounted for exactly once, and nothing was
    # invented: the rule may only keep, drop, or add.
    handled = set(reading.kept) | set(reading.dropped)
    assert handled == set(selection.selected), (
        "the rule lost or invented steps relative to the map's selection"
    )
    assert not (set(reading.added) & set(selection.selected)), (
        "a step the map already selected was also reported as an addition"
    )
    # And it actually did something, rather than abstaining on everything.
    assert reading.added or reading.dropped, (
        "the rule neither added nor dropped on a real narrow diff; it is inert"
    )


def test_selected_paths_lines_up_with_selected_rules(state):
    """The positional contract the rule depends on, checked on real output.

    `read_pr` walks reasons and paths together. If the map ever emits lists of
    different lengths, a step gets weighed against another reason's files and
    the drop is unjustifiable -- but nothing raises.
    """
    selection = select(state, ["vllm/v1/attention/selector.py", "tests/conftest.py"])
    assert selection.selected_paths, "no attribution at all; the contract is moot"

    for step_id, paths in selection.selected_paths.items():
        rules = selection.selected_rules[step_id]
        assert len(paths) == len(rules), (
            f"{step_id}: {len(paths)} path entries against {len(rules)} rules"
        )
        for entry in paths:
            assert entry is None or isinstance(entry, list), (
                f"{step_id}: a path entry is neither None nor a list of paths"
            )
