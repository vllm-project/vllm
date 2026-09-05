# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Co-location routing: the mirror walk, the union, and the guard under it.

The layout convention this rests on is neither enforced nor documented (see the
module docstring in `codemap/colocation.py`), so these pin the two behaviours
that make the failures survivable: it climbs rather than gives up, and it
declines rather than guesses.
"""

import pytest
from ci_selector.codemap import colocation
from ci_selector.codemap.repo import build_test_index


class _FakeState:
    """Enough RepoState to drive the two public functions."""

    def __init__(self, tests, importers=None):
        self._index = build_test_index(frozenset(tests))
        self._importers = {k: frozenset(v) for k, v in (importers or {}).items()}

    def test_index(self):
        return self._index

    def direct_test_importers(self):
        return self._importers


def test_the_deepest_existing_directory_wins():
    """Climbing stops at the first level that holds tests, so a depth-3 miss
    becomes a depth-2 answer rather than a depth-1 one."""
    state = _FakeState(["tests/a/test_shallow.py", "tests/a/b/test_deep.py"])
    tests, directory = colocation.colocated_tests(state, "vllm/a/b/c/thing.py")
    assert directory == "tests/a/b/"
    assert tests == frozenset({"tests/a/b/test_deep.py"})


def test_a_matching_test_name_routes_without_any_directory():
    state = _FakeState(["tests/wherever/test_thing.py"])
    tests, directory = colocation.colocated_tests(state, "vllm/a/b/thing.py")
    assert directory is None
    assert tests == frozenset({"tests/wherever/test_thing.py"})


def test_it_never_widens_to_bare_tests():
    """`tests/` is a run-all wearing a directory's clothes, and the walk stops
    one level above it."""
    state = _FakeState(["tests/unrelated/test_x.py"])
    assert colocation.colocated_tests(state, "vllm/nomirror/thing.py") == (
        frozenset(),
        None,
    )


def test_a_top_level_source_file_has_no_directory_to_mirror():
    state = _FakeState(["tests/a/test_x.py"])
    assert colocation.colocated_tests(state, "vllm/thing.py")[0] == frozenset()


def test_non_python_and_non_source_paths_are_not_routed():
    state = _FakeState(["tests/a/test_x.py"])
    assert colocation.colocated_tests(state, "vllm/a/data.json")[0] == frozenset()
    assert colocation.colocated_tests(state, "csrc/a/kernel.cu")[0] == frozenset()


def test_the_union_widens_a_firing_mirror():
    state = _FakeState(
        ["tests/a/test_beside.py", "tests/elsewhere/test_importer.py"],
        {"vllm/a/thing.py": {"tests/elsewhere/test_importer.py"}},
    )
    tests, directory = colocation.implicated_tests(state, "vllm/a/thing.py")
    assert directory == "tests/a/"
    assert tests == frozenset(
        {"tests/a/test_beside.py", "tests/elsewhere/test_importer.py"}
    )


def test_an_empty_mirror_stays_empty_even_when_importers_exist():
    """The guard, and the only reason the union is safe. Firing on direct
    importers alone would select far less than the graph fallback, so a file
    with no co-located tests must decline and let the graph rule answer."""
    state = _FakeState(
        ["tests/elsewhere/test_importer.py"],
        {"vllm/nomirror/thing.py": {"tests/elsewhere/test_importer.py"}},
    )
    assert colocation.implicated_tests(state, "vllm/nomirror/thing.py") == (
        frozenset(),
        None,
    )


def test_the_union_only_ever_adds():
    state = _FakeState(
        ["tests/a/test_beside.py"], {"vllm/a/thing.py": {"tests/x/test_other.py"}}
    )
    colocated, _ = colocation.colocated_tests(state, "vllm/a/thing.py")
    united, _ = colocation.implicated_tests(state, "vllm/a/thing.py")
    assert colocated <= united


def test_the_env_switch_reads_three_modes_and_rejects_anything_else(monkeypatch):
    """A typo must kill the run. Read as "off" it would send every cycle file
    back to graph reach, which looks exactly like the rule doing nothing.
    `cycle-only` is the pre-extension behavior, kept as the measurement arm."""
    monkeypatch.delenv(colocation.ENV_VAR, raising=False)
    assert colocation.mode() == "on"
    monkeypatch.setenv(colocation.ENV_VAR, "off")
    assert colocation.mode() == "off"
    monkeypatch.setenv(colocation.ENV_VAR, "cycle-only")
    assert colocation.mode() == "cycle-only"
    monkeypatch.setenv(colocation.ENV_VAR, "1")
    with pytest.raises(ValueError, match="expected one of"):
        colocation.mode()


def test_the_underscore_suffix_is_tried_when_the_plain_name_holds_no_tests():
    """`tests/utils_` and `tests/tokenizers_` are collision avoidance, not
    typos, so the rule derives them rather than listing them."""
    state = _FakeState(["tests/utils_/test_x.py", "tests/plugins_tests/test_y.py"])
    assert (
        colocation.colocated_tests(state, "vllm/utils/thing.py")[1] == "tests/utils_/"
    )
    assert (
        colocation.colocated_tests(state, "vllm/plugins/thing.py")[1]
        == "tests/plugins_tests/"
    )


def test_the_literal_name_wins_over_an_alias():
    """A suffixed directory is a fallback for a name that is taken, never a
    replacement for one that works."""
    state = _FakeState(["tests/lora/test_a.py", "tests/lora_/test_b.py"])
    tests, directory = colocation.colocated_tests(state, "vllm/lora/thing.py")
    assert directory == "tests/lora/"
    assert tests == frozenset({"tests/lora/test_a.py"})


def test_the_one_hardcoded_alias():
    """`compilation` -> `compile` is a truncation no suffix rule produces, and
    it is the largest single group of files that would otherwise fall back."""
    state = _FakeState(["tests/compile/test_a.py"])
    assert (
        colocation.colocated_tests(state, "vllm/compilation/backends.py")[1]
        == "tests/compile/"
    )


def test_an_alias_only_applies_to_the_first_segment():
    """The collisions all happen at the top level, where a directory shares a
    namespace with tests/*.py and with installed packages. Aliasing deeper would
    be guessing, so `tests/a/b_/` is not offered as a spelling of `vllm/a/b`;
    the walk climbs past it to `tests/a/` like any other miss.
    """
    state = _FakeState(["tests/a/b_/test_x.py"])
    _, directory = colocation.colocated_tests(state, "vllm/a/b/thing.py")
    assert directory == "tests/a/"


def test_the_real_tree_climbs_to_the_nearest_mirror(state):
    """End to end on the checkout, and on the drift the walk exists for:
    `tests/lora/` is flat, so a file under `vllm/lora/layers/` climbs one level
    rather than finding nothing."""
    path = "vllm/lora/layers/base.py"
    assert path in state.full.import_cycle().reach_blind
    assert "tests/lora/layers/" not in state.test_index().dirs
    tests, directory = colocation.implicated_tests(state, path)
    assert directory == "tests/lora/"
    assert tests
    assert all(f.startswith("tests/") for f in tests)


def test_a_file_with_no_mirror_declines_on_the_real_tree(state):
    """`vllm/device_allocator` is in the cycle and has no
    `tests/device_allocator` counterpart at HEAD, so it declines and lets graph
    reach answer rather than routing somewhere merely plausible."""
    path = "vllm/device_allocator/cumem.py"
    assert path in state.full.import_cycle().reach_blind
    assert "tests/device_allocator/" not in state.test_index().dirs
    tests, _ = colocation.implicated_tests(state, path)
    assert tests == frozenset()
