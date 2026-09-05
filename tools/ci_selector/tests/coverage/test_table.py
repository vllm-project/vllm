# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Every way of failing to read a table has to end in "keep the step".

The one thing this must never do is answer an unanswerable question with
silence, because silence here means "no step runs your change", which means drop
everything. So each test names the state it is checking and asserts the verdict,
not just the absence of a crash.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
from ci_selector.coverage.changed_funcs import Attribution, FileQuery, Query
from ci_selector.coverage.model import Stamp, digest_of
from ci_selector.coverage.phase import PhaseMode
from ci_selector.coverage.table import Evidence, apply, load
from ci_selector.scripts.build import merge_build, write_table

from .helpers import Build, Repo, process_file


def restamp(table_path: Path, key: str, patch: dict) -> None:
    """Give a row a stamp property, legitimately.

    The signature covers the stamp, so editing a field on disk and leaving the
    digest alone is tampering, and the row is rejected outright -- which is its
    own test below. These tests want a row that genuinely HAS the property, so
    they re-sign.
    """
    payload = json.loads(table_path.read_text())
    blob = payload["rows"][key]
    blob["stamp"].update(patch)
    functions = {p: frozenset(n) for p, n in blob["functions"].items()}
    blob["stamp"]["digest"] = digest_of(functions, Stamp(**blob["stamp"]))
    table_path.write_text(json.dumps(payload))


def query_for(path: str, *names: str, **kwargs) -> Query:
    return Query(
        base="base",
        head="head",
        files=[
            FileQuery(
                path=path,
                status=kwargs.pop("status", Attribution.ATTRIBUTED),
                head_names=frozenset(names),
                **kwargs,
            )
        ],
    )


@pytest.fixture
def table_path(tmp_path: Path, tmp_repo: Repo) -> Path:
    build = Build(tmp_path / "sweep" / "b1", "1", tmp_repo.head())
    process_file(
        build.job("j0", step_key="runs-it") / "fn.a.txt", [("mod.py", "plain")]
    )
    process_file(
        build.job("j1", step_key="runs-other") / "fn.b.txt",
        [("mod.py", "Holder.method")],
    )
    process_file(build.job("j2", step_key="runs-nothing") / "fn.c.txt", [])
    out = tmp_path / "table.json"
    write_table(merge_build(build.finish(), tmp_repo.root), out)
    return out


class TestReading:
    def test_a_missing_file_authorizes_nothing(self, tmp_path: Path):
        table = load(tmp_path / "absent.json")
        assert not table.available
        assert (
            table.look_up("anything", query_for("vllm/mod.py", "plain")).evidence
            is Evidence.NO_TABLE
        )

    def test_unparsable_bytes_authorize_nothing(self, tmp_path: Path):
        broken = tmp_path / "t.json"
        broken.write_text("{not json")
        assert not load(broken).available

    def test_a_truncated_gzip_authorizes_nothing(self, tmp_path: Path):
        broken = tmp_path / "t.json.gz"
        broken.write_bytes(gzip.compress(b'{"rows": {}}')[:20])
        assert not load(broken).available

    def test_a_table_with_no_rows_is_broken_not_an_answer(self, tmp_path: Path):
        # The failure the whole module exists to prevent: an empty table reads as
        # "no step runs any changed code", which is a licence to drop everything.
        empty = tmp_path / "t.json"
        empty.write_text(json.dumps({"version": 1, "rows": {}}))
        table = load(empty)
        assert not table.available
        assert table.look_up("runs-it", query_for("vllm/mod.py", "plain")).keep

    def test_a_good_table_loads(self, table_path: Path):
        table = load(table_path)
        assert table.available and len(table) == 3 and not table.rejected


class TestStampVerification:
    def _tamper(self, table_path: Path, mutate) -> Path:
        payload = json.loads(table_path.read_text())
        mutate(payload["rows"]["runs-it"])
        table_path.write_text(json.dumps(payload))
        return table_path

    def test_content_that_disagrees_with_the_digest_is_rejected(self, table_path: Path):
        def mutate(row):
            row["functions"]["vllm/mod.py"] = ["plain", "smuggled"]
            row["stamp"]["n_functions"] = 2

        table = load(self._tamper(table_path, mutate))
        assert "runs-it" in table.rejected
        verdict = table.look_up("runs-it", query_for("vllm/mod.py", "plain"))
        assert verdict.evidence is Evidence.ROW_REJECTED and verdict.keep

    def test_a_dropped_function_is_caught_by_the_counts(self, table_path: Path):
        # Truncation is the dangerous direction: a row missing functions looks
        # like a step that never ran them.
        table = load(self._tamper(table_path, lambda row: row["functions"].clear()))
        assert "runs-it" in table.rejected

    def test_a_table_whose_every_row_fails_is_unavailable(self, table_path: Path):
        payload = json.loads(table_path.read_text())
        for row in payload["rows"].values():
            row["stamp"]["digest"] = "0" * 64
        table_path.write_text(json.dumps(payload))
        table = load(table_path)
        assert not table.available and len(table.rejected) == 3

    def test_a_blank_digest_is_refused_rather_than_waved_through(
        self, table_path: Path
    ):
        # The tests above all write a WRONG digest. Blanking it used to skip
        # verification entirely, so scrubbed health fields rode through and the
        # row went on to authorize a drop.
        def mutate(row):
            row["stamp"]["jobs_ran_no_tests"] = 0
            row["stamp"]["digest"] = ""

        table = load(self._tamper(table_path, mutate))
        assert table.rejected.get("runs-it") == "unsigned row"
        verdict = table.look_up("runs-it", query_for("vllm/mod.py", "plain"))
        assert verdict.evidence is Evidence.ROW_REJECTED and verdict.keep


class TestVerdicts:
    def test_a_row_holding_the_changed_function_keeps_the_step(self, table_path: Path):
        verdict = load(table_path).look_up("runs-it", query_for("vllm/mod.py", "plain"))
        assert verdict.evidence is Evidence.EXECUTES_CHANGE and verdict.keep

    def test_a_row_holding_none_of_them_is_the_only_drop(self, table_path: Path):
        verdict = load(table_path).look_up(
            "runs-other", query_for("vllm/mod.py", "plain")
        )
        assert verdict.evidence is Evidence.ABSENT_FROM_ROW
        assert verdict.evidence.authorizes_drop and not verdict.keep

    def test_a_step_with_no_row_is_kept(self, table_path: Path):
        verdict = load(table_path).look_up(
            "never-recorded", query_for("vllm/mod.py", "plain")
        )
        assert verdict.evidence is Evidence.NO_ROW and verdict.keep

    def test_a_row_with_no_functions_is_kept(self, table_path: Path):
        # A complete, clean recording that entered no vLLM code says nothing
        # about coverage. Reading it as absence would drop the step forever.
        verdict = load(table_path).look_up(
            "runs-nothing", query_for("vllm/mod.py", "plain")
        )
        assert verdict.evidence is Evidence.ROW_EMPTY and verdict.keep

    def test_a_row_from_another_python_minor_is_kept(self, table_path: Path):
        restamp(table_path, "runs-other", {"interpreters": ["3.9.18"]})
        verdict = load(table_path).look_up(
            "runs-other", query_for("vllm/mod.py", "plain")
        )
        assert verdict.evidence is Evidence.INTERPRETER_MISMATCH and verdict.keep

    def test_a_wholly_dead_table_says_so_out_loud(self, table_path: Path):
        # The day CI moves to a new Python minor, every row dies at once and
        # the only symptom is `-0 dropped`, which is what a quiet PR looks like
        # too. Silence is the failure mode here, so the table has to volunteer.
        for key in ("runs-it", "runs-other", "runs-nothing"):
            restamp(table_path, key, {"interpreters": ["3.9.18"]})
        note = load(table_path).dead_interpreter
        assert "3.9" in note and "No step can be dropped" in note

    def test_one_dead_row_is_not_a_dead_table(self, table_path: Path):
        # The floor. Announcing on a single stale row would train people to
        # ignore the line, which costs exactly what it was added to buy.
        restamp(table_path, "runs-other", {"interpreters": ["3.9.18"]})
        assert not load(table_path).dead_interpreter

    def test_a_patch_level_difference_is_not_a_mismatch(self, table_path: Path):
        # 3.12.13 recordings against a 3.12.12 extractor measured zero drift over
        # 27,490 names, so patch level must not gate anything.
        import sys

        other = f"{sys.version_info.major}.{sys.version_info.minor}.99"
        restamp(table_path, "runs-other", {"interpreters": [other]})
        assert (
            load(table_path)
            .look_up("runs-other", query_for("vllm/mod.py", "plain"))
            .evidence
            is Evidence.ABSENT_FROM_ROW
        )

    def test_deleting_a_health_counter_no_longer_flips_a_row_healthy(
        self, table_path: Path
    ):
        """The digest covers the stamp, not just the functions.

        Signing only the functions left every field droppability turns on
        unsigned. Deleting a counter makes `Stamp(**blob)` fall back to its
        default, and every default is the healthy value, so a row that recorded
        no test came back clean and droppable.
        """
        restamp(table_path, "runs-other", {"jobs_ran_no_tests": 1})
        thin = load(table_path).look_up("runs-other", query_for("vllm/mod.py", "plain"))
        assert thin.evidence is Evidence.ROW_THIN

        payload = json.loads(table_path.read_text())
        del payload["rows"]["runs-other"]["stamp"]["jobs_ran_no_tests"]
        table_path.write_text(json.dumps(payload))
        verdict = load(table_path).look_up(
            "runs-other", query_for("vllm/mod.py", "plain")
        )
        assert verdict.evidence is Evidence.ROW_REJECTED and verdict.keep

    def test_a_table_of_an_unknown_version_is_refused(self, table_path: Path):
        """An older table's stamp has fewer fields, and every missing field
        defaults to its healthy value, so it would read healthier than it was
        recorded. Refusing is the only reading that fails safe."""
        payload = json.loads(table_path.read_text())
        payload["version"] = 1
        table_path.write_text(json.dumps(payload))
        table = load(table_path)
        assert not table.available
        assert "re-merge" in table.unavailable


class TestThinRow:
    """A silence is only worth as much as the run behind it.

    `merge.Stamp` has computed `thin` since Phase 3 and says in terms that it
    must never authorize a drop, but only `recall.py` was reading it, so the
    reader and the merge disagreed about the same rows.
    """

    @pytest.mark.parametrize(
        "patch, why",
        [
            ({"clean_exits": 0}, "no contributing process exited cleanly"),
            ({"jobs_ran_no_tests": 1}, "executed no test"),
            (
                {"shards_expected": {"1": 4}, "shards_seen": {"1": [0, 1]}},
                "declared shard",
            ),
            ({"lost_lines": True}, "lost lines"),
        ],
    )
    def test_a_thin_row_cannot_authorize_a_drop(
        self, table_path: Path, patch: dict, why: str
    ):
        restamp(table_path, "runs-other", patch)
        verdict = load(table_path).look_up(
            "runs-other", query_for("vllm/mod.py", "plain")
        )
        assert verdict.evidence is Evidence.ROW_THIN and verdict.keep
        assert why in verdict.detail

    def test_an_empty_row_keeps_reporting_the_sharper_reason(self, table_path: Path):
        # `thin` subsumes "no functions", so ordering decides which reason a
        # reader sees. ROW_EMPTY is the specific one and must win.
        verdict = load(table_path).look_up(
            "runs-nothing", query_for("vllm/mod.py", "plain")
        )
        assert verdict.evidence is Evidence.ROW_EMPTY

    def test_a_healthy_row_is_untouched_by_the_gate(self, table_path: Path):
        # Detection floor for the gate itself: if this ever flips to ROW_THIN,
        # the fixture's rows have gone thin and every other drop test is vacuous.
        verdict = load(table_path).look_up(
            "runs-other", query_for("vllm/mod.py", "plain")
        )
        assert verdict.evidence is Evidence.ABSENT_FROM_ROW


class TestQueryFailsOpen:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"status": Attribution.FAILED},
            {"residue": True},
            {"in_recorder_scope": False},
            {"unfaithful": True},
        ],
        ids=["unparsable", "residue", "outside-recorder-root", "unfaithful"],
    )
    def test_an_unanswerable_file_blocks_every_drop(self, table_path: Path, kwargs):
        query = query_for("vllm/mod.py", "plain", **kwargs)
        verdict = load(table_path).look_up("runs-other", query)
        assert verdict.evidence is Evidence.QUERY_FAILS_OPEN and verdict.keep

    def test_one_bad_file_in_a_mixed_diff_blocks_the_others(self, table_path: Path):
        query = Query(
            base="b",
            head="h",
            files=[
                FileQuery(
                    path="vllm/mod.py",
                    status=Attribution.ATTRIBUTED,
                    head_names=frozenset({"plain"}),
                ),
                FileQuery(path="vllm/other.py", status=Attribution.FAILED),
            ],
        )
        assert load(table_path).look_up("runs-other", query).keep


class TestUnfaithfulPropagation:
    def test_a_file_flagged_in_any_row_fails_open_everywhere(
        self, tmp_path: Path, tmp_repo: Repo
    ):
        # The cute-dsl case: the row holds a name the source cannot produce, so
        # matching on names for that file is meaningless for every step.
        build = Build(tmp_path / "sweep" / "b1", "1", tmp_repo.head())
        process_file(
            build.job("j0", step_key="kernels") / "fn.a.txt",
            [("mod.py", "plain"), ("mod.py", "if_region_0")],
        )
        process_file(
            build.job("j1", step_key="other") / "fn.b.txt",
            [("mod.py", "Holder.method")],
        )
        out = tmp_path / "table.json"
        write_table(merge_build(build.finish(), tmp_repo.root), out)

        table = load(out)
        assert table.unfaithful_paths == {"vllm/mod.py"}
        verdicts = apply(table, query_for("vllm/mod.py", "plain"), ["kernels", "other"])
        assert all(v.evidence is Evidence.QUERY_FAILS_OPEN for v in verdicts)

    def test_a_faithful_file_still_produces_a_real_absence(self, table_path: Path):
        # Otherwise the guard above is indistinguishable from blanket fail-open.
        table = load(table_path)
        assert table.unfaithful_paths == set()
        (verdict,) = apply(table, query_for("vllm/mod.py", "plain"), ["runs-other"])
        assert verdict.evidence.authorizes_drop


@pytest.fixture
def phase_table_path(tmp_path: Path, tmp_repo: Repo) -> Path:
    """One row that only imported mod.py, one that also ran a function in it.

    `table_path` records no `<module>` at all, so every row it builds has an
    empty `import_time` and `contains_call` cannot be told from `contains`.
    Nothing in that fixture can see a phase, which is why this exists.
    """
    build = Build(tmp_path / "sweep" / "b1", "1", tmp_repo.head())
    process_file(
        build.job("j0", step_key="imports-only") / "fn.a.txt", [("mod.py", "<module>")]
    )
    process_file(
        build.job("j1", step_key="calls-plain") / "fn.b.txt",
        [("mod.py", "<module>"), ("mod.py", "plain")],
    )
    out = tmp_path / "phase-table.json"
    write_table(merge_build(build.finish(), tmp_repo.root), out)
    return out


class TestPhaseInLookUp:
    """`look_up` is the second of the two sites reading a phase mode.

    It re-makes the keep check's match further down, so it needs its own
    coverage at every mode. `rules.py` pins that the two sites agree; this pins
    what the answer actually is.
    """

    MIXED = ("<module>", "plain")

    @pytest.mark.parametrize(
        "mode,expected",
        [
            (PhaseMode.OFF, Evidence.EXECUTES_CHANGE),
            (PhaseMode.CARVED, Evidence.EXECUTES_CHANGE),
            (PhaseMode.STRICT, Evidence.NOTHING_CALLABLE),
        ],
    )
    def test_a_module_only_diff(self, phase_table_path, mode, expected):
        """No call evidence can exist for this diff, so CARVED keeps reading
        the row, which still separates importers from everyone else. STRICT
        cannot, and says so with its own verdict rather than borrowing
        ABSENT_FROM_ROW: the drop is the same either way, but only one of them
        is a reading, and the other has to stay countable.
        """
        table = load(phase_table_path)
        query = query_for(
            "vllm/mod.py", "<module>", import_time=frozenset({"<module>"})
        )
        assert table.look_up("imports-only", query, mode).evidence is expected

    @pytest.mark.parametrize(
        "mode,expected",
        [
            (PhaseMode.OFF, Evidence.EXECUTES_CHANGE),
            (PhaseMode.CARVED, Evidence.ABSENT_FROM_ROW),
            (PhaseMode.STRICT, Evidence.ABSENT_FROM_ROW),
        ],
    )
    def test_a_mixed_diff_against_a_row_that_only_imported(
        self, phase_table_path, mode, expected
    ):
        """Where the modes pay. The row holds `<module>` for the changed file
        and nothing else, which under OFF reads as evidence the step is
        relevant."""
        table = load(phase_table_path)
        query = query_for(
            "vllm/mod.py", *self.MIXED, import_time=frozenset({"<module>"})
        )
        assert table.look_up("imports-only", query, mode).evidence is expected

    def test_a_row_that_ran_the_function_keeps_at_every_mode(self, phase_table_path):
        table = load(phase_table_path)
        query = query_for(
            "vllm/mod.py", *self.MIXED, import_time=frozenset({"<module>"})
        )
        for mode in PhaseMode:
            verdict = table.look_up("calls-plain", query, mode)
            assert verdict.evidence is Evidence.EXECUTES_CHANGE, mode


class TestEmptyChangedFunctionSet:
    """A diff that names no function must not satisfy the skip rule for free.

    The dangerous shape is a file that is nameless by nature AND inside the
    recorder's root: nothing fails open, because nameless is a legitimate empty,
    and there is no name to look up, so "the row contains none of them" is true
    vacuously and every step drops. Two PRs in the 300-PR benchmark were exactly
    this, both tuned-kernel config JSON under vllm/.
    """

    def test_a_nameless_in_scope_file_drops_nothing(self, table_path: Path):
        query = Query(
            base="b",
            head="h",
            files=[
                FileQuery(
                    path="vllm/model_executor/layers/fused_moe/configs/E=256.json",
                    status=Attribution.NAMELESS,
                    in_recorder_scope=True,
                )
            ],
        )
        verdict = load(table_path).look_up("runs-other", query)
        assert verdict.evidence is Evidence.NAMELESS_IN_SCOPE and verdict.keep

    def test_a_nameless_in_scope_file_is_not_ignored_beside_a_real_one(
        self, table_path: Path
    ):
        # The mixed case, and the one per-step scoping newly exposes. Alone the
        # config file keeps the step; next to a real change it would otherwise
        # be silently ignored while being a reason the step was selected.
        query = Query(
            base="b",
            head="h",
            files=[
                FileQuery(
                    path="vllm/mod.py",
                    status=Attribution.ATTRIBUTED,
                    head_names=frozenset({"plain"}),
                ),
                FileQuery(
                    path="vllm/model_executor/layers/fused_moe/configs/E=256.json",
                    status=Attribution.NAMELESS,
                    in_recorder_scope=True,
                ),
            ],
        )
        verdict = load(table_path).look_up("runs-other", query)
        assert verdict.evidence is Evidence.NAMELESS_IN_SCOPE and verdict.keep

    def test_an_empty_diff_drops_nothing(self, table_path: Path):
        verdict = load(table_path).look_up("runs-other", Query(base="b", head="h"))
        assert verdict.evidence is Evidence.NOTHING_TO_MATCH and verdict.keep

    def test_a_real_change_still_drops(self, table_path: Path):
        # The guard must not blunt the one path that removes a step.
        verdict = load(table_path).look_up(
            "runs-other", query_for("vllm/mod.py", "plain")
        )
        assert verdict.evidence.authorizes_drop
