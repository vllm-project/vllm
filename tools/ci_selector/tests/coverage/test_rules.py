# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The payoff estimator, and the bracket that keeps it honest.

The measurement's own bias is the subject. Our table was recorded after every
measured PR merged, so it holds code those PRs added, and reading it straight
credits drops production could not make. The pessimistic pass removes that
credit; these pin both ends and the invariant between them.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ci_selector.codemap.selection import Selection
from ci_selector.coverage.changed_funcs import Attribution, FileQuery, Query
from ci_selector.coverage.model import MIN_ROWS_FOR_BREADTH
from ci_selector.coverage.phase import DEFAULT_MODE, PhaseMode
from ci_selector.coverage.rules import (
    RowKeys,
    read_pr,
    row_key_for,
    unknown_names,
)

from .helpers import Repo, make_table


class TestRowKeyFor:
    def test_strips_the_pipeline_prefix(self):
        assert row_key_for("vllm_ci:lora") == "lora"

    def test_a_keyless_label_with_colons_survives(self):
        assert row_key_for("vllm_ci:AMD: Kernels MoE Test") == "AMD: Kernels MoE Test"

    def test_a_mirror_does_not_resolve_to_its_parent(self):
        # The invariant, in the one place it could be broken by accident. The
        # hardware suffix stays on, so the key matches no row, so the mirror is
        # kept. No list of mirror names to maintain.
        assert row_key_for("vllm_ci:lora:amd") != row_key_for("vllm_ci:lora")

    def test_an_id_without_a_pipeline_has_no_row(self):
        assert row_key_for("lora") is None


class TestRowKeys:
    """A bare key is unique only WITHIN a pipeline.

    `vllm_intel_ci` defines its own `engine`, `quantization`, `regression` and
    ten more in a directory of its own, on Intel GPUs. Stripping the prefix
    hands those steps the NVIDIA rows of the same name, which is the mirror
    invariant broken one level up.
    """

    OWNED = {"vllm_ci": 1.0, "vllm_intel_ci": 0.06, "vllm_rocm_ci": 0.0}

    def test_the_owning_pipeline_keeps_its_rows(self):
        keys = RowKeys.from_match_rates(self.OWNED)
        assert keys.owners == {"vllm_ci"}
        assert keys.key_for("vllm_ci:engine") == "engine"

    def test_another_pipeline_reusing_the_key_gets_no_row(self):
        keys = RowKeys.from_match_rates(self.OWNED)
        assert keys.key_for("vllm_intel_ci:engine") is None

    def test_a_mirror_still_does_not_resolve_to_its_parent(self):
        keys = RowKeys.from_match_rates(self.OWNED)
        assert keys.key_for("vllm_ci:lora:amd") != keys.key_for("vllm_ci:lora")

    def test_a_mirror_reads_its_own_row_once_the_steps_are_resolved(self):
        # Real CI spells the mirror `amd-lora`, so the shape alone finds no row
        # and every AMD step is kept. Resolving through the step recovers it,
        # and the invariant holds: NVIDIA's `lora` row is still a different row.
        keys = RowKeys.from_match_rates(
            self.OWNED, {"vllm_ci:lora-amd:amd": "amd-lora", "vllm_ci:lora": "lora"}
        )
        assert keys.key_for("vllm_ci:lora-amd:amd") == "amd-lora"
        assert keys.key_for("vllm_ci:lora") == "lora"

    def test_a_step_the_checkout_does_not_explain_reads_no_row(self):
        keys = RowKeys.from_match_rates(self.OWNED, {"vllm_ci:lora": "lora"})
        assert keys.key_for("vllm_ci:since-deleted") is None

    def test_the_owner_is_whichever_pipeline_explains_the_most_rows(self):
        keys = RowKeys.from_match_rates({"vllm_ci": 0.4, "vllm_intel_ci": 0.3})
        assert keys.owners == {"vllm_ci"}

    def test_no_match_rates_means_no_owner_and_so_no_rows(self):
        # The degraded reading has to keep everything rather than stop the run:
        # no owner, so no step resolves to a row, so the table changes nothing.
        keys = RowKeys.from_match_rates({})
        assert keys.owners == set()
        assert keys.key_for("vllm_ci:engine") is None


@pytest.fixture
def table(tmp_path: Path, tmp_repo: Repo):
    return make_table(
        tmp_path,
        tmp_repo,
        {
            "runs-plain": [("mod.py", "plain"), ("mod.py", "<module>")],
            "runs-method": [("mod.py", "Holder.method"), ("mod.py", "<module>")],
            "elsewhere": [("other.py", "<module>"), ("other.py", "elsewhere")],
        },
    )


def result_for(*step_ids: str, paths: tuple[str, ...] = ("vllm/mod.py",), **extra):
    """A `Selection`, as the code map would have produced it.

    Attribution is present by default, as in production. Absent attribution
    means "not droppable", so a fixture that omitted it would keep every step
    and test nothing. Pass `selected_paths={}` to exercise that deliberately.
    """
    fields = {
        "selected": {s: [] for s in step_ids},
        "selected_paths": {sid: [list(paths)] for sid in step_ids},
    }
    # `paths=` shapes the default attribution; the rest are Selection fields.
    fields.update({k: v for k, v in extra.items() if k != "paths"})
    return Selection(**fields)


def query_for(path: str, *names: str, **kwargs) -> Query:
    return Query(
        base="base",
        head="head",
        files=[
            FileQuery(
                path=path,
                status=Attribution.ATTRIBUTED,
                head_names=frozenset(names),
                **kwargs,
            )
        ],
    )


UNION = {
    "vllm/mod.py": frozenset({"plain", "<module>", "Holder.method"}),
    "vllm/other.py": frozenset({"<module>", "elsewhere"}),
}
KNOWN = frozenset(UNION)
ALL_STEPS = ("vllm_ci:runs-plain", "vllm_ci:runs-method", "vllm_ci:elsewhere")


OWNER = RowKeys({"vllm_ci"}, {"vllm_ci": 1.0})


class FakeStep:
    """The slice of `Step` that `RowKeys.candidates()` and `restrict_to` read."""

    def __init__(self, manual_only: bool = False, identity: tuple | None = None):
        self.manual_only = manual_only
        # Rename-tolerant, so a test can pose the same step under two labels.
        self.identity = identity


# CI outcomes are not part of a Selection; they reach the rule as keywords.
_OUTCOMES = ("failed_ran", "matched_slugs", "failed_missed")


def read(table, query, *steps, fresh=None, **extra):
    # Attribute each step to the files the query is actually about, so a fixture
    # can never disagree with its own query. Override with `paths=` or
    # `selected_paths=` to test a mismatch on purpose.
    extra.setdefault("paths", tuple(f.path for f in query.files))
    outcomes = {k: extra.pop(k) for k in _OUTCOMES if k in extra}
    stale = frozenset(extra.pop("stale", ()))
    mode = extra.pop("mode", DEFAULT_MODE)
    return read_pr(
        table,
        result_for(*(steps or ALL_STEPS), **extra),
        query,
        unknown_names(query, UNION, fresh or {}),
        KNOWN,
        OWNER,
        stale,
        mode=mode,
        **outcomes,
    )


class TestFreshnessGate:
    """A row may only drop a step it still describes.

    The gate itself lives in `freshness.py` and has its own tests; this pins the
    check inside `read_pr` that consumes it, which had no test at all -- nothing
    anywhere passed `stale`.
    """

    def test_a_stale_step_is_never_dropped(self, table):
        query = query_for("vllm/mod.py", "plain")
        without = read(table, query, "vllm_ci:elsewhere")
        assert without.dropped == ["vllm_ci:elsewhere"], "fixture: nothing to gate"

        gated = read(table, query, "vllm_ci:elsewhere", stale={"vllm_ci:elsewhere"})
        assert gated.dropped == []
        assert "vllm_ci:elsewhere" in gated.kept
        assert gated.reasons["row-is-stale"] == 1

    def test_staleness_does_not_block_a_positive_match(self, table):
        """Only the DROP direction is gated. A row showing the step ran the
        changed code is still proof it is relevant, stale or not -- and the
        keep it produces is credited to the evidence, not to the gate."""
        reading = read(
            table,
            query_for("vllm/mod.py", "plain"),
            "vllm_ci:runs-plain",
            stale={"vllm_ci:runs-plain"},
        )
        assert reading.reasons["row-executes-a-changed-function"] == 1
        assert not reading.reasons["row-is-stale"]


class TestTheAdditiveHalf:
    """The record SELECTS, it does not only remove.

    The record shipped as a one-way narrower, and the `authorizes_drop` single
    authority encoded that: a row could only ever take a step away. But a row
    showing that a step executed a changed file is proof the step is relevant,
    whatever the map concluded, and that direction is the cheaper one --
    presence needs a single observation where absence needs the recording to
    be complete.
    """

    # `candidates()` walks the resolved steps and reads `manual_only` off each,
    # so the fixtures have to be step-shaped rather than bare sentinels.
    OWNER_WITH_STEPS = RowKeys(
        {"vllm_ci"}, {"vllm_ci": 1.0}, steps={s: FakeStep() for s in ALL_STEPS}
    )

    def test_a_manual_only_step_is_never_added(self, table):
        """The sweeps unblocked the optional steps so the table would cover
        them, so it holds rows for nightlies and AMD mirrors CI does not run on
        a PR. Adding one is not over-selection, it is naming a step the
        generator will not emit -- and it was 43% of the additive half's cost
        before this filter. The subtractive direction needs no equivalent: its
        population is the map's selection, which never holds a manual-only
        step.
        """
        owner = RowKeys(
            {"vllm_ci"},
            {"vllm_ci": 1.0},
            steps={"vllm_ci:elsewhere": FakeStep(manual_only=True)},
        )
        assert owner.candidates() == []
        reading = read_pr(
            table,
            result_for("vllm_ci:runs-plain"),
            query_for("vllm/other.py", "<module>"),
            {},
            KNOWN,
            owner,
        )
        assert reading.added == []

    def test_a_step_absent_at_base_is_never_added(self, table):
        """The two-clock bug. These ids were resolved at the TABLE's commit and
        the emitter names steps from the base, so a label rename on main leaves
        the record adding a step the emitter cannot name. Its answer is to omit
        the variable, and omission means run everything: one stale row buys a
        full pipeline. Measured at HEAD, several keyless ids existed that main
        did not, most of them carrying large rows.
        """
        owner = RowKeys(
            {"vllm_ci"}, {"vllm_ci": 1.0}, steps={s: FakeStep() for s in ALL_STEPS}
        )
        owner.restrict_to(frozenset({"vllm_ci:runs-plain"}))
        assert "vllm_ci:elsewhere" not in owner.candidates()

        reading = read_pr(
            table,
            result_for("vllm_ci:runs-plain", paths=("vllm/other.py",)),
            query_for("vllm/other.py", "<module>"),
            {},
            KNOWN,
            owner,
        )
        assert reading.added == []

    def test_a_renamed_step_keeps_its_add(self):
        """The defect the two tests around this one structurally cannot see.

        Both compare hand-written literal ids, so a rename is not expressible
        in them. Here the step exists at base under a different label, hence a
        different `step_id`, and only `identity` says it is the same step. Id
        alone deleted live candidates on most PRs of a measured sample.
        """
        renamed = FakeStep(identity=("vllm_ci", "a.yaml", ("pytest x",), None))
        owner = RowKeys(
            {"vllm_ci"},
            {"vllm_ci": 1.0},
            steps={"vllm_ci:Old Label": renamed},
        )
        owner.restrict_to(
            frozenset({"vllm_ci:New Label"}), frozenset({renamed.identity})
        )
        assert "vllm_ci:Old Label" in owner.candidates()
        assert owner.restricted_away == []

    def test_a_step_that_really_left_is_still_dropped_and_counted(self):
        """Widening the survival test must not disarm it. A step present in
        neither spelling is gone, and the drop is now countable rather than a
        silent absence from the histogram."""
        gone = FakeStep(identity=("vllm_ci", "gone.yaml", ("pytest y",), None))
        owner = RowKeys(
            {"vllm_ci"}, {"vllm_ci": 1.0}, steps={"vllm_ci:elsewhere": gone}
        )
        owner.restrict_to(
            frozenset({"vllm_ci:runs-plain"}),
            frozenset({("vllm_ci", "other.yaml", ("pytest z",), None)}),
        )
        assert owner.candidates() == []
        assert owner.restricted_away == ["vllm_ci:elsewhere"]

    def test_an_unreadable_base_pipeline_changes_nothing(self, table):
        """The floor, and it matters: an empty set is "we could not look", not
        "no step exists". Treating the two alike would silently disable the
        whole add side on any base we failed to parse."""
        owner = RowKeys(
            {"vllm_ci"}, {"vllm_ci": 1.0}, steps={s: FakeStep() for s in ALL_STEPS}
        )
        owner.restrict_to(frozenset())
        assert "vllm_ci:elsewhere" in owner.candidates()

    def _read(self, table, query, *selected):
        return read_pr(
            table,
            result_for(*selected, paths=tuple(f.path for f in query.files)),
            query,
            unknown_names(query, UNION, {}),
            KNOWN,
            self.OWNER_WITH_STEPS,
        )

    def test_a_row_adds_a_step_the_map_never_selected(self, table):
        """`elsewhere` recorded `vllm/other.py`, and the map picked only
        `runs-plain`. The observation outranks the map's silence.

        On a CALL frame: since 2A the add side refuses import-time names, and
        `<module>` -- what this used to assert on -- is one."""
        reading = self._read(
            table, query_for("vllm/other.py", "elsewhere"), "vllm_ci:runs-plain"
        )
        assert "vllm_ci:elsewhere" in reading.added

    def test_an_import_time_name_alone_never_adds(self, table):
        """The 2A asymmetry, on one reading. `<module>` is in every (row, file)
        pair there is, so it says "this step imported the file" and cannot
        separate one row from another. The drop side still honours it, on
        purpose."""
        reading = self._read(
            table, query_for("vllm/other.py", "<module>"), "vllm_ci:runs-plain"
        )
        assert reading.added == []

    def test_a_step_already_selected_is_not_added_twice(self, table):
        reading = self._read(
            table, query_for("vllm/other.py", "<module>"), "vllm_ci:elsewhere"
        )
        assert reading.added == []

    def test_a_step_with_no_row_is_never_added(self, table):
        """The third arm. No row is not evidence of anything, so the record
        must stay quiet rather than add on a hunch. The subtractive twin of
        this lives in test_decide.py under nearly the same name."""
        owner = RowKeys(
            {"vllm_ci"}, {"vllm_ci": 1.0}, steps={"vllm_ci:never-recorded": object()}
        )
        reading = read_pr(
            table,
            result_for("vllm_ci:runs-plain"),
            query_for("vllm/mod.py", "plain"),
            {},
            KNOWN,
            owner,
        )
        assert reading.added == []

    def test_the_gate_is_counted_in_the_reason_histogram(self, table):
        """`row-adds-a-step-the-map-missed` is what says whether this half ran
        at all, and it was pinned by nothing. It counts adds AFTER the breadth
        gate, so a gated PR reads as "fired and found nothing" rather than
        "never ran"."""
        reading = self._read(
            table, query_for("vllm/other.py", "<module>"), "vllm_ci:runs-plain"
        )
        assert reading.reasons["row-adds-a-step-the-map-missed"] == len(reading.added)

    def test_the_two_halves_decide_independently(self, table):
        """The asymmetry, pinned on one reading. Same diff, same table: one
        step is dropped on an absence while another is added on a presence.
        Neither answer derives from the other, which is the point of splitting
        the authority: a single keep-or-drop verdict per selected step could
        not have produced the second."""
        reading = read_pr(
            table,
            result_for("vllm_ci:runs-plain", paths=("vllm/other.py",)),
            query_for("vllm/other.py", "elsewhere"),
            {},
            KNOWN,
            self.OWNER_WITH_STEPS,
        )
        assert reading.dropped == ["vllm_ci:runs-plain"]
        assert "vllm_ci:elsewhere" in reading.added


class TestPerReasonScoping:
    """One unanswerable file must not silence the whole PR.

    But narrowing is only safe on a path set that is COMPLETE for the step:
    `look_up` is anti-monotone in the query, so removing a file can only remove
    a fail-open or remove a matching name, and both turn a keep into a drop.
    These pin the two directions.
    """

    MIXED = Query(
        base="base",
        head="head",
        files=[
            FileQuery(
                path="vllm/mod.py",
                status=Attribution.ATTRIBUTED,
                head_names=frozenset({"plain"}),
            ),
            FileQuery(
                path="tests/test_thing.py",
                status=Attribution.ATTRIBUTED,
                head_names=frozenset({"test_thing"}),
                in_recorder_scope=False,
            ),
        ],
    )

    def test_an_unrelated_unanswerable_file_no_longer_blocks(self, table):
        # The whole point. `runs-method` does not execute `plain`, and the
        # tests/ file had nothing to do with why it was selected.
        reading = read(
            table,
            self.MIXED,
            "vllm_ci:runs-method",
            selected_paths={"vllm_ci:runs-method": [["vllm/mod.py"]]},
        )
        assert reading.dropped == ["vllm_ci:runs-method"]

    def test_the_unanswerable_file_still_blocks_its_own_steps(self, table):
        # Two gates would each catch this and the earlier one wins: the file's
        # names are in no row at all, so the unknown-code gate fires before the
        # table is consulted. Both are keeps; what matters is that scoping did
        # not release the step whose reason really was the tests/ file.
        reading = read(
            table,
            self.MIXED,
            "vllm_ci:runs-method",
            selected_paths={"vllm_ci:runs-method": [["tests/test_thing.py"]]},
        )
        assert reading.dropped == []
        assert reading.reasons["unknown-code-blocks-narrowing"] == 1

    def test_an_out_of_scope_file_alone_fails_the_query_open(self, table):
        query = Query(
            base="base",
            head="head",
            files=[
                FileQuery(
                    path="tests/test_thing.py",
                    status=Attribution.ATTRIBUTED,
                    in_recorder_scope=False,
                )
            ],
        )
        reading = read(
            table,
            query,
            "vllm_ci:runs-method",
            selected_paths={"vllm_ci:runs-method": [["tests/test_thing.py"]]},
        )
        assert reading.dropped == []
        assert reading.reasons["query-cannot-answer"] == 1

    def test_one_not_droppable_reason_holds_the_step(self, table):
        # A declared dep, hardware tagging, run-all, preflight or always-run.
        # It stands on its own no matter what the rows say about the rest.
        reading = read(
            table,
            self.MIXED,
            "vllm_ci:runs-method",
            selected_paths={"vllm_ci:runs-method": [["vllm/mod.py"], None]},
        )
        assert reading.dropped == []
        assert reading.reasons["not droppable-reason"] == 1

    def test_no_attributed_file_is_a_keep(self, table):
        reading = read(
            table,
            self.MIXED,
            "vllm_ci:runs-method",
            selected_paths={"vllm_ci:runs-method": [[]]},
        )
        assert reading.dropped == []
        assert reading.reasons["no-attributed-file"] == 1

    def test_a_row_running_an_out_of_scope_change_still_keeps(self, table):
        """Scoping may narrow what you examine, never who may be doubted.

        The map cited `other.py` for this step, so scoping alone would examine
        only that and read the row's silence about it as a drop -- while the
        row demonstrably executes `plain` from `mod.py`, which the same diff
        changed. 221 drops in the benchmark had this shape.
        """
        both = Query(
            base="base",
            head="head",
            files=[
                FileQuery(
                    path="vllm/mod.py",
                    status=Attribution.ATTRIBUTED,
                    head_names=frozenset({"plain"}),
                ),
                FileQuery(
                    path="vllm/other.py",
                    status=Attribution.ATTRIBUTED,
                    head_names=frozenset({"plain"}),
                ),
            ],
        )
        reading = read(table, both, "vllm_ci:runs-plain", paths=("vllm/other.py",))
        assert reading.dropped == []
        assert reading.reasons["row-executes-a-changed-function"] == 1

    @pytest.mark.parametrize(
        "mode,kept",
        [(PhaseMode.OFF, True), (PhaseMode.CARVED, True), (PhaseMode.STRICT, False)],
    )
    def test_an_import_time_name_counts_as_use(self, table, mode, kept):
        """An import is use under OFF, and that reading is the expensive half
        of a measured trade.

        The stricter reading is that import-time frames run in every step that
        imports the file, so they separate nothing and cannot be positive
        evidence. True as far as it goes, and it caused half of the harmful
        drops when those were judged one at a time: a step that only imports a
        changed file can still break on it, and nearly every row records the
        most-touched file at import time alone, which made it permanently
        droppable.

        CARVED keeps that rescue for exactly this shape, a file whose changed
        names are ALL import-time, where no call evidence could ever exist.
        STRICT refuses it, which is why STRICT is a ceiling and not a rule: no
        row in any table can match, so the verdict is a constant.
        """
        module_only = Query(
            base="base",
            head="head",
            files=[
                FileQuery(
                    path="vllm/mod.py",
                    status=Attribution.ATTRIBUTED,
                    head_names=frozenset({"<module>"}),
                    import_time=frozenset({"<module>"}),
                ),
            ],
        )
        reading = read(table, module_only, "vllm_ci:runs-plain", mode=mode)
        if kept:
            assert reading.reasons["row-executes-a-changed-function"] == 1
            assert reading.dropped == []
        else:
            assert reading.dropped == ["vllm_ci:runs-plain"]
            assert reading.reasons["diff-named-no-callable-function"] == 1

    def test_a_crosscheck_without_attribution_cannot_refute_anything(self, table):
        """Absent attribution is an authority question, not a scope one.

        This used to fall back to the whole diff, described as "strictly more
        conservative than any subset" -- true of the scope and false of what it
        granted. On an input missing the key, every step became droppable,
        preflight-forced and always-run ones included.
        """
        reading = read(table, self.MIXED, "vllm_ci:runs-method", selected_paths={})
        assert reading.dropped == []
        assert reading.reasons["no-attribution"] == 1


class TestPhaseModes:
    """The three readings of "the row ran this", and that they stay distinct.

    The design hazard is CARVED collapsing onto OFF. A per-NAME carve-out does
    exactly that: `import_time_names` always yields `<module>`, so `<module>`
    is in `FileQuery.import_time` whenever it is in `names`, and a per-name
    rule could never deny the one name that matters. The carve-out is per FILE
    for that reason, and these tests are what hold it there.
    """

    @staticmethod
    def _mixed() -> Query:
        """A file that changed a real function AND its module body.

        The common shape, not a corner: `<module>` owns the `def` line, so it
        joins the changed set whenever a signature moves. `runs-method`
        recorded `<module>` for mod.py and never `plain`, so it is the row that
        separates "imported the file" from "ran the change".
        """
        return query_for(
            "vllm/mod.py", "<module>", "plain", import_time=frozenset({"<module>"})
        )

    @pytest.mark.parametrize(
        "mode,dropped",
        [(PhaseMode.OFF, False), (PhaseMode.CARVED, True), (PhaseMode.STRICT, True)],
    )
    def test_a_file_that_changed_a_function_loses_its_import_rescue(
        self, table, mode, dropped
    ):
        """The arm that pays for itself. Under OFF the `<module>` entry keeps a
        step that only ever imported the file; once the file changed a real
        function, CARVED stops accepting that."""
        reading = read(table, self._mixed(), "vllm_ci:runs-method", mode=mode)
        assert (reading.dropped == ["vllm_ci:runs-method"]) is dropped

    def test_both_predicate_sites_move_together(self, table):
        """The trap this whole design exists to prevent.

        `look_up` re-makes the keep check's match further down. Swap only the
        check and `look_up` keeps the step anyway; swap only `look_up` and the
        check never reaches it. Both halves read as "the mode buys nothing", so
        assert the drop AND the vanished reason: the first fails if `look_up`
        was missed, the second if the check was.
        """
        reading = read(
            table, self._mixed(), "vllm_ci:runs-method", mode=PhaseMode.CARVED
        )
        assert reading.dropped == ["vllm_ci:runs-method"]
        assert "row-executes-a-changed-function" not in reading.reasons

    def test_a_row_that_ran_the_function_keeps_at_every_mode(self, table):
        """The floor. A mode may only withdraw evidence that never
        discriminated; real call evidence has to survive all three."""
        for mode in PhaseMode:
            reading = read(table, self._mixed(), "vllm_ci:runs-plain", mode=mode)
            assert reading.dropped == [], mode
            assert reading.reasons["row-executes-a-changed-function"] == 1, mode


class TestRestrict:
    def test_a_rename_matches_on_either_side(self):
        q = Query(
            base="b",
            head="h",
            files=[
                FileQuery(
                    path="vllm/new.py",
                    old_path="vllm/old.py",
                    status=Attribution.ATTRIBUTED,
                )
            ],
        )
        # The map sees both sides of a rename; the diff yields one record under
        # the head path, so attribution by the old path must still find it.
        assert len(q.restrict({"vllm/old.py"}).files) == 1
        assert len(q.restrict({"vllm/new.py"}).files) == 1
        assert q.restrict({"vllm/other.py"}).files == []

    def test_marks_set_after_restriction_are_still_visible(self):
        from ci_selector.coverage.changed_funcs import mark_unfaithful

        q = Query(
            base="b",
            head="h",
            files=[FileQuery(path="vllm/mod.py", status=Attribution.ATTRIBUTED)],
        )
        narrowed = q.restrict({"vllm/mod.py"})
        mark_unfaithful(q, {"vllm/mod.py"})
        # Shared objects, not copies: the unfaithful flag is what keeps the
        # AST-rewriting kernel files from reading as real absences.
        assert narrowed.files[0].unfaithful and narrowed.fail_open


class TestReadings:
    def test_a_step_running_the_change_is_kept_and_others_drop(self, table):
        reading = read(table, query_for("vllm/mod.py", "plain"))
        assert reading.kept == ["vllm_ci:runs-plain"]
        assert sorted(reading.dropped) == ["vllm_ci:elsewhere", "vllm_ci:runs-method"]

    def test_a_mirror_is_never_cleared_by_its_parents_row(self, table):
        reading = read(
            table, query_for("vllm/mod.py", "plain"), "vllm_ci:runs-method:amd"
        )
        assert reading.dropped == []
        assert reading.reasons["no-row"] == 1


class TestAbsenceOfEvidence:
    """A function missing from EVERY row is no data at all.

    Missing from one step's row is evidence about that step. Missing everywhere
    is a gap in the recordings, and the design forbids narrowing on it. This is
    not a bias correction, so it binds both ends of the bracket.
    """

    def test_a_name_no_row_has_ever_seen_falls_back_to_the_file(self, table):
        reading = read(table, query_for("vllm/mod.py", "never_recorded"))
        # Steps that run the file keep, because the unknown code could be
        # reached from what they do run. The step that never touches it drops.
        assert reading.dropped == ["vllm_ci:elsewhere"]
        assert reading.reasons["file-projection-keeps"] == 2

    def test_a_name_in_a_file_no_row_has_seen_stops_everything(self, table):
        reading = read(table, query_for("vllm/newfile.py", "anything"))
        assert reading.dropped == []
        assert reading.reasons["unknown-code-blocks-narrowing"] == 3


class TestBracket:
    """The measurement bias, isolated: a name the table only knows because it
    was recorded after the PR merged."""

    def test_new_at_base_is_the_only_difference_between_the_ends(self, table):
        query = query_for("vllm/mod.py", "plain")
        optimistic = read(table, query)
        pessimistic = read(table, query, fresh={"vllm/mod.py": frozenset({"plain"})})

        # Straight read: nobody but runs-plain ran it, so the rest go.
        assert sorted(optimistic.dropped) == [
            "vllm_ci:elsewhere",
            "vllm_ci:runs-method",
        ]
        # Honest read: the function did not exist at the base, so it is unknown
        # and steps running its file keep.
        assert pessimistic.dropped == ["vllm_ci:elsewhere"]
        assert len(pessimistic.dropped) <= len(optimistic.dropped)


class TestSafety:
    def test_dropping_a_step_that_actually_failed_is_recorded(self, table):
        # The number that outranks every ratio in the report.
        reading = read(
            table,
            query_for("vllm/mod.py", "plain"),
            "vllm_ci:elsewhere",
            failed_ran={"elsewhere-job": "FAILURE"},
            matched_slugs={"vllm_ci:elsewhere": ["elsewhere-job"]},
        )
        assert reading.dropped_and_failed == ["vllm_ci:elsewhere"]

    def test_an_unanswerable_file_drops_nothing(self, table):
        reading = read(
            table,
            query_for("vllm/mod.py", "plain", in_recorder_scope=False),
            "vllm_ci:runs-method",
            "vllm_ci:elsewhere",
        )
        assert reading.dropped == []


class TestTheBreadthGate:
    """A name nearly every row holds is not evidence about any of them.

    `Row.contains` is a membership test, so 194 of 196 rows reads exactly like
    1 of 196. `<module>` sits in every (row, file) pair there is, which made the
    add side assert only "the step imports the changed file", measured at over
    a thousand steps added and none dropped against a narrower code map.

    Only this side is gated. On the drop side the same breadth argues the other
    way, on purpose.
    """

    # Enough rows for share to mean something. `tmp_repo` writes only mod.py and
    # other.py, so every pair has to name one of those.
    # `plain` is a CALL frame, so the add side will look at it; `<module>` is
    # refused on phase before breadth is consulted, which is a different gate.
    WIDE = {f"row{i}": [("mod.py", "plain")] for i in range(24)}
    WIDE["narrow"] = [("mod.py", "plain"), ("mod.py", "Holder.method")]

    @pytest.fixture
    def wide_table(self, tmp_path: Path, tmp_repo: Repo):
        return make_table(tmp_path, tmp_repo, self.WIDE)

    @staticmethod
    def _owner(keys):
        return RowKeys(
            {"vllm_ci"},
            {"vllm_ci": 1.0},
            steps={f"vllm_ci:{k}": FakeStep() for k in keys},
        )

    def _read(self, table, keys, query, *selected):
        return read_pr(
            table,
            result_for(*selected, paths=tuple(f.path for f in query.files)),
            query,
            unknown_names(query, UNION, {}),
            KNOWN,
            self._owner(keys),
        )

    def test_a_name_nearly_every_row_holds_does_not_authorize_an_add(self, wide_table):
        """`<module>` is in all 25 rows. Every one of them "contains" the
        changed name, which is exactly why none of them may be added on it."""
        reading = self._read(
            wide_table,
            self.WIDE,
            query_for("vllm/mod.py", "plain"),
            "vllm_ci:row0",
        )
        assert reading.added == []

    def test_a_name_one_row_holds_still_authorizes_an_add(self, wide_table):
        """The gate removes evidence that separates nothing, not the half. One
        row ran `plain`; that observation still outranks the map's silence."""
        reading = self._read(
            wide_table,
            self.WIDE,
            query_for("vllm/mod.py", "Holder.method"),
            "vllm_ci:row0",
        )
        assert "vllm_ci:narrow" in reading.added

    def test_the_gate_stands_down_on_a_table_too_small_to_measure(
        self, tmp_path: Path, tmp_repo: Repo
    ):
        """On three rows every pair is 33%, 67% or 100%, so a share threshold
        would be reading noise. Below the floor it behaves as it always did:
        fail-open, which is the safe direction for an add."""
        keys = {f"row{i}": [("mod.py", "plain")] for i in range(3)}
        small = make_table(tmp_path, tmp_repo, keys)
        assert len(small) < MIN_ROWS_FOR_BREADTH
        reading = self._read(
            small, keys, query_for("vllm/mod.py", "plain"), "vllm_ci:row0"
        )
        assert "vllm_ci:row1" in reading.added


class TestProxyEvidence:
    """Wrapper stand-ins for a csrc change: the drop side weighs them, the
    add side ignores them, and an unrecorded name keeps the step."""

    def _proxy_query(self, *names, status=Attribution.ATTRIBUTED):
        return Query(
            base="base",
            head="head",
            files=[
                FileQuery(
                    path="vllm/mod.py",
                    status=status,
                    head_names=frozenset(names),
                    proxy=True,
                )
            ],
        )

    def test_the_positive_check_reads_proxy_wrappers(self, table):
        reading = read(table, self._proxy_query("plain"), "vllm_ci:runs-plain")
        assert "vllm_ci:runs-plain" in reading.kept
        assert reading.reasons["row-executes-a-changed-function"] == 1

    def test_a_silent_row_drops_on_proxy_evidence(self, table):
        reading = read(table, self._proxy_query("plain"), "vllm_ci:elsewhere")
        assert "vllm_ci:elsewhere" in reading.dropped
        assert reading.reasons["proxy-evidence-consulted"] == 1
        assert reading.reasons["proxy-evidence-drop"] == 1

    def test_an_unrecorded_proxy_name_fails_open(self, table):
        reading = read(
            table,
            self._proxy_query("ghost_wrapper", status=Attribution.FAILED),
            "vllm_ci:elsewhere",
        )
        assert "vllm_ci:elsewhere" in reading.kept
        assert not reading.dropped
        assert reading.reasons["query-cannot-answer"] == 1

    def test_a_proxy_never_adds(self, table):
        keys = RowKeys(
            {"vllm_ci"}, {"vllm_ci": 1.0}, steps={s: FakeStep() for s in ALL_STEPS}
        )

        def run(query):
            return read_pr(
                table,
                result_for("vllm_ci:elsewhere", paths=("vllm/mod.py",)),
                query,
                unknown_names(query, UNION, {}),
                KNOWN,
                keys,
            )

        control = run(query_for("vllm/mod.py", "plain"))
        assert "vllm_ci:runs-plain" in control.added  # the detection floor
        proxied = run(self._proxy_query("plain"))
        assert "vllm_ci:runs-plain" not in proxied.added
