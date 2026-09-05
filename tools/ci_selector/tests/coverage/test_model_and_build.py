# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What the merger has to get right, and what it must never quietly get wrong.

Most of these encode something the raw recordings taught us rather than
something the design assumed. Where that is so, the docstring says which.
"""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import inspect
from pathlib import Path

import pytest
from ci_selector.coverage.model import (
    STAMP_SHAPE,
    TABLE_VERSION,
    Row,
    Stamp,
    UnresolvableCommit,
    read_process,
    row_key,
)
from ci_selector.scripts.build import (
    BuildCensus,
    merge_build,
    merge_builds,
    union_rows,
)

from .helpers import ROOT, Build, Repo, process_file


class TestRowKey:
    def test_step_key_wins(self):
        assert row_key({"step_key": "lora", "label": "LoRA 1"}) == ("lora", True)

    def test_keyless_shard_suffix_is_stripped(self):
        # The suffix exists only because the yaml label says %N. Buildkite does
        # not add it, so it cannot be relied on, only recognised.
        key, keyed = row_key(
            {
                "step_key": None,
                "label": "CPU-Multi-Modal Model Tests 3",
                "parallel_index": 2,
                "parallel_total": 4,
            }
        )
        assert (key, keyed) == ("CPU-Multi-Modal Model Tests", False)

    def test_keyless_shards_without_a_suffix_already_agree(self):
        # Kernels Core Operation Test ships three shards under one identical
        # label, because its yaml omits %N.
        keys = {
            row_key(
                {
                    "step_key": None,
                    "label": "Kernels Core Operation Test",
                    "parallel_index": i,
                    "parallel_total": 3,
                }
            )[0]
            for i in range(3)
        }
        assert keys == {"Kernels Core Operation Test"}

    def test_a_trailing_number_that_is_not_this_shard_is_left_alone(self):
        assert (
            row_key(
                {
                    "step_key": None,
                    "label": "Model Tests 4",
                    "parallel_index": 0,
                    "parallel_total": 2,
                }
            )[0]
            == "Model Tests 4"
        )

    def test_unsharded_label_keeps_its_trailing_number(self):
        assert (
            row_key({"step_key": None, "label": "Distributed 2"})[0] == "Distributed 2"
        )

    def test_mirror_keeps_its_own_identity(self):
        # The invariant: NVIDIA evidence must never clear an AMD job. The mirror
        # label differs, so this holds; the test exists so it keeps holding.
        parent = row_key(
            {
                "step_key": None,
                "label": "Kernels MoE Test 1",
                "parallel_index": 0,
                "parallel_total": 5,
            }
        )[0]
        mirror = row_key(
            {
                "step_key": None,
                "label": "AMD: Kernels MoE Test 1 (mi300_1)",
                "parallel_index": 0,
                "parallel_total": 5,
            }
        )[0]
        assert parent != mirror


class TestReadProcess:
    def test_data_after_end_is_still_recorded(self, tmp_path: Path):
        # 1,768 of 8,546 real processes write teardown entries after #end,
        # because _end is idempotent and interpreter shutdown keeps entering
        # vLLM functions. Stopping there would discard exactly __del__ and close.
        p = tmp_path / "fn.a.txt"
        process_file(p, [("mod.py", "plain")], after_end=[("mod.py", "Holder.method")])
        record = read_process(p)
        assert record.functions["vllm/mod.py"] == {"plain", "Holder.method"}
        assert record.data_lines == 2

    def test_post_end_data_is_not_read_as_loss(self, tmp_path: Path):
        p = tmp_path / "fn.a.txt"
        process_file(p, [("mod.py", "plain")], after_end=[("mod.py", "Holder.method")])
        assert not read_process(p).lost_lines

    def test_missing_end_is_not_damage(self, tmp_path: Path):
        # About 30% of processes are killed by design. Reading that as damage
        # would condemn nearly every row.
        p = tmp_path / "fn.a.txt"
        process_file(p, [("mod.py", "plain")], clean_exit=False, counter=1)
        record = read_process(p)
        assert not record.clean_exit and not record.lost_lines

    def test_fewer_lines_than_the_counter_is_loss(self, tmp_path: Path):
        # The positive control. Across all 8,546 real process files this check
        # never fires, so without a synthetic case it would be untested rather
        # than validated.
        p = tmp_path / "fn.a.txt"
        process_file(p, [("mod.py", "plain")], counter=9)
        assert read_process(p).lost_lines

    def test_root_comes_from_the_root_line_not_the_header(self, tmp_path: Path):
        # The header is written before the root is knowable and carries an
        # env-derived guess. One real process has an empty header root and a
        # #root of /vllm-workspace/vllm/.
        p = tmp_path / "fn.a.txt"
        process_file(
            p, [("mod.py", "plain")], root="/vllm-workspace/vllm/", header_root=""
        )
        record = read_process(p)
        assert record.root == "/vllm-workspace/vllm/"
        assert record.functions == {"vllm/mod.py": {"plain"}}

    def test_lines_outside_the_root_are_counted_not_silently_skipped(
        self, tmp_path: Path
    ):
        p = tmp_path / "fn.a.txt"
        p.write_text(
            f"#start\troot={ROOT}\tpy=3.12.13\n"
            f"#root\t{ROOT}\tt=1\n"
            f"/elsewhere/other.py\tthing\t1\n"
            f"{ROOT}mod.py\tplain\t1\n"
        )
        record = read_process(p)
        assert record.outside_root == 1
        assert record.functions == {"vllm/mod.py": {"plain"}}

    def test_a_process_with_no_resolvable_root_is_dropped(self, tmp_path: Path):
        p = tmp_path / "fn.a.txt"
        p.write_text("#start\tpid=1\n")
        assert read_process(p) is None


@pytest.fixture
def build(tmp_path: Path, tmp_repo: Repo) -> Build:
    return Build(tmp_path / "sweep" / "b1", "1", tmp_repo.head())


class TestMergeBuild:
    def test_processes_and_shards_fold_into_one_row(self, build: Build, tmp_repo: Repo):
        for index in range(2):
            d = build.job(
                f"j{index}",
                step_key="lora",
                label=f"LoRA {index + 1}",
                parallel_index=index,
                parallel_total=2,
            )
            process_file(d / "fn.a.txt", [("mod.py", "plain")])
            process_file(d / "fn.b.txt", [("mod.py", "Holder.method")])
        rows = merge_build(build.finish(), tmp_repo.root)

        assert set(rows) == {"lora"}
        assert rows["lora"].functions == {
            "vllm/mod.py": frozenset({"plain", "Holder.method"})
        }
        assert rows["lora"].stamp.processes == 4
        assert rows["lora"].stamp.shards_seen == {"1": [0, 1]}
        assert rows["lora"].stamp.shards_complete

    def test_a_missing_shard_shows_as_short(self, build: Build, tmp_repo: Repo):
        d = build.job("j0", step_key="lora", parallel_index=0, parallel_total=3)
        process_file(d / "fn.a.txt", [("mod.py", "plain")])
        build.job(
            "j1", step_key="lora", parallel_index=1, parallel_total=3, recorded=False
        )
        rows = merge_build(build.finish(), tmp_repo.root)

        assert rows["lora"].stamp.shards_seen == {"1": [0]}
        assert rows["lora"].stamp.shards_expected == {"1": 3}
        assert not rows["lora"].stamp.shards_complete

    def test_a_retried_shard_does_not_stand_in_for_a_missing_one(
        self, build: Build, tmp_repo: Repo
    ):
        # The live route is a manual retry: Buildkite returns both attempts, and
        # counting jobs read two attempts of shard 0 as "both shards recorded",
        # so a function only shard 1 would have run became droppable.
        for job_id in ("j0", "j0-retry"):
            d = build.job(job_id, step_key="lora", parallel_index=0, parallel_total=2)
            process_file(d / "fn.a.txt", [("mod.py", "plain")])
        rows = merge_build(build.finish(), tmp_repo.root)

        assert rows["lora"].stamp.shards_seen == {"1": [0]}
        assert not rows["lora"].stamp.shards_complete

    def test_a_job_that_did_not_pass_disqualifies_the_row(
        self, build: Build, tmp_repo: Repo
    ):
        # Nothing held this in place before: deleting the check left the suite
        # green.
        d = build.job("j0", step_key="lora", state="broken", exit_status=1)
        process_file(d / "fn.a.txt", [("mod.py", "plain")])
        rows = merge_build(build.finish(), tmp_repo.root)

        assert rows["lora"].stamp.failed_jobs == ["j0"]

    def test_a_bad_exit_status_disqualifies_even_when_state_says_passed(
        self, build: Build, tmp_repo: Repo
    ):
        # Two fields, so neither one being wrong, absent or newly spelled by
        # Buildkite can promote a partial recording on its own.
        d = build.job("j0", step_key="lora", state="passed", exit_status=2)
        process_file(d / "fn.a.txt", [("mod.py", "plain")])
        rows = merge_build(build.finish(), tmp_repo.root)

        assert rows["lora"].stamp.failed_jobs == ["j0"]

    def test_a_healthy_job_is_not_disqualified(self, build: Build, tmp_repo: Repo):
        # The floor for the two above, so they cannot pass by condemning
        # everything.
        d = build.job("j0", step_key="lora")
        process_file(d / "fn.a.txt", [("mod.py", "plain")])
        rows = merge_build(build.finish(), tmp_repo.root)

        assert rows["lora"].stamp.failed_jobs == []
        assert not rows["lora"].stamp.thin

    def test_an_unreadable_log_makes_the_row_thin(self, build: Build, tmp_repo: Repo):
        # Without this the log-derived health counters are simply absent, and
        # absent reads identically to healthy.
        d = build.job("j0", step_key="lora", log=None)
        process_file(d / "fn.a.txt", [("mod.py", "plain")])
        rows = merge_build(build.finish(), tmp_repo.root)

        assert rows["lora"].stamp.logs_unreadable == 1
        assert rows["lora"].stamp.thin

    def test_a_truncated_log_does_not_kill_the_merge(
        self, build: Build, tmp_repo: Repo
    ):
        # A partially uploaded .gz raises EOFError, not an OSError, so one bad
        # log used to take down the whole sweep.
        d = build.job("j0", step_key="lora")
        (d.parent / "job.log.gz").write_bytes(
            gzip.compress(b"collected 3 items\n")[:12]
        )
        process_file(d / "fn.a.txt", [("mod.py", "plain")])
        rows = merge_build(build.finish(), tmp_repo.root)

        assert rows["lora"].stamp.logs_unreadable == 1

    def test_a_commit_the_repo_cannot_see_is_fatal(self, build: Build, tmp_repo: Repo):
        # Not tolerated, because every symptom is a lie: each file reads as
        # absent at that commit, so the build merges to nothing while still
        # lending a unioned row its completeness and its vintage.
        build.commit = "0" * 40
        d = build.job("j0", step_key="lora")
        process_file(d / "fn.a.txt", [("mod.py", "plain")])
        with pytest.raises(UnresolvableCommit):
            merge_build(build.finish(), tmp_repo.root)

    def test_an_unread_world_is_counted_not_silently_empty(
        self, build: Build, tmp_repo: Repo
    ):
        # No build.json, so build_env is unknown. That has to be told apart
        # from a build where every variable was genuinely unset, since
        # "no NIGHTLY" separates a hand-triggered sweep from a production one.
        d = build.job("j0", step_key="lora")
        process_file(d / "fn.a.txt", [("mod.py", "plain")])
        rows = merge_build(build.finish(), tmp_repo.root)

        assert rows["lora"].stamp.worlds_unread == 1
        assert rows["lora"].stamp.build_env == {}

    def test_a_step_that_enters_no_vllm_code_still_gets_a_row(
        self, build: Build, tmp_repo: Repo
    ):
        # torch-stable-abi-audit is real: clean exit, root=0, 566 functions
        # entered, none of them ours. A complete and empty row is the most
        # dangerous shape there is, so it must be visible rather than absent.
        d = build.job("j0", step_key="abi-audit")
        process_file(d / "fn.a.txt", [])
        rows = merge_build(build.finish(), tmp_repo.root)

        assert "abi-audit" in rows
        assert not rows["abi-audit"].stamp.has_evidence

    def test_a_job_with_no_recording_gets_no_row(self, build: Build, tmp_repo: Repo):
        build.job("j0", step_key="never-instrumented", recorded=False)
        assert merge_build(build.finish(), tmp_repo.root) == {}

    def test_a_build_that_recorded_nothing_is_counted_not_silent(
        self, build: Build, tmp_repo: Repo
    ):
        """A build can lose most of its jobs and still merge into a table that
        looks structurally fine. The skip is per job and was two bare
        `continue`s, so nothing held the denominator."""
        for i in range(4):
            build.job(f"j{i}", step_key=f"step-{i}", recorded=False)
        census = BuildCensus()
        assert merge_build(build.finish(), tmp_repo.root, census=census) == {}

        assert (census.recorded, census.attempted) == (0, 4)
        assert census.rate == 0.0
        assert census.collapsed
        assert len(census.no_dir) == 4

    def test_the_three_no_recording_causes_are_told_apart(
        self, build: Build, tmp_repo: Repo
    ):
        """They mean different things: nothing delivered, a failed install, and
        our own parser. Lumping them together is how the second one hid."""
        build.job("j0", step_key="nothing-delivered", recorded=False)

        installer_failed = build.job("j1", step_key="install-failed")
        (installer_failed / "install.err").write_text("boom\n")

        no_root = build.job("j2", step_key="no-root")
        (no_root / "fn.a.txt").write_text("#start\tpid=1\n")

        good = build.job("j3", step_key="fine")
        process_file(good / "fn.a.txt", [("mod.py", "plain")])

        census = BuildCensus()
        rows = merge_build(build.finish(), tmp_repo.root, census=census)

        assert set(rows) == {"fine"}
        assert census.recorded == 1
        assert len(census.no_dir) == 1
        assert len(census.no_record_files) == 1
        assert len(census.no_resolvable_root) == 1

    def test_a_job_that_never_started_is_not_counted_against_the_build(
        self, build: Build, tmp_repo: Repo
    ):
        """A blocked job could not have recorded, so counting it would
        understate the rate."""
        good = build.job("j0", step_key="fine")
        process_file(good / "fn.a.txt", [("mod.py", "plain")])
        build.job(
            "j1", step_key="blocked", recorded=False, started=False, state="blocked"
        )

        census = BuildCensus()
        merge_build(build.finish(), tmp_repo.root, census=census)

        assert (census.recorded, census.attempted) == (1, 1)
        assert not census.collapsed

    def test_reading_the_index_after_the_shard_loop_still_works(
        self, build: Build, tmp_repo: Repo
    ):
        """`index` used to be rebound to an int by the shard accounting, so any
        post-loop read raised TypeError on every sharded build -- which no
        fixture without shards would catch."""
        for i in range(2):
            d = build.job(f"j{i}", step_key="lora", parallel_index=i, parallel_total=2)
            process_file(d / "fn.a.txt", [("mod.py", "plain")])
        census = BuildCensus()
        merge_build(build.finish(), tmp_repo.root, census=census)
        assert census.n_jobs == 2

    def test_the_census_is_not_a_stamp_field(self, build: Build, tmp_repo: Repo):
        """It is never stored, which is why no TABLE_VERSION bump was needed.

        A version pin exists to stop an older stored table reading healthier
        than recorded, since `load` fills a missing field with its healthy
        default. Nothing here is stored, so no older table can be missing it.
        """
        names = {f.name for f in dataclasses.fields(Stamp)}
        assert not names & {f.name for f in dataclasses.fields(BuildCensus)}

    def test_the_breadth_index_is_derived_and_never_stored(
        self, tmp_path: Path, tmp_repo: Repo
    ):
        """Same argument as the census, for `Table._breadth`.

        Recomputed from rows already stored, so no older table can be missing it.
        Three things would invert that: writing it in `write_table`, making it a
        `Stamp` field, or feeding it into the digest. This pins the first two;
        the third has no path, since `digest_of` signs functions and stamp
        fields only.
        """
        from ci_selector.coverage.table import Table

        assert "_breadth" not in {f.name for f in dataclasses.fields(Stamp)}
        row = Row(
            key="a",
            keyed=True,
            functions={"vllm/mod.py": frozenset({"plain"})},
            stamp=Stamp(),
        )
        assert Table({"a": row})._breadth == {"vllm/mod.py": {"plain": 1}}
        # An empty table has no share to read, so the gate cannot fire on it.
        assert Table({}).discriminates("vllm/mod.py", "plain")

    def test_files_absent_at_the_commit_are_dropped(self, build: Build, tmp_repo: Repo):
        # _version.py and third_party/** are generated at build time, so no diff
        # can ever name them: 168 of 2,205 files in the real sweeps.
        d = build.job("j0", step_key="lora")
        process_file(d / "fn.a.txt", [("mod.py", "plain"), ("_version.py", "<module>")])
        rows = merge_build(build.finish(), tmp_repo.root)

        assert set(rows["lora"].functions) == {"vllm/mod.py"}
        assert rows["lora"].stamp.dropped_absent_files == 1

    def test_a_name_the_source_cannot_produce_flags_the_file(
        self, build: Build, tmp_repo: Repo
    ):
        # The cute-dsl shape: the row holds `kernel` where the source says
        # `Sm100ChunkOKernel.kernel`, so matching on names there is meaningless.
        d = build.job("j0", step_key="kernels")
        process_file(d / "fn.a.txt", [("mod.py", "plain"), ("mod.py", "if_region_0")])
        rows = merge_build(build.finish(), tmp_repo.root)

        assert rows["kernels"].stamp.unfaithful_files == ["vllm/mod.py"]

    def test_a_faithful_file_is_not_flagged(self, build: Build, tmp_repo: Repo):
        d = build.job("j0", step_key="kernels")
        process_file(d / "fn.a.txt", [("mod.py", "plain"), ("mod.py", "Holder.method")])
        rows = merge_build(build.finish(), tmp_repo.root)

        assert rows["kernels"].stamp.unfaithful_files == []


class TestUnionCoversEveryStampField:
    """`union_rows` rebuilds `Stamp` field by field, and nothing else notices.

    Coverage is complete today, which is what makes it dangerous: a new field
    left out reverts to its default for rows drawn from more than one build
    **and only those**, so a single-build table looks correct while a merged one
    quietly loses it. `_verify` cannot catch it either, because it recomputes
    the counts and the digest from whatever it was handed.
    """

    RECOMPUTED = frozenset({"n_files", "n_functions", "n_import_time", "digest"})

    def test_union_rows_carries_every_field(self):
        source = inspect.getsource(union_rows)
        missing = [
            f.name
            for f in dataclasses.fields(Stamp)
            if f.name not in self.RECOMPUTED and f"{f.name}=" not in source
        ]
        assert not missing, (
            f"union_rows does not carry {missing}; a row from two builds would "
            "silently take the default"
        )

    def test_the_oracle_has_not_gone_empty(self):
        # Guard the guard: a renamed or emptied Stamp passes the test above
        # vacuously.
        assert len(dataclasses.fields(Stamp)) > 20


class TestStampShapeIsPinned:
    """Changing `Stamp` without bumping `TABLE_VERSION` is the one edit that
    breaks old tables silently.

    `load` fills any field an older table lacks with its default, and every
    default here is the healthy value, so the old table reads as pristine and
    authorizes drops it never earned. The version guard already stops that, but
    only for someone who remembers to bump it. This makes forgetting loud.
    """

    @staticmethod
    def shape() -> str:
        parts = []
        for f in dataclasses.fields(Stamp):
            if f.default is not dataclasses.MISSING:
                default = repr(f.default)
            elif f.default_factory is not dataclasses.MISSING:
                default = f.default_factory.__name__
            else:
                default = "-"
            parts.append(f"{f.name}:{f.type}:{default}")
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    def test_the_stamp_matches_its_pinned_shape(self):
        actual = self.shape()
        assert actual == STAMP_SHAPE, (
            f"\n\nStamp's fields changed (shape {actual}, pinned {STAMP_SHAPE}).\n\n"
            "A table written before this change is missing the field, `load` "
            "gives it the healthy default, and that row then authorizes drops "
            "it never earned.\n\n"
            "Fix, both in this commit:\n"
            "  - bump TABLE_VERSION in ci_selector/coverage/model.py "
            f"(now {TABLE_VERSION})\n"
            f"  - set STAMP_SHAPE beside it to {actual!r}\n"
        )

    def test_the_shape_is_not_vacuous(self):
        # Guard the guard: an emptied or renamed Stamp must not hash to
        # something the pin happens to accept.
        assert len(dataclasses.fields(Stamp)) > 20
        assert self.shape() != hashlib.sha256(b"").hexdigest()[:16]


class TestCrossBuild:
    def test_the_same_step_in_two_builds_unions(self, tmp_path: Path, tmp_repo: Repo):
        # The two real sweeps share no step key, so nothing has ever exercised
        # this path against live data. It ships tested anyway.
        commit = tmp_repo.head()
        first = Build(tmp_path / "s" / "b1", "1", commit)
        process_file(
            first.job("j0", step_key="lora") / "fn.a.txt", [("mod.py", "plain")]
        )
        second = Build(tmp_path / "s" / "b2", "2", commit)
        process_file(
            second.job("j1", step_key="lora") / "fn.a.txt",
            [("mod.py", "Holder.method")],
        )

        rows = merge_builds([first.finish(), second.finish()], tmp_repo.root)
        row = rows["lora"]
        assert row.functions == {"vllm/mod.py": frozenset({"plain", "Holder.method"})}
        assert row.stamp.builds == ["1", "2"]
        assert sorted(row.stamp.jobs) == ["j0", "j1"]
        assert row.stamp.n_functions == 2

    def test_union_refuses_to_mix_two_different_steps(
        self, tmp_path: Path, tmp_repo: Repo
    ):
        commit = tmp_repo.head()
        b = Build(tmp_path / "s" / "b1", "1", commit)
        process_file(b.job("j0", step_key="a") / "fn.a.txt", [("mod.py", "plain")])
        process_file(b.job("j1", step_key="b") / "fn.a.txt", [("mod.py", "plain")])
        rows = merge_build(b.finish(), tmp_repo.root)
        with pytest.raises(ValueError):
            union_rows(rows["a"], rows["b"])


# The cross-build path, synthetically. It replaced a test that read real
# recordings from a git-excluded directory, which made it unrunnable: its path
# was wrong for months and the skip read as normal on a machine without the
# data. It also earned little, since both sides of its comparison read the SAME
# bytes, so a parse bug cancels out. This asserts strictly more, adding the
# shard bookkeeping and the absent-file asymmetry the original ran past.
SHARDS = {
    0: [("mod.py", "plain")],
    1: [("mod.py", "plain"), ("gone.py", "<module>")],
    2: [("mod.py", "Holder.method"), ("gone.py", "<module>")],
    3: [("other.py", "elsewhere")],
    4: [("mod.py", "Holder.method")],
}


def _sharded(root: Path, number: str, commit: str, indexes) -> Path:
    """One build carrying a chosen subset of a five-shard step's jobs."""
    build = Build(root, number, commit)
    for i in indexes:
        d = build.job(
            f"j{i}",
            step_key="moe",
            label=f"Kernels MoE Test {i + 1}",
            parallel_index=i,
            parallel_total=5,
        )
        process_file(d / "fn.a.txt", SHARDS[i])
        process_file(d / "fn.b.txt", [("other.py", "elsewhere")])
    return build.finish()


class TestCrossBuildShards:
    """Splitting a sharded step across two builds must lose nothing.

    Shard contents are deliberately asymmetric across the split, `plain` only
    on the left and `Holder.method` only on the right, so a merge that replaces
    instead of unioning cannot pass by coincidence.
    """

    def test_union_of_halves_equals_the_whole(self, tmp_path: Path, tmp_repo: Repo):
        commit = tmp_repo.head()
        whole = _sharded(tmp_path / "w", "1", commit, range(5))
        left = _sharded(tmp_path / "l", "1", commit, [0, 1])
        right = _sharded(tmp_path / "r", "2", commit, [2, 3, 4])

        one = merge_build(whole, tmp_repo.root)["moe"]
        split = merge_builds([left, right], tmp_repo.root)["moe"]

        assert split.functions == one.functions
        assert split.functions == {
            "vllm/mod.py": frozenset({"plain", "Holder.method"}),
            "vllm/other.py": frozenset({"elsewhere"}),
        }
        assert split.stamp.n_functions == one.stamp.n_functions == 3
        assert sorted(split.stamp.jobs) == sorted(one.stamp.jobs)
        assert split.stamp.builds == ["1", "2"]
        assert split.stamp.processes == one.stamp.processes == 10

    def test_the_split_is_visibly_short_of_shards(self, tmp_path: Path, tmp_repo: Repo):
        """Where "union equals the whole" stops being true, and it stops on the
        fields that authorize a drop. Each half declares five shards and holds
        two or three, so neither is complete and the merged row is THIN even
        though its functions match the whole exactly."""
        commit = tmp_repo.head()
        whole = _sharded(tmp_path / "w", "1", commit, range(5))
        left = _sharded(tmp_path / "l", "1", commit, [0, 1])
        right = _sharded(tmp_path / "r", "2", commit, [2, 3, 4])

        one = merge_build(whole, tmp_repo.root)["moe"]
        split = merge_builds([left, right], tmp_repo.root)["moe"]

        assert one.stamp.shards_seen == {"1": [0, 1, 2, 3, 4]}
        assert split.stamp.shards_seen == {"1": [0, 1], "2": [2, 3, 4]}
        # Both builds must carry the expectation. Keeping only the left one
        # survives the whole suite otherwise.
        assert split.stamp.shards_expected == {"1": 5, "2": 5}
        assert one.stamp.shards_complete and not split.stamp.shards_complete
        assert not one.stamp.thin and split.stamp.thin

    def test_a_file_absent_at_the_commit_is_counted_once_per_build(
        self, tmp_path: Path, tmp_repo: Repo
    ):
        """`gone.py` is recorded but never existed, so each build drops it. The
        count is per build and does not deduplicate, which is why the whole
        reports one and the split two. Pinned as a known asymmetry."""
        commit = tmp_repo.head()
        whole = _sharded(tmp_path / "w", "1", commit, range(5))
        left = _sharded(tmp_path / "l", "1", commit, [0, 1])
        right = _sharded(tmp_path / "r", "2", commit, [2, 3, 4])

        one = merge_build(whole, tmp_repo.root)["moe"]
        split = merge_builds([left, right], tmp_repo.root)["moe"]
        assert one.stamp.dropped_absent_files == 1
        assert split.stamp.dropped_absent_files == 2

    def test_two_directories_reporting_one_build_keep_both_shard_sets(
        self, tmp_path: Path, tmp_repo: Repo
    ):
        """What `_union_shards` exists for. A plain dict merge lets the second
        row's indexes replace the first's, and the row then reads complete on
        half the shards."""
        commit = tmp_repo.head()
        a = _sharded(tmp_path / "a", "1", commit, [0, 1, 2])
        b = _sharded(tmp_path / "b", "1", commit, [3, 4])
        row = merge_builds([a, b], tmp_repo.root)["moe"]
        assert row.stamp.shards_seen == {"1": [0, 1, 2, 3, 4]}
        assert row.stamp.shards_complete
