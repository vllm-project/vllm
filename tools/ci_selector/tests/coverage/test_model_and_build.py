# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What the merger has to get right, and what it must never quietly get wrong.

Most of these encode something the raw recordings taught us rather than
something the design assumed. Where that is so, the docstring says which.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
from pathlib import Path

import pytest
from ci_selector.coverage.build import merge_build, merge_builds, union_rows
from ci_selector.coverage.model import (
    STAMP_SHAPE,
    TABLE_VERSION,
    Stamp,
    read_process,
    row_key,
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
        assert rows["lora"].stamp.shards_seen == {"1": 2}
        assert rows["lora"].stamp.shards_complete

    def test_a_missing_shard_shows_as_short(self, build: Build, tmp_repo: Repo):
        d = build.job("j0", step_key="lora", parallel_index=0, parallel_total=3)
        process_file(d / "fn.a.txt", [("mod.py", "plain")])
        build.job(
            "j1", step_key="lora", parallel_index=1, parallel_total=3, recorded=False
        )
        rows = merge_build(build.finish(), tmp_repo.root)

        assert rows["lora"].stamp.shards_seen == {"1": 1}
        assert rows["lora"].stamp.shards_expected == {"1": 3}
        assert not rows["lora"].stamp.shards_complete

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

    RECOMPUTED = frozenset({"n_files", "n_functions", "digest"})

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


SWEEPS = Path(__file__).resolve().parents[2] / "covspike" / "sweeps"


@pytest.mark.skipif(not SWEEPS.is_dir(), reason="raw sweeps not present")
class TestAgainstRealRecordings:
    """The cross-build path on real bytes, by splitting one sharded step's five
    shards into two pretend builds. Union of the halves must equal the whole."""

    def test_kernels_moe_shards_split_across_builds_rejoin(self, tmp_path: Path):
        source = SWEEPS / "vllm-ci-82772"
        index = json.loads((source / "index.json").read_text())
        shards = [j for j in index["jobs"] if j.get("step_key") == "kernels-moe-test"]
        assert len(shards) == 5

        tmp_repo = Path(__file__).resolve().parents[4]
        whole = _stage(tmp_path / "whole", "1", index["commit"], shards, source)
        left = _stage(tmp_path / "left", "1", index["commit"], shards[:2], source)
        right = _stage(tmp_path / "right", "2", index["commit"], shards[2:], source)

        one = merge_build(whole, tmp_repo)["kernels-moe-test"]
        split = merge_builds([left, right], tmp_repo)["kernels-moe-test"]

        assert split.functions == one.functions
        assert split.stamp.n_functions == one.stamp.n_functions
        assert sorted(split.stamp.jobs) == sorted(one.stamp.jobs)
        assert split.stamp.builds == ["1", "2"]


def _stage(
    root: Path, number: str, commit: str, jobs: list[dict], source: Path
) -> Path:
    """A build directory pointing at real recordings, without copying 2GB."""
    import os

    build = Build(root, number, commit)
    for job in jobs:
        target = build.job(
            job["job"],
            step_key=job["step_key"],
            label=job["label"],
            parallel_index=job["parallel_index"],
            parallel_total=job["parallel_total"],
        )
        for src in (source / "jobs" / job["job"] / "fnrec").glob("fn.*.txt"):
            os.symlink(src, target / src.name)
    return build.finish()
