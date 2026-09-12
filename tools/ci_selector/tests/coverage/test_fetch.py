# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The downloader had no tests, which is most of why a build that delivered
nothing still exited 0.

Three delivery shapes have to normalise to one job directory, and the floor has
to fire on the shape that actually happened without firing on the worst healthy
sweep. Both are pinned here against measured numbers, not invented ones.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest
from ci_selector.scripts.fetch import (
    EXIT_FLOOR_BREACHED,
    audit,
    classify_artifact,
    collect_job,
    group_artifacts,
    place_raw,
    unpack,
)

JOB = "01a038f6-9638-43b5-89e8-ddbb6cfe743f"
OTHER = "01a038f6-0000-0000-0000-000000000000"
RECORD = "#start\tpid=1\n#root\t/usr/lib/vllm/\tt=1\n"


def art(path, job_id=JOB, url=None):
    return {"path": path, "job_id": job_id, "download_url": url or f"https://x/{path}"}


def tarball(names):
    """A .tar.gz holding `names`, as the packer produces it."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name in names:
            body = RECORD.encode()
            info = tarfile.TarInfo(name=name)
            info.size = len(body)
            tf.addfile(info, io.BytesIO(body))
    return buf.getvalue()


class FakeTransport:
    """A dict instead of a socket. Records what was asked for."""

    def __init__(self, build=None, artifacts=(), blobs=None, log=b"ok"):
        self._build = build or {}
        self._artifacts = list(artifacts)
        self._blobs = blobs or {}
        self._log = log
        self.requested: list[str] = []

    def get_json(self, url):
        return self._build

    def paged(self, url, per_page=100):
        yield from self._artifacts

    def get_bytes(self, url, accept="application/json"):
        self.requested.append(url)
        if url.endswith("/log"):
            return self._log
        return self._blobs.get(url)


def _ctx(tmp_path, transport):
    return {
        "base": "https://api/builds/1",
        "transport": transport,
        "jobs_dir": str(tmp_path / "jobs"),
        "build_number": "1",
        "commit": "abc123",
        "logs": "none",
        "force": False,
    }


class TestMatching:
    def test_the_legacy_root_tarball_still_matches(self):
        """Every stored sweep is in this shape, and they are the evidence behind
        every number we quote. A rewrite that stops reading them loses all of it
        silently."""
        assert classify_artifact(art(f"{JOB}.tar.gz")) == (JOB, "tar-legacy")

    def test_the_packed_tarball_under_fnrec_matches(self):
        assert classify_artifact(art(f".fnrec/{JOB}.tar.gz")) == (JOB, "tar")

    def test_raw_files_match_and_name_their_owner(self):
        assert classify_artifact(art(f".fnrec/{JOB}/fn.host.abc.1.txt")) == (JOB, "raw")
        assert classify_artifact(art(f".fnrec/{JOB}/install.err")) == (JOB, "raw")

    @pytest.mark.parametrize(
        "path",
        [
            "dist/wheel.tar.gz",
            ".fnrec",
            ".fnrec/",
            f".fnrec/{JOB}/nested/deeper.txt",
            "artifacts/amd-gpu-diagnostics/x/diagnostics.log",
        ],
    )
    def test_an_unrelated_artifact_is_ignored(self, path):
        assert classify_artifact(art(path)) is None

    def test_raw_files_group_under_one_job(self):
        arts = [art(f".fnrec/{JOB}/fn.host.abc.{pid}.txt") for pid in range(24)]
        grouped, foreign = group_artifacts(arts)
        assert set(grouped) == {JOB}
        assert len(grouped[JOB]) == 24
        assert foreign == 0

    def test_a_tarball_wins_over_raw_files_for_the_same_job(self):
        """The tarball is the same bytes, taken before packing deleted them, so
        preferring it turns a 24-download job into one."""
        arts = [art(f".fnrec/{JOB}/fn.host.abc.{p}.txt") for p in range(24)]
        arts.append(art(f".fnrec/{JOB}.tar.gz"))
        grouped, _ = group_artifacts(arts)
        assert len(grouped[JOB]) == 1
        assert grouped[JOB][0]["shape"] == "tar"

    def test_an_artifact_uploaded_by_another_job_is_attributed_by_path(self):
        """The checkout outlives the job, so a directory a crashed job left
        behind can be uploaded by an unrelated one.

        Filing it under the uploader makes the owner read as covering less than
        it does, and a diff touching only those functions would then drop it.
        That is under-selection, from a shape the agent can already produce.
        """
        grouped, foreign = group_artifacts(
            [art(f".fnrec/{JOB}/fn.host.abc.1.txt", job_id=OTHER)]
        )
        assert set(grouped) == {JOB}
        assert foreign == 1

    def test_the_same_file_uploaded_twice_lands_once(self):
        arts = [
            art(f".fnrec/{JOB}/fn.host.abc.1.txt"),
            art(f".fnrec/{JOB}/fn.host.abc.1.txt", job_id=OTHER),
        ]
        grouped, _ = group_artifacts(arts)
        assert len(grouped[JOB]) == 1


class TestPlacement:
    @pytest.mark.parametrize("name", ["../../etc/passwd", "/abs/evil.txt", "a/b/c.txt"])
    def test_a_tarball_member_escaping_the_directory_lands_flat(self, tmp_path, name):
        """The defence existed and was never held by a test. The name comes from
        an uploader we do not control, so only its last component is written."""
        dest = tmp_path / "fnrec"
        written, _ = unpack(io.BytesIO(tarball([name])), str(dest))
        assert written == 1
        assert [p.name for p in dest.iterdir()] == [Path(name).name]
        assert not (tmp_path / "etc").exists()

    @pytest.mark.parametrize("name", ["../../etc/passwd", "/abs/evil.txt"])
    def test_a_raw_artifact_path_escaping_the_directory_lands_flat(
        self, tmp_path, name
    ):
        dest = tmp_path / "fnrec"
        written, _ = place_raw(f".fnrec/{JOB}/{name}", b"x", str(dest))
        assert written == 1
        assert [p.name for p in dest.iterdir()] == [Path(name).name]

    def test_two_members_with_one_basename_are_counted_not_silently_merged(
        self, tmp_path
    ):
        dest = tmp_path / "fnrec"
        written, dups = unpack(
            io.BytesIO(tarball(["a/fn.x.txt", "b/fn.x.txt"])), str(dest)
        )
        assert (written, dups) == (1, 1)


class TestCollect:
    def test_the_three_shapes_produce_the_same_directory(self, tmp_path):
        """The load-bearing assertion: however a job delivered, the merger sees
        one layout. If either half drifts, this fails."""
        names = ["fn.host.abc.1.txt", "fn.host.abc.2.txt"]
        shapes = {
            "tar-legacy": [art(f"{JOB}.tar.gz")],
            "tar": [art(f".fnrec/{JOB}.tar.gz")],
            "raw": [art(f".fnrec/{JOB}/{n}") for n in names],
        }
        trees = {}
        for shape, arts in shapes.items():
            blobs = {
                a["download_url"]: (
                    RECORD.encode()
                    if a["shape"] == "raw"
                    else tarball([f"{JOB}/{n}" for n in names])
                )
                for a in group_artifacts(arts)[0][JOB]
            }
            root = tmp_path / shape
            transport = FakeTransport(blobs=blobs)
            meta = collect_job(
                {"id": JOB, "state": "passed"},
                group_artifacts(arts)[0][JOB],
                _ctx(root, transport),
            )
            fnrec = root / "jobs" / JOB / "fnrec"
            trees[shape] = {p.name: p.read_bytes() for p in sorted(fnrec.iterdir())}
            assert meta["n_records"] == 2

        assert trees["tar"] == trees["raw"] == trees["tar-legacy"]

    def test_meta_names_the_shape_so_a_dead_packer_is_a_grep(self, tmp_path):
        """Delivery working while packing stopped is invisible from every other
        signal, and it is the state the fleet is in until the producer ships."""
        arts = group_artifacts([art(f".fnrec/{JOB}/fn.host.abc.1.txt")])[0][JOB]
        transport = FakeTransport(blobs={arts[0]["download_url"]: RECORD.encode()})
        meta = collect_job({"id": JOB}, arts, _ctx(tmp_path, transport))
        assert meta["artifact"] == "raw"

    def test_install_err_alone_is_not_a_recording(self, tmp_path):
        """Six jobs across the stored sweeps are in this shape: counted as
        recorded by the fetcher and skipped without a word by the merger."""
        arts = group_artifacts([art(f".fnrec/{JOB}/install.err")])[0][JOB]
        transport = FakeTransport(blobs={arts[0]["download_url"]: b"boom\n"})
        meta = collect_job({"id": JOB}, arts, _ctx(tmp_path, transport))
        assert meta["n_files"] == 1
        assert meta["n_records"] == 0
        assert meta["artifact"] == "no records"

    def test_an_empty_tarball_is_not_a_recording(self, tmp_path):
        arts = group_artifacts([art(f".fnrec/{JOB}.tar.gz")])[0][JOB]
        transport = FakeTransport(blobs={arts[0]["download_url"]: tarball([])})
        meta = collect_job({"id": JOB}, arts, _ctx(tmp_path, transport))
        assert meta["artifact"] == "empty tarball"

    def test_a_missing_download_is_absent_not_fatal(self, tmp_path):
        arts = group_artifacts([art(f".fnrec/{JOB}.tar.gz")])[0][JOB]
        meta = collect_job({"id": JOB}, arts, _ctx(tmp_path, FakeTransport()))
        assert meta["artifact"] == "download 404"

    def test_an_existing_meta_short_circuits_the_job(self, tmp_path):
        """The resume contract, previously untested: a finished job is not
        refetched, and --force overrides."""
        arts = group_artifacts([art(f".fnrec/{JOB}.tar.gz")])[0][JOB]
        blobs = {arts[0]["download_url"]: tarball([f"{JOB}/fn.host.abc.1.txt"])}
        ctx = _ctx(tmp_path, FakeTransport(blobs=blobs))
        collect_job({"id": JOB}, arts, ctx)
        before = len(ctx["transport"].requested)

        collect_job({"id": JOB}, arts, ctx)
        assert len(ctx["transport"].requested) == before

        ctx["force"] = True
        collect_job({"id": JOB}, arts, ctx)
        assert len(ctx["transport"].requested) > before

    def test_meta_is_written_last_so_a_broken_job_is_redone(self, tmp_path):
        """Not skipped forever: the sentinel must not exist for a partial job."""

        class Exploding(FakeTransport):
            def get_bytes(self, url, accept="application/json"):
                raise RuntimeError("connection reset")

        arts = group_artifacts([art(f".fnrec/{JOB}.tar.gz")])[0][JOB]
        with pytest.raises(RuntimeError):
            collect_job({"id": JOB}, arts, _ctx(tmp_path, Exploding()))
        assert not (tmp_path / "jobs" / JOB / "meta.json").exists()


def _jobs(started, recorded, shape="tar"):
    """A build's worth of index entries at a given recording rate."""
    out = []
    for i in range(started):
        out.append(
            {
                "job": f"j{i}",
                "state": "passed",
                "step_key": f"step-{i}",
                "started_at": "2026-08-26T00:00:00Z",
                "n_records": 1 if i < recorded else 0,
                "artifact": shape if i < recorded else "no artifact uploaded",
            }
        )
    return out


class TestFloor:
    def test_the_real_85489_shape_breaches(self):
        """87 of 340 started jobs, measured. The collapse this whole change
        exists to make impossible to miss."""
        lines, code, rec, att = audit(_jobs(340, 87))
        assert (rec, att) == (87, 340)
        assert code == EXIT_FLOOR_BREACHED
        assert "delivery failure" in lines[0]

    def test_the_real_82772_shape_clears(self):
        """The floor for the floor. 165 of 248 is the worst HEALTHY sweep, so a
        threshold tuned only against the collapse would fail every good build."""
        _, code, _, _ = audit(_jobs(248, 165))
        assert code == 0

    def test_blocked_jobs_are_not_in_the_denominator(self):
        """A job that never started could not have recorded. 85489 had 27 of
        these, and counting them would understate an already bad rate."""
        jobs = _jobs(10, 10) + [
            {"job": "b", "state": "blocked", "started_at": None, "n_records": 0}
            for _ in range(27)
        ]
        _, code, rec, att = audit(jobs)
        assert (rec, att, code) == (10, 10, 0)

    def test_zero_recorded_is_a_breach_not_a_clean_run(self):
        """`0/369` exited 0 before this."""
        lines, code, _, _ = audit(_jobs(369, 0))
        assert code == EXIT_FLOOR_BREACHED
        assert "NOTHING recorded" in lines[0]

    def test_a_build_where_nothing_started_is_a_breach(self):
        lines, code, _, _ = audit(
            [{"job": "b", "state": "blocked", "started_at": None, "n_records": 0}]
        )
        assert code == EXIT_FLOOR_BREACHED

    def test_a_raw_delivery_says_the_packer_stopped(self):
        """Delivery fine, packing dead. The only other symptom is a slow fetch."""
        lines, code, _, _ = audit(_jobs(10, 10, shape="raw"))
        assert code == 0
        assert any("packing step is not running" in line for line in lines)

    def test_a_legacy_delivery_says_the_producer_is_stale(self):
        lines, _, _, _ = audit(_jobs(10, 10, shape="tar-legacy"))
        assert any("legacy in-container upload" in line for line in lines)

    def test_an_index_without_n_records_falls_back_to_n_files(self):
        """Absent means unknown, not zero. Indexes written before the shape
        vocabulary existed must keep reading."""
        old = [
            {"job": "j", "state": "passed", "started_at": "t", "n_files": 3}
            for _ in range(4)
        ]
        _, code, rec, _ = audit(old)
        assert (rec, code) == (4, 0)
