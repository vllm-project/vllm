from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPT = Path(__file__).parents[1] / "scripts" / "rl_source_snapshot.py"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main", str(repo)],
        check=True,
        capture_output=True,
    )
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / "tracked.py").write_text("value = 1\n")
    _git(repo, "add", "tracked.py")
    _git(repo, "commit", "-m", "base")
    return repo, _git(repo, "rev-parse", "HEAD")


def _stack(path: Path, base: str, patches: list[dict] | None = None) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "base": {
                    "source_commit": base,
                    "image_digest": "sha256:test",
                },
                "patches": patches or [],
            },
            sort_keys=False,
        )
    )
    return path


def _run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def test_prepare_verify_and_cleanup_preserve_dirty_source(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path)
    stack = _stack(tmp_path / "stack.yaml", base)
    (repo / "tracked.py").write_text("value = 2\n")
    result = tmp_path / "result"
    worktrees = tmp_path / "worktrees"

    prepared = _run(
        "prepare",
        "--repo",
        str(repo),
        "--run-id",
        "run-1",
        "--result-dir",
        str(result),
        "--worktree-root",
        str(worktrees),
        "--patch-stack",
        str(stack),
    )
    manifest_path = Path(prepared.stdout.strip())
    manifest = json.loads(manifest_path.read_text())
    snapshot = Path(manifest["snapshot_worktree"])

    assert _git(repo, "status", "--short") == "M tracked.py"
    assert (snapshot / "tracked.py").read_text() == "value = 2\n"
    assert _git(snapshot, "status", "--short") == ""
    _run("verify", "--manifest", str(manifest_path))
    assert (
        _run("image-digest", "--manifest", str(manifest_path)).stdout.strip()
        == "sha256:test"
    )

    check = _run(
        "render-check",
        "--manifest",
        str(manifest_path),
        "--source-root",
        str(snapshot),
    ).stdout
    subprocess.run(["bash"], input=check, text=True, check=True)
    mounted = tmp_path / "mounted"
    shutil.copytree(snapshot, mounted, symlinks=True)
    (mounted / "tracked.py").write_text("wrong = 1\n")
    assert (
        subprocess.run(
            ["bash"],
            input=check.replace(str(snapshot), str(mounted)),
            text=True,
        ).returncode
        != 0
    )

    _run("cleanup", "--manifest", str(manifest_path), "--confirm")
    assert not snapshot.exists()
    assert (
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "show-ref",
                "--verify",
                "--quiet",
                "refs/heads/run-snapshot/run-1",
            ]
        ).returncode
        != 0
    )


def test_untracked_source_must_be_explicit(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path)
    stack = _stack(tmp_path / "stack.yaml", base)
    (repo / "new.py").write_text("value = 3\n")
    common = [
        "prepare",
        "--repo",
        str(repo),
        "--run-id",
        "run-2",
        "--result-dir",
        str(tmp_path / "result"),
        "--worktree-root",
        str(tmp_path / "worktrees"),
        "--patch-stack",
        str(stack),
    ]

    failed = _run(*common, check=False)
    assert failed.returncode == 1
    assert "untracked files are not captured" in failed.stderr

    prepared = _run(*common, "--include-untracked", "new.py")
    manifest_path = Path(prepared.stdout.strip())
    manifest = json.loads(manifest_path.read_text())
    assert (Path(manifest["snapshot_worktree"]) / "new.py").read_text() == "value = 3\n"


def test_patch_stack_must_match_first_parent_history(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path)
    (repo / "tracked.py").write_text("value = 2\n")
    _git(repo, "add", "tracked.py")
    _git(repo, "commit", "-m", "patch")
    patch = _git(repo, "rev-parse", "HEAD")
    common = [
        "prepare",
        "--repo",
        str(repo),
        "--run-id",
        "run-3",
        "--result-dir",
        str(tmp_path / "result"),
        "--worktree-root",
        str(tmp_path / "worktrees"),
    ]

    empty_stack = _stack(tmp_path / "empty.yaml", base)
    failed = _run(*common, "--patch-stack", str(empty_stack), check=False)
    assert failed.returncode == 1
    assert "patch stack does not match" in failed.stderr

    stack = _stack(
        tmp_path / "stack.yaml",
        base,
        [{"id": "patch", "source": "internal", "commits": [patch]}],
    )
    prepared = _run(*common, "--patch-stack", str(stack))
    manifest = json.loads(Path(prepared.stdout.strip()).read_text())
    lock = yaml.safe_load(
        Path(manifest["artifacts"]["patch_stack_lock"]["path"]).read_text()
    )
    assert lock["history_commits"] == [patch]
