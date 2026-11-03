#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Create and verify clean source snapshots for RL experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import regex as re

try:
    import yaml
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyYAML is required; run this script with the project virtualenv"
    ) from exc


class SnapshotError(RuntimeError):
    pass


def _run_git(
    repo: Path,
    args: list[str],
    *,
    env: dict[str, str] | None = None,
) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise SnapshotError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_root(path: Path) -> Path:
    root = Path(_run_git(path, ["rev-parse", "--show-toplevel"]).strip())
    return root.resolve()


def _resolve_commit(repo: Path, revision: str) -> str:
    return _run_git(repo, ["rev-parse", "--verify", f"{revision}^{{commit}}"]).strip()


def _load_patch_stack(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise SnapshotError(f"cannot read patch stack {path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("version") != 1:
        raise SnapshotError("patch stack must be a mapping with version: 1")
    base = data.get("base")
    patches = data.get("patches")
    if not isinstance(base, dict) or not isinstance(base.get("source_commit"), str):
        raise SnapshotError("patch stack requires base.source_commit")
    if not isinstance(patches, list):
        raise SnapshotError("patch stack requires a patches list")
    return data


def _validate_patch_stack(
    repo: Path,
    path: Path,
    head: str,
) -> tuple[dict[str, Any], str, list[str]]:
    stack = _load_patch_stack(path)
    base = _resolve_commit(repo, stack["base"]["source_commit"])
    ancestor = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", base, head]
    )
    if ancestor.returncode != 0:
        raise SnapshotError(f"patch-stack base {base} is not an ancestor of {head}")

    expected: list[str] = []
    seen_ids: set[str] = set()
    resolved_patches: list[dict[str, Any]] = []
    for index, patch in enumerate(stack["patches"]):
        if not isinstance(patch, dict) or not isinstance(patch.get("id"), str):
            raise SnapshotError(f"patches[{index}] requires a string id")
        patch_id = patch["id"]
        if patch_id in seen_ids:
            raise SnapshotError(f"duplicate patch id: {patch_id}")
        seen_ids.add(patch_id)
        revisions = patch.get("commits", patch.get("commit"))
        if isinstance(revisions, str):
            revisions = [revisions]
        if (
            not isinstance(revisions, list)
            or not revisions
            or not all(isinstance(item, str) for item in revisions)
        ):
            raise SnapshotError(f"patch {patch_id} requires commit or commits")
        commits = [_resolve_commit(repo, item) for item in revisions]
        expected.extend(commits)
        resolved_patches.append({**patch, "commits": commits})

    actual_text = _run_git(
        repo, ["rev-list", "--reverse", "--first-parent", f"{base}..{head}"]
    )
    actual = [line for line in actual_text.splitlines() if line]
    if expected != actual:
        raise SnapshotError(
            "patch stack does not match first-parent history\n"
            f"expected: {expected}\nactual:   {actual}"
        )
    stack["base"]["source_commit"] = base
    stack["patches"] = resolved_patches
    return stack, base, actual


def _safe_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise SnapshotError(
            "run id must start with an alphanumeric character and contain only "
            "letters, digits, '.', '_' or '-'"
        )
    return value


def _relative_to_repo(repo: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else repo / path
    try:
        return candidate.resolve().relative_to(repo)
    except ValueError as exc:
        raise SnapshotError(f"path is outside repository: {path}") from exc


def _untracked_files(repo: Path) -> list[str]:
    output = _run_git(repo, ["ls-files", "--others", "--exclude-standard", "-z"])
    return sorted(item for item in output.split("\0") if item)


def _covered_untracked(
    repo: Path,
    untracked: list[str],
    includes: list[Path],
) -> tuple[list[str], list[str]]:
    roots = [_relative_to_repo(repo, path) for path in includes]
    covered: list[str] = []
    excluded: list[str] = []
    for name in untracked:
        rel = Path(name)
        if any(rel == root or root in rel.parents for root in roots):
            covered.append(name)
        else:
            excluded.append(name)
    return covered, excluded


def _snapshot_tree(
    repo: Path,
    includes: list[str],
    marker: bytes,
) -> str:
    with tempfile.TemporaryDirectory(prefix="rl-source-index-") as temp:
        index = Path(temp) / "index"
        env = {**os.environ, "GIT_INDEX_FILE": str(index)}
        _run_git(repo, ["read-tree", "HEAD"], env=env)
        _run_git(repo, ["add", "-u", "--", "."], env=env)
        if includes:
            _run_git(repo, ["add", "-f", "--", *includes], env=env)
        blob = subprocess.run(
            ["git", "-C", str(repo), "hash-object", "-w", "--stdin"],
            input=marker,
            capture_output=True,
            env=env,
        )
        if blob.returncode != 0:
            raise SnapshotError(
                f"git hash-object failed: {blob.stderr.decode().strip()}"
            )
        _run_git(
            repo,
            [
                "update-index",
                "--add",
                "--cacheinfo",
                "100644",
                blob.stdout.decode().strip(),
                ".rl-source-snapshot.json",
            ],
            env=env,
        )
        return _run_git(repo, ["write-tree"], env=env).strip()


def _create_commit(
    repo: Path,
    tree: str,
    parent: str,
    run_id: str,
    stack_hash: str,
) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": os.environ.get("GIT_AUTHOR_NAME", "RL Snapshot"),
        "GIT_AUTHOR_EMAIL": os.environ.get("GIT_AUTHOR_EMAIL", "rl-snapshot@localhost"),
        "GIT_COMMITTER_NAME": os.environ.get("GIT_COMMITTER_NAME", "RL Snapshot"),
        "GIT_COMMITTER_EMAIL": os.environ.get(
            "GIT_COMMITTER_EMAIL", "rl-snapshot@localhost"
        ),
    }
    message = f"run snapshot: {run_id}\n\nPatch-stack-sha256: {stack_hash}\n"
    result = subprocess.run(
        ["git", "-C", str(repo), "commit-tree", tree, "-p", parent],
        input=message,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        raise SnapshotError(f"git commit-tree failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _changed_file_records(
    repo: Path,
    base: str,
    snapshot: str,
    worktree: Path,
) -> list[tuple[str, str, str]]:
    changed = _run_git(
        repo,
        [
            "diff",
            "--name-only",
            "-z",
            "--diff-filter=ACMRTUXB",
            base,
            snapshot,
            "--",
        ],
    )
    deleted = _run_git(
        repo,
        [
            "diff",
            "--name-only",
            "-z",
            "--diff-filter=D",
            base,
            snapshot,
            "--",
        ],
    )
    records: list[tuple[str, str, str]] = []
    for name in sorted(item for item in changed.split("\0") if item):
        path = worktree / name
        if path.is_symlink():
            digest = _sha256_bytes(os.readlink(path).encode())
            records.append(("L", digest, name))
        elif path.is_file():
            records.append(("F", _sha256_file(path), name))
        else:
            raise SnapshotError(f"changed path is not a file: {name}")
    records.extend(
        ("D", "-", name)
        for name in sorted(item for item in deleted.split("\0") if item)
    )
    return records


def _write_records(
    path: Path,
    records: list[tuple[str, str, str]],
) -> None:
    for _, _, name in records:
        if "\t" in name or "\n" in name:
            raise SnapshotError(f"unsupported tab/newline in path: {name!r}")
    text = "".join(f"{kind}\t{digest}\t{name}\n" for kind, digest, name in records)
    path.write_text(text)


def _read_records(path: Path) -> list[tuple[str, str, str]]:
    records: list[tuple[str, str, str]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        parts = line.split("\t", 2)
        if len(parts) != 3 or parts[0] not in {"F", "L", "D"}:
            raise SnapshotError(f"invalid source hash record at {path}:{line_number}")
        records.append((parts[0], parts[1], parts[2]))
    return records


def _verify_records(
    root: Path,
    records: list[tuple[str, str, str]],
) -> None:
    for kind, expected, name in records:
        path = root / name
        if kind == "D":
            if path.exists() or path.is_symlink():
                raise SnapshotError(f"expected deleted path exists: {path}")
        elif kind == "L":
            if not path.is_symlink():
                raise SnapshotError(f"expected symlink is missing: {path}")
            actual = _sha256_bytes(os.readlink(path).encode())
            if actual != expected:
                raise SnapshotError(f"symlink target hash mismatch: {path}")
        else:
            if not path.is_file() or path.is_symlink():
                raise SnapshotError(f"expected file is missing: {path}")
            if _sha256_file(path) != expected:
                raise SnapshotError(f"file hash mismatch: {path}")


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _prepare(args: argparse.Namespace) -> Path:
    repo = _repo_root(args.repo.resolve())
    run_id = _safe_run_id(args.run_id)
    head = _resolve_commit(repo, "HEAD")
    branch = f"run-snapshot/{run_id}"
    subprocess.run(
        ["git", "check-ref-format", "--branch", branch],
        check=True,
        capture_output=True,
        text=True,
    )
    if (
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "show-ref",
                "--verify",
                "--quiet",
                f"refs/heads/{branch}",
            ]
        ).returncode
        == 0
    ):
        raise SnapshotError(f"branch already exists: {branch}")

    patch_stack = args.patch_stack.resolve()
    stack, base, history = _validate_patch_stack(repo, patch_stack, head)
    stack_hash = _sha256_file(patch_stack)
    marker_path = repo / ".rl-source-snapshot.json"
    if marker_path.exists() or marker_path.is_symlink():
        raise SnapshotError(
            "reserved snapshot marker already exists: .rl-source-snapshot.json"
        )

    covered, excluded = _covered_untracked(
        repo, _untracked_files(repo), args.include_untracked
    )
    if excluded:
        preview = "\n".join(f"  {item}" for item in excluded[:20])
        suffix = "\n  ..." if len(excluded) > 20 else ""
        raise SnapshotError(
            "untracked files are not captured; pass --include-untracked for "
            f"intentional source files:\n{preview}{suffix}"
        )

    marker = (
        json.dumps(
            {
                "version": 1,
                "run_id": run_id,
                "source_head": head,
                "patch_stack_sha256": stack_hash,
            },
            indent=2,
            sort_keys=True,
        ).encode()
        + b"\n"
    )
    tree = _snapshot_tree(repo, covered, marker)
    snapshot = _create_commit(repo, tree, head, run_id, stack_hash)
    worktree = (args.worktree_root.resolve() / run_id).resolve()
    result_dir = args.result_dir.resolve()
    if worktree.exists():
        raise SnapshotError(f"snapshot worktree already exists: {worktree}")

    created_branch = False
    created_worktree = False
    try:
        _run_git(repo, ["branch", branch, snapshot])
        created_branch = True
        worktree.parent.mkdir(parents=True, exist_ok=True)
        _run_git(repo, ["worktree", "add", "--quiet", str(worktree), branch])
        created_worktree = True

        result_dir.mkdir(parents=True, exist_ok=True)
        patch_path = result_dir / "source.patch"
        patch_path.write_text(
            _run_git(repo, ["diff", "--binary", base, snapshot, "--"])
        )
        hashes_path = result_dir / "source-files.tsv"
        records = _changed_file_records(repo, base, snapshot, worktree)
        _write_records(hashes_path, records)

        lock = {
            "version": 1,
            "generated_at": datetime.now(UTC).isoformat(),
            "manifest": {
                "path": str(patch_stack),
                "sha256": stack_hash,
            },
            "base": stack["base"],
            "patches": stack["patches"],
            "history_commits": history,
            "snapshot": {
                "run_id": run_id,
                "branch": branch,
                "parent_commit": head,
                "commit": snapshot,
                "tree": tree,
            },
        }
        lock_path = result_dir / "rl-patch-stack.lock.yaml"
        _write_yaml(lock_path, lock)

        manifest = {
            "version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "run_id": run_id,
            "repo": str(repo),
            "base_commit": base,
            "base_image_digest": stack["base"].get("image_digest"),
            "source_head": head,
            "snapshot_branch": branch,
            "snapshot_commit": snapshot,
            "snapshot_tree": tree,
            "snapshot_worktree": str(worktree),
            "included_untracked": covered,
            "patch_stack": str(patch_stack),
            "artifacts": {
                "patch": {
                    "path": str(patch_path),
                    "sha256": _sha256_file(patch_path),
                },
                "source_files": {
                    "path": str(hashes_path),
                    "sha256": _sha256_file(hashes_path),
                },
                "patch_stack_lock": {
                    "path": str(lock_path),
                    "sha256": _sha256_file(lock_path),
                },
            },
        }
        manifest_path = result_dir / "source-manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return manifest_path
    except Exception:
        if created_worktree:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "worktree",
                    "remove",
                    "--force",
                    str(worktree),
                ],
                capture_output=True,
            )
        if created_branch:
            subprocess.run(
                ["git", "-C", str(repo), "branch", "-D", branch],
                capture_output=True,
            )
        raise


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"cannot read source manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise SnapshotError("source manifest must be version 1")
    return manifest


def _artifact_path(manifest: dict[str, Any], name: str) -> tuple[Path, str]:
    artifact = manifest.get("artifacts", {}).get(name)
    if not isinstance(artifact, dict):
        raise SnapshotError(f"manifest is missing artifact: {name}")
    path = Path(artifact["path"])
    expected = artifact["sha256"]
    return path, expected


def _verify_manifest(path: Path) -> dict[str, Any]:
    manifest = _load_manifest(path)
    repo = _repo_root(Path(manifest["repo"]))
    worktree = Path(manifest["snapshot_worktree"])
    branch = manifest["snapshot_branch"]
    snapshot = manifest["snapshot_commit"]
    if _resolve_commit(repo, branch) != snapshot:
        raise SnapshotError(f"snapshot branch moved: {branch}")
    if _resolve_commit(worktree, "HEAD") != snapshot:
        raise SnapshotError(f"snapshot worktree HEAD moved: {worktree}")
    if _run_git(worktree, ["status", "--porcelain"]).strip():
        raise SnapshotError(f"snapshot worktree is dirty: {worktree}")
    tree = _run_git(worktree, ["rev-parse", "HEAD^{tree}"]).strip()
    if tree != manifest["snapshot_tree"]:
        raise SnapshotError(f"snapshot tree mismatch: {worktree}")

    for name in ("patch", "source_files", "patch_stack_lock"):
        artifact_path, expected = _artifact_path(manifest, name)
        if _sha256_file(artifact_path) != expected:
            raise SnapshotError(f"artifact hash mismatch: {artifact_path}")
    patch_path, _ = _artifact_path(manifest, "patch")
    expected_patch = _run_git(
        repo,
        [
            "diff",
            "--binary",
            manifest["base_commit"],
            manifest["snapshot_commit"],
            "--",
        ],
    ).encode()
    if patch_path.read_bytes() != expected_patch:
        raise SnapshotError(f"source patch content mismatch: {patch_path}")

    lock_path, _ = _artifact_path(manifest, "patch_stack_lock")
    try:
        lock = yaml.safe_load(lock_path.read_text())
    except yaml.YAMLError as exc:
        raise SnapshotError(f"invalid patch-stack lock: {lock_path}") from exc
    snapshot_lock = lock.get("snapshot", {}) if isinstance(lock, dict) else {}
    if (
        snapshot_lock.get("branch") != branch
        or snapshot_lock.get("commit") != snapshot
        or snapshot_lock.get("tree") != manifest["snapshot_tree"]
    ):
        raise SnapshotError(f"patch-stack lock does not match snapshot: {lock_path}")

    records_path, _ = _artifact_path(manifest, "source_files")
    _verify_records(worktree, _read_records(records_path))
    return manifest


def _render_check(args: argparse.Namespace) -> str:
    manifest = _verify_manifest(args.manifest.resolve())
    records_path, _ = _artifact_path(manifest, "source_files")
    root = shlex.quote(str(args.source_root))
    lines = [
        "set -euo pipefail",
        f"root={root}",
        '[[ -d "$root" ]] || { echo "source root missing: $root" >&2; exit 1; }',
    ]
    for kind, expected, name in _read_records(records_path):
        quoted_path = f'"$root"/{shlex.quote(name)}'
        if kind == "D":
            lines.append(
                f"[[ ! -e {quoted_path} && ! -L {quoted_path} ]] || "
                f"{{ echo 'expected deleted path exists: {name}' >&2; exit 1; }}"
            )
        elif kind == "L":
            lines.extend(
                [
                    f"[[ -L {quoted_path} ]] || "
                    f"{{ echo 'missing symlink: {name}' >&2; exit 1; }}",
                    f'actual=$(printf %s "$(readlink {quoted_path})" | '
                    "sha256sum | awk '{print $1}')",
                    f'[[ "$actual" == {shlex.quote(expected)} ]] || '
                    f"{{ echo 'symlink hash mismatch: {name}' >&2; exit 1; }}",
                ]
            )
        else:
            lines.extend(
                [
                    f"[[ -f {quoted_path} && ! -L {quoted_path} ]] || "
                    f"{{ echo 'missing file: {name}' >&2; exit 1; }}",
                    f"actual=$(sha256sum {quoted_path} | awk '{{print $1}}')",
                    f'[[ "$actual" == {shlex.quote(expected)} ]] || '
                    f"{{ echo 'file hash mismatch: {name}' >&2; exit 1; }}",
                ]
            )
    lines.append(f"echo 'source snapshot PASS: {manifest['snapshot_commit']}'")
    return "\n".join(lines) + "\n"


def _print_image_digest(args: argparse.Namespace) -> None:
    manifest = _load_manifest(args.manifest.resolve())
    value = manifest.get("base_image_digest")
    if value:
        print(value)


def _cleanup(args: argparse.Namespace) -> None:
    if not args.confirm:
        raise SnapshotError("cleanup requires --confirm")
    manifest = _verify_manifest(args.manifest.resolve())
    repo = Path(manifest["repo"])
    branch = manifest["snapshot_branch"]
    if not branch.startswith("run-snapshot/"):
        raise SnapshotError(f"refusing to delete non-snapshot branch: {branch}")
    worktree = Path(manifest["snapshot_worktree"])
    _run_git(repo, ["worktree", "remove", str(worktree)])
    _run_git(repo, ["branch", "-D", branch])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--repo", type=Path, required=True)
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--result-dir", type=Path, required=True)
    prepare.add_argument("--worktree-root", type=Path, required=True)
    prepare.add_argument("--patch-stack", type=Path, required=True)
    prepare.add_argument("--include-untracked", action="append", type=Path, default=[])

    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)

    render = subparsers.add_parser("render-check")
    render.add_argument("--manifest", type=Path, required=True)
    render.add_argument("--source-root", type=Path, required=True)

    image = subparsers.add_parser("image-digest")
    image.add_argument("--manifest", type=Path, required=True)

    cleanup = subparsers.add_parser("cleanup")
    cleanup.add_argument("--manifest", type=Path, required=True)
    cleanup.add_argument("--confirm", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "prepare":
            print(_prepare(args))
        elif args.command == "verify":
            manifest = _verify_manifest(args.manifest.resolve())
            print(
                f"source snapshot PASS: {manifest['snapshot_commit']} "
                f"({manifest['snapshot_branch']})"
            )
        elif args.command == "render-check":
            sys.stdout.write(_render_check(args))
        elif args.command == "image-digest":
            _print_image_digest(args)
        else:
            _cleanup(args)
    except (SnapshotError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
