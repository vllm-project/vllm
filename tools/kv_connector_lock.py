# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep the CUDA 13 CI image's KV connector lock honest.

Subcommands:
    align: check that requirements/test/kv_connectors_cuda.txt satisfies the
        release requirements in requirements/kv_connectors.txt (pre-commit).
    plan LOCK OUT: write the lock entries that are not installed yet to OUT.
        Preinstalled packages at other versions are reported, not replaced:
        later image layers can still change them, so `verify` judges the
        finished image.
    verify LOCK: check that every lock entry is installed at its locked
        version and that the installed environment satisfies the requirements
        of every locked package.

Only the exceptions listed below are tolerated.
"""

import argparse
import importlib.metadata as metadata
import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

RELEASE_REQUIREMENTS = Path("requirements/kv_connectors.txt")
CI_INPUT = Path("requirements/test/kv_connectors_cuda.in")
CI_LOCK = Path("requirements/test/kv_connectors_cuda.txt")

# Release package -> the CUDA 13 wheel the CI image installs instead.
CUDA13_VARIANTS = {"mooncake-transfer-engine": "mooncake-transfer-engine-cuda13"}
# CI-only additions to the release requirements.
CI_ONLY = {"nixl-cu13"}

# Preinstalled at a version other than the lock's; their compatibility with
# the locked packages is still checked by `verify`.
IMAGE_OWNED = {
    # The vLLM wheel install upgrades it for cuda-core and nccl4py (>= 1.5.4).
    "cuda-pathfinder",
}
# (package, dependency) requirements that are intentionally not met.
ALLOWED_UNMET = {
    # The CUDA 13 image ships nixl-cu13 only.
    ("nixl", "nixl-cu12"),
    # The image pins NCCL itself (NCCL_VERSION + UV_OVERRIDE).
    ("torch", "nvidia-nccl-cu13"),
}


def read_lock(path: Path) -> dict[str, tuple[str, str]]:
    """Map each locked name to (version, entry text including hashes)."""
    entries: dict[str, tuple[str, str]] = {}
    name = None
    for line in path.read_text().splitlines(keepends=True):
        if line[:1].isalnum() and "==" in line:
            raw_name, rest = line.split("==", 1)
            name = canonicalize_name(raw_name)
            entries[name] = (rest.split()[0], line)
        elif name and line[:1].isspace():
            version, text = entries[name]
            entries[name] = (version, text + line)
        else:
            name = None
    return entries


def read_requirements(path: Path) -> list[Requirement]:
    lines = (line.split("#", 1)[0].strip() for line in path.read_text().splitlines())
    return [Requirement(line) for line in lines if line]


def installed_version(name: str) -> str | None:
    try:
        # The first distribution on sys.path, i.e. the one that is imported.
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def version_mismatches(lock: dict[str, tuple[str, str]]) -> list[str]:
    errors = []
    for name, (locked, _) in sorted(lock.items()):
        installed = installed_version(name)
        if installed not in (None, locked) and name not in IMAGE_OWNED:
            errors.append(f"{name}: image has {installed}, lock has {locked}")
    return errors


def align() -> list[str]:
    lock = read_lock(CI_LOCK)
    errors = []
    expected = set(CI_ONLY)
    for req in read_requirements(RELEASE_REQUIREMENTS):
        name = canonicalize_name(req.name)
        name = CUDA13_VARIANTS.get(name, name)
        expected.add(name)
        locked = lock.get(name, (None, ""))[0]
        if locked is None or not req.specifier.contains(locked, prereleases=True):
            errors.append(
                f"{RELEASE_REQUIREMENTS} requires {req}, but {CI_LOCK} has "
                f"{name}=={locked}; update {CI_INPUT} and recompile"
            )
    ci_names = {canonicalize_name(r.name) for r in read_requirements(CI_INPUT)}
    for name in sorted(ci_names - expected):
        errors.append(f"{CI_INPUT} has {name}, which {RELEASE_REQUIREMENTS} lacks")
    return errors


def plan(lock_path: Path, out: Path) -> list[str]:
    lock = read_lock(lock_path)
    for mismatch in version_mismatches(lock):
        print(f"not installing {mismatch}; checked by verify", file=sys.stderr)
    missing = [text for name, (_, text) in lock.items() if not installed_version(name)]
    out.write_text("".join(missing))
    return []


def verify(lock_path: Path) -> list[str]:
    lock = read_lock(lock_path)
    errors = version_mismatches(lock)
    errors += [f"{name}: not installed" for name in lock if not installed_version(name)]
    for name in sorted(lock):
        if not installed_version(name):
            continue
        for req in metadata.requires(name) or []:
            dep = Requirement(req)
            if dep.marker and not dep.marker.evaluate({"extra": ""}):
                continue
            dep_name = canonicalize_name(dep.name)
            if (name, dep_name) in ALLOWED_UNMET:
                continue
            have = installed_version(dep_name)
            if have is None or not dep.specifier.contains(have, prereleases=True):
                errors.append(f"{name} requires {req}, but image has {have}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("align")
    plan_parser = sub.add_parser("plan")
    plan_parser.add_argument("lock", type=Path)
    plan_parser.add_argument("out", type=Path)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("lock", type=Path)
    args = parser.parse_args()

    if args.command == "align":
        errors = align()
    elif args.command == "plan":
        errors = plan(args.lock, args.out)
    else:
        errors = verify(args.lock)
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
