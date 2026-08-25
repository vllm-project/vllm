#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-check tests/'s multi_gpu_test/gpu_tier_mark/pytest.mark.distributed
`num_gpus` requirements against the `num_devices` Buildkite actually
provisions for them, across `.buildkite/test_areas/*.yaml` and the legacy
`.buildkite/test-amd.yaml`.

A test can be wired into a non-optional CI job and still never execute a
code path, because its own GPU-count skip guard fires every run when the
job's `num_devices:` is below what the test declares it needs. This shows
up as "skipped," not "failed," so it never trips CI red -- see
`docs/... World Size Num Devices Audit` in the vLLM notes repo for the
investigation this tool came out of.

Only catches tests that declare their requirement via `multi_gpu_test`,
`gpu_tier_mark`, or `pytest.mark.distributed(num_gpus=N)` with a literal
int. Tests that compute their GPU requirement at runtime from parametrize
values (e.g. `world_size = tp_size * dp_size`) aren't visible to this tool
until migrated to one of those markers.
"""

import argparse
import ast
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"
TEST_AREAS_DIR = REPO_ROOT / ".buildkite" / "test_areas"
TEST_AMD_YAML = REPO_ROOT / ".buildkite" / "test-amd.yaml"

_ENV_PREFIX_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=\S+\s+")
_VALUE_FLAGS = {"-m", "-k", "--shard-id", "--num-shards"}


@dataclass
class Coverage:
    """Best `num_devices` seen for a test file, and which job(s) provided it."""

    max_num_devices: int = 0
    jobs: list[str] = field(default_factory=list)


def _call_name(func: ast.expr) -> str | None:
    """Return a call's bare function/method name, e.g. `foo` for both
    `foo(...)` and `obj.foo(...)`; `None` for anything else callable."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _is_pytest_mark_distributed(func: ast.expr) -> bool:
    """True for the `pytest.mark.distributed` attribute chain specifically,
    disambiguating it from any other `*.distributed(...)` call `_call_name`
    would also match (e.g. `torch.distributed(...)`)."""
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "distributed"
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "mark"
    )


def _extract_int_kwarg(call: ast.Call, names: tuple[str, ...]) -> int | None:
    """Return the literal int passed to `call` under any keyword in `names`,
    falling back to the first literal-int positional arg (covers
    `multi_gpu_test(4)` as well as `multi_gpu_test(num_gpus=4)`)."""
    for kw in call.keywords:
        if (
            kw.arg in names
            and isinstance(kw.value, ast.Constant)
            and isinstance(kw.value.value, int)
        ):
            return kw.value.value
    for arg in call.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
            return arg.value
    return None


def find_gpu_requirements() -> dict[str, int]:
    """AST-scan tests/ for literal-int GPU requirements. Returns
    {repo-relative path: max num_gpus required across all test functions
    in that file}. Files with no declared requirement > 1 are omitted."""
    requirements: dict[str, int] = {}
    for path in sorted(TESTS_DIR.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue

        max_n = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node.func)
            if name == "distributed":
                if not _is_pytest_mark_distributed(node.func):
                    continue
                n = _extract_int_kwarg(node, ("num_gpus",))
            elif name == "multi_gpu_test":
                n = _extract_int_kwarg(node, ("num_gpus",))
            elif name == "gpu_tier_mark":
                n = _extract_int_kwarg(node, ("min_gpus",))
            else:
                continue
            if n is not None:
                max_n = max(max_n, n)

        if max_n > 1:
            requirements[path.relative_to(REPO_ROOT).as_posix()] = max_n

    return requirements


def _strip_env_prefix(cmd: str) -> str:
    """Strip leading inline env-var assignments (e.g. `VLLM_USE_V1=1 pytest
    ...` -> `pytest ...`) so the pytest-token search below isn't fooled by
    an `=`-containing value that happens to precede the real command."""
    while True:
        m = _ENV_PREFIX_RE.match(cmd)
        if not m:
            return cmd
        cmd = cmd[m.end() :]


def _parse_pytest_command(cmd: str) -> tuple[list[str], list[str]] | None:
    """Return (targets, ignored) for a `pytest ...` command string, or None
    if this command doesn't invoke pytest at all (e.g. a `torchrun` line)."""
    cmd = _strip_env_prefix(cmd.strip())
    # commands are sometimes chained, e.g. "cd .. && pytest ...": only take
    # the part from the first "pytest" token onward.
    idx = cmd.find("pytest")
    if idx == -1:
        return None
    cmd = cmd[idx:]
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        return None
    if not tokens or tokens[0] != "pytest":
        return None

    targets: list[str] = []
    ignored: list[str] = []
    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--ignore="):
            ignored.append(tok[len("--ignore=") :])
        elif tok == "--ignore":
            i += 1
            if i < len(tokens):
                ignored.append(tokens[i])
        elif tok in _VALUE_FLAGS:
            i += 1  # skip this flag's value, not a target
        elif tok.startswith("-"):
            pass  # flag we don't need the value of (-v, -s, -x, ...)
        else:
            targets.append(tok)
        i += 1
    return targets, ignored


def _normalize_to_tests_relative(target: str) -> str:
    """Commands are written relative to `tests/` (the usual working_dir) or
    already carry a leading `tests/` (when working_dir is the repo root).
    Normalize both to a path relative to `tests/`."""
    target = target.split("::", 1)[0]
    if target.startswith("tests/"):
        target = target[len("tests/") :]
    return target.strip("/")


def _resolve_target_files(target: str, ignored: list[str]) -> list[Path]:
    rel = _normalize_to_tests_relative(target)
    if not rel:
        return []
    abs_target = TESTS_DIR / rel
    ignored_rel = {_normalize_to_tests_relative(i) for i in ignored}

    if abs_target.is_file():
        return [abs_target]
    if not abs_target.exists():
        return []

    files = []
    for f in abs_target.rglob("test_*.py"):
        f_rel = f.relative_to(TESTS_DIR).as_posix()
        if any(f_rel == ig or f_rel.startswith(ig + "/") for ig in ignored_rel):
            continue
        files.append(f)
    return files


def _record_coverage(
    coverage: dict[str, Coverage], files: list[Path], num_devices: int, job_label: str
) -> None:
    for f in files:
        rel = f.relative_to(REPO_ROOT).as_posix()
        c = coverage.setdefault(rel, Coverage())
        if num_devices > c.max_num_devices:
            c.max_num_devices = num_devices
            c.jobs = [job_label]
        elif num_devices == c.max_num_devices:
            c.jobs.append(job_label)


def _process_step(
    coverage: dict[str, Coverage],
    label: str,
    commands: list[str],
    num_devices: int,
) -> None:
    for cmd in commands:
        parsed = _parse_pytest_command(cmd)
        if parsed is None:
            continue
        targets, ignored = parsed
        for target in targets:
            files = _resolve_target_files(target, ignored)
            _record_coverage(coverage, files, num_devices, label)


def find_ci_coverage() -> dict[str, Coverage]:
    coverage: dict[str, Coverage] = {}

    for yaml_path in sorted(TEST_AREAS_DIR.glob("*.yaml")):
        doc = yaml.safe_load(yaml_path.read_text())
        if not doc or "steps" not in doc:
            continue
        for step in doc["steps"]:
            if not isinstance(step, dict) or "commands" not in step:
                continue
            label = step.get("label", "<unlabeled>")
            num_devices = step.get("num_devices", 1)
            _process_step(coverage, label, step["commands"], num_devices)

            mirror = step.get("mirror", {}).get("amd") if step.get("mirror") else None
            if mirror:
                amd_label = mirror.get("label", f"{label} (amd)")
                amd_commands = mirror.get("commands", step["commands"])
                amd_num_devices = mirror.get("num_devices", num_devices)
                _process_step(coverage, amd_label, amd_commands, amd_num_devices)

    if TEST_AMD_YAML.exists():
        doc = yaml.safe_load(TEST_AMD_YAML.read_text())
        for step in doc.get("steps", []):
            if not isinstance(step, dict) or "commands" not in step:
                continue
            label = step.get("label", "<unlabeled>")
            num_devices = step.get("num_gpus", 1)
            _process_step(coverage, label, step["commands"], num_devices)

    return coverage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verbose", action="store_true", help="also print files with no gap"
    )
    args = parser.parse_args()

    requirements = find_gpu_requirements()
    coverage = find_ci_coverage()

    gaps = []
    for rel_path, needed in sorted(requirements.items()):
        c = coverage.get(rel_path)
        have = c.max_num_devices if c else 0
        if have < needed:
            gaps.append((rel_path, needed, have, c.jobs if c else []))
        elif args.verbose:
            print(f"OK    {rel_path}: needs {needed}, covered at {have} ({c.jobs})")

    if not gaps:
        print("No world_size/num_devices gaps found.")
        return 0

    print(f"Found {len(gaps)} file(s) with unsatisfied GPU requirements:\n")
    for rel_path, needed, have, jobs in gaps:
        if have == 0:
            print(f"  {rel_path}: needs {needed} GPU(s), not run by any CI job")
        else:
            print(
                f"  {rel_path}: needs {needed} GPU(s), best covering job(s) "
                f"{jobs} only provide {have}"
            )
    return 1


if __name__ == "__main__":
    sys.exit(main())
