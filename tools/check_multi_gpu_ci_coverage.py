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
`multi_gpu_marks`, `gpu_tier_mark`, or `pytest.mark.distributed(num_gpus=N)`
with a literal int. Tests that compute their GPU requirement at runtime
from parametrize values (e.g. `world_size = tp_size * dp_size`) aren't
visible to this tool until migrated to one of those markers.

Also accounts for `-m` marker-expression selection: pytest's `distributed`
marker matching is by exact-value equality, not `>=` -- a job running
`-m 'distributed(num_gpus=2)'` selects *only* tests marked exactly
`num_gpus=2`, never `num_gpus=4`, even though 2 < 4 might suggest partial
coverage. A job's `num_devices:` alone is therefore not sufficient to prove
coverage; the `-m` expression's exact-match value has to agree with what
the test declares too. Only the three literal forms this repo's CI configs
actually use are interpreted (`distributed`, `distributed(num_gpus=N)`,
`not distributed`); any other `-m`/`-k` expression is treated as
unconstrained rather than guessed at, to avoid false gap reports.

Also flags a second, separate finding category: hand-rolled runtime GPU-count
skip guards (`if <condition mentioning device_count/world_size/num_gpus/...>:
pytest.skip(...)`) that bypass the marker convention entirely. These are the
root cause of the blind spots above -- a test using one isn't just missing
from a `-m`/`num_devices` cross-check, it's invisible to this tool (and any
other static analysis) from the start. This is a heuristic, not proof of a
bug: it flags the *pattern*, not whether the guard's actual value is
correct or already covered. Human judgment decides whether to migrate.
"""

import argparse
import ast
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"
TEST_AREAS_DIR = REPO_ROOT / ".buildkite" / "test_areas"
TEST_AMD_YAML = REPO_ROOT / ".buildkite" / "test-amd.yaml"

_ENV_PREFIX_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=\S+\s+")
_EXACT_NUM_GPUS_RE = re.compile(r"distributed\(num_gpus=(\d+)\)")
# "-m" is handled separately (its value is semantically meaningful, see
# GpuMarkConstraint); these are skipped as opaque flag values only.
_VALUE_FLAGS = {"-k", "--shard-id", "--num-shards"}


@dataclass
class GpuMarkConstraint:
    """What a step's `-m` expression implies about which `distributed`-marked
    tests it actually selects. `covers_any=False` means the expression
    excludes distributed tests entirely (`not distributed`); `exact_n` set
    means it selects only tests whose `num_gpus` equals that value exactly
    (pytest mark matching is equality, not `>=`); `exact_n=None` with
    `covers_any=True` means no constraint on the value (no `-m` flag, or a
    bare `distributed`)."""

    covers_any: bool = True
    exact_n: int | None = None


def _parse_mark_gpu_constraint(mark_expr: str | None) -> GpuMarkConstraint:
    """Interpret a step's `-m` expression string, handling exactly the forms
    this repo's CI configs use (`distributed`, `distributed(num_gpus=N)`,
    `not distributed`). Anything else -- no `-m` flag, or a compound/other
    expression -- is treated as unconstrained rather than guessed at."""
    if mark_expr is None:
        return GpuMarkConstraint()

    stripped = mark_expr.strip()
    if stripped == "not distributed":
        return GpuMarkConstraint(covers_any=False)

    m = _EXACT_NUM_GPUS_RE.fullmatch(stripped)
    if m:
        return GpuMarkConstraint(covers_any=True, exact_n=int(m.group(1)))

    return GpuMarkConstraint()


@dataclass
class CoveringStep:
    """One CI step that targets a given test file, and what it actually
    selects/provisions for it."""

    num_devices: int
    constraint: GpuMarkConstraint
    label: str


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


def find_gpu_requirements() -> dict[str, set[int]]:
    """AST-scan tests/ for literal-int GPU requirements. Returns
    {repo-relative path: set of distinct num_gpus values required by test
    functions in that file}. Files with no declared requirement > 1 are
    omitted. A file can require more than one distinct value (e.g. some
    tests need 2 GPUs, others need 4) -- each is checked against CI
    coverage independently, since `-m 'distributed(num_gpus=N)'` selection
    is exact-match, not `>=`."""
    requirements: dict[str, set[int]] = {}
    for path in sorted(TESTS_DIR.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue

        needed: set[int] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node.func)
            if name == "distributed":
                if not _is_pytest_mark_distributed(node.func):
                    continue
                n = _extract_int_kwarg(node, ("num_gpus",))
            elif name in ("multi_gpu_test", "multi_gpu_marks"):
                # multi_gpu_marks is the primitive multi_gpu_test wraps
                # (tests/utils.py); also called directly for pytest.param
                # marks=, e.g. test_common.py.
                n = _extract_int_kwarg(node, ("num_gpus",))
            elif name == "gpu_tier_mark":
                n = _extract_int_kwarg(node, ("min_gpus",))
            else:
                continue
            if n is not None and n > 1:
                needed.add(n)

        if needed:
            requirements[path.relative_to(REPO_ROOT).as_posix()] = needed

    return requirements


_GPU_GUARD_IDENTIFIERS = {
    "device_count",
    "world_size",
    "num_gpus",
    "gpu_count",
    "n_gpu",
    "num_devices",
    "gpu_tier",
}


def _mentions_gpu_guard(node: ast.expr) -> bool:
    """True if any Name/Attribute inside `node` looks GPU-count-related.
    Substring match, not exact -- catches fixture/variable names like
    `num_gpus_available` that embed a guard keyword without equaling it."""
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            identifier = n.id
        elif isinstance(n, ast.Attribute):
            identifier = n.attr
        else:
            continue
        if any(kw in identifier for kw in _GPU_GUARD_IDENTIFIERS):
            return True
    return False


def _is_pytest_skip_call(func: ast.expr) -> bool:
    """True for `pytest.skip(...)` specifically (not `@pytest.mark.skip`,
    which is a decorator, not a runtime call, and already static)."""
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "skip"
        and isinstance(func.value, ast.Name)
        and func.value.id == "pytest"
    )


def find_hand_rolled_skip_guards() -> dict[str, list[int]]:
    """AST-scan tests/ for `if <condition mentioning GPU count>: ...
    pytest.skip(...)` -- runtime skip guards that bypass the marker
    convention (multi_gpu_test/gpu_tier_mark/multi_gpu_marks/
    pytest.mark.distributed) this tool relies on to see requirements at
    all. Returns {repo-relative path: [line numbers of the `if`]}.

    Heuristic, not proof: flags the *pattern* of a hand-rolled GPU-count
    guard so it can be migrated to a statically-visible marker; doesn't
    evaluate whether the guard's value is itself correct or already
    covered by CI."""
    findings: dict[str, list[int]] = {}
    for path in sorted(TESTS_DIR.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue

        lines: list[int] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            if not any(isinstance(n, ast.Compare) for n in ast.walk(node.test)):
                continue
            if not _mentions_gpu_guard(node.test):
                continue
            has_skip_call = any(
                isinstance(stmt, ast.Call) and _is_pytest_skip_call(stmt.func)
                for stmt in ast.walk(node)
            )
            if has_skip_call:
                lines.append(node.lineno)

        if lines:
            findings[path.relative_to(REPO_ROOT).as_posix()] = lines

    return findings


def _strip_env_prefix(cmd: str) -> str:
    """Strip leading inline env-var assignments (e.g. `VLLM_USE_V1=1 pytest
    ...` -> `pytest ...`) so the pytest-token search below isn't fooled by
    an `=`-containing value that happens to precede the real command."""
    while True:
        m = _ENV_PREFIX_RE.match(cmd)
        if not m:
            return cmd
        cmd = cmd[m.end() :]


def _parse_pytest_command(
    cmd: str,
) -> tuple[list[str], list[str], str | None] | None:
    """Return (targets, ignored, mark_expr) for a `pytest ...` command
    string, or None if this command doesn't invoke pytest at all (e.g. a
    `torchrun` line). `mark_expr` is the raw `-m` value, if present."""
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
    mark_expr: str | None = None
    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--ignore="):
            ignored.append(tok[len("--ignore=") :])
        elif tok == "--ignore":
            i += 1
            if i < len(tokens):
                ignored.append(tokens[i])
        elif tok == "-m":
            i += 1
            if i < len(tokens):
                mark_expr = tokens[i]
        elif tok.startswith("-m="):
            mark_expr = tok[len("-m=") :]
        elif tok in _VALUE_FLAGS:
            i += 1  # skip this flag's value, not a target
        elif tok.startswith("-"):
            pass  # flag we don't need the value of (-v, -s, -x, ...)
        else:
            targets.append(tok)
        i += 1
    return targets, ignored, mark_expr


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
    coverage: dict[str, list[CoveringStep]],
    files: list[Path],
    num_devices: int,
    job_label: str,
    constraint: GpuMarkConstraint,
) -> None:
    for f in files:
        rel = f.relative_to(REPO_ROOT).as_posix()
        coverage.setdefault(rel, []).append(
            CoveringStep(num_devices, constraint, job_label)
        )


def _process_step(
    coverage: dict[str, list[CoveringStep]],
    label: str,
    commands: list[str],
    num_devices: int,
) -> None:
    for cmd in commands:
        parsed = _parse_pytest_command(cmd)
        if parsed is None:
            continue
        targets, ignored, mark_expr = parsed
        constraint = _parse_mark_gpu_constraint(mark_expr)
        for target in targets:
            files = _resolve_target_files(target, ignored)
            _record_coverage(coverage, files, num_devices, label, constraint)


def find_ci_coverage() -> dict[str, list[CoveringStep]]:
    coverage: dict[str, list[CoveringStep]] = {}

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


def _covers(step: CoveringStep, needed: int) -> bool:
    """True if `step` actually selects and satisfies a test that declares
    `num_gpus=needed` -- both the `-m` expression's exact-match constraint
    (if any) and the provisioned device count have to agree."""
    if not step.constraint.covers_any:
        return False
    if step.constraint.exact_n is not None and step.constraint.exact_n != needed:
        return False
    return step.num_devices >= needed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verbose", action="store_true", help="also print files with no gap"
    )
    args = parser.parse_args()

    requirements = find_gpu_requirements()
    coverage = find_ci_coverage()

    gaps = []
    for rel_path, needed_values in sorted(requirements.items()):
        steps = coverage.get(rel_path, [])
        for needed in sorted(needed_values):
            covering = [s for s in steps if _covers(s, needed)]
            if covering:
                if args.verbose:
                    best = max(s.num_devices for s in covering)
                    jobs = sorted({s.label for s in covering if s.num_devices == best})
                    print(
                        f"OK    {rel_path} (num_gpus={needed}): covered at "
                        f"{best} ({jobs})"
                    )
                continue

            candidates = [s for s in steps if s.constraint.covers_any]
            selecting = [
                s
                for s in candidates
                if s.constraint.exact_n is None or s.constraint.exact_n == needed
            ]
            if not steps:
                reason = "not run by any CI job"
            elif not candidates:
                reason = "only covering job(s) exclude distributed tests entirely (`-m 'not distributed'`)"
            elif not selecting:
                other_n = sorted(
                    {s.constraint.exact_n for s in candidates if s.constraint.exact_n}
                )
                reason = (
                    f"covering job(s) select `-m 'distributed(num_gpus=N)'` "
                    f"for N={other_n}, never num_gpus={needed}"
                )
            else:
                have = max(s.num_devices for s in selecting)
                jobs = sorted({s.label for s in selecting if s.num_devices == have})
                reason = f"best covering job(s) {jobs} only provision {have}"
            gaps.append((rel_path, needed, reason))

    hand_rolled = find_hand_rolled_skip_guards()

    if not gaps and not hand_rolled:
        print("No world_size/num_devices gaps found.")
        return 0

    exit_code = 0

    if gaps:
        exit_code = 1
        print(f"Found {len(gaps)} unsatisfied GPU requirement(s):\n")
        for rel_path, needed, reason in gaps:
            print(f"  {rel_path} (num_gpus={needed}): {reason}")

    if hand_rolled:
        exit_code = 1
        if gaps:
            print()
        total = sum(len(lines) for lines in hand_rolled.values())
        print(
            f"Found {total} hand-rolled GPU-count skip guard(s) in "
            f"{len(hand_rolled)} file(s) -- these bypass the marker "
            "convention and are invisible to the checks above; consider "
            "migrating to multi_gpu_test/gpu_tier_mark/multi_gpu_marks/"
            "pytest.mark.distributed(num_gpus=N):\n"
        )
        for rel_path, lines in sorted(hand_rolled.items()):
            line_list = ", ".join(f"L{n}" for n in lines)
            print(f"  {rel_path}: {line_list}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
