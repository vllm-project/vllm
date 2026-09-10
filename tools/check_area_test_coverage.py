# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test-file coverage guard for Buildkite test-area YAMLs.

Reads a test-area YAML (e.g. ``.buildkite/test_areas/models_language.yaml``,
optionally plus the lane YAMLs that also run the area, such as
``.buildkite/hardware_tests/cpu.yaml``) and checks two things:

1. Coverage: every ``test_*.py`` under the area's test tree is inside a
   directory that some job command runs WHOLE (``pytest <dir>`` with no
   filter flags). An unclaimed file is reported with its path and the
   nearest job whose target would need to claim it.
2. Partition filters: no job command carries a ``-k`` expression, a
   per-file target inside the area tree, a partitioning ``-m`` expression,
   or pytest-shard flags (``--num-shards``/``--shard-id``) combined with any
   of those filters — anything that divides one directory between sibling
   jobs on the same lane. Shard flags on an otherwise whole-directory
   command are accepted: they split one job across its own ``parallelism:``
   instances, not between sibling jobs.

Lane-filter exception: a lane-capability marker expression (a single marker
from ``LANE_FILTER_MARKERS``, e.g. the CPU lane's recursive ``-m cpu_model``)
is allowed ONLY when every in-tree directory the command targets is also run
whole by some job (its "home lane"). The marker expresses what the lane's
hardware can run — possibly sharded for wall-clock on that lane — not a
partition of a directory between sibling jobs that each own a slice of it.
A command that selects a directory whole with no filter is always fine. A
``-m``/``-k``/shard combination that splits a directory between jobs of the
same lane, with no home lane running the directory whole, fails.

Commands are parsed from ``steps[*].commands`` and ``mirror.*.commands``
(each mirror is a separate lane); a ``parallelism:`` key without shard flags
in the command re-runs the same selection and is not a partition. Only
commands with at least one target inside ``--tree`` are checked, so shared
hardware YAMLs (cpu.yaml) can be passed without flagging other areas.

Entry point (wired into CI by ticket 08, when the language YAML is rewritten
to the one-job-per-directory shape; until then the real language YAML fails
this guard by design):

    python -m tools.check_area_test_coverage --tree tests/models/language \
        .buildkite/test_areas/models_language.yaml \
        .buildkite/hardware_tests/cpu.yaml
"""

import argparse
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Markers that express what a lane's hardware can run. Anything else in -m
# partitions a directory by test content, not by lane capability.
LANE_FILTER_MARKERS = frozenset({"cpu_model", "distributed"})

_VALUE_FLAGS = (
    "-m",
    "-k",
    "--num-shards",
    "--shard-id",
    "--ignore",
    "--deselect",
    "--ignore-glob",
    "--rootdir",
    "-p",
    "-o",
)


@dataclass
class PytestCommand:
    """One pytest invocation found in a job command."""

    lane: str
    step: str
    raw: str
    targets: list[str] = field(default_factory=list)  # tree-relative paths
    m_expr: str | None = None
    k_expr: str | None = None
    sharded: bool = False

    @property
    def file_targets(self) -> list[str]:
        return [t for t in self.targets if "::" in t or t.endswith(".py")]

    @property
    def dir_targets(self) -> list[str]:
        return [t for t in self.targets if t not in self.file_targets]

    @property
    def is_whole_dir(self) -> bool:
        """True if the command runs its directory targets with no filter.

        Pytest-shard flags do not disqualify a command: they split the
        directory across the parallel instances of ONE job (Buildkite
        ``parallelism:``), so the step still claims the whole directory.
        """
        return (
            bool(self.dir_targets)
            and not self.file_targets
            and self.m_expr is None
            and self.k_expr is None
        )


def _normalize_target(token: str, tree_rel: str) -> str | None:
    """Normalize a pytest target to a tree-relative path, or None if outside.

    Commands in test-area YAMLs run from ``tests/`` (e.g. ``models/language``)
    while hardware YAMLs run from the repo root (e.g. ``tests/models/...``).
    """
    token = token.split("::")[0].rstrip("/")
    if token.startswith("tests/"):
        token = token[len("tests/") :]
    if token == tree_rel or token.startswith(tree_rel + "/"):
        return token
    return None


def find_pytest_invocations(command: str) -> list[str]:
    """Extract pytest invocations from a shell command string.

    Handles multi-line commands and pytest calls embedded in quoted wrappers
    (e.g. cpu.yaml's ``bash run-cpu-test.sh 25m "...pytest..."``). Commented
    text (``# ...`` before the invocation on the same line) is skipped.
    """
    invocations = []
    for line in command.splitlines():
        start = 0
        while True:
            idx = line.find("pytest", start)
            if idx == -1:
                break
            before = line[:idx]
            preceded_ok = idx == 0 or not (
                line[idx - 1].isalnum() or line[idx - 1] in "/_-"
            )
            if "#" in before or not preceded_ok:
                start = idx + 1
                continue
            invocations.append(line[idx:])
            break
    return invocations


def parse_invocation(text: str, lane: str, step: str, tree_rel: str) -> PytestCommand:
    """Parse one pytest invocation into targets and filter flags."""
    try:
        tokens = shlex.split(text)
    except ValueError:
        # Unbalanced quote from an enclosing wrapper (e.g. a trailing '"' that
        # closes an outer bash -c string); strip trailing quotes and retry.
        tokens = shlex.split(text.rstrip("\"'"))
    cmd = PytestCommand(lane=lane, step=step, raw=text.strip())
    i = 1  # skip "pytest"
    while i < len(tokens):
        tok = tokens[i]
        if tok in _VALUE_FLAGS:
            value = tokens[i + 1] if i + 1 < len(tokens) else ""
            if tok == "-m":
                cmd.m_expr = value
            elif tok == "-k":
                cmd.k_expr = value
            elif tok in ("--num-shards", "--shard-id"):
                cmd.sharded = True
            i += 2
            continue
        if tok.startswith("--num-shards") or tok.startswith("--shard-id"):
            cmd.sharded = True
        elif tok.startswith("-m") and len(tok) > 2:
            cmd.m_expr = tok[2:].lstrip("=")
        elif tok.startswith("-k") and len(tok) > 2:
            cmd.k_expr = tok[2:].lstrip("=")
        elif not tok.startswith("-"):
            target = _normalize_target(tok, tree_rel)
            if target is not None:
                cmd.targets.append(target)
        i += 1
    return cmd


def load_commands(yaml_paths: list[Path], tree_rel: str) -> list[PytestCommand]:
    """Parse all in-tree pytest commands from the given YAML files."""
    commands = []
    for path in yaml_paths:
        doc = yaml.safe_load(path.read_text())
        for step in doc.get("steps") or []:
            label = step.get("key") or step.get("label") or "<unnamed>"
            base_commands = step.get("commands") or []
            lanes = [("base", base_commands)]
            for lane, mirror in (step.get("mirror") or {}).items():
                # A mirror without its own commands inherits the base lane's.
                lanes.append((lane, mirror.get("commands") or base_commands))
            for lane, cmd_list in lanes:
                for cmd in cmd_list:
                    for text in find_pytest_invocations(str(cmd)):
                        parsed = parse_invocation(text, lane, label, tree_rel)
                        if parsed.targets:
                            commands.append(parsed)
    return commands


def _test_files_under(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("test_*.py") if p.is_file())


def _is_single_lane_marker(expr: str) -> bool:
    return expr.isidentifier() and expr in LANE_FILTER_MARKERS


def check_coverage(
    tests_root: Path, yaml_paths: list[Path]
) -> tuple[list[str], list[str]]:
    """Run the guard. Returns (violations, notes)."""
    tests_dir = tests_root.parents[1]  # the tests/ directory
    tree_rel = tests_root.relative_to(tests_dir).as_posix()
    commands = load_commands(yaml_paths, tree_rel)
    whole_dirs = {t for c in commands if c.is_whole_dir for t in c.dir_targets}
    all_files = _test_files_under(tests_root)

    def covered(path: Path) -> bool:
        rel = path.parent.relative_to(tests_dir).as_posix()
        return any(rel == d or rel.startswith(d + "/") for d in whole_dirs)

    violations: list[str] = []
    notes: list[str] = []

    for f in all_files:
        if not covered(f):
            nearest = _nearest_job(f, tests_dir, commands)
            violations.append(
                f"unclaimed test file: {f.relative_to(tests_dir)} "
                f"(nearest job that would need to claim it: {nearest})"
            )

    for cmd in commands:
        where = f"{cmd.step} [{cmd.lane}]"
        problems = []
        in_tree_files = cmd.file_targets
        if in_tree_files:
            problems.append(f"per-file target(s): {', '.join(in_tree_files)}")
        if cmd.k_expr is not None:
            problems.append(f"-k expression: {cmd.k_expr!r}")
        lane_filter = cmd.m_expr is not None and _is_single_lane_marker(cmd.m_expr)
        if cmd.m_expr is not None and not lane_filter:
            problems.append(f"partitioning -m expression: {cmd.m_expr!r}")
        if cmd.sharded and problems:
            problems.append("pytest-shard flags")
        if not problems:
            if lane_filter:
                notes.append(
                    f"{where}: accepted lane filter -m {cmd.m_expr} "
                    f"(directory run whole by its home lane)"
                )
            elif cmd.sharded:
                notes.append(
                    f"{where}: accepted pytest-shard flags on a "
                    "whole-directory target (parallel instances of one job, "
                    "not a partition between sibling jobs)"
                )
            continue
        # Lane-filter exception: a single lane-capability marker, no -k, no
        # in-tree per-file targets, and every in-tree directory target fully
        # covered by home-lane whole-dir jobs.
        if (
            lane_filter
            and cmd.k_expr is None
            and not in_tree_files
            and all(
                all(covered(f) for f in _test_files_under(tests_dir / d))
                for d in cmd.dir_targets
            )
        ):
            notes.append(
                f"{where}: accepted lane filter -m {cmd.m_expr} "
                f"(directory run whole by its home lane)"
            )
            continue
        violations.append(
            f"partition filter in {where}: {'; '.join(problems)}\n"
            f"    command: {cmd.raw}"
        )

    return violations, notes


def _nearest_job(file: Path, tests_parent: Path, commands: list[PytestCommand]) -> str:
    """The job whose directory target is closest to the unclaimed file."""
    rel = file.parent.relative_to(tests_parent).as_posix()
    file_parts = Path(rel).parts
    best_label, best_len = "<none>", -1
    for cmd in commands:
        for target in cmd.dir_targets:
            target_parts = Path(target).parts
            common = 0
            for a, b in zip(file_parts, target_parts):
                if a != b:
                    break
                common += 1
            if common > best_len:
                best_len = common
                best_label = f"{cmd.step} [{cmd.lane}] (targets {target})"
    return best_label


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("yaml", nargs="+", type=Path, help="area/lane YAML files")
    parser.add_argument(
        "--tree",
        type=Path,
        required=True,
        help="area test tree rooted at the repo, e.g. tests/models/language",
    )
    args = parser.parse_args(argv)

    violations, notes = check_coverage(args.tree, args.yaml)
    for note in notes:
        print(f"note: {note}")
    if violations:
        print(f"\n{len(violations)} coverage violation(s):")
        for v in violations:
            print(f"- {v}")
        return 1
    print("coverage guard: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
