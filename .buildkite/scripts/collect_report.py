# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Report on how the Buildkite job groups divide up the test tree.

`check_test_tethering.py` answers "does *any* job run this test?" and stops
there. This answers the neighbouring questions, reusing that checker's parser
so the two tools cannot disagree about what a command collects:

* ``--runs``        the same file collected by more than one job group
* ``--subsets``     pairs where one job's file set contains another's
* ``--dead``        ``--ignore`` / ``--deselect`` arguments that match nothing
* ``--untethered``  files no job collects, annotated with allowlist status

No vLLM install is needed - it only reads pipeline yamls and the tests tree.

Known limitation: collection is per *file*. Two jobs that run
``pytest models/language -m core_model`` and
``pytest models/language -m 'core_model and slow_test'`` have identical file
sets and share no tests, because markers are applied at collection time.
Separating those needs test *ids*, which means running ``pytest --collect-only``
per job in an environment where the modules import.

Most of ``--runs`` is expected to be intentional: vendor mirrors, per
architecture jobs, and GPU/CPU pairs split by ``-m cpu_test``. A file collected
by two groups is a question to look into, not a finding on its own.

Usage::

    python.buildkite / scripts / collect_report.py - -runs
    python.buildkite / scripts / collect_report.py - -dead - -untethered
"""

import argparse
import fnmatch
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER = REPO_ROOT / "tools" / "pre_commit" / "check_test_tethering.py"


def set_repo_root(path):
    """Point the checker at a different checkout.

    The tool is sometimes run from outside the checkout it is analysing - the
    measurement pass copies tests/ somewhere isolated so pytest's rootdir
    cannot shadow the installed vllm with an unbuilt source tree. Resolving the
    checkout from this file's location would then look in the wrong place.
    """
    global REPO_ROOT, CHECKER
    REPO_ROOT = Path(path).resolve()
    CHECKER = REPO_ROOT / "tools" / "pre_commit" / "check_test_tethering.py"
    return REPO_ROOT


def load_checker():
    """Import the tethering checker as a module so its parser is reused as-is."""
    spec = importlib.util.spec_from_file_location("check_test_tethering", CHECKER)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {CHECKER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # the checker resolves its own yamls relative to its file location; keep it
    # aligned with the checkout this tool was pointed at
    module.REPO_ROOT = REPO_ROOT
    return module


def job_records(checker):
    """(job key, commands, selections) for every job in every pipeline yaml.

    The checker's own loader discards the job identity, so walk the yamls here
    and reuse `_iter_job_steps` / `_parse_command` for the parsing itself.
    """
    records = []
    for path in checker._pipeline_yaml_paths():
        source = Path(path).relative_to(REPO_ROOT).as_posix()
        try:
            doc = checker.yaml.safe_load(Path(path).read_text())
        except checker.yaml.YAMLError as error:
            print(f"warning: skipping {source}: {error}", file=sys.stderr)
            continue
        if not doc:
            continue
        for step in checker._iter_job_steps(doc):
            commands = step.get("commands") or step.get("command") or []
            if isinstance(commands, str):
                commands = [commands]
            commands = [c for c in commands if isinstance(c, str)]
            selections = []
            for command in commands:
                selections.extend(checker._parse_command(command))
            key = (
                source,
                step.get("label", "?"),
                step.get("device", ""),
                step.get("working_dir") or "",
            )
            records.append({"key": key, "commands": commands, "selections": selections})
    return records


def files_of(record, checker, test_modules):
    """Test files a job collects, as repo-relative ``tests/...`` paths."""
    collected = set()
    for selection in record["selections"]:
        for module in test_modules:
            # the checker's selections are relative to tests/
            if selection.runs(module[len("tests/") :]):
                collected.add(module)
    return collected


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #


def report_runs(job_files):
    per_file = defaultdict(list)
    for key, files in job_files.items():
        for path in files:
            per_file[path].append(key)

    shared = {p: keys for p, keys in per_file.items() if len(keys) > 1}
    print(f"{len(shared)} test files are collected by more than one job group\n")
    ranked = sorted(shared.items(), key=lambda kv: -len(kv[1]))
    for path, keys in ranked:
        print(f"{len(keys):3d}x  {path}")
        for source, label, device, _ in sorted(keys):
            print(f"        {label}  [{Path(source).name} {device}]")
    return shared


def report_subsets(job_files, min_size=20):
    """Job pairs where one collects a strict subset of the other's files.

    A subset is a candidate for merging, but only a candidate: the smaller job
    may carry different markers, a different device, or a shard id.
    """
    items = [(k, v) for k, v in job_files.items() if v]
    print(f"\nstrict-subset job pairs (>= {min_size} files):\n")
    found = 0
    for i, (key_a, a) in enumerate(items):
        for key_b, b in items[i + 1 :]:
            if a == b or len(a) < min_size and len(b) < min_size:
                continue
            small, large = (a, b) if len(a) < len(b) else (b, a)
            skey, lkey = (key_a, key_b) if len(a) < len(b) else (key_b, key_a)
            if not small <= large:
                continue
            found += 1
            print(f"  {len(small)} of {len(large)} files")
            print(f"    subset: {skey[1]}  [{Path(skey[0]).name} {skey[2]}]")
            print(f"    superset: {lkey[1]}  [{Path(lkey[0]).name} {lkey[2]}]")
            print(f"    e.g. {sorted(small)[:3]}")
            print()
    if not found:
        print("  none\n")
    return found


def report_dead(records, checker, test_modules):
    """--ignore / --ignore-glob / bare --deselect values that match nothing."""
    test_set = set(test_modules)
    dead = []
    for record in records:
        _, label, _, _ = record["key"]
        source = Path(record["key"][0]).name
        for selection in record["selections"]:
            paths = list(getattr(selection, "ignored_paths", []))
            globs = list(getattr(selection, "ignored_globs", []))
            for raw in paths:
                spec = checker.normalize_test_path(str(raw))
                if not spec:
                    continue
                if not any(
                    module == f"tests/{spec}"
                    or module.startswith(f"tests/{spec.rstrip('/')}/")
                    for module in test_set
                ):
                    dead.append((label, source, "--ignore", str(raw)))
            for raw in globs:
                spec = checker.normalize_test_path(str(raw))
                hit = any(
                    fnmatch.fnmatch(module[len("tests/") :], spec)
                    for module in test_set
                )
                if not hit:
                    dead.append((label, source, "--ignore-glob", str(raw)))

    print(f"\nignored paths that match no test file: {len(dead)}\n")
    for label, source, kind, raw in dead:
        print(f"  {kind}={raw}")
        print(f"      {label}  [{source}]")
    return dead


def report_untethered(job_files, test_modules):
    """Files no job collects, annotated with whether the allowlist covers them.

    A file here that is *not* allowlisted is the same condition
    `check_test_tethering.py --all` reports as an error, so this doubles as a
    cross-check that the two tools agree.
    """
    collected = set()
    for files in job_files.values():
        collected |= files
    missing = [m for m in test_modules if m not in collected]
    allowlist = load_allowlist()
    unlisted = [m for m in missing if m not in allowlist]

    print(
        f"\nno job collects {len(missing)} test files "
        f"({len(missing) - len(unlisted)} allowlisted, {len(unlisted)} not)\n"
    )
    for path in missing:
        mark = "allowlisted" if path in allowlist else "NOT ALLOWLISTED"
        print(f"  [{mark}] {path}")
    if unlisted:
        print(
            f"\n{len(unlisted)} file(s) are collected by no job and not "
            "allowlisted - check_test_tethering.py --all should report these"
        )
    return missing


def load_allowlist():
    path = REPO_ROOT / "tools" / "pre_commit" / "test_tethering_allowlist.txt"
    entries = set()
    for line in path.read_text().splitlines():
        entry = line.split("#", 1)[0].strip()
        if entry:
            entries.add(entry)
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--runs", action="store_true", help="files run by >1 job group")
    parser.add_argument("--subsets", action="store_true", help="subset job pairs")
    parser.add_argument("--dead", action="store_true", help="inert ignore arguments")
    parser.add_argument(
        "--untethered",
        action="store_true",
        help="files no job collects (should be allowlist-only)",
    )
    parser.add_argument(
        "--min-files",
        type=int,
        default=20,
        help="ignore jobs smaller than this in --subsets",
    )
    args = parser.parse_args()

    if not (args.runs or args.subsets or args.dead or args.untethered):
        parser.error("pick at least one of --runs / --subsets / --dead / --untethered")

    checker = load_checker()
    records = job_records(checker)
    test_modules = checker.all_test_modules()
    job_files = {r["key"]: files_of(r, checker, test_modules) for r in records}

    print(f"{len(records)} job groups, {len(test_modules)} test modules\n")
    if args.runs:
        report_runs(job_files)
    if args.subsets:
        report_subsets(job_files, args.min_files)
    if args.dead:
        report_dead(records, checker, test_modules)
    if args.untethered:
        report_untethered(job_files, test_modules)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
