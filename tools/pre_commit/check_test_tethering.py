# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check that test files are wired into ("tethered to") the Buildkite CI.

A test file is *tethered* when at least one Buildkite job actually collects it -
some job's ``commands`` run ``pytest`` (or ``python`` / ``torchrun``, or a
``find ... | xargs pytest`` pipeline) over a path that includes the file. A test
that is collected by no job silently never runs: it passes review, merges, and
rots.

How it works
------------
1. Parse every job in ``.buildkite/test_areas/*.yaml`` (and its ``mirror.amd``
   sub-step) plus the legacy ``.buildkite/test-amd.yaml``, and turn each shell
   command into a :class:`Selection` describing which test files it runs.
2. Build the inventory of test files under ``tests/``.
3. A file is a problem if no selection runs it and it is not in the allowlist
   (``tools/pre_commit/test_tethering_allowlist.txt`` - the set of pre-existing
   gaps, which is only ever meant to shrink). Every allowlist entry must carry a
   trailing ``# <reason>``; a bare path line is itself an error.

Modes
-----
* **changed-files** (default): pre-commit passes the changed test files as
  arguments; each one must be tethered or allowlisted.
* **full scan**: ``--all``, or automatically whenever the change set touches a
  pipeline yaml or the allowlist itself - because editing a job command can
  orphan a test that is not otherwise in the diff. The full scan also reports
  allowlist entries that have since become tethered, or whose file is gone, so
  the list stays honest.

Coverage from either ``.buildkite/test_areas/`` or the legacy
``.buildkite/test-amd.yaml`` counts. ``test-amd.yaml`` is being folded into
``test_areas/`` a group at a time; when it is gone, this checker stops parsing
it and reverts to ``test_areas/`` only.

Usage::

    python tools/pre_commit/check_test_tethering.py [FILES...]
    python tools/pre_commit/check_test_tethering.py --all
"""

import argparse
import fnmatch
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_AREAS_DIR = REPO_ROOT / ".buildkite" / "test_areas"
# The legacy hand-maintained AMD pipeline. It is being migrated into
# ``test_areas/`` piecemeal; until that finishes, a test wired only into this
# file is still genuinely run by CI, so it counts as tethered. When the file is
# removed, drop it from ``_pipeline_yaml_paths()`` and the checker is back to
# ``test_areas/`` only, flagging anything the migration left behind.
TEST_AMD_YAML = REPO_ROOT / ".buildkite" / "test-amd.yaml"
TESTS_DIR = REPO_ROOT / "tests"
ALLOWLIST_PATH = Path(__file__).resolve().parent / "test_tethering_allowlist.txt"

# `pytest` options that take a value in the *next* token, so that token must not
# be mistaken for a test path (e.g. `-k expr`, `--shard-id 3`). Options that only
# ever take their value with `=` (`--cov=...`) don't belong here - a bare
# `--cov` takes no value and would wrongly swallow the following test path.
PYTEST_OPTIONS_WITH_VALUE = {
    "-k", "-m", "-p", "-n", "-c", "-o", "-W", "-r",
    "--shard-id", "--num-shards", "--dist", "--durations", "--maxfail",
    "--timeout", "--rootdir", "--junitxml", "--tb",
}  # fmt: skip

# `pytest` options whose value is a path that is *removed* from collection at
# file granularity. ``--ignore`` takes a path prefix; ``--ignore-glob`` takes a
# glob that pytest matches against the whole path with ``fnmatch`` (so its ``*``
# *does* cross a ``/``), which is why the two are modelled separately.
PYTEST_IGNORE_OPTIONS = {"--ignore", "--ignore-glob"}

# `--deselect` takes a value too, but it removes individual node IDs - a file
# with one parametrization deselected is still collected and still "run".
PYTEST_DESELECT_OPTION = "--deselect"

PYTEST_COMMANDS = {"pytest", "py.test"}

# A leading ``$VAR/`` or ``${VAR}/`` path prefix - CI shell scripts routinely
# name a test by an absolute path rooted at a shell var
# (``"${GIT_ROOT}/tests/v1/..."``). The prefix is dropped before the path is
# interpreted; an unresolved ``$`` anywhere else still makes the token unusable.
_LEADING_VAR_PREFIX = re.compile(r"^\$\{?\w+\}?/")

# Commands that execute a Python file directly. Only the *first* path argument is
# the script that runs; everything after it is that script's own argv, not more
# test files (``python tensorize.py --suffix v1`` runs one file, not ``tests/v1``).
FILE_RUNNER_COMMANDS = {"python", "python3", "torchrun", "coverage"}


# --------------------------------------------------------------------------- #
# Shell-token helpers
# --------------------------------------------------------------------------- #


def _is_env_assignment(token: str) -> bool:
    """True for a leading ``FOO=bar`` shell env-var assignment - the kind some
    commands put before ``pytest`` (``VLLM_TEST_FORCE_LOAD_FORMAT=auto pytest``).
    """
    if "=" not in token:
        return False
    name = token.split("=", 1)[0]
    return bool(name) and name.replace("_", "").isupper()


def normalize_test_path(raw: str) -> str:
    """Reduce a path as it appears in a command to one comparable form:
    relative to ``tests/``, no surrounding quotes, no ``::nodeid`` selector, no
    trailing slash.

    Commands run from either the repo root or from ``tests/``, so both a
    ``tests/`` and a ``/vllm-workspace/tests/`` prefix are stripped, as is a
    leading ``$VAR/`` (``"${GIT_ROOT}/tests/..."``).
    """
    path = raw.strip().strip("'\"")
    path = path.split("::", 1)[0]
    path = _LEADING_VAR_PREFIX.sub("", path)
    for prefix in ("./", "tests/", "/vllm-workspace/tests/"):
        if path.startswith(prefix):
            path = path[len(prefix) :]
    return path.rstrip("/")


def _token_is_test_path(token: str) -> bool:
    """True if a shell token is a path argument rather than a flag, an env-var
    assignment, a Buildkite ``$VAR``, or a ``-m`` / ``-k`` expression value.

    A token with a ``/`` or a ``.py`` ending is always a path. A bare word
    (``samplers``) only counts if it is really a directory under ``tests/``. A
    leading ``$VAR/`` is dropped first, but a ``$`` anywhere else means the token
    can't be resolved to a known path.
    """
    if not token or token.startswith("-"):
        return False
    if _is_env_assignment(token):
        return False
    token = _LEADING_VAR_PREFIX.sub("", token)
    if "$" in token:
        return False
    if "/" in token or token.endswith(".py"):
        return True
    return (TESTS_DIR / normalize_test_path(token)).is_dir()


def _path_is_within(parent: str, path: str) -> bool:
    """True if ``path`` is ``parent`` itself or nested underneath it. A ``''`` /
    ``'.'`` parent (a command run with no path arg) contains everything."""
    if parent in ("", "."):
        return True
    return path == parent or path.startswith(parent + "/")


def _glob_match_path(pattern: str, path: str) -> bool:
    """Like :func:`fnmatch.fnmatch`, but ``*`` and ``?`` do not match across a
    ``/`` - the way the shell and ``pytest`` treat an unexpanded glob path
    argument (``kernels/test_foo_*.py`` selects one directory, not the tree).
    A ``**`` component keeps the cross-separator meaning."""
    if not any(ch in pattern for ch in "*?["):
        return pattern == path
    if "**" in pattern:
        return fnmatch.fnmatch(path, pattern)
    pattern_parts = pattern.split("/")
    path_parts = path.split("/")
    return len(pattern_parts) == len(path_parts) and all(
        fnmatch.fnmatch(part, glob) for part, glob in zip(path_parts, pattern_parts)
    )


# --------------------------------------------------------------------------- #
# Selections - "which test files does this command run?"
# --------------------------------------------------------------------------- #


@dataclass
class PytestSelection:
    """A ``pytest``-style command: run everything under ``included_paths``
    except what an ``--ignore`` (path prefix) or ``--ignore-glob`` (whole-path
    ``fnmatch``) removes."""

    included_paths: list[str] = field(default_factory=list)
    ignored_paths: list[str] = field(default_factory=list)
    ignored_globs: list[str] = field(default_factory=list)

    def runs(self, test_file: str) -> bool:
        included = any(self._arg_covers(arg, test_file) for arg in self.included_paths)
        if not included:
            return False
        if any(self._arg_covers(arg, test_file) for arg in self.ignored_paths):
            return False
        # `--ignore-glob` is a plain `fnmatch` over the whole path - `*_tp.py`
        # drops `kernels/nested/test_a_tp.py`, unlike a positional glob arg.
        return not any(
            fnmatch.fnmatch(test_file, normalize_test_path(glob))
            for glob in self.ignored_globs
        )

    @staticmethod
    def _arg_covers(path_arg: str, test_file: str) -> bool:
        """True if a single pytest path argument selects ``test_file`` - by
        exact match, by being a parent directory, or as a glob."""
        path_arg = normalize_test_path(path_arg)
        if _path_is_within(path_arg, test_file):
            return True
        return _glob_match_path(path_arg, test_file)


@dataclass
class FindSelection:
    """A ``find <root> -name ... [-not -name ...] [-maxdepth N] | xargs pytest``
    pipeline. ``pytest`` runs one file per ``find`` hit, so the ``find``
    expression alone defines the selected set."""

    root: str
    name_globs: list[str] = field(default_factory=list)
    exclude_name_globs: list[str] = field(default_factory=list)
    max_depth: int | None = None

    def runs(self, test_file: str) -> bool:
        root = normalize_test_path(self.root)
        if not _path_is_within(root, test_file):
            return False

        # `find <root>` puts a direct child at depth 1, whether <root> is a real
        # subdir ("compile") or the tests/ root itself ("" / ".").
        below = test_file if root in ("", ".") else test_file[len(root) :]
        depth_below_root = below.lstrip("/").count("/") + 1
        if self.max_depth is not None and depth_below_root > self.max_depth:
            return False

        filename = test_file.rsplit("/", 1)[-1]
        if self.name_globs and not _glob_any(filename, self.name_globs):
            return False
        return not _glob_any(filename, self.exclude_name_globs)


Selection = PytestSelection | FindSelection


def _glob_any(name: str, globs: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, glob) for glob in globs)


# --------------------------------------------------------------------------- #
# Parsing CI commands into selections
# --------------------------------------------------------------------------- #


def _parse_pytest_command(tokens: list[str]) -> PytestSelection:
    """Read a ``pytest ...`` invocation's tokens into a :class:`PytestSelection`.

    Positional path args are the included paths; ``--ignore`` values (and a
    bare-path ``--deselect``) are ignored paths, ``--ignore-glob`` values are
    ignored globs. Value-taking options and their values, and every other flag,
    are skipped. ``-k`` / ``-m`` filtering is intentionally *not* modelled - a
    file that a marker expression narrows is still "run".
    """
    selection = PytestSelection()

    # Advance past the pytest token itself (and any env-var / wrapper prefix).
    tokens = iter(tokens)
    for token in tokens:
        if token in PYTEST_COMMANDS:
            break

    for token in tokens:
        opt, eq, inline = token.partition("=")
        if opt in PYTEST_IGNORE_OPTIONS:
            value = inline if eq else next(tokens, "")
            if opt == "--ignore-glob":
                selection.ignored_globs.append(value)
            else:
                selection.ignored_paths.append(value)
        elif opt == PYTEST_DESELECT_OPTION:
            value = inline if eq else next(tokens, "")
            # A `path::node_id` deselect leaves the file collected; only a
            # bare-path deselect removes anything at file granularity.
            if value and "::" not in value:
                selection.ignored_paths.append(value)
        elif token in PYTEST_OPTIONS_WITH_VALUE:
            next(tokens, None)  # consume and discard the option's value
        elif token.startswith("-"):
            continue
        elif _token_is_test_path(token):
            selection.included_paths.append(token)

    return selection


def _parse_find_command(tokens: list[str]) -> FindSelection | None:
    """Read a ``find <root> -name ...`` command's tokens into a
    :class:`FindSelection`, or return ``None`` if there is no usable ``find``."""
    if "find" not in tokens:
        return None

    root: str | None = None
    name_globs: list[str] = []
    exclude_name_globs: list[str] = []
    max_depth: int | None = None

    tokens = iter(tokens[tokens.index("find") + 1 :])
    negated = False
    for token in tokens:
        if token in ("-not", "!"):
            negated = True
            continue
        if token in ("-name", "-iname"):
            glob = next(tokens, "").strip("'\"")
            (exclude_name_globs if negated else name_globs).append(glob)
        elif token == "-maxdepth":
            depth = next(tokens, "")
            max_depth = int(depth) if depth.isdigit() else None
        elif (
            root is None and not token.startswith("-") and token not in PYTEST_COMMANDS
        ):
            root = token
        negated = False

    if root is None:
        return None
    return FindSelection(root, name_globs, exclude_name_globs, max_depth)


def _parse_shell_script(tokens: list[str], visited: set[str]) -> list[Selection]:
    """Follow a ``bash tests/foo.sh`` command: read the script and parse the
    pytest / runner lines inside it. ``visited`` stops a script that (directly or
    otherwise) refers back to itself; nesting is one level deep in practice.
    """
    script_arg = next((t for t in tokens[1:] if t.endswith(".sh")), None)
    if script_arg is None:
        return []

    rel = script_arg.strip().strip("'\"").replace("/vllm-workspace/", "")
    rel = rel.replace("vllm-workspace/", "").removeprefix("./").removeprefix("/")
    candidates = [REPO_ROOT / rel, TESTS_DIR / normalize_test_path(rel)]
    script_path = next((p for p in candidates if p.is_file()), None)
    if script_path is None or str(script_path) in visited:
        return []
    visited.add(str(script_path))

    # Join `\`-continued lines (a `pytest ... \` with its path on the next line
    # is common), then track literal `VAR=value` assignments so a later
    # `pytest "$VAR"` - the import-canary pattern - resolves to a real path.
    text = script_path.read_text().replace("\\\n", " ")
    var_values: dict[str, str] = {}
    selections: list[Selection] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        assignment = re.match(r"^(\w+)=(\S+)$", line)
        if assignment:
            var_values[assignment.group(1)] = assignment.group(2).strip("'\"")
            continue
        if var_values:
            line = re.sub(
                r"\$\{(\w+)\}|\$(\w+)",
                lambda m: var_values.get(m.group(1) or m.group(2), m.group(0)),
                line,
            )
        selections.extend(_parse_command(line, visited))
    return selections


# Shell tokens (or runs of them) that separate one sub-command from the next.
_SHELL_SEP_CHARS = set("|&;()<>")
_RUNNER_KEYWORDS = PYTEST_COMMANDS | FILE_RUNNER_COMMANDS


def _split_shell_stages(command: str) -> list[tuple[str, list[str]]]:
    """Tokenize a shell line - honouring quotes - and split it at the shell
    operators, tagging each stage with the operator token that *precedes* it
    (``""`` for the first stage, then ``"|"`` / ``"&&"`` / ``"||"`` / ``";"`` /
    ``"&"`` / a subshell paren). Knowing which operator joins two stages is what
    lets a ``find`` sweep count only when it is genuinely piped into ``pytest``,
    not merely on the same line as one. A line that cannot be tokenized
    (unbalanced quote, ...) yields nothing: better to miss a selection than to
    misparse one. Splitting *after* tokenizing means an operator inside a quoted
    argument stays part of that argument."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return []

    stages: list[tuple[str, list[str]]] = [("", [])]
    for token in tokens:
        if token and set(token) <= _SHELL_SEP_CHARS:
            stages.append((token, []))
        else:
            stages[-1][1].append(token)
    return [(sep, sub) for sep, sub in stages if sub]


def _looks_like_command_string(token: str) -> bool:
    """True for a single token that is itself a command line - e.g. a quoted
    ``"pytest a.py && torchrun b.py"`` passed as an argument to a runner
    script."""
    return " " in token and any(word in _RUNNER_KEYWORDS for word in token.split())


def _classify_subcommand(tokens: list[str], visited: set[str]) -> list[Selection]:
    """Map one already-tokenized sub-command to the selections it implies:
    find / pytest / ``bash <script>`` / direct file runner. Anything else
    contributes nothing - better to miss a real selection than to invent
    coverage that isn't there."""
    if not tokens:
        return []

    is_find = "find" in tokens and any(t in ("-name", "-iname") for t in tokens)
    if is_find:
        find_selection = _parse_find_command(tokens)
        return [find_selection] if find_selection is not None else []

    if any(token in PYTEST_COMMANDS for token in tokens):
        return [_parse_pytest_command(tokens)]

    # `bash <script>`, possibly behind a wrapper (`env FOO=1 bash x.sh`,
    # `if ! bash x.sh; then`) - look for the interpreter or a `.sh` anywhere in
    # the stage, not just at token 0. `_parse_shell_script` still needs a real
    # `.sh` file to do anything.
    if (
        "bash" in tokens
        or "sh" in tokens
        or any(t.rsplit("/", 1)[-1].endswith(".sh") for t in tokens)
    ):
        selections = _parse_shell_script(tokens, visited)
        # Runner scripts (run-multi-node-test.sh, ...) take the real test
        # commands as quoted string arguments - parse those too.
        for arg in tokens[1:]:
            if _looks_like_command_string(arg):
                selections.extend(_parse_command(arg, visited))
        return selections

    # `python` / `torchrun` / `coverage <file>.py`, possibly after an env-var
    # prefix - so search for the runner rather than assuming it is token 0.
    runner_index = next(
        (
            i
            for i, t in enumerate(tokens)
            if t.rsplit("/", 1)[-1] in FILE_RUNNER_COMMANDS
        ),
        None,
    )
    if runner_index is not None:
        # `python foo.py a b/c --suffix v1` executes only `foo.py`; the tokens
        # after it are that script's argv - treating them as test paths tethers
        # whatever directory an argument happens to name (`v1` -> `tests/v1`).
        script = next(
            (t for t in tokens[runner_index + 1 :] if _token_is_test_path(t)), None
        )
        if script is not None:
            return [PytestSelection(included_paths=[script])]

    return []


def _find_feeds_pytest(stages: list[tuple[str, list[str]]], find_index: int) -> bool:
    """True if the ``find`` stage at ``find_index`` actually pipes its matches
    into ``pytest``: an ``-exec pytest`` in the ``find`` itself, or an
    ``xargs pytest`` reached through one or more consecutive ``|`` pipes. A
    ``find`` piped to ``wc`` / ``grep``, or one merely sharing a line with a
    ``pytest`` in a separate ``;`` / ``&&`` command, feeds it nothing."""
    find_tokens = stages[find_index][1]
    for i, token in enumerate(find_tokens[:-1]):
        if token in ("-exec", "-execdir") and find_tokens[i + 1] in PYTEST_COMMANDS:
            return True
    for sep, tokens in stages[find_index + 1 :]:
        if sep != "|":
            break
        if "xargs" in tokens and any(t in PYTEST_COMMANDS for t in tokens):
            return True
    return False


def _parse_command(command: str, visited: set[str] | None = None) -> list[Selection]:
    """Turn one yaml ``commands:`` entry into the selections it implies.

    The line is tokenized (respecting quotes) and split into stages at the shell
    operators, and each stage is classified independently so that, e.g., a
    ``find`` and a ``pytest`` on the same line are both handled. A ``find -name``
    sweep is kept only when :func:`_find_feeds_pytest` confirms its output
    actually reaches ``pytest`` - otherwise it would invent coverage.
    """
    visited = visited if visited is not None else set()
    stages = _split_shell_stages(command)
    selections: list[Selection] = []
    for index, (_, tokens) in enumerate(stages):
        for selection in _classify_subcommand(tokens, visited):
            if isinstance(selection, FindSelection) and not _find_feeds_pytest(
                stages, index
            ):
                continue
            selections.append(selection)
    return selections


def _iter_job_steps(yaml_doc: dict):
    """Yield each job step in a test_areas yaml, plus a merged view of each
    ``mirror.*`` sub-step. The AMD mirror can override ``commands``, so it has to
    be checked as its own step."""
    for step in yaml_doc.get("steps") or []:
        if not isinstance(step, dict):
            continue
        yield step

        for mirror_step in (step.get("mirror") or {}).values():
            if isinstance(mirror_step, dict):
                merged = {**step, **mirror_step}
                merged.pop("mirror", None)
                yield merged


def _pipeline_yaml_paths() -> list[Path]:
    """Every Buildkite pipeline yaml whose jobs count as CI coverage: the
    ``test_areas/`` set plus the legacy ``test-amd.yaml`` (see ``TEST_AMD_YAML``).
    """
    paths = sorted(TEST_AREAS_DIR.glob("*.yaml"))
    if TEST_AMD_YAML.is_file():
        paths.append(TEST_AMD_YAML)
    return paths


def load_selections() -> list[Selection]:
    """Parse every CI pipeline yaml into the full list of selections - the CI's
    complete picture of which test files it runs.

    A yaml that doesn't parse is fatal: silently skipping it would drop whatever
    coverage it defines and produce false "untethered" reports.
    """
    selections: list[Selection] = []
    for yaml_path in _pipeline_yaml_paths():
        try:
            yaml_doc = yaml.safe_load(yaml_path.read_text()) or {}
        except yaml.YAMLError as e:
            rel = yaml_path.relative_to(REPO_ROOT)
            raise SystemExit(f"error: {rel} is not valid YAML: {e}") from None
        if not isinstance(yaml_doc, dict):
            rel = yaml_path.relative_to(REPO_ROOT)
            raise SystemExit(
                f"error: {rel} is not a Buildkite pipeline "
                f"(top-level {type(yaml_doc).__name__}, expected a mapping)"
            )
        for step in _iter_job_steps(yaml_doc):
            commands = list(step.get("commands") or [])
            if step.get("command"):  # some steps use the singular form
                commands.append(step["command"])
            for command in commands:
                if isinstance(command, str):
                    selections.extend(_parse_command(command))
    return selections


def is_tethered(test_file: str, selections: list[Selection]) -> bool:
    """True if any CI selection runs ``test_file`` (path relative to ``tests/``
    or to the repo root - both are accepted)."""
    normalized = normalize_test_path(test_file)
    return any(selection.runs(normalized) for selection in selections)


# --------------------------------------------------------------------------- #
# Test-file inventory and allowlist
# --------------------------------------------------------------------------- #


def is_test_module(repo_path: str) -> bool:
    """True if ``repo_path`` is a pytest test module CI is expected to run - a
    ``tests/**/test_*.py`` or ``*_test.py``, but not ``conftest.py`` /
    ``__init__.py`` or other helper modules."""
    if not repo_path.startswith("tests/") or not repo_path.endswith(".py"):
        return False
    filename = repo_path.rsplit("/", 1)[-1]
    if filename in ("conftest.py", "__init__.py"):
        return False
    return filename.startswith("test_") or filename.endswith("_test.py")


def all_test_modules() -> list[str]:
    """Every tracked test module under ``tests/``, sorted. Tracked-only (via
    ``git ls-files``) so a developer's local scratch files are never flagged."""
    tracked = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "tests/"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return sorted(path for path in tracked.splitlines() if is_test_module(path))


def load_allowlist() -> set[str]:
    """Read the allowlist file into a set of repo-relative paths. Blank lines and
    ``#`` comments (whole-line or trailing) are ignored."""
    if not ALLOWLIST_PATH.exists():
        return set()

    paths = set()
    for line in ALLOWLIST_PATH.read_text().splitlines():
        path = line.split("#", 1)[0].strip()
        if path:
            paths.add(path)
    return paths


def allowlist_entries_missing_reason() -> list[str]:
    """Allowlist paths with no trailing ``# reason`` on their own line. Every gap
    has to say why the test can't be wired into a job, so a bare ``path`` line
    (the easy way to silence the gate) is itself an error."""
    if not ALLOWLIST_PATH.exists():
        return []

    offenders = []
    for line in ALLOWLIST_PATH.read_text().splitlines():
        path, _, comment = line.partition("#")
        if path.strip() and not comment.strip():
            offenders.append(path.strip())
    return offenders


# --------------------------------------------------------------------------- #
# The two check modes
# --------------------------------------------------------------------------- #


def run_full_scan(
    selections: list[Selection], allowlist: set[str], *, strict: bool
) -> int:
    """Scan every test module. Untethered files that aren't allowlisted are
    always errors. Allowlist entries that have become tethered, or whose file no
    longer exists, are reported either way but are only errors when ``strict``
    (an explicit ``--all`` run, as opposed to a PR that happened to touch CI
    config - which shouldn't be blocked by unrelated allowlist cleanup).
    """
    untethered = [
        path
        for path in all_test_modules()
        if path not in allowlist and not is_tethered(path, selections)
    ]
    now_tethered = sorted(path for path in allowlist if is_tethered(path, selections))
    deleted = sorted(path for path in allowlist if not (REPO_ROOT / path).exists())

    for path in untethered:
        print(f"error: {path} is not run by any Buildkite job")
    for path in now_tethered:
        print(f"note: {path} is now tethered - remove it from the allowlist")
    for path in deleted:
        print(f"note: {path} no longer exists - remove it from the allowlist")

    if untethered:
        print(
            f"\n{len(untethered)} test file(s) exist but run in no CI job. Wire "
            "each into a job's `commands` in .buildkite/test_areas/ (or "
            ".buildkite/test-amd.yaml), or add it "
            f"to {ALLOWLIST_PATH.relative_to(REPO_ROOT)} with a reason."
        )

    error_count = len(untethered)
    if strict:
        error_count += len(now_tethered) + len(deleted)
    return 1 if error_count else 0


def run_changed_files_check(
    paths: list[str], selections: list[Selection], allowlist: set[str]
) -> int:
    """Check only the given paths (what pre-commit passes). Non-test files, and
    paths that don't exist on disk, are ignored."""
    untethered = []
    for raw_path in paths:
        repo_path = _to_repo_relative(raw_path)
        if not is_test_module(repo_path) or not (REPO_ROOT / repo_path).is_file():
            continue
        if repo_path in allowlist or is_tethered(repo_path, selections):
            continue
        untethered.append(repo_path)

    for path in untethered:
        print(f"error: {path} is not run by any Buildkite job")
    if untethered:
        print(
            "\nAdd the test to a job's `commands` (and `source_file_dependencies`)"
            " in .buildkite/test_areas/, or, if it is intentionally not run in "
            f"CI, add it to {ALLOWLIST_PATH.relative_to(REPO_ROOT)} with a "
            "comment explaining why."
        )
    return 1 if untethered else 0


def _to_repo_relative(path: str) -> str:
    if Path(path).is_absolute():
        path = str(Path(path).resolve().relative_to(REPO_ROOT))
    return path.replace("\\", "/")


def _change_set_touches_ci_config(paths: list[str]) -> bool:
    """A change to a pipeline yaml or to the allowlist can orphan a test that is
    not itself in the change set, so those changes force a full scan."""
    return any(
        ".buildkite/test_areas/" in path
        or path.endswith("test-amd.yaml")
        or path.endswith("test_tethering_allowlist.txt")
        for path in paths
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="test files to check")
    parser.add_argument(
        "--all", action="store_true", help="scan every test file in the tree"
    )
    args = parser.parse_args()

    selections = load_selections()
    allowlist = load_allowlist()

    unreasoned = allowlist_entries_missing_reason()
    for path in unreasoned:
        print(
            f"error: {ALLOWLIST_PATH.relative_to(REPO_ROOT)}: {path} has no "
            "trailing '# <reason>' - say why it can't be wired into a job"
        )

    if args.all or _change_set_touches_ci_config(args.files):
        rc = run_full_scan(selections, allowlist, strict=args.all)
    else:
        rc = run_changed_files_check(args.files, selections, allowlist)
    return 1 if (unreasoned or rc) else 0


if __name__ == "__main__":
    sys.exit(main())
