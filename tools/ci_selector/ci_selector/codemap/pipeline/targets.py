# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Map a step's commands to the test files and scripts it invokes.

Every command is a test target, benign, or unparsable. Only unparsable widens
selection, and a test keeps that bucket empty.

Narrowing flags are recorded but never shrink the target set. Over-running a
step is fine; proving nothing invokes a file must not rest on a filter that
only applies at runtime.
"""

from __future__ import annotations

import posixpath
import shlex
from dataclasses import dataclass, field
from pathlib import Path

import regex as re

from ...handwritten import (
    BENIGN_CMDS,
    CONTAINER_WORKSPACE,
    PYTEST_VALUE_FLAGS,
    UNWRAP_BARE_FLAGS,
    UNWRAP_CMDS,
    UNWRAP_VALUE_FLAGS,
)
from .step import Step

VAR_PREFIX_RE = re.compile(r"^\"?\$\{?\w+\}?\"?/")

FIND_XARGS_RE = re.compile(
    r"\bfind\s+(?P<dir>\S+)(?P<args>.*?)\|\s*xargs\b.*?\bpytest\b"
)
NOT_NAME_RE = re.compile(r"-not\s+-name\s+'?\"?([\w*.\[\]-]+)")
IF_THEN_RE = re.compile(
    r"^if\s+.*?;\s*then\s+(?P<then>.*?)(?:;\s*else\s+(?P<else>.*?))?;\s*fi\s*;?\s*$"
)
OPERATORS = {"&&", "||", ";"}


@dataclass
class Target:
    path: str  # repo-relative posix path (file or directory)
    kind: str  # "pytest" | "script"
    narrowing: list[str] = field(default_factory=list)  # -m/-k expressions
    via: str | None = None  # shell script that carries the invocation


@dataclass
class StepTargets:
    step_id: str
    targets: list[Target] = field(default_factory=list)
    data_files: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)
    benign: list[str] = field(default_factory=list)
    unparsable: list[str] = field(default_factory=list)
    dangling: list[str] = field(default_factory=list)
    # Test paths in a container payload that are missing from this checkout.
    # Reported, never escalated: an image-only path and a renamed one look the
    # same here, and escalating both runs the step on every PR.
    container_tests: list[str] = field(default_factory=list)
    scripts_seen: list[str] = field(default_factory=list)
    # The step's commands plus every script body it reaches. Registered keys
    # are matched here, because a job can pick a backend by name in argv
    # instead of importing it.
    haystack: str = ""

    def add_target(self, path: str | None, kind: str, narrowing=(), via=None):
        if path is None:
            return
        self.targets.append(
            Target(path=path, kind=kind, narrowing=list(narrowing), via=via)
        )


def working_dir_to_repo_rel(working_dir: str) -> str:
    """Map the container working dir onto a repo-relative cwd ('' = root)."""
    wd = working_dir.rstrip("/")
    if wd in ("", ".", CONTAINER_WORKSPACE):
        return ""
    if wd.startswith(f"{CONTAINER_WORKSPACE}/"):
        return wd[len(CONTAINER_WORKSPACE) + 1 :]
    return ""  # unknown absolute dir: fall back to repo root


class CommandParser:
    """Parse one step's commands. Script recursion is delegated to the scanner,
    which calls back into parse_pytest, resolve_path, chdir, cwd and out."""

    def __init__(self, repo: Path, step: Step, script_scanner=None):
        self.repo = repo
        self.step = step
        self.out = StepTargets(step_id=step.step_id)
        self.cwd = working_dir_to_repo_rel(step.working_dir)
        self.script_scanner = script_scanner

    def run(self) -> StepTargets:
        for command in self.step.commands:
            self._process_command(command)
        self.out.haystack = "\n".join(self.step.commands) + self.out.haystack
        return self.out

    def _process_command(self, command: str) -> None:
        try:
            tokens = shlex.split(command, comments=True)
        except ValueError:
            tokens = None
        if tokens is not None and any("\n" in t for t in tokens):
            # One invocation carrying a quoted multi-line block argument.
            self._process_segment(tokens, command)
            return
        for line in command.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            self._process_line(line)

    def _process_line(self, line: str) -> None:
        m = IF_THEN_RE.match(line)
        if m:
            for branch in (m.group("then"), m.group("else")):
                if branch:
                    self._process_line(branch.strip())
            return
        m = FIND_XARGS_RE.search(line)
        if m:
            excluded = NOT_NAME_RE.findall(m.group("args"))
            path = self.resolve_path(m.group("dir"))
            self.out.add_target(
                path,
                "pytest",
                narrowing=[f"find-exclude:{x}" for x in excluded],
            )
            if path is None:
                self.out.dangling.append(line)
            return
        try:
            tokens = shlex.split(line, comments=True)
        except ValueError:
            self.out.unparsable.append(line)
            return
        for segment in _split_segments(tokens):
            self._process_segment(segment, line)

    def _process_segment(self, tokens: list[str], raw: str) -> None:
        tokens = _strip_env_prefix(tokens)
        if not tokens:
            self.out.benign.append(raw)
            return
        cmd = tokens[0]
        if cmd in UNWRAP_CMDS:
            sub = UNWRAP_CMDS[cmd]
            # Global flags can precede the subcommand (uv -q run ...), so
            # strip on both sides of it.
            rest = _strip_wrapper_flags(tokens[1:])
            if sub is None or (rest and rest[0] == sub):
                if sub is not None:
                    rest = rest[1:]
                rest = _strip_wrapper_flags(rest)
                if rest:
                    self._process_segment(rest, raw)
                    return
            # bare wrapper, or a subcommand we do not unwrap (uv pip ...):
            # fall through to the benign check below
        if cmd in ("docker", "docker-compose"):
            # Walking docker's argv grammar is unreliable, so a docker form
            # wrapping a test call is unparsable and preflight force-selects.
            wrapped_test = any("pytest" in t or t.endswith(".py") for t in tokens[1:])
            bucket = self.out.unparsable if wrapped_test else self.out.benign
            bucket.append(raw)
            return
        if cmd == "cd" and len(tokens) > 1:
            self.chdir(tokens[1])
            return
        if cmd == "pytest":
            self.parse_pytest(tokens[1:])
            return
        if cmd in ("python", "python3"):
            self._parse_python(tokens[1:], raw)
            return
        if cmd == "torchrun":
            self._parse_torchrun(tokens[1:], raw)
            return
        if cmd in ("bash", "sh") or cmd.endswith(".sh"):
            args = tokens[1:] if cmd in ("bash", "sh") else tokens
            self._parse_script_invocation(args, raw)
            return
        if cmd in BENIGN_CMDS or "=" in cmd:
            self.out.benign.append(raw)
            return
        self.out.unparsable.append(raw)

    def parse_pytest(
        self, args: list[str], via: str | None = None, container: bool = False
    ) -> None:
        narrowing: list[str] = []
        positionals: list[str] = []
        config_lists: list[str] = []
        i = 0
        while i < len(args):
            tok = args[i]
            if tok.startswith("--") and "=" in tok:
                flag, value = tok.split("=", 1)
                self._handle_pytest_value_flag(flag, value, narrowing, config_lists)
                i += 1
            elif tok in PYTEST_VALUE_FLAGS:
                value = args[i + 1] if i + 1 < len(args) else ""
                self._handle_pytest_value_flag(tok, value, narrowing, config_lists)
                i += 2
            elif tok.startswith("-"):
                i += 1
            else:
                positionals.append(tok)
                i += 1
        for pos in positionals:
            base = pos.split("::", 1)
            func = base[1] if len(base) > 1 else None
            narrow = list(narrowing)
            if func:
                narrow.append(f"func:{func}")
            if "*" in base[0]:
                expanded = self._expand_glob(base[0])
                if not expanded:
                    # A glob matching nothing is the same hole as a rename.
                    self._mark_dangling(pos, container)
                    continue
                for path in expanded:
                    self.out.add_target(path, "pytest", narrowing=narrow, via=via)
                continue
            path = self.resolve_path(base[0])
            if path is None:
                self._mark_dangling(pos, container)
                continue
            self.out.add_target(path, "pytest", narrowing=narrow, via=via)
        for value in config_lists:
            self._resolve_config_list(value)

    def _handle_pytest_value_flag(
        self,
        flag: str,
        value: str,
        narrowing: list[str],
        config_lists: list[str],
    ) -> None:
        if flag in ("-m", "-k"):
            narrowing.append(f"{flag}:{value}")
        elif flag in ("--ignore", "--deselect"):
            resolved = self.resolve_path(value.split("::", 1)[0])
            if resolved:
                self.out.ignored.append(resolved)
        elif flag == "--config-list-file":
            config_lists.append(value)

    def _resolve_config_list(self, value: str) -> None:
        resolved = self.resolve_path(value)
        if resolved is None:
            # The gsm8k harness resolves --config-list-file relative to the
            # test file's own directory, not the shell cwd.
            for t in self.out.targets:
                base = (
                    t.path if not t.path.endswith(".py") else posixpath.dirname(t.path)
                )
                candidate = posixpath.normpath(posixpath.join(base, value))
                if (self.repo / candidate).is_file():
                    resolved = candidate
                    break
        if resolved:
            self.out.data_files.append(resolved)
        else:
            self._mark_dangling(value)

    def _parse_python(self, args: list[str], raw: str) -> None:
        i = 0
        while i < len(args):
            tok = args[i]
            if tok == "-c":
                self.out.benign.append(raw)  # inline snippet, not a test file
                return
            if tok == "-m":
                module = args[i + 1] if i + 1 < len(args) else ""
                if module == "pytest":
                    self.parse_pytest(args[i + 2 :])
                else:
                    # `python -m vllm...` server drivers: not a test file.
                    self.out.benign.append(raw)
                return
            if tok.endswith(".py"):
                path = self.resolve_path(tok)
                if path is None:
                    self._mark_dangling(tok)
                else:
                    self.out.add_target(path, "script")
                # data-file args after the script (e.g. `-c models.txt`)
                self._collect_file_args(args[i + 1 :])
                return
            i += 1
        self.out.unparsable.append(raw)

    def _parse_torchrun(self, args: list[str], raw: str) -> None:
        for tok in args:
            if tok.endswith(".py"):
                path = self.resolve_path(tok)
                if path is None:
                    self._mark_dangling(tok)
                else:
                    self.out.add_target(path, "script")
                return
        self.out.unparsable.append(raw)

    def _parse_script_invocation(self, args: list[str], raw: str) -> None:
        # Skip shell flags (`bash -lc '<cmds>'` etc.) to find the payload.
        while args and args[0].startswith("-"):
            args = args[1:]
        if not args:
            self.out.unparsable.append(raw)
            return
        script_tok, rest = args[0], args[1:]
        if not script_tok.endswith(".sh"):
            # `bash -c/-lc` style: payload is an inline command string.
            self._process_block(script_tok)
            return
        script = self.resolve_path(script_tok) or self._resolve_from_root(script_tok)
        if script is None:
            self._mark_dangling(script_tok)
            return
        self.out.scripts_seen.append(script)
        if self.script_scanner is not None:
            self.script_scanner(script, self)
        for tok in rest:
            if " " in tok or "\n" in tok:
                # A quoted command string passed as one argv token. Always
                # parsed, or a block holding only a nested `bash x.sh` leaves
                # its step with no targets at all.
                self._process_block(tok)
            else:
                self._collect_file_args([tok])

    def _process_block(self, block: str) -> None:
        """Parse a quoted command block passed as an argv string. cwd is kept:
        resolve_path falls back to the repo root anyway, so both root-relative
        and step-relative blocks resolve."""
        saved = self.cwd
        for line in block.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                self._process_line(line)
        self.cwd = saved

    def _expand_glob(self, token: str) -> list[str]:
        base = self._join(token)
        if base is None or base.startswith(".."):
            return []
        return sorted(p.relative_to(self.repo).as_posix() for p in self.repo.glob(base))

    def _collect_file_args(self, args: list[str]) -> None:
        for tok in args:
            if tok.startswith("-") or "$" in tok or "/" not in tok:
                continue
            resolved = self.resolve_path(tok)
            if resolved and (self.repo / resolved).is_file():
                self.out.data_files.append(resolved)

    def chdir(self, arg: str) -> None:
        if "$" in arg:
            return  # variable cd: keep current cwd best-effort
        joined = self._join(arg)
        if joined is not None and (self.repo / joined).is_dir():
            self.cwd = joined
            return
        # `cd tests` inside a block whose tracked cwd is already tests/:
        # fall back to root-relative.
        root_rel = posixpath.normpath(arg)
        if not root_rel.startswith("..") and (self.repo / root_rel).is_dir():
            self.cwd = root_rel

    def _join(self, token: str) -> str | None:
        if token.startswith("/"):
            # An absolute path outside the workspace is not a repo path. The
            # repo-root fallback is for the working_dir field only: on a path
            # token it returns '', which exists, so a foreign path became a
            # target that matched nothing and silenced the warning.
            wd = token.rstrip("/")
            if wd != CONTAINER_WORKSPACE and not wd.startswith(
                f"{CONTAINER_WORKSPACE}/"
            ):
                return None
            return working_dir_to_repo_rel(token)
        joined = posixpath.normpath(posixpath.join(self.cwd, token))
        return "" if joined == "." else joined

    def resolve_path(self, token: str) -> str | None:
        """Resolve a path token against cwd; None if it doesn't exist."""
        if "\n" in token or " " in token or "{}" in token:
            return None
        if "$" in token:
            # `${GIT_ROOT}/tests/...`: the var is a repo-root prefix, so try
            # the rest from root.
            stripped = VAR_PREFIX_RE.sub("", token)
            if "$" in stripped:
                return None
            return self._resolve_from_root(stripped)
        joined = self._join(token)
        if (
            joined is not None
            and not joined.startswith("..")
            and (self.repo / joined).exists()
        ):
            return joined
        return self._resolve_from_root(token)

    def _resolve_from_root(self, token: str) -> str | None:
        if "$" in token or "{}" in token:
            return None
        candidate = posixpath.normpath(token)
        if (
            candidate
            and candidate not in (".", "..")
            and not candidate.startswith("../")
            and (self.repo / candidate).exists()
        ):
            return candidate
        return None

    def _mark_dangling(self, token: str, container: bool = False) -> None:
        # Variable/placeholder tokens are unknowable statically, not findings.
        if "$" in token or "{}" in token:
            return
        # An unresolvable container path is reported, not escalated. See
        # StepTargets.container_tests.
        if container:
            self.out.container_tests.append(token)
        else:
            self.out.dangling.append(token)


def map_step(repo: Path, step: Step, script_scanner=None) -> StepTargets:
    return CommandParser(repo, step, script_scanner).run()


def _split_segments(tokens: list[str]) -> list[list[str]]:
    """Split on shell operators. The right side of a pipe is skipped, but `&&`
    and `;` resume, so `torchrun x.py | grep ok && pytest y` still yields the
    pytest call."""
    segments: list[list[str]] = [[]]
    skipping = False
    for tok in tokens:
        if tok in OPERATORS:
            segments.append([])
            skipping = False
        elif tok == "|":
            skipping = True  # find|xargs was already handled at line level
        elif not skipping:
            segments[-1].append(tok)
    return [s for s in segments if s]


def _strip_env_prefix(tokens: list[str]) -> list[str]:
    i = 0
    while i < len(tokens) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[i]):
        i += 1
    return tokens[i:]


def _strip_wrapper_flags(tokens: list[str]) -> list[str]:
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in UNWRAP_VALUE_FLAGS:
            i += 2
        elif t in UNWRAP_BARE_FLAGS or (t.startswith("-") and "=" in t):
            i += 1
        else:
            break
    return tokens[i:]
