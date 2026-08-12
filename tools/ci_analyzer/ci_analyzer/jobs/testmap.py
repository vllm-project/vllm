# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Map a step's commands to the test files/scripts it invokes.

Classification contract: every command unit is TEST-TARGET, BENIGN
(allowlisted non-test shapes), or UNPARSABLE. Only UNPARSABLE ever triggers
conservative selection; the enforced bar is UNPARSABLE empty at HEAD.

Markers (-m), -k, --ignore and --deselect are recorded as narrowing metadata
but never shrink the target set: selection may over-run a step, and uninvoked
proof never relies on narrowing.
"""

from __future__ import annotations

import posixpath
import shlex
from dataclasses import dataclass, field
from pathlib import Path

import regex as re

from ..curated import (
    BENIGN_CMDS,
    PYTEST_VALUE_FLAGS,
    UNWRAP_BARE_FLAGS,
    UNWRAP_CMDS,
    UNWRAP_VALUE_FLAGS,
)
from .model import Step

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
    scripts_seen: list[str] = field(default_factory=list)
    # Raw searchable text: the step's commands plus every recursed script
    # body. Registered string keys (e.g. "NixlConnector") are matched here,
    # because e2e jobs select connectors/backends by name in argv/config
    # blobs, not by importing them.
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
    if wd in ("", ".", "/vllm-workspace"):
        return ""
    if wd.startswith("/vllm-workspace/"):
        return wd[len("/vllm-workspace/") :]
    return ""  # unknown absolute dir: fall back to repo root


class CommandParser:
    """Parses one step's commands; script recursion is delegated.

    Public surface consumed by jobs/scripts.py's scanner: parse_pytest,
    resolve_path, chdir, cwd, out."""

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
            # A single invocation carrying a quoted multi-line block arg
            # (e.g. `bash run-cpu-test.sh 30m "<pytest lines>"`).
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
            # Global flags may precede the subcommand (uv -q run ...), so
            # strip before AND after matching it.
            rest = _strip_wrapper_flags(tokens[1:])
            if sub is None or (rest and rest[0] == sub):
                if sub is not None:
                    rest = rest[1:]
                rest = _strip_wrapper_flags(rest)
                if rest:
                    self._process_segment(rest, raw)
                    return
            # bare wrapper or a non-unwrapped subcommand (uv pip ...):
            # fall through to the BENIGN check below
        if cmd in ("docker", "docker-compose"):
            # ANY docker form (exec/run/compose run/global-flag variants)
            # wrapping a test invocation: argv-skipping docker's grammar is
            # unreliable, so go UNPARSABLE and let preflight force-select.
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

    def parse_pytest(self, args: list[str], via: str | None = None) -> None:
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
                    # Matching nothing is the same stale hole as a renamed
                    # file: recording neither a target nor a dangling made it
                    # invisible to the preflight escalation.
                    self._mark_dangling(pos)
                    continue
                for path in expanded:
                    self.out.add_target(path, "pytest", narrowing=narrow, via=via)
                continue
            path = self.resolve_path(base[0])
            if path is None:
                self._mark_dangling(pos)
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
                    # e.g. `python -m vllm...` server drivers: spawn-edge
                    # territory; benign for job mapping.
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
                # Quoted command string passed as an argv block (multi-line
                # v2 shape, or the legacy single-line `&&`-joined shape).
                # Always parsed: the per-line classifier handles benign lines
                # safely, and blocks whose only content is a nested `bash
                # x.sh` (the CPU-Distributed smoke shape) would otherwise
                # leave their steps with zero targets.
                self._process_block(tok)
            else:
                self._collect_file_args([tok])

    def _process_block(self, block: str) -> None:
        """Parse a quoted command block passed as an argv string.

        cwd is kept: resolve_path falls back to repo root anyway, which covers
        blocks that address tests/... explicitly (run-cpu-test.sh $2) while
        step-cwd-relative blocks (run-multi-node-test.sh) still resolve.
        """
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
            # A container path outside the workspace root is not a repo path.
            # working_dir_to_repo_rel's repo-root fallback exists for the
            # working_dir FIELD; applying it to a path token yielded '', which
            # `(repo / "").exists()` accepts, so a foreign absolute path became
            # a phantom root target that matched nothing while silencing both
            # the dangling escalation and the zero-target warning.
            wd = token.rstrip("/")
            if wd != "/vllm-workspace" and not wd.startswith("/vllm-workspace/"):
                return None
            return working_dir_to_repo_rel(token)
        joined = posixpath.normpath(posixpath.join(self.cwd, token))
        return "" if joined == "." else joined

    def resolve_path(self, token: str) -> str | None:
        """Resolve a path token against cwd; None if it doesn't exist."""
        if "\n" in token or " " in token or "{}" in token:
            return None
        if "$" in token:
            # `${GIT_ROOT}/tests/...` in scripts: the var is a repo-root
            # prefix; try the remainder from root.
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

    def _mark_dangling(self, token: str) -> None:
        # Variable/placeholder tokens are unknowable statically, not findings.
        if "$" not in token and "{}" not in token:
            self.out.dangling.append(token)


def map_step(repo: Path, step: Step, script_scanner=None) -> StepTargets:
    return CommandParser(repo, step, script_scanner).run()


def _split_segments(tokens: list[str]) -> list[list[str]]:
    """Split on shell operators; pipe RHS is skipped but `&&`/`;` resume.

    `torchrun x.py | grep ok && pytest y` must yield [torchrun x.py],
    [pytest y]: dropping everything after the first pipe would lose real
    test invocations (the multi-node distributed blocks have this shape).
    """
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
