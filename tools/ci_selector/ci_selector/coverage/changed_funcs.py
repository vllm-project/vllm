# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Turn a diff into the question the record asks: which functions changed, per
file.

Names come from CPython rather than our own AST walk: compile the source and
read the qualnames and line tables off the code objects. That is the mechanism
that produced the recordings, so every name is spelled the way the recorder
wrote it, with no version rules to keep in sync.

A changed line belongs to every code object claiming it, so boundary lines are
owned twice over. That is correct, not merely careful: a default argument
really does run at import time.

Two things this will not do. It never matches on a first line number, which
shifts when anything above a function moves. And it never reads an empty result
as an answer, because "no Python in this file" and "we could not read it" must
not look the same downstream.
"""

from __future__ import annotations

import subprocess
import types
from dataclasses import dataclass, field
from enum import Enum
from inspect import CO_OPTIMIZED
from pathlib import Path

import regex as re

from ..handwritten import (
    RECORDER_SCOPE,
)

MODULE = "<module>"

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class Attribution(str, Enum):
    """Why a file's name sets look the way they do.

    ATTRIBUTED   compiled, with real names read off it
    NAMELESS     nothing to extract, so a kernel or a data file
    FAILED       Python we could not compile or could not read
    """

    ATTRIBUTED = "attributed"
    NAMELESS = "nameless-by-nature"
    FAILED = "could-not-attribute"


@dataclass
class FileQuery:
    """One changed file, and the functions it changed on each side of the diff."""

    path: str  # repo-relative; the head-side path, or the old path for a delete
    status: Attribution
    base_names: frozenset[str] = frozenset()
    head_names: frozenset[str] = frozenset()
    old_path: str | None = None
    # The subset of `names` that runs at import: `<module>` and class bodies.
    # Every step importing the file executes these, so they narrow nothing.
    import_time: frozenset[str] = frozenset()
    # A changed line we could not place in any scope. Means the diff and the
    # source disagree, so the file is not answerable.
    residue: bool = False
    # False for anything the recorder never watches (tests/, benchmarks/, csrc/).
    # The names may be perfectly good; no row can ever hold them.
    in_recorder_scope: bool = True
    # Set from the table: the recorder's names for this file are not faithful to
    # its source, so matching on names is meaningless here. See merge.py.
    unfaithful: bool = False
    # A stand-in for a file the recorder cannot see, not part of the diff.
    # The drop side weighs it like any file; the add side skips it.
    proxy: bool = False
    note: str = ""  # why FAILED, for diagnosis; never load-bearing

    @property
    def names(self) -> frozenset[str]:
        return self.base_names | self.head_names

    @property
    def function_names(self) -> frozenset[str]:
        """Changed names that need a real call, not merely an import."""
        return self.names - self.import_time

    @property
    def fail_open(self) -> bool:
        """True when this file cannot authorize narrowing anything."""
        return (
            self.status is Attribution.FAILED
            or self.residue
            or self.unfaithful
            or not self.in_recorder_scope
        )


@dataclass
class Query:
    base: str
    head: str | None
    files: list[FileQuery] = field(default_factory=list)

    @property
    def fail_open(self) -> bool:
        return any(f.fail_open for f in self.files)

    def restrict(self, paths: set[str]) -> Query:
        """The same query, narrowed to the files one step was selected for.

        Without this, one unanswerable `tests/` file keeps every step on the
        PR. Narrowing cuts both ways though: dropping a file can only remove a
        fail-open or a matching name, and both turn a keep into a drop. So it
        is safe only on a path set COMPLETE for the step, which is what
        `Selection.selected_paths` guarantees. A csrc claim carries its
        wrapper files instead of its own path, so completeness there rests on
        a kernel being reachable only through its wrappers.

        Matches either side of a rename. The FileQuery objects are shared, not
        copied, so a flag set on the query afterwards shows through here too.
        """
        return Query(
            base=self.base,
            head=self.head,
            files=[
                f
                for f in self.files
                if f.path in paths or (f.old_path and f.old_path in paths)
            ],
        )

    def flat_names(self) -> frozenset[str]:
        """(path, qualname) pairs, the shape a row is keyed on.

        A projection of the per-file structure, never a replacement for it:
        Phase 5 needs the per-file link to scope abstention by reason.
        """
        return frozenset(f"{f.path}\t{name}" for f in self.files for name in f.names)


def code_objects(code: types.CodeType):
    yield code
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from code_objects(const)


def names_of(source: str, path: str) -> frozenset[str]:
    """Every qualname CPython produces for this source, as the recorder spells it.

    Raises whatever compile() raises; callers decide that means fail-open.
    """
    return frozenset(c.co_qualname for c in code_objects(compile(source, path, "exec")))


def import_time_names(source: str, path: str) -> frozenset[str]:
    """Names whose code runs on import: the module body and every class body.

    Read off the compiler rather than guessed from the shape of a name. The
    distinction carries weight: every step importing a file runs these, so a
    change to one says almost nothing, while a change inside a real function is
    what lets the record narrow.
    """
    return frozenset(
        c.co_qualname
        for c in code_objects(compile(source, path, "exec"))
        if not c.co_flags & CO_OPTIMIZED
    )


def _spans(code: types.CodeType, total_lines: int) -> dict[str, tuple[int, int]]:
    """qualname -> the inclusive line range it covers, nested scopes included.

    Only used to place a line that no code object claims: a comment, a blank, a
    docstring. `<module>` always spans the whole file so the fallback terminates.
    """
    spans: dict[str, tuple[int, int]] = {}

    def visit(c: types.CodeType) -> tuple[int, int]:
        low = high = c.co_firstlineno
        for _s, _e, lineno in c.co_lines():
            if lineno:
                low = min(low, lineno)
                high = max(high, lineno)
        for const in c.co_consts:
            if isinstance(const, types.CodeType):
                clow, chigh = visit(const)
                low, high = min(low, clow), max(high, chigh)
        prev = spans.get(c.co_qualname)
        spans[c.co_qualname] = (
            (low, high) if prev is None else (min(prev[0], low), max(prev[1], high))
        )
        return low, high

    visit(code)
    spans[MODULE] = (1, max(total_lines, spans.get(MODULE, (1, 1))[1]))
    return spans


def _owners(
    source: str, path: str
) -> tuple[dict[int, set[str]], dict[str, tuple[int, int]]]:
    code = compile(source, path, "exec")
    owners: dict[int, set[str]] = {}
    for c in code_objects(code):
        for _s, _e, lineno in c.co_lines():
            if lineno:
                owners.setdefault(lineno, set()).add(c.co_qualname)
    return owners, _spans(code, source.count("\n") + 1)


def attribute(source: str, path: str, lines: set[int]) -> tuple[frozenset[str], bool]:
    """(names owning those lines, residue). Residue means a changed line sits
    outside every scope in the file, which only happens when the diff and the
    source disagree. Not a gap to paper over, a reason to stop trusting the
    file."""
    if not lines:
        return frozenset(), False
    owners, spans = _owners(source, path)
    found: set[str] = set()
    residue = False
    for line in lines:
        direct = owners.get(line)
        if direct:
            found |= direct
            continue
        # Innermost scope containing the line: smallest span wins, ties kept.
        containing = [
            (hi - lo, name) for name, (lo, hi) in spans.items() if lo <= line <= hi
        ]
        if not containing:
            residue = True
            continue
        best = min(width for width, _ in containing)
        found |= {name for width, name in containing if width == best}
    return frozenset(found), residue


def hunks(
    repo: Path, base: str, head: str | None
) -> dict[str, tuple[set[int], set[int]]]:
    """path -> (base-side changed lines, head-side changed lines). Keyed under
    both sides of a rename so either path finds it. A zero count on one side
    means nothing changed there, which is what a pure add or delete looks
    like."""
    args = [
        "git",
        "-c",
        "core.quotepath=false",
        "-C",
        str(repo),
        "diff",
        "-U0",
        "-M",
        "--no-prefix",
        base,
    ]
    if head:
        args.append(head)
    out = subprocess.run(args, capture_output=True, text=True, check=True).stdout

    found: dict[str, tuple[set[int], set[int]]] = {}
    old_path = new_path = None
    for line in out.splitlines():
        if line.startswith("--- "):
            old_path = None if line[4:] == "/dev/null" else line[4:]
        elif line.startswith("+++ "):
            new_path = None if line[4:] == "/dev/null" else line[4:]
        elif line.startswith("@@"):
            m = _HUNK.match(line)
            if not m:
                continue
            ostart, ocount, nstart, ncount = (
                int(m.group(1)),
                1 if m.group(2) is None else int(m.group(2)),
                int(m.group(3)),
                1 if m.group(4) is None else int(m.group(4)),
            )
            for key in (old_path, new_path):
                if key is None:
                    continue
                base_lines, head_lines = found.setdefault(key, (set(), set()))
                base_lines |= set(range(ostart, ostart + ocount))
                head_lines |= set(range(nstart, nstart + ncount))
    return found


def _read(repo: Path, ref: str | None, path: str) -> str | None:
    """File content at a ref, or from the working tree when ref is None."""
    if ref is None:
        try:
            return (repo / path).read_text()
        except OSError:
            return None
    proc = subprocess.run(
        ["git", "-C", str(repo), "show", f"{ref}:{path}"], capture_output=True
    )
    if proc.returncode != 0:
        return None
    try:
        return proc.stdout.decode()
    except UnicodeDecodeError:
        return None


def _side(
    repo: Path, ref: str | None, path: str | None, lines: set[int]
) -> tuple[frozenset[str], bool, str, frozenset[str]]:
    """(names, residue, note, import-time subset) for one side of one file."""
    if path is None or not lines:
        return frozenset(), False, "", frozenset()
    source = _read(repo, ref, path)
    if source is None:
        return frozenset(), False, f"unreadable at {ref or 'worktree'}", frozenset()
    try:
        names, residue = attribute(source, path, lines)
        at_import = names & import_time_names(source, path)
    except Exception as exc:  # compile() on anything we cannot handle: fail open
        return frozenset(), False, f"{type(exc).__name__}: {exc}", frozenset()
    return names, residue, "", at_import


def build(repo: Path, base: str, head: str | None = None) -> Query:
    """The query for a diff: per changed file, the functions it touched."""
    from ..gitdiff import diff_files

    by_path = hunks(repo, base, head)
    query = Query(base=base, head=head)

    for changed in diff_files(repo, base, head):
        path, old_path = changed.path, changed.old_path
        base_lines, head_lines = set(), set()
        for key in (path, old_path):
            if key and key in by_path:
                base_lines |= by_path[key][0]
                head_lines |= by_path[key][1]

        shown = path if changed.status != "D" else (old_path or path)
        in_scope = shown.startswith(RECORDER_SCOPE)

        if not shown.endswith(".py"):
            query.files.append(
                FileQuery(
                    path=shown,
                    old_path=old_path,
                    status=Attribution.NAMELESS,
                    in_recorder_scope=in_scope,
                )
            )
            continue

        # A Python file git reported with no hunks at all: a mode change, a
        # binary blob, or a path our hunk parse failed to match. We cannot tell
        # which, so we do not get to say "nothing changed here".
        if not base_lines and not head_lines:
            query.files.append(
                FileQuery(
                    path=shown,
                    old_path=old_path,
                    status=Attribution.FAILED,
                    in_recorder_scope=in_scope,
                    note="no hunks parsed",
                )
            )
            continue

        base_side = None if changed.status == "A" else (old_path or path)
        head_side = None if changed.status == "D" else path
        base_names, base_residue, base_note, base_import = _side(
            repo, base, base_side, base_lines
        )
        head_names, head_residue, head_note, head_import = _side(
            repo, head, head_side, head_lines
        )

        note = "; ".join(n for n in (base_note, head_note) if n)
        query.files.append(
            FileQuery(
                path=shown,
                old_path=old_path,
                status=Attribution.FAILED if note else Attribution.ATTRIBUTED,
                base_names=base_names,
                head_names=head_names,
                residue=base_residue or head_residue,
                import_time=base_import | head_import,
                in_recorder_scope=in_scope,
                note=note,
            )
        )
    return query


def mark_unfaithful(query: Query, unfaithful_paths: set[str]) -> Query:
    """Flag files whose recorded names the recorder cannot spell from source.

    The table knows this, the diff cannot. Applied as a separate step so the
    query stays a pure function of the diff.
    """
    for f in query.files:
        if f.path in unfaithful_paths or (
            f.old_path and f.old_path in unfaithful_paths
        ):
            f.unfaithful = True
    return query


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("repo", type=Path)
    ap.add_argument("ref", help="'A...B' merge-base diff, 'A..B', or 'A' vs worktree")
    ap.add_argument("--names", action="store_true", help="print every name")
    args = ap.parse_args()

    from ..gitdiff import resolve_diff_ref

    base, head = resolve_diff_ref(args.repo, args.ref)
    query = build(args.repo, base, head)
    for f in sorted(query.files, key=lambda f: f.path):
        flags = [f.status.value]
        if f.residue:
            flags.append("residue")
        if not f.in_recorder_scope:
            flags.append("out-of-scope")
        if f.note:
            flags.append(f.note)
        print(f"{f.path}  [{', '.join(flags)}]  {len(f.names)} names")
        if args.names:
            for name in sorted(f.names):
                print(f"    {name}")
    print(f"\n{len(query.files)} files, fail_open={query.fail_open}")


if __name__ == "__main__":
    main()
