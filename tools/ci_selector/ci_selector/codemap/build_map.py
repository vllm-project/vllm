# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which device families compile a csrc/cmake file, from CMake's own guards.

A file copied into an image normally selects every step running that image,
so one CUDA kernel drags in the CPU and XPU suites. But the build system
already records who compiles what: CMakeLists.txt wraps each source list in
device guards, and an ``include()``d cmake file inherits the guards of its
include site. A family whose build never compiles a file cannot ship it, so
editing that file cannot affect its steps.

The walker is a line-level stack machine over a tiny grammar. Anything it
does not recognize fails open: an unknown condition inherits the enclosing
families, a ``return()`` subtracts only when every enclosing condition was
recognized, and a file whose families come out empty or complete is left
unmapped, which keeps today's wider answer. Headers map through the files
that include them, then through their directory, then not at all.

So an entry in ``BuildMap.families`` is always something worth narrowing by.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import regex as re

ENV_VAR = "CI_SELECTOR_BUILD_MAP"
MODES = ("on", "off")

# "cuda" is the main image and its steps, which carry no device token at all.
# "other" is xpu/hpu/npu/tpu: they build no csrc today but the device block
# still names them.
CUDA = "cuda"
AMD = "amd"
CPU = "cpu"
OTHER = "other"
ALL_FAMILIES = frozenset({CUDA, AMD, CPU, OTHER})

_LIT = re.compile(
    r'["\s(](?:\$\{CMAKE_CURRENT_SOURCE_DIR\}/)?'
    r"(csrc/[\w./+-]+\.(?:cu|cpp|cc|c|hip|py|h|hpp|cuh))"
)
_GLOB = re.compile(r'file\(GLOB\s+\w+\s+"(csrc/[^"]+)"')
_IF = re.compile(r"^\s*if\s*\(", re.I)
_ELSEIF = re.compile(r"^\s*elseif\s*\(", re.I)
_ELSE = re.compile(r"^\s*else\s*\(\s*\)", re.I)
_ENDIF = re.compile(r"^\s*endif", re.I)
_RETURN = re.compile(r"^\s*return\s*\(\s*\)", re.I)
_INCLUDE = re.compile(
    r"^\s*include\s*\(\s*(?:\$\{CMAKE_CURRENT_LIST_DIR\}/)?(cmake/[\w./]+)\s*\)", re.I
)
_HDR_INCLUDE = re.compile(r'#include\s+"([^"]+)"')

_TU_EXTS = (".cu", ".cpp", ".cc", ".c", ".hip")
_HDR_EXTS = (".h", ".hpp", ".cuh")


def mode() -> str:
    """Unset means "on". An unrecognized value raises rather than defaulting,
    since a swallowed typo looks exactly like the switch doing nothing."""
    raw = os.environ.get(ENV_VAR)
    if raw is None or raw == "":
        return "on"
    if raw in MODES:
        return raw
    raise ValueError(f"{ENV_VAR}={raw!r}, expected one of: {', '.join(MODES)}")


def _classify_cond(cond: str) -> frozenset[str] | None:
    """Families under which this condition can hold. None = unrecognized."""
    c = cond.upper()
    has_cuda = 'VLLM_GPU_LANG STREQUAL "CUDA"' in c
    has_hip = 'VLLM_GPU_LANG STREQUAL "HIP"' in c
    if has_cuda or has_hip:
        fams = set()
        if has_cuda:
            fams.add(CUDA)
        if has_hip:
            fams.add(AMD)
        return frozenset(fams)
    # Anything that is neither cuda nor rocm goes to the cpu build.
    if (
        'NOT VLLM_TARGET_DEVICE STREQUAL "CUDA"' in c
        and 'NOT VLLM_TARGET_DEVICE STREQUAL "ROCM"' in c
    ):
        return frozenset({CPU, OTHER})
    if 'VLLM_TARGET_DEVICE STREQUAL "CPU"' in c:
        return frozenset({CPU})
    return None


def _effective(stack: list[list], ambient: set[str]) -> set[str]:
    eff = set(ambient)
    for branch, _taken, _known in stack:
        if branch is not None:
            eff &= branch
    return eff


def _walk(text: str, ambient_in: frozenset[str]):
    """Yield the file's (literal, families, kind) triples and include sites.

    Stack frames are [families_of_this_branch | None, families_taken, known].
    A ``return()`` narrows only when every enclosing condition was recognized:
    subtracting under a guard we could not read is the one move here that can
    select too little.
    """
    lines = text.splitlines()
    out: list[tuple[str, frozenset[str], str]] = []
    includes: list[tuple[str, frozenset[str]]] = []
    stack: list[list] = []
    ambient = set(ambient_in)
    i = 0
    while i < len(lines):
        line = lines[i]
        if _IF.match(line) or _ELSEIF.match(line):
            cond = line
            depth = cond.count("(") - cond.count(")")
            while depth > 0 and i + 1 < len(lines):
                i += 1
                cond += " " + lines[i]
                depth = cond.count("(") - cond.count(")")
            fams = _classify_cond(cond)
            branch = set(fams) if fams is not None else None
            if _ELSEIF.match(line):
                if stack:
                    prev, taken, prev_known = stack[-1]
                    taken = taken | (prev if prev is not None else set())
                    # One unreadable arm anywhere in the chain makes the whole
                    # chain unreadable: it could have matched any family, so
                    # the else-arm and any return() below can no longer be
                    # trusted to subtract.
                    stack[-1] = [branch, taken, prev_known and fams is not None]
            else:
                stack.append([branch, set(), fams is not None])
        elif _ELSE.match(line):
            if stack:
                prev, taken, known = stack[-1]
                taken = taken | (prev if prev is not None else set())
                if known:
                    stack[-1] = [_effective(stack[:-1], ambient) - taken, taken, True]
                else:
                    stack[-1] = [None, taken, False]
        elif _ENDIF.match(line):
            if stack:
                stack.pop()
        elif _RETURN.match(line):
            if all(known for _b, _t, known in stack):
                ambient -= _effective(stack, ambient)
        else:
            eff = frozenset(_effective(stack, ambient))
            m = _INCLUDE.match(line)
            if m:
                includes.append((m.group(1), eff))
            for lit in _LIT.findall(line):
                out.append((lit, eff, "lit"))
            for pattern in _GLOB.findall(line):
                out.append((pattern, eff, "glob"))
        i += 1
    return out, includes


@dataclass
class BuildMap:
    """csrc/cmake path -> the families that compile it, never all of them.

    Absence means unmapped: leave the file alone. ``error`` records a walk
    that blew up, which also leaves everything unmapped."""

    families: dict[str, frozenset[str]] = field(default_factory=dict)
    unresolved_headers: int = 0
    total_headers: int = 0
    error: str | None = None

    @classmethod
    def build(cls, repo: Path) -> BuildMap:
        try:
            return cls._build(repo)
        except Exception as exc:  # noqa: BLE001 - any parse failure fails open
            return cls(error=f"{type(exc).__name__}: {exc}")

    @classmethod
    def _build(cls, repo: Path) -> BuildMap:
        root = repo / "CMakeLists.txt"
        if not root.is_file():
            return cls(error="no CMakeLists.txt")
        raw: dict[str, set[str]] = {}

        def add(path: str, fams: frozenset[str]) -> None:
            raw.setdefault(path, set()).update(fams)

        def collect(triples) -> None:
            for lit, fams, kind in triples:
                if kind == "lit":
                    add(lit, fams)
                    continue
                for match in repo.glob(lit):
                    if match.is_file():
                        add(match.relative_to(repo).as_posix(), fams)

        top, top_includes = _walk(root.read_text(errors="replace"), ALL_FAMILIES)
        collect(top)
        for inc, fams in top_includes:
            add(inc, fams)
            inc_path = repo / inc
            if not inc_path.is_file():
                continue
            sub, sub_includes = _walk(inc_path.read_text(errors="replace"), fams)
            collect(sub)
            for inc2, fams2 in sub_includes:
                add(inc2, fams2)

        result = cls()
        cls._map_headers(repo, raw, result)
        if result.total_headers and (
            result.unresolved_headers / result.total_headers > 0.2
        ):
            # Too many headers went unresolved to trust any of them, so drop
            # the headers as a group. Source and cmake entries are unaffected.
            for p in [p for p in raw if p.endswith(_HDR_EXTS)]:
                del raw[p]
        result.families = {
            p: frozenset(f)
            for p, f in raw.items()
            if f and not frozenset(f) >= ALL_FAMILIES
        }
        return result

    @staticmethod
    def _map_headers(repo: Path, raw: dict[str, set[str]], result: BuildMap) -> None:
        """A header inherits from every source file that includes it, then
        from its own directory, then stays unmapped."""
        graph = csrc_include_graph(repo)
        if graph is None:
            return
        rev, tu_rels, hdr_rels, _texts = graph
        result.total_headers = len(hdr_rels)
        for h in hdr_rels:
            fams: set[str] = set()
            seen: set[str] = set()
            frontier = [h]
            while frontier:
                cur = frontier.pop()
                for parent in rev.get(cur, ()):
                    if parent in seen:
                        continue
                    seen.add(parent)
                    if parent in tu_rels:
                        fams |= raw.get(parent, set())
                    else:
                        frontier.append(parent)
            if not fams:
                hdir = h.rsplit("/", 1)[0] + "/"
                for t in tu_rels:
                    if t.startswith(hdir):
                        fams |= raw.get(t, set())
            if fams:
                raw.setdefault(h, set()).update(fams)
            else:
                result.unresolved_headers += 1


def csrc_include_graph(
    repo: Path,
) -> tuple[dict[str, set[str]], set[str], list[str], dict[str, str]] | None:
    """(file -> files that #include it, source paths, header paths, path ->
    text) for the csrc tree, or None when there is no csrc tree.

    Shared with the native-ops map, since both route headers the same way. An
    include that matches several files takes all of them, which can only widen
    what a header inherits."""
    csrc = repo / "csrc"
    if not csrc.is_dir():
        return None
    files = [
        p for p in csrc.rglob("*") if p.is_file() and p.suffix in _TU_EXTS + _HDR_EXTS
    ]
    rels = {p: p.relative_to(repo).as_posix() for p in files}
    by_suffix: dict[str, set[str]] = {}
    for rel in rels.values():
        by_suffix.setdefault(rel.rsplit("/", 1)[-1], set()).add(rel)
    rev: dict[str, set[str]] = {}
    texts: dict[str, str] = {}
    for p, rel in rels.items():
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        texts[rel] = text
        for inc in _HDR_INCLUDE.findall(text):
            resolved: set[str] = set()
            for cand in (p.parent / inc, csrc / inc):
                try:
                    c = cand.resolve().relative_to(repo.resolve()).as_posix()
                except (ValueError, OSError):
                    continue
                if c in by_suffix.get(c.rsplit("/", 1)[-1], set()):
                    resolved.add(c)
            if not resolved:
                resolved = by_suffix.get(inc.rsplit("/", 1)[-1], set())
            for r in resolved:
                rev.setdefault(r, set()).add(rel)
    tu_rels = {rel for p, rel in rels.items() if p.suffix in _TU_EXTS}
    hdr_rels = [rel for p, rel in rels.items() if p.suffix in _HDR_EXTS]
    return rev, tu_rels, hdr_rels, texts
