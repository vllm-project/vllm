# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which ops a csrc file implements, and the Python wrappers that call them.

A coverage row cannot see compiled code, but every kernel is reached through
a Python wrapper holding ``torch.ops.<ns>.<op>``, which the recorder sees like
any other function. So a changed csrc file stands in as: its ops -> their
wrapper qualnames -> call frames in rows.

Missing an op is the dangerous direction, because the stand-in would then
cover less than the file really reaches. So attribution only ever widens: a
source file owns the ops it registers, the ops whose impl symbols it defines,
and, when it registers none itself, the ops of files calling a symbol only it
defines. Headers inherit from the files that include them.

Deriving no ops at all means never droppable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import regex as re

from .build_map import csrc_include_graph

ENV_VAR = "CI_SELECTOR_CSRC_OPS"
MODES = ("on", "off")

_DEF = re.compile(r'\b\w+\.def\(\s*"(\w+)')
_IMPL = re.compile(r'\b\w+\.impl\(\s*"(\w+)(?:\.\w+)?"\s*,(?:[^&;]*)&(\w+)')
_DEF_FN = re.compile(r'\b\w+\.def\(\s*"(\w+)"\s*,\s*(?:[^&;]*)&(\w+)')
_TORCH_OPS = re.compile(r"\btorch\s*\.\s*ops\s*\.\s*(\w+)\s*\.\s*(\w+)")

# Our own extension namespaces, derived from the build files (see
# _derive_namespaces). A torch.ops chain landing anywhere else is upstream
# torch, not a wrapper.
_CMAKE_COMMENT = re.compile(r"#[^\n]*")
_EXT_TARGET = re.compile(r"\bdefine_extension_target\s*\(\s*(\w+)")
_EXT_NAME_DEF = re.compile(r"-DTORCH_EXTENSION_NAME=(\w+)")
_LIB_REG = re.compile(
    r"\b(?:STABLE_)?TORCH_LIBRARY(?:_FRAGMENT|_IMPL)?(?:_EXPAND)?\s*\(\s*(\w+)"
)
_REG_EXT = re.compile(r"\bREGISTER_EXTENSION\s*\(\s*(\w+)\s*\)")
_NS_CONCAT = re.compile(r"\bCONCAT\s*\(\s*TORCH_EXTENSION_NAME\s*,\s*(\w+)\s*\)")
# Macro-parameter placeholders the csrc regexes see in registration.h's own
# definitions; TORCH_EXTENSION_NAME itself resolves through the cmake legs.
_NS_PLACEHOLDERS = frozenset({"NAME", "TORCH_EXTENSION_NAME", "CONCAT"})


def mode() -> str:
    """Unset means "on". An unrecognized value raises."""
    raw = os.environ.get(ENV_VAR)
    if raw is None or raw == "":
        return "on"
    if raw in MODES:
        return raw
    raise ValueError(f"{ENV_VAR}={raw!r}, expected one of: {', '.join(MODES)}")


def _definition_re(sym: str) -> re.Pattern:
    """Where `sym` is defined: no ; or { in the parameters, then the body."""
    return re.compile(r"\b" + re.escape(sym) + r"\s*\((?:[^;{}])*\)\s*\{")


_LINE_COMMENT = re.compile(r"//[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)


def _strip_comments(text: str) -> str:
    """A registration inside a comment would mint an op no wrapper can hold,
    and one such op blocks its file and every header including it."""
    return _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub("", text))


@dataclass
class NativeOps:
    """Absence from `file_ops` means never droppable."""

    # csrc path -> the ops it implements
    file_ops: dict[str, frozenset[str]] = field(default_factory=dict)
    # op -> {(vllm path, wrapper qualname)}
    wrappers: dict[str, frozenset[tuple[str, str]]] = field(default_factory=dict)
    # op -> test files whose text names it
    op_test_files: dict[str, frozenset[str]] = field(default_factory=dict)
    # our extension namespaces, derived from cmake targets + csrc registrations
    namespaces: frozenset[str] = frozenset()
    op_count: int = 0
    error: str | None = None

    def owns(self, path: str) -> bool:
        """csrc/cpu/ and csrc/rocm/ already route narrowly, so leave them."""
        return path.startswith("csrc/") and not path.startswith(
            ("csrc/cpu/", "csrc/rocm/")
        )

    def proxies_for(self, path: str) -> dict[str, frozenset[str]] | None:
        """Wrapper file -> qualnames standing in for `path`, or None when its
        ops are unknown or any one of them has no wrapper."""
        ops = self.file_ops.get(path)
        if not ops:
            return None
        out: dict[str, set[str]] = {}
        for op in ops:
            pairs = self.wrappers.get(op)
            if not pairs:
                return None  # one op without a wrapper blocks the whole file
            for f, q in pairs:
                out.setdefault(f, set()).add(q)
        return {f: frozenset(qs) for f, qs in out.items()}

    def test_files_for(self, path: str) -> frozenset[str]:
        ops = self.file_ops.get(path, frozenset())
        return frozenset(
            t for op in ops for t in self.op_test_files.get(op, frozenset())
        )

    @classmethod
    def build(cls, repo: Path, test_catalog: list[str] | None = None) -> NativeOps:
        try:
            return cls._build(repo, test_catalog or [])
        except Exception as exc:  # noqa: BLE001 - any parse failure fails open
            return cls(error=f"{type(exc).__name__}: {exc}")

    @classmethod
    def _build(cls, repo: Path, test_catalog: list[str]) -> NativeOps:
        graph = csrc_include_graph(repo)
        if graph is None:
            return cls(error="no csrc tree")
        rev, tu_rels, hdr_rels, texts = graph
        texts = {rel: _strip_comments(t) for rel, t in texts.items()}

        ops: set[str] = set()
        file_ops: dict[str, set[str]] = {}
        impl_syms: dict[str, set[str]] = {}
        for rel, text in texts.items():
            found = set(_DEF.findall(text))
            for op, sym in _IMPL.findall(text):
                found.add(op)
                impl_syms.setdefault(op, set()).add(sym)
            for op, sym in _DEF_FN.findall(text):
                found.add(op)
                impl_syms.setdefault(op, set()).add(sym)
            if found:
                ops |= found
                file_ops.setdefault(rel, set()).update(found)

        # A file also owns the ops whose impl symbol it defines. Definitions
        # only: a file that merely calls the symbol reaches less than the op.
        for op, syms in impl_syms.items():
            for sym in syms:
                rx = _definition_re(sym)
                for rel in tu_rels:
                    text = texts.get(rel, "")
                    if sym in text and rx.search(text):
                        file_ops.setdefault(rel, set()).add(op)

        # A file registering no ops of its own inherits from the files that
        # call a symbol only it defines. One hop over a sorted snapshot, so
        # the result cannot depend on hash order. Only uniquely-defined names
        # count: a shared helper would spread every op onto every kernel.
        name_def = re.compile(r"\b(\w+)\s*\((?:[^;{}])*\)\s*\{")
        defined_in: dict[str, set[str]] = {}
        for rel in sorted(tu_rels):
            for name in set(name_def.findall(texts.get(rel, ""))):
                defined_in.setdefault(name, set()).add(rel)
        bearers = sorted((rel, frozenset(o)) for rel, o in file_ops.items())
        for rel in sorted(tu_rels):
            if rel in file_ops:
                continue
            own_defs = {
                n
                for n, where in defined_in.items()
                if where == {rel} and n not in ("if", "for", "while", "switch")
            }
            if not own_defs:
                continue
            defs_rx = re.compile(
                r"\b(?:" + "|".join(re.escape(d) for d in sorted(own_defs)) + r")\s*\("
            )
            inherited: set[str] = set()
            for bearer, bearer_ops in bearers:
                if bearer == rel or bearer not in texts:
                    continue
                if defs_rx.search(texts[bearer]):
                    inherited |= bearer_ops
            if inherited:
                file_ops[rel] = set(inherited)

        # A header holds the ops of everything that includes it.
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
                        fams |= file_ops.get(parent, set())
                    else:
                        frontier.append(parent)
            if fams:
                file_ops[h] = fams

        namespaces = cls._derive_namespaces(repo, texts)
        wrappers = cls._wrapper_join(repo, ops, namespaces)
        op_tests = cls._test_refs(repo, ops, test_catalog)
        return cls(
            file_ops={p: frozenset(o) for p, o in file_ops.items() if o},
            wrappers=wrappers,
            op_test_files=op_tests,
            namespaces=namespaces,
            op_count=len(ops),
        )

    @staticmethod
    def _derive_namespaces(repo: Path, texts: dict[str, str]) -> frozenset[str]:
        """Every namespace our extensions register under, from the build files.

        New kernels land as new cmake files (flashkda, qutlass), so a hand
        list here goes stale on the normal contribution pattern. Neither
        source alone is complete: cmake names the extension targets (and so
        every TORCH_EXTENSION_NAME value), csrc holds the literal
        (STABLE_)TORCH_LIBRARY sub-namespaces. `texts` must already be
        comment-stripped; cmake comments are #-style and stripped here. The
        CONCAT products over all targets are mostly phantoms, which is safe:
        a namespace only matters when vllm/ actually names it in a torch.ops
        chain, and the join still requires the op be csrc-minted."""
        targets: set[str] = set()
        cmake_files = [repo / "CMakeLists.txt"]
        cmake_dir = repo / "cmake"
        if cmake_dir.is_dir():
            cmake_files += sorted(cmake_dir.rglob("*.cmake"))
        for cm in cmake_files:
            try:
                text = _CMAKE_COMMENT.sub("", cm.read_text(errors="replace"))
            except OSError:
                continue
            targets.update(_EXT_TARGET.findall(text))
            targets.update(_EXT_NAME_DEF.findall(text))
        literals: set[str] = set()
        suffixes: set[str] = set()
        for text in texts.values():
            literals.update(_LIB_REG.findall(text))
            literals.update(_REG_EXT.findall(text))
            suffixes.update(_NS_CONCAT.findall(text))
        literals -= _NS_PLACEHOLDERS
        expanded = {t + s for t in targets for s in suffixes}
        return frozenset(targets | literals | expanded)

    @staticmethod
    def _wrapper_join(repo: Path, ops: set[str], namespaces: frozenset[str]) -> dict:
        """op -> {(vllm file, qualname)} for functions calling that op.
        Qualnames are spelled the way the recorder spells them."""
        import ast

        def walk(node, qual, in_func, text, rel, out):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    sep = ".<locals>." if in_func else "."
                    q = f"{qual}{sep}{child.name}" if qual else child.name
                    seg = ast.get_source_segment(text, child) or ""
                    for ns, op in _TORCH_OPS.findall(seg):
                        if ns in namespaces and op in ops:
                            out.setdefault(op, set()).add((rel, q))
                    walk(child, q, True, text, rel, out)
                elif isinstance(child, ast.ClassDef):
                    sep = ".<locals>." if in_func else "."
                    q = f"{qual}{sep}{child.name}" if qual else child.name
                    walk(child, q, False, text, rel, out)
                else:
                    walk(child, qual, in_func, text, rel, out)

        out: dict[str, set[tuple[str, str]]] = {}
        for py in (repo / "vllm").rglob("*.py"):
            rel = py.relative_to(repo).as_posix()
            try:
                text = py.read_text()
            except OSError:
                continue
            if "torch.ops" not in text:
                continue
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            walk(tree, "", False, text, rel, out)
        return {op: frozenset(pairs) for op, pairs in out.items()}

    @staticmethod
    def _test_refs(repo: Path, ops: set[str], catalog: list[str]) -> dict:
        """op -> test files naming it. A test calling an op directly is not
        recorded, so any mention has to keep its steps. Over-matching here
        only shrinks what may drop."""
        if not ops:
            return {}
        rx = re.compile(r"\b(" + "|".join(re.escape(o) for o in sorted(ops)) + r")\b")
        out: dict[str, set[str]] = {}
        for rel in catalog:
            try:
                text = (repo / rel).read_text(errors="replace")
            except OSError:
                continue
            if "torch.ops" not in text and "_custom_ops" not in text:
                continue
            for m in set(rx.findall(text)):
                out.setdefault(m, set()).add(rel)
        return {op: frozenset(t) for op, t in out.items()}
