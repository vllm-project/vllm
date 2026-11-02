# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lint docstring style that matters once docstrings render as Markdown.

- `[a][b]` is a reference link that mkdocs-autorefs tries to resolve, so shapes
  and indexing written outside code spans produce broken cross-references:

      [block][2][16]        # BUG: looks up targets "2" and "16"
      `[block][2][16]`      # OK: code span
      [LLM][vllm.LLM]       # OK: intended cross-reference
      [int][]               # OK: stdlib cross-reference
"""

import ast
import inspect
import sys

import regex as re

FENCE = re.compile(r"^\s*(```|~~~)")
DOCTEST = re.compile(r"^\s*>>>")
CODE_SPAN = re.compile(r"(`+).+?\1", re.DOTALL)
REF_LINK = re.compile(r"\[([^\[\]\n]+)\]\[([^\[\]\n]*)\]")
# Fully qualified Python paths, e.g. `vllm.LLM` or `torch.Tensor`
TARGET = re.compile(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)+")
# Collapsed `[label][]` may also name a builtin, e.g. `[int][]`
COLLAPSED_TARGET = re.compile(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*")

DEFS = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def prose_mask(lines: list[str]) -> list[bool]:
    """Whether each docstring line is Markdown prose (not a code block)."""
    indent = min(
        (len(line) - len(line.lstrip()) for line in lines[1:] if line.strip()),
        default=0,
    )
    mask, fence, indented, doctest, prev_blank = [], None, False, False, True
    for i, line in enumerate(lines):
        text = line if i == 0 else line[indent:]
        blank = not text.strip()
        if fence:
            mask.append(False)
            if text.strip().startswith(fence):
                fence = None
        elif match := FENCE.match(text):
            fence = match.group(1)
            mask.append(False)
        else:
            indented = (
                not blank and text.startswith("    ") and (prev_blank or indented)
            ) or (blank and indented)
            doctest = not blank and (doctest or bool(DOCTEST.match(text)))
            mask.append(not (indented or doctest))
        prev_blank = blank
    return mask


def prose(doc: str) -> str:
    lines = inspect.cleandoc(doc).splitlines()
    return "\n".join(line for line, p in zip(lines, prose_mask(lines)) if p)


def docstring_nodes(tree: ast.Module) -> list[ast.Constant]:
    """Docstring constants that mkdocstrings renders (not nested functions)."""
    nodes, stack = [], [tree]
    while stack:
        node = stack.pop()
        nodes.append(node)
        if isinstance(node, (ast.Module, ast.ClassDef)):
            stack.extend(c for c in node.body if isinstance(c, DEFS))
    return [
        n.body[0].value for n in nodes if ast.get_docstring(n, clean=False) is not None
    ]


def crossref_violations(filepath: str, tree: ast.Module) -> list[str]:
    violations = []
    for node in docstring_nodes(tree):
        for match in REF_LINK.finditer(CODE_SPAN.sub("", prose(node.value))):
            label, target = match.groups()
            if target and TARGET.fullmatch(target):
                continue
            if not target and COLLAPSED_TARGET.fullmatch(label):
                continue
            violations.append(f"{filepath}:{node.lineno}: {match.group()}")
    return violations


def main() -> int:
    violations = []
    for filepath in sys.argv[1:]:
        try:
            with open(filepath, encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=filepath)
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        violations.extend(crossref_violations(filepath, tree))

    if violations:
        print("❌ Docstring text that mkdocs will read as a cross-reference:")
        print("\n".join(f"   {v}" for v in violations))
        print(
            "   Wrap shapes and indexing in backticks (`[block][2][16]`), or "
            "use a fully qualified target ([LLM][vllm.LLM], [int][])."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
