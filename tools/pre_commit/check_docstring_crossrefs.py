# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lint: detect accidental cross-references in docstrings.

Docstrings render as Markdown in the API docs, where `[a][b]` is a reference
link that mkdocs-autorefs tries to resolve. Shapes and indexing written outside
code spans therefore produce broken cross-references:

    [block][2][16]        # BUG: looks up targets "2" and "16"
    `[block][2][16]`      # OK: code span
    [LLM][vllm.LLM]       # OK: intended cross-reference
"""

import ast
import inspect
import sys

import regex as re

FENCE = re.compile(r"^\s*(```|~~~).*?^\s*\1", re.DOTALL | re.MULTILINE)
CODE_SPAN = re.compile(r"(`+).+?\1", re.DOTALL)
REF_LINK = re.compile(r"\[([^\[\]\n]+)\]\[([^\[\]\n]*)\]")
# Fully qualified Python paths, e.g. `vllm.LLM` or `torch.Tensor`
TARGET = re.compile(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)+")


def strip_indented_code(doc: str) -> str:
    """Drop Markdown indented code blocks (4+ spaces after a blank line)."""
    kept, in_code, prev_blank = [], False, True
    for line in inspect.cleandoc(doc).splitlines():
        if not line.strip():
            prev_blank = True
            kept.append(line)
            continue
        in_code = line.startswith("    ") and (prev_blank or in_code)
        prev_blank = False
        if not in_code:
            kept.append(line)
    return "\n".join(kept)


def iter_docstrings(node: ast.AST):
    """Yield docstrings that mkdocstrings renders (not nested functions)."""
    doc = ast.get_docstring(node, clean=False)
    if doc is not None:
        yield node.body[0].lineno, doc
    if isinstance(node, (ast.Module, ast.ClassDef)):
        for child in node.body:
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                yield from iter_docstrings(child)


def check_file(filepath: str) -> list[str]:
    try:
        with open(filepath, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=filepath)
    except (OSError, UnicodeDecodeError, SyntaxError):
        return []

    violations = []
    for lineno, doc in iter_docstrings(tree):
        text = CODE_SPAN.sub("", FENCE.sub("", strip_indented_code(doc)))
        for match in REF_LINK.finditer(text):
            label, target = match.groups()
            if not TARGET.fullmatch(target or label):
                violations.append(f"{filepath}:{lineno}: {match.group()}")
    return violations


def main() -> int:
    violations = []
    for filepath in sys.argv[1:]:
        violations.extend(check_file(filepath))

    if violations:
        print("❌ Docstring text that mkdocs will read as a cross-reference:")
        print("\n".join(f"   {v}" for v in violations))
        print(
            "   Wrap shapes and indexing in backticks (`[block][2][16]`), or "
            "use a fully qualified target ([LLM][vllm.LLM])."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
