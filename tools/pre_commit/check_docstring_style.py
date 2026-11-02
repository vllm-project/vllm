# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lint and fix docstring style that matters once docstrings render as Markdown.

- Code spans use single backticks. ``x`` is rewritten to `x` in place.
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
DOUBLE_BACKTICKS = re.compile(r"(?<!`)``([^`\n]+?)``(?!`)")
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


def docstring_nodes(tree: ast.Module, rendered_only: bool) -> list[ast.Constant]:
    """Docstring constants; `rendered_only` skips what mkdocstrings won't show."""
    if rendered_only:
        nodes, stack = [], [tree]
        while stack:
            node = stack.pop()
            nodes.append(node)
            if isinstance(node, (ast.Module, ast.ClassDef)):
                stack.extend(c for c in node.body if isinstance(c, DEFS))
    else:
        nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.Module, *DEFS))]
    return [
        n.body[0].value for n in nodes if ast.get_docstring(n, clean=False) is not None
    ]


def fix_backticks(source: str, tree: ast.Module) -> str:
    lines = source.splitlines(keepends=True)
    nodes = docstring_nodes(tree, rendered_only=False)
    # Bottom up, so splicing a docstring doesn't shift the ones still to come
    for node in sorted(nodes, key=lambda n: n.lineno, reverse=True):
        first, last = node.lineno - 1, node.end_lineno - 1
        # AST column offsets count UTF-8 bytes
        head = lines[first].encode()[: node.col_offset].decode()
        tail = lines[last].encode()[node.end_col_offset :].decode()
        body = lines[first : last + 1]
        body[-1] = body[-1].encode()[: node.end_col_offset].decode()
        body[0] = body[0][len(head) :]
        mask = prose_mask([line.rstrip("\n") for line in body])
        body = [
            DOUBLE_BACKTICKS.sub(r"`\1`", line) if p else line
            for line, p in zip(body, mask)
        ]
        lines[first : last + 1] = [head + "".join(body) + tail]
    return "".join(lines)


def crossref_violations(filepath: str, tree: ast.Module) -> list[str]:
    violations = []
    for node in docstring_nodes(tree, rendered_only=True):
        for match in REF_LINK.finditer(CODE_SPAN.sub("", prose(node.value))):
            label, target = match.groups()
            if target and TARGET.fullmatch(target):
                continue
            if not target and COLLAPSED_TARGET.fullmatch(label):
                continue
            violations.append(f"{filepath}:{node.lineno}: {match.group()}")
    return violations


def main() -> int:
    fixed, violations = [], []
    for filepath in sys.argv[1:]:
        try:
            with open(filepath, encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source, filename=filepath)
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        new_source = fix_backticks(source, tree)
        if new_source != source:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(new_source)
            fixed.append(filepath)
            tree = ast.parse(new_source, filename=filepath)
        violations.extend(crossref_violations(filepath, tree))

    if fixed:
        print("Rewrote ``code`` as `code` in docstrings:")
        print("\n".join(f"   {f}" for f in fixed))
    if violations:
        print("❌ Docstring text that mkdocs will read as a cross-reference:")
        print("\n".join(f"   {v}" for v in violations))
        print(
            "   Wrap shapes and indexing in backticks (`[block][2][16]`), or "
            "use a fully qualified target ([LLM][vllm.LLM], [int][])."
        )
    return 1 if fixed or violations else 0


if __name__ == "__main__":
    sys.exit(main())
