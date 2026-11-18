# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lint and fix docstring style that matters once docstrings render as Markdown.

- Code spans use single backticks: ``x`` is rewritten to `x` in place.
- Python Sphinx roles become code spans of the text Sphinx would display, e.g.
  :class:`~vllm.LLM` is rewritten to `LLM`. Other roles must be fixed by hand.
- Sections are Google style (`Args:`), not NumPy style (`Parameters` + `---`).
- Admonitions, code blocks and links are Markdown (`!!! note`, fences,
  `[text](url)`), not reStructuredText (`.. note::`, `.. code-block::`,
  `` `text <url>`_ ``).
- Code spans do not start with a space or start or end at a line break, which
  is how a span broken across lines or missing its closing backtick shows up.
- Docstrings with escapes or line continuations cannot be fixed in place, so
  any of the above in them is reported; a raw string fixes that.
- `[a][b]` is a reference link that mkdocs-autorefs tries to resolve, so shapes
  and indexing written outside code spans produce broken cross-references:

      [block][2][16]        # BUG: looks up targets "2" and "16"
      `[block][2][16]`      # OK: code span
      [LLM][vllm.LLM]       # OK: intended cross-reference
      [int][]               # OK: stdlib cross-reference

All rules read docstrings as source text, one Markdown paragraph at a time, and
skip code blocks, so reported line numbers point at the offending line.
"""

import ast
import codecs
import io
import sys
import tokenize
from collections.abc import Iterator

import regex as re

FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
DOCTEST = re.compile(r"^\s*>>>")
SECTION = re.compile(r"^\s*([A-Za-z][A-Za-z ]*):\s*$")
ADMONITION = re.compile(r"^\s*(!!!|\?\?\?\+?)\s")
BACKTICKS = re.compile(r"`+")
REF_LINK = re.compile(
    r"(?<!\\)\[((?:[^\[\]]|\[[^\[\]]*\])+)\](?: |\n[ \t]*)?\[([^\[\]]*)\]"
)
REF_DEF = re.compile(r"^\s*\[([^\]]+)\]:\s", re.MULTILINE)
# A Sphinx role directly before a code span, e.g. :class: or :py:meth:
ROLE = re.compile(r"(?<![\w:`]):((?:[a-z]+:)?[a-z]+):$")
# Explicit title, e.g. :meth:`generate <vllm.LLM.generate>`
TITLED_ROLE = re.compile(r"\s*(.*?)\s*<[^<>]+>\s*", re.DOTALL)
PY_ROLES = {
    f"{domain}{role}"
    for domain in ("", "py:")
    for role in ("attr", "class", "const", "data", "exc", "func", "meth", "mod",
                 "obj", "type")
}  # fmt: skip
NUMPY_SECTION = re.compile(
    r"^[ \t]*(Parameters|Other Parameters|Returns|Yields|Receives|Raises|Warns|"
    r"Attributes|Notes)[ \t]*\n[ \t]*-{3,}[ \t]*$",
    re.MULTILINE,
)
RST_DIRECTIVE = re.compile(r"^[ \t]*\.\. ([a-z][a-z-]*)::", re.MULTILINE)
RST_LINK = re.compile(r"`([^`<>]+?) ?<https?://[^<>`]+>`__?")
# What `fix_code_spans` would rewrite, for docstrings it cannot edit
RST_MARKUP = re.compile(r"``[^`]|:(?:[a-z]+:)?[a-z]+:`")
QUOTE = re.compile(r"""^[rRuU]?("{3}|'{3}|"|')""")
# Fully qualified Python paths, e.g. `vllm.LLM` or `torch.Tensor`
TARGET = re.compile(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)+")
# Collapsed `[label][]` may also name a builtin, e.g. `[int][]`
COLLAPSED_TARGET = re.compile(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*")

# Google-style sections griffe parses; item sections nest one level deeper
ITEM_SECTIONS = {
    "args", "arguments", "params", "parameters", "keyword args",
    "keyword arguments", "other args", "other arguments", "other params",
    "other parameters", "raises", "exceptions", "warns", "receives",
    "attributes", "functions", "methods", "classes", "modules",
}  # fmt: skip
SECTIONS = ITEM_SECTIONS | {
    "returns", "yields", "examples", "example", "note", "notes", "warning",
    "warnings", "todo", "see also", "references",
}  # fmt: skip

DEFS = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
BLOCKS = (ast.If, ast.Try, ast.With, ast.AsyncWith)


def prose_mask(lines: list[str]) -> list[bool]:
    """Whether each docstring line is Markdown prose (not code or blank)."""
    lines = [line.expandtabs() for line in lines]
    indent = min(
        (len(line) - len(line.lstrip()) for line in lines[1:] if line.strip()),
        default=0,
    )
    mask: list[bool] = []
    fence = None
    container: tuple[int, int] | None = None  # (header indent, code indent)
    code = doctest = False
    prev_blank = True
    for i, line in enumerate(lines):
        text = QUOTE.sub("", line) if i == 0 else line[indent:]
        stripped = text.strip()
        if fence:
            mask.append(False)
            if len(stripped) >= len(fence) and set(stripped) == {fence[0]}:
                fence = None
            continue
        if not stripped:
            mask.append(False)
            doctest, prev_blank = False, True
            continue
        ind = len(text) - len(text.lstrip())
        if container and ind <= container[0]:
            container = None
        code = ind >= (container[1] if container else 4) and (prev_blank or code)
        prev_blank = False
        if not code and (match := FENCE.match(text)):
            fence = match.group(1)
            mask.append(False)
            continue
        doctest = not code and (doctest or bool(DOCTEST.match(text)))
        prose = not (code or doctest)
        mask.append(prose)
        section = SECTION.match(text) if prose else None
        if section and (title := section.group(1).lower()) in SECTIONS:
            container = (ind, ind + (12 if title in ITEM_SECTIONS else 8))
        elif prose and ADMONITION.match(text):
            container = (ind, ind + 8)
    return mask


def paragraphs(lines: list[str]) -> Iterator[tuple[int, int]]:
    """Line ranges of the prose paragraphs in a docstring."""
    start = None
    for i, prose in enumerate([*prose_mask(lines), False]):
        if prose and start is None:
            start = i
        elif not prose and start is not None:
            yield start, i
            start = None


def code_spans(text: str) -> Iterator[tuple[int, int, int]]:
    """(start, end, backtick count) of each Markdown code span in a paragraph."""
    runs = [m.span() for m in BACKTICKS.finditer(text)]
    i = 0
    while i < len(runs):
        start, end = runs[i]
        width = end - start
        close = next(
            (j for j in range(i + 1, len(runs)) if runs[j][1] - runs[j][0] == width),
            None,
        )
        if (start and text[start - 1] == "\\") or close is None:
            i += 1
            continue
        yield start, runs[close][1], width
        i = close + 1


def docstrings(tree: ast.Module, rendered_only: bool) -> list[ast.Constant]:
    """Docstring constants; `rendered_only` keeps what mkdocstrings shows.

    That is module, class, function and attribute docstrings (including
    `self.x` in `__init__`), under `if`/`try`/`with` blocks too, but not other
    docstrings inside function bodies.
    """
    found: dict[int, ast.Constant] = {}

    def is_str(node: ast.AST) -> bool:
        return isinstance(node, ast.Constant) and isinstance(node.value, str)

    def visit(node: ast.AST) -> None:
        body = getattr(node, "body", [])
        if body and isinstance(body[0], ast.Expr) and is_str(body[0].value):
            found[id(body[0].value)] = body[0].value
        if isinstance(node, (ast.Module, ast.ClassDef)) or (
            isinstance(node, ast.FunctionDef) and node.name == "__init__"
        ):
            scope(body)

    def scope(body: list[ast.stmt]) -> None:
        prev = None
        for stmt in body:
            if (
                isinstance(stmt, ast.Expr)
                and is_str(stmt.value)
                and isinstance(prev, (ast.Assign, ast.AnnAssign))
            ):
                found[id(stmt.value)] = stmt.value
            if isinstance(stmt, DEFS):
                visit(stmt)
            elif isinstance(stmt, BLOCKS):
                scope(stmt.body)
                scope(getattr(stmt, "orelse", []))
                scope(getattr(stmt, "finalbody", []))
                for handler in getattr(stmt, "handlers", []):
                    scope(handler.body)
            prev = stmt

    visit(tree)
    if not rendered_only:
        for node in ast.walk(tree):
            if isinstance(node, DEFS):
                visit(node)
    return list(found.values())


def segment(lines: list[str], node: ast.Constant) -> tuple[str, list[str], str] | None:
    """Split a docstring's source lines into (head, string lines, tail).

    Returns None unless the docstring is a single string token whose lines
    match its value's (no implicit concatenation, `\\n` escapes or line
    continuations), so the source text is what Markdown sees.
    """
    assert node.end_lineno is not None and node.end_col_offset is not None
    first, last = node.lineno - 1, node.end_lineno - 1
    body = lines[first : last + 1]
    # AST column offsets count UTF-8 bytes
    end = body[-1].encode()
    tail = end[node.end_col_offset :].decode()
    body[-1] = end[: node.end_col_offset].decode()
    head = lines[first].encode()[: node.col_offset].decode()
    body[0] = body[0][len(head) :]
    try:
        tokens = [
            token
            for token in tokenize.generate_tokens(io.StringIO("\n".join(body)).readline)
            if token.type not in (tokenize.NEWLINE, tokenize.NL, tokenize.ENDMARKER)
        ]
    except (tokenize.TokenError, SyntaxError):
        return None
    if len(tokens) != 1 or tokens[0].type != tokenize.STRING:
        return None
    if node.value.count("\n") != len(body) - 1:
        return None
    return head, body, tail


def role_text(content: str) -> str:
    """The text Sphinx displays for a role's content, e.g. `~a.B` -> `B`."""
    if titled := TITLED_ROLE.fullmatch(content):
        return titled.group(1).strip()
    content = content.strip().lstrip("!")
    return content[1:].rsplit(".", 1)[-1] if content.startswith("~") else content


def fix_code_spans(
    filepath: str, source: str, tree: ast.Module
) -> tuple[str, list[str], list[str]]:
    """Rewrite ``code`` and Python Sphinx roles as `code`.

    Returns the new source, the fixes made and the roles left for a human.
    """
    lines = source.split("\n")
    fixed: dict[int, str] = {}
    unfixable = []
    for node in docstrings(tree, rendered_only=False):
        if (parts := segment(lines, node)) is None:
            if match := RST_MARKUP.search(node.value):
                lineno = node.lineno + node.value.count("\n", 0, match.start())
                unfixable.append(
                    f"{filepath}:{lineno}: reStructuredText markup in a docstring "
                    "with escapes or line continuations; make it a raw string"
                )
            continue
        head, body, tail = parts
        crs = [line.endswith("\r") for line in body]
        body = [line.removesuffix("\r") for line in body]
        for a, b in paragraphs(body):
            text = "\n".join(body[a:b])
            out, pos = [], 0
            for start, end, width in code_spans(text):
                content = text[start + width : end - width]
                lineno = node.lineno + a + text.count("\n", 0, start)
                if "`" in content:
                    continue
                lead = content[: len(content) - len(content.lstrip())]
                trail = content[len(content.rstrip()) :]
                if content.strip() and ("\n" in lead + trail or (lead and not trail)):
                    unfixable.append(
                        f"{filepath}:{lineno}: code span starts or ends with "
                        "whitespace in a docstring; check for a misplaced backtick"
                    )
                    continue
                if role := ROLE.search(text, pos, start):
                    if role.group(1) not in PY_ROLES or "\n" in content:
                        unfixable.append(
                            f"{filepath}:{lineno}: `{role.group()}` Sphinx role in "
                            "a docstring; use Markdown instead"
                        )
                        continue
                    out += [text[pos : role.start()], "`", role_text(content), "`"]
                    fixed[lineno] = "rewrote :role:`code` as `code`"
                elif width == 2:
                    out += [text[pos:start], "`", content, "`"]
                    fixed.setdefault(lineno, "rewrote ``code`` as `code`")
                else:
                    continue
                pos = end
            if pos:
                body[a:b] = "".join([*out, text[pos:]]).split("\n")
        body = [line + "\r" * cr for line, cr in zip(body, crs)]
        body[0] = head + body[0]
        body[-1] += tail
        lines[node.lineno - 1 : node.end_lineno] = body
    fixes = [f"{filepath}:{n}: {m} in a docstring" for n, m in sorted(fixed.items())]
    return "\n".join(lines), fixes, unfixable


def crossref_violations(filepath: str, source: str, tree: ast.Module) -> list[str]:
    lines = source.split("\n")
    violations = []
    for node in docstrings(tree, rendered_only=True):
        if (parts := segment(lines, node)) is None:
            continue
        body = [line.removesuffix("\r") for line in parts[1]]
        paras = [(a, "\n".join(body[a:b])) for a, b in paragraphs(body)]
        defined = {
            " ".join(m.group(1).split()).lower()
            for _, text in paras
            for m in REF_DEF.finditer(text)
        }
        for a, text in paras:
            chars = list(text)
            for start, end, _ in code_spans(text):
                chars[start:end] = [c if c == "\n" else "x" for c in text[start:end]]
            text = "".join(chars)
            for match in REF_LINK.finditer(text):
                label, target = match.groups()
                if target and TARGET.fullmatch(target):
                    continue
                if not target and COLLAPSED_TARGET.fullmatch(label):
                    continue
                if " ".join((target or label).split()).lower() in defined:
                    continue
                lineno = node.lineno + a + text.count("\n", 0, match.start())
                link = " ".join(match.group().split())
                violations.append(
                    f"{filepath}:{lineno}: `{link}` in a docstring is a "
                    "cross-reference; wrap it in backticks or use a qualified target"
                )
    return violations


def section_violations(filepath: str, tree: ast.Module) -> list[str]:
    violations = []
    for node in docstrings(tree, rendered_only=True):
        for pattern, found, fix in (
            (NUMPY_SECTION, "NumPy-style `{}` section", "use Google style"),
            (RST_DIRECTIVE, "reStructuredText `.. {}::`", "use `!!!` or a fence"),
            (RST_LINK, "reStructuredText link `{}`", "use `[text](url)`"),
        ):
            for match in pattern.finditer(node.value):
                lineno = node.lineno + node.value.count("\n", 0, match.start())
                found_text = found.format(match.group(1))
                violations.append(
                    f"{filepath}:{lineno}: {found_text} in a docstring; {fix}"
                )
    return violations


def main() -> int:
    failures = []
    for filepath in sys.argv[1:]:
        try:
            with open(filepath, "rb") as f:
                data = f.read()
            bom = codecs.BOM_UTF8 if data.startswith(codecs.BOM_UTF8) else b""
            source = data[len(bom) :].decode("utf-8")
            tree = ast.parse(source, filename=filepath)
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        # A lone \r is a line break to `ast` but not to `str.split("\n")`
        if re.search(r"\r(?!\n)", source):
            continue
        new_source, fixes, unfixable = fix_code_spans(filepath, source, tree)
        if fixes:
            with open(filepath, "wb") as f:
                f.write(bom + new_source.encode("utf-8"))
            source, tree = new_source, ast.parse(new_source, filename=filepath)
        failures += fixes + unfixable + crossref_violations(filepath, source, tree)
        failures += section_violations(filepath, tree)
    print("\n".join(failures), end="\n" if failures else "")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
