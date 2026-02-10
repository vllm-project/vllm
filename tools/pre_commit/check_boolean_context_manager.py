# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lint: detect `with a() and b():` (boolean op in with-statement context).

Using `and`/`or` to combine context managers is almost always a bug:

    with ctx_a() and ctx_b():   # BUG: only ctx_b is entered
    with ctx_a() or  ctx_b():   # BUG: only ctx_a is entered

The correct way to combine context managers is:

    with ctx_a(), ctx_b():          # comma-separated
    with (ctx_a(), ctx_b()):        # parenthesized (Python 3.10+)
    with contextlib.ExitStack() ... # ExitStack
"""

import ast
import sys


def check_file(filepath: str) -> list[str]:
    try:
        with open(filepath) as f:
            source = f.read()
    except (OSError, UnicodeDecodeError):
        return []

    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError:
        return []

    violations = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if isinstance(item.context_expr, ast.BoolOp):
                    op = "and" if isinstance(item.context_expr.op, ast.And) else "or"
                    violations.append(
                        f"{filepath}:{item.context_expr.lineno}: "
                        f"boolean `{op}` used to combine context managers "
                        f"in `with` statement — use a comma instead"
                    )
    return violations


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: check_boolean_context_manager.py <file> ...", file=sys.stderr)
        return 1

    all_violations = []
    for filepath in sys.argv[1:]:
        all_violations.extend(check_file(filepath))

    if all_violations:
        print(
            "❌ Boolean operator used to combine context managers in `with` "
            "statement.\n"
            "   `with a() and b():` only enters `b()` as a context manager.\n"
            "   Use `with a(), b():` or `with (a(), b()):` instead.\n"
        )
        for v in all_violations:
            print(f"  {v}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
