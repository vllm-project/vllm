# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ensure we perform lazy loading in vllm/__init__.py.
i.e: appears only within the `if typing.TYPE_CHECKING:` guard,
**except** for a short whitelist.
"""

import ast
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Final

INIT_PATH: Final = Path("vllm/__init__.py")

# If you need to add items to whitelist, do it here.
ALLOWED_IMPORTS: Final[frozenset[str]] = frozenset(
    {
        "vllm.env_override",
    }
)
ALLOWED_FROM_MODULES: Final[frozenset[str]] = frozenset(
    {
        ".version",
    }
)


def _is_internal(name: str | None, *, level: int = 0) -> bool:
    if level > 0:
        return True
    if name is None:
        return False
    return name.startswith("vllm.") or name == "vllm"


def _fail(violations: Iterable[tuple[int, str]]) -> None:
    print("ERROR: Disallowed eager imports in vllm/__init__.py:\n", file=sys.stderr)
    for lineno, msg in violations:
        print(f"  Line {lineno}: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    source = INIT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(INIT_PATH))

    violations: list[tuple[int, str]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            super().__init__()
            self._in_type_checking = False

        def visit_If(self, node: ast.If) -> None:
            guard_is_type_checking = False
            test = node.test
            if isinstance(test, ast.Attribute) and isinstance(test.value, ast.Name):
                guard_is_type_checking = (
                    test.value.id == "typing" and test.attr == "TYPE_CHECKING"
                )
            elif isinstance(test, ast.Name):
                guard_is_type_checking = test.id == "TYPE_CHECKING"

            if guard_is_type_checking:
                prev = self._in_type_checking
                self._in_type_checking = True
                for child in node.body:
                    self.visit(child)
                self._in_type_checking = prev
                for child in node.orelse:
                    self.visit(child)
            else:
                self.generic_visit(node)

        def visit_Import(self, node: ast.Import) -> None:
            if self._in_type_checking:
                return
            for alias in node.names:
                module_name = alias.name
                if _is_internal(module_name) and module_name not in ALLOWED_IMPORTS:
                    violations.append(
                        (
                            node.lineno,
                            f"import '{module_name}' must be inside typing.TYPE_CHECKING",  # noqa: E501
                        )
                    )

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if self._in_type_checking:
                return
            module_as_written = ("." * node.level) + (node.module or "")
            if (
                _is_internal(node.module, level=node.level)
                and module_as_written not in ALLOWED_FROM_MODULES
            ):
                violations.append(
                    (
                        node.lineno,
                        f"from '{module_as_written}' import ... must be inside typing.TYPE_CHECKING",  # noqa: E501
                    )
                )

    Visitor().visit(tree)

    if violations:
        _fail(violations)


if __name__ == "__main__":
    main()
