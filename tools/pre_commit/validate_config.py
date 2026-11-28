# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ensures all fields in a config dataclass have default values
and that each field has a docstring.
"""

import ast
import inspect
import sys
from itertools import pairwise

import regex as re


def get_attr_docs(cls_node: ast.ClassDef) -> dict[str, str]:
    """Get any docstrings placed after attribute assignments in a class body.

    Adapted from https://davidism.com/attribute-docstrings/
    https://davidism.com/mit-license/
    """
    out = {}

    # Consider each pair of nodes.
    for a, b in pairwise(cls_node.body):
        # Must be an assignment then a constant string.
        if (
            not isinstance(a, (ast.Assign, ast.AnnAssign))
            or not isinstance(b, ast.Expr)
            or not isinstance(b.value, ast.Constant)
            or not isinstance(b.value.value, str)
        ):
            continue

        doc = inspect.cleandoc(b.value.value)

        # An assignment can have multiple targets (a = b = v), but an
        # annotated assignment only has one target.
        targets = a.targets if isinstance(a, ast.Assign) else [a.target]

        for target in targets:
            # Must be assigning to a plain name.
            if not isinstance(target, ast.Name):
                continue

            out[target.id] = doc

    return out


def _name(node: ast.expr) -> str | None:
    """Return the trailing name of `Name` or (qualified) `Attribute` nodes."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _subscript_name(node: ast.expr) -> str | None:
    return _name(node.value) if isinstance(node, ast.Subscript) else None


def _union_members(annotation: ast.expr) -> list[ast.expr]:
    """Return the flattened members of `Union[...]`, `Optional[...]` or PEP 604
    `A | B` annotations, recursing so nested unions are fully flattened."""
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        return _union_members(annotation.left) + _union_members(annotation.right)
    if _subscript_name(annotation) in {"Union", "Optional"}:
        members = annotation.slice
        elts = members.elts if isinstance(members, ast.Tuple) else [members]
        return [m for elt in elts for m in _union_members(elt)]
    return [annotation]


class ConfigValidator(ast.NodeVisitor):
    def __init__(self): ...

    def visit_ClassDef(self, node):
        # Validate classes with a @config decorator
        decorators = [
            _name(d.func if isinstance(d, ast.Call) else d) for d in node.decorator_list
        ]

        if "config" in decorators:
            if len(decorators) > 1:
                fail(f"config decorator for {node.name} should be used alone", node)
            validate_class(node)

        self.generic_visit(node)


def validate_class(class_node: ast.ClassDef):
    attr_docs = get_attr_docs(class_node)

    for stmt in class_node.body:
        # A field is defined as a class variable that has a type annotation.
        if isinstance(stmt, ast.AnnAssign):
            # Skip ClassVar and InitVar
            # see https://docs.python.org/3/library/dataclasses.html#class-variables
            # and https://docs.python.org/3/library/dataclasses.html#init-only-variables
            if _subscript_name(stmt.annotation) in {"ClassVar", "InitVar"}:
                continue

            if isinstance(stmt.target, ast.Name):
                field_name = stmt.target.id
                if stmt.value is None:
                    fail(
                        f"Field '{field_name}' in {class_node.name} must have "
                        "a default value.",
                        stmt,
                    )

                if field_name not in attr_docs:
                    fail(
                        f"Field '{field_name}' in {class_node.name} must have "
                        "a docstring.",
                        stmt,
                    )

                literal_args = [
                    arg
                    for arg in _union_members(stmt.annotation)
                    if _subscript_name(arg) == "Literal"
                ]
                if len(literal_args) > 1:
                    fail(
                        f"Field '{field_name}' in {class_node.name} must "
                        "use a single "
                        "Literal type. Please use 'Literal[Literal1, "
                        "Literal2]' instead of 'Union[Literal1, Literal2]'"
                        ".",
                        stmt,
                    )


def validate_ast(tree: ast.stmt):
    ConfigValidator().visit(tree)


def validate_file(file_path: str):
    try:
        print(f"Validating {file_path} config dataclasses ", end="")
        with open(file_path, encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source, filename=file_path)
        validate_ast(tree)
    except ValueError as e:
        print(e)
        raise SystemExit(1) from e
    else:
        print("✅")


def fail(message: str, node: ast.stmt):
    raise ValueError(f"❌ line({node.lineno}): {message}")


def main():
    for filename in sys.argv[1:]:
        # Only run for Python files in vllm/ or tests/
        if not re.match(r"^(vllm|tests)/.*\.py$", filename):
            continue
        # Only run if the file contains @config
        with open(filename, encoding="utf-8") as f:
            if re.search(r"@(?:\w+\.)*config\b", f.read()):
                validate_file(filename)


if __name__ == "__main__":
    main()
