#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Validation script for vllm/envs/_variables.py

This script validates that:
1. No direct imports of vllm.envs._variables exist in the codebase
   (except in envs/__init__.py which needs it for TYPE_CHECKING)
2. All variables in _variables.py have valid type annotations (no Any allowed)
3. Default values match their declared types (basic check)

Note: Factory function type consistency is handled by mypy.
"""

import ast
import sys
from pathlib import Path


class DirectImportChecker(ast.NodeVisitor):
    """AST visitor to check for direct imports of _variables module.

    Direct imports of vllm.envs._variables are not allowed anywhere,
    including in TYPE_CHECKING blocks (except in envs/__init__.py).
    """

    def __init__(self, filename: str):
        self.filename = filename
        self.errors = []

    def visit_Import(self, node: ast.Import) -> None:
        """Check Import nodes for direct imports of _variables."""
        for alias in node.names:
            if alias.name == "vllm.envs._variables" or alias.name.startswith(
                "vllm.envs._variables."
            ):
                self.errors.append(
                    f"{self.filename}:{node.lineno}: "
                    f"Direct import of vllm.envs._variables is not allowed. "
                    f"Use 'import vllm.envs as envs' instead."
                )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Check ImportFrom nodes for direct imports of _variables."""
        # Check absolute imports (e.g., from vllm.envs._variables import X)
        if node.module and (
            node.module == "vllm.envs._variables"
            or node.module.startswith("vllm.envs._variables.")
        ):
            self.errors.append(
                f"{self.filename}:{node.lineno}: "
                f"Direct import from vllm.envs._variables is not allowed. "
                f"Use 'import vllm.envs as envs' instead."
            )

        # Check relative imports (e.g., from ._variables import X)
        if node.level > 0:
            # from ._variables import X or from ._variables.submodule import X
            if node.module and (
                node.module == "_variables" or node.module.startswith("_variables.")
            ):
                self.errors.append(
                    f"{self.filename}:{node.lineno}: "
                    f"Relative import from _variables is not allowed. "
                    f"Use 'import vllm.envs as envs' instead."
                )
            # from . import _variables
            if node.module is None or node.module == "":
                for alias in node.names:
                    if alias.name == "_variables":
                        self.errors.append(
                            f"{self.filename}:{node.lineno}: "
                            f"Relative import of _variables is not allowed. "
                            f"Use 'import vllm.envs as envs' instead."
                        )


def check_no_direct_imports(vllm_root: Path) -> list[str]:
    """Check all Python files for direct imports of _variables module.

    Args:
        vllm_root: Root directory of vllm package

    Returns:
        List of error messages
    """
    errors = []

    # Check all Python files
    for py_file in vllm_root.rglob("*.py"):
        # Skip envs/__init__.py - it's allowed to import _variables for TYPE_CHECKING
        if py_file.name == "__init__.py" and py_file.parent.name == "envs":
            continue

        # Skip _variables.py itself
        if py_file.name == "_variables.py":
            continue

        try:
            with open(py_file) as f:
                tree = ast.parse(f.read(), filename=str(py_file))

            checker = DirectImportChecker(str(py_file.relative_to(vllm_root)))
            checker.visit(tree)
            errors.extend(checker.errors)
        except SyntaxError:
            # Skip files with syntax errors (might be Python 2 or broken)
            continue

    return errors


def _is_any_type(annotation: ast.expr) -> bool:
    """Check if an annotation is the Any type."""
    # Check for bare 'Any'
    if isinstance(annotation, ast.Name) and annotation.id == "Any":
        return True
    # Check for 'typing.Any'
    return (
        isinstance(annotation, ast.Attribute)
        and annotation.attr == "Any"
        and isinstance(annotation.value, ast.Name)
        and annotation.value.id == "typing"
    )


def _get_annotation_base_type(annotation: ast.expr) -> str | None:
    """Extract the base type name from an annotation.

    Returns the base type name (e.g., 'str', 'int', 'bool', 'list', 'dict')
    or None if it cannot be determined.
    """
    if isinstance(annotation, ast.Name):
        return annotation.id
    if isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name):
        # Handle generic types like list[str], dict[str, int], Optional[str]
        return annotation.value.id
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        # Union type using | syntax (e.g., str | None)
        # Return None as we can't easily validate unions
        return None
    return None


def _get_default_value_type(value: ast.expr) -> str | None:
    """Get the Python type name of a default value from AST.

    Returns the type name or None if it cannot be determined.
    """
    if isinstance(value, ast.Constant):
        if value.value is None:
            return "NoneType"
        return type(value.value).__name__
    if isinstance(value, ast.List):
        return "list"
    if isinstance(value, ast.Dict):
        return "dict"
    if isinstance(value, ast.Tuple):
        return "tuple"
    if isinstance(value, ast.Set):
        return "set"
    # For calls (like EnvFactory(...)), we can't easily determine the type
    return None


def _types_are_compatible(annotation_type: str, value_type: str) -> bool:
    """Check if a value type is compatible with an annotation type."""
    # Direct match
    if annotation_type.lower() == value_type.lower():
        return True

    # Optional types allow None
    if annotation_type == "Optional" and value_type == "NoneType":
        return True

    # Literal types are compatible with their underlying type (usually str)
    if annotation_type == "Literal":
        return True

    # Common type aliases
    type_aliases = {
        "List": "list",
        "Dict": "dict",
        "Tuple": "tuple",
        "Set": "set",
        "Str": "str",
        "Int": "int",
        "Float": "float",
        "Bool": "bool",
    }

    norm_annotation = type_aliases.get(annotation_type, annotation_type).lower()
    norm_value = type_aliases.get(value_type, value_type).lower()

    return norm_annotation == norm_value


def check_variable_type_annotations(variables_file: Path) -> list[str]:
    """Check that all variables in _variables.py have valid type annotations.

    Validates:
    - All uppercase variables have type annotations
    - No variables use 'Any' type
    - Default values match their declared types (when determinable)

    Args:
        variables_file: Path to _variables.py

    Returns:
        List of error messages
    """
    errors = []

    with open(variables_file) as f:
        content = f.read()

    tree = ast.parse(content)

    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            # This is a type-annotated assignment
            if isinstance(node.target, ast.Name) and node.target.id.isupper():
                var_name = node.target.id

                # Check for missing annotation
                if node.annotation is None:
                    errors.append(f"Variable {var_name} is missing type annotation")
                    continue

                # Check for Any type
                if _is_any_type(node.annotation):
                    errors.append(
                        f"Variable {var_name} uses 'Any' type which is not allowed. "
                        f"Please use a specific type annotation."
                    )

                # Check default value type compatibility
                if node.value is not None:
                    annotation_type = _get_annotation_base_type(node.annotation)
                    value_type = _get_default_value_type(node.value)

                    if (
                        annotation_type is not None
                        and value_type is not None
                        and value_type != "NoneType"  # Allow None for Optional
                        and not _types_are_compatible(annotation_type, value_type)
                    ):
                        errors.append(
                            f"Variable {var_name} has type annotation "
                            f"'{annotation_type}' but default value is of type "
                            f"'{value_type}'"
                        )

        elif isinstance(node, ast.Assign):
            # This is a plain assignment without type annotation
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id.isupper()
                    and not target.id.startswith("_")
                ):
                    var_name = target.id
                    errors.append(
                        f"Variable {var_name} must have a type annotation. "
                        f"Use 'VAR_NAME: type = value' instead of 'VAR_NAME = value'"
                    )

    return errors


def main() -> int:
    """Run all validation checks.

    Returns:
        0 if all checks pass, 1 otherwise
    """
    # Find vllm root directory
    script_dir = Path(__file__).parent
    vllm_root = script_dir.parent / "vllm"
    variables_file = vllm_root / "envs" / "_variables.py"

    if not variables_file.exists():
        print(f"Error: {variables_file} not found")
        return 1

    print("Validating environment variable declarations...")
    print()

    all_errors = []

    # Check 1: No direct imports
    print("1. Checking for direct imports of _variables module...")
    import_errors = check_no_direct_imports(vllm_root)
    if import_errors:
        all_errors.extend(import_errors)
        print(f"   ✗ Found {len(import_errors)} direct import(s)")
        for error in import_errors[:5]:  # Show first 5
            print(f"     - {error}")
        if len(import_errors) > 5:
            print(f"     ... and {len(import_errors) - 5} more")
    else:
        print("   ✓ No direct imports found")

    # Check 2: Type annotations
    print("\n2. Checking type annotations...")
    type_errors = check_variable_type_annotations(variables_file)
    if type_errors:
        all_errors.extend(type_errors)
        print(f"   ✗ Found {len(type_errors)} type annotation error(s)")
        for error in type_errors:
            print(f"     - {error}")
    else:
        print("   ✓ All variables have valid type annotations")

    print()
    if all_errors:
        print(f"Validation failed with {len(all_errors)} error(s)")
        return 1
    else:
        print("All validation checks passed! ✓")
        return 0


if __name__ == "__main__":
    sys.exit(main())
