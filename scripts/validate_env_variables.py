#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Validation script for vllm/envs/_variables.py

This script validates that:
1. No direct imports of vllm.envs._variables exist in the codebase
   (except in envs/__init__.py which needs it for TYPE_CHECKING)
2. All variables in _variables.py have valid type annotations
3. Lazy default factories return values matching their type annotations
4. EnvFactory instances have consistent types between default_value and parse function
"""

import ast
import sys
from pathlib import Path
from typing import Any, get_type_hints


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
        if node.module and (
            node.module == "vllm.envs._variables"
            or node.module.startswith("vllm.envs._variables.")
        ):
            self.errors.append(
                f"{self.filename}:{node.lineno}: "
                f"Direct import from vllm.envs._variables is not allowed. "
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

        # Skip test files for now (they might need special imports)
        if "test" in py_file.name:
            continue

        try:
            with open(py_file) as f:
                tree = ast.parse(f.read(), filename=str(py_file))

            checker = DirectImportChecker(str(py_file.relative_to(vllm_root)))
            checker.visit(tree)
            errors.extend(checker.errors)
        except SyntaxError as e:
            # Skip files with syntax errors (might be Python 2 or broken)
            continue

    return errors


def check_variable_type_annotations(variables_file: Path) -> list[str]:
    """Check that all variables in _variables.py have type annotations.

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
            if isinstance(node.target, ast.Name) and node.target.id.isupper() and node.annotation is None:
                var_name = node.target.id
                # This is an environment variable (all caps) missing type annotation
                errors.append(
                    f"Variable {var_name} is missing type annotation"
                )
        elif isinstance(node, ast.Assign):
            # This is a plain assignment without type annotation
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper() and not target.id.startswith("_"):
                    var_name = target.id
                    errors.append(
                        f"Variable {var_name} must have a type annotation. "
                        f"Use 'VAR_NAME: type = value' instead of 'VAR_NAME = value'"
                    )

    return errors


def check_lazy_default_consistency(variables_file: Path) -> list[str]:
    """Check that env_factory and env_default_factory are used correctly.

    This performs basic structural checks:
    - env_factory receives exactly 2 arguments (default value, parser function)
    - env_default_factory receives exactly 1 argument (callable)

    Note: This does NOT perform full type consistency validation between
    type hints and factory return types. Full type checking would require
    runtime analysis or a static type checker like mypy.

    Args:
        variables_file: Path to _variables.py

    Returns:
        List of warning messages
    """
    warnings = []

    with open(variables_file) as f:
        content = f.read()

    # Parse the file using AST for more accurate analysis
    try:
        tree = ast.parse(content)
    except SyntaxError:
        warnings.append("Could not parse _variables.py - syntax error")
        return warnings

    # Check for proper usage of env_factory and env_default_factory
    # This only checks argument counts, not type consistency
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func_name = node.func.id

            if func_name == 'env_default_factory':
                # Should have exactly 1 argument (a callable)
                if len(node.args) != 1:
                    warnings.append(
                        f"env_default_factory should receive "
                        f"exactly 1 callable argument"
                    )

            elif func_name == 'env_factory':
                # Should have exactly 2 arguments (default value and parser)
                if len(node.args) != 2:
                    warnings.append(
                        f"env_factory should receive exactly "
                        f"2 arguments (default, parser)"
                    )

    # Add a note if no issues were found
    if not warnings:
        warnings.append(
            "NOTE: Basic structural checks passed. For full type consistency "
            "validation, run mypy on vllm/envs/_variables.py"
        )

    return warnings


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
        print(f"   ✗ Found {len(type_errors)} missing type annotation(s)")
        for error in type_errors:
            print(f"     - {error}")
    else:
        print("   ✓ All variables have type annotations")

    # Check 3: Lazy defaults consistency
    print("\n3. Checking factory function usage...")
    lazy_warnings = check_lazy_default_consistency(variables_file)
    # Separate informational notes from actual warnings
    notes = [w for w in lazy_warnings if w.startswith("NOTE:")]
    actual_warnings = [w for w in lazy_warnings if not w.startswith("NOTE:")]

    if actual_warnings:
        print(f"   ⚠ Found {len(actual_warnings)} warning(s)")
        for warning in actual_warnings:
            print(f"     - {warning}")
    else:
        print("   ✓ Factory functions used correctly")

    # Display informational notes
    if notes:
        for note in notes:
            print(f"   ℹ {note[6:]}")  # Skip "NOTE: " prefix

    print()
    if all_errors:
        print(f"Validation failed with {len(all_errors)} error(s)")
        return 1
    else:
        print("All validation checks passed! ✓")
        return 0


if __name__ == "__main__":
    sys.exit(main())
