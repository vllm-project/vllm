#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Validate env var declarations in vllm/envs_impl/_variables.py."""

import ast
import importlib
import sys
from pathlib import Path


class DirectImportChecker(ast.NodeVisitor):
    """Check for disallowed direct imports of _variables module."""

    def __init__(self, filename: str):
        self.filename = filename
        self.errors = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "vllm.envs_impl._variables" or alias.name.startswith(
                "vllm.envs_impl._variables."
            ):
                self.errors.append(
                    f"{self.filename}:{node.lineno}: "
                    f"Direct import of vllm.envs_impl._variables is not allowed. "
                    f"Use 'import vllm.envs_impl as envs' instead."
                )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and (
            node.module == "vllm.envs_impl._variables"
            or node.module.startswith("vllm.envs_impl._variables.")
        ):
            self.errors.append(
                f"{self.filename}:{node.lineno}: "
                f"Direct import from vllm.envs_impl._variables is not allowed. "
                f"Use 'import vllm.envs_impl as envs' instead."
            )

        if node.level > 0:
            if node.module and (
                node.module == "_variables" or node.module.startswith("_variables.")
            ):
                self.errors.append(
                    f"{self.filename}:{node.lineno}: "
                    f"Relative import from _variables is not allowed. "
                    f"Use 'import vllm.envs_impl as envs' instead."
                )
            if node.module is None or node.module == "":
                for alias in node.names:
                    if alias.name == "_variables":
                        self.errors.append(
                            f"{self.filename}:{node.lineno}: "
                            f"Relative import of _variables is not allowed. "
                            f"Use 'import vllm.envs_impl as envs' instead."
                        )


def check_no_direct_imports(vllm_root: Path) -> list[str]:
    errors = []

    # Check all Python files
    for py_file in vllm_root.rglob("*.py"):
        if py_file.name == "__init__.py" and py_file.parent.name == "envs_impl":
            continue

        if py_file.name == "_variables.py":
            continue

        try:
            with open(py_file) as f:
                tree = ast.parse(f.read(), filename=str(py_file))

            checker = DirectImportChecker(str(py_file.relative_to(vllm_root)))
            checker.visit(tree)
            errors.extend(checker.errors)
        except SyntaxError:
            continue

    return errors


def _is_any_type(annotation: ast.expr) -> bool:
    if isinstance(annotation, ast.Name) and annotation.id == "Any":
        return True
    return (
        isinstance(annotation, ast.Attribute)
        and annotation.attr == "Any"
        and isinstance(annotation.value, ast.Name)
        and annotation.value.id == "typing"
    )


def _get_annotation_base_type(annotation: ast.expr) -> str | None:
    if isinstance(annotation, ast.Name):
        return annotation.id
    if isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name):
        return annotation.value.id
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        return None
    return None


def _get_default_value_type(value: ast.expr) -> str | None:
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
    return None


def _types_are_compatible(annotation_type: str, value_type: str) -> bool:
    if annotation_type.lower() == value_type.lower():
        return True

    if annotation_type == "Optional" and value_type == "NoneType":
        return True

    if annotation_type == "Literal":
        return True

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
    errors = []

    with open(variables_file) as f:
        content = f.read()

    tree = ast.parse(content)

    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id.isupper():
                var_name = node.target.id

                if node.annotation is None:
                    errors.append(f"Variable {var_name} is missing type annotation")
                    continue

                if _is_any_type(node.annotation):
                    errors.append(
                        f"Variable {var_name} uses 'Any' type which is not allowed. "
                        f"Please use a specific type annotation."
                    )

                if node.value is not None:
                    annotation_type = _get_annotation_base_type(node.annotation)
                    value_type = _get_default_value_type(node.value)

                    if (
                        annotation_type is not None
                        and value_type is not None
                        and value_type != "NoneType"
                        and not _types_are_compatible(annotation_type, value_type)
                    ):
                        errors.append(
                            f"Variable {var_name} has type annotation "
                            f"'{annotation_type}' but default value is of type "
                            f"'{value_type}'"
                        )

        elif isinstance(node, ast.Assign):
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


def _get_declared_variables(variables_file: Path) -> set[str]:
    with open(variables_file) as f:
        tree = ast.parse(f.read())

    return {
        node.target.id
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id.isupper()
            and not node.target.id.startswith("_")
        )
    }


def check_env_var_list_sync(repo_root: Path, variables_file: Path) -> list[str]:
    errors = []

    declared_vars = _get_declared_variables(variables_file)

    sys.path.insert(0, str(repo_root))
    envs_module = importlib.import_module("vllm.envs_impl")
    exposed_vars = set(envs_module.__dir__())

    missing_in_variables = sorted(exposed_vars - declared_vars)
    extra_in_variables = sorted(declared_vars - exposed_vars)

    if missing_in_variables:
        errors.append(
            "_variables.py is missing env vars defined in vllm.envs_impl: "
            f"{', '.join(missing_in_variables)}"
        )
    if extra_in_variables:
        errors.append(
            "_variables.py has env vars not exposed by vllm.envs_impl: "
            f"{', '.join(extra_in_variables)}"
        )

    return errors


def main() -> int:
    script_dir = Path(__file__).parent
    vllm_root = script_dir.parent / "vllm"
    variables_file = vllm_root / "envs_impl" / "_variables.py"

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

    # Check 3: Env var list sync
    print("\n3. Checking env var list sync with vllm.envs_impl...")
    sync_errors = check_env_var_list_sync(script_dir.parent, variables_file)
    if sync_errors:
        all_errors.extend(sync_errors)
        print(f"   ✗ Found {len(sync_errors)} sync error(s)")
        for error in sync_errors:
            print(f"     - {error}")
    else:
        print("   ✓ Env var list is in sync")

    print()
    if all_errors:
        print(f"Validation failed with {len(all_errors)} error(s)")
        return 1
    else:
        print("All validation checks passed! ✓")
        return 0


if __name__ == "__main__":
    sys.exit(main())
