#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Pre-commit script to enforce import restrictions in vllm/ir directory.

vLLM IR code should be independent from the rest of vLLM, except for:
- vllm.ir.* (internal IR imports)
- vllm.logger (centralized logging)
- vllm.logging_utils (logging utilities)

This script checks all Python files in vllm/ir/ and reports any violations.
"""

import ast
import sys
from pathlib import Path


class ImportChecker(ast.NodeVisitor):
    """AST visitor to check for disallowed imports from vllm."""

    def __init__(self, filepath: Path, vllm_root: Path):
        self.filepath = filepath
        self.vllm_root = vllm_root
        self.violations: list[tuple[int, str]] = []
        # Calculate the module path for this file
        self.module_parts = self._get_module_parts()

    def _get_module_parts(self) -> list[str]:
        """Get the module path parts for this file.
        Example: ['vllm', 'ir', 'ops']
        """
        try:
            rel_path = self.filepath.relative_to(self.vllm_root)
            # Remove .py extension and convert path to module parts
            module_path = rel_path.with_suffix("").parts
            return list(module_path)
        except ValueError:
            return []

    def visit_Import(self, node: ast.Import) -> None:
        """Check 'import vllm.xxx' statements."""
        for alias in node.names:
            if self._is_disallowed_import(alias.name):
                self.violations.append((node.lineno, f"import {alias.name}"))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Check 'from vllm.xxx import ...' and 'from . import ...' statements."""
        # Resolve relative imports to absolute module names
        module_name = self._resolve_import(node.module, node.level)

        if module_name and self._is_disallowed_import(module_name):
            imported = ", ".join(alias.name for alias in node.names)
            self.violations.append(
                (node.lineno, f"from {module_name} import {imported}")
            )
        self.generic_visit(node)

    def _resolve_import(self, module: str | None, level: int) -> str | None:
        """
        Resolve a relative or absolute import to its absolute module name.

        Args:
            module: The module name (None for 'from . import x')
            level: The relative import level (0 for absolute, 1+ for relative)

        Returns:
            The absolute module name, or None if it cannot be resolved
        """
        if level == 0:
            # Absolute import
            return module

        # Relative import - resolve based on current module path
        if not self.module_parts or self.module_parts[0] != "vllm":
            # Not in a vllm module, can't resolve
            return None

        base_parts = self.module_parts[:-1]  # Remove file name (or __init__)

        # Go up 'level - 1' more levels
        if level - 1 > len(base_parts):
            return None

        base_parts = base_parts[: -(level - 1)] if level > 1 else base_parts

        if module:
            return ".".join(base_parts + module.split("."))
        else:
            return ".".join(base_parts) if base_parts else None

    def _is_disallowed_import(self, module_name: str) -> bool:
        """
        Check if an import from this module is disallowed.

        Allowed:
        - vllm.ir.* (internal IR imports)
        - vllm.logger.* (centralized logging and submodules)
        - vllm.logging_utils.* (logging utilities and submodules)

        Disallowed:
        - All other vllm.* imports
        """
        # Only check imports from the 'vllm' package.
        if module_name != "vllm" and not module_name.startswith("vllm."):
            return False

        # Allowed: vllm.ir, vllm.logger, vllm.logging_utils
        for allowed in ("vllm.ir", "vllm.logger", "vllm.logging_utils"):
            if module_name == allowed or module_name.startswith(allowed + "."):
                return False

        return True


def check_file(filepath: Path, vllm_root: Path) -> list[tuple[int, str]]:
    """Check a single Python file for disallowed imports."""
    try:
        with open(filepath, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(filepath))

        checker = ImportChecker(filepath, vllm_root)
        checker.visit(tree)
        return checker.violations
    except SyntaxError as e:
        print(f"Syntax error in {filepath}: {e}", file=sys.stderr)
        return []


def main() -> int:
    """Main entry point for the pre-commit script."""
    vllm_root = Path(__file__).parent.parent.parent
    ir_dir = vllm_root / "vllm" / "ir"

    # Use files passed as arguments (from pre-commit), or find all if none
    if len(sys.argv) > 1:
        python_files = [Path(f).resolve() for f in sys.argv[1:]]
    else:
        if not ir_dir.exists():
            print(
                f"Error: vllm/ir directory not found at {ir_dir}",
                file=sys.stderr,
            )
            return 1
        python_files = sorted(ir_dir.rglob("*.py"))

    if not python_files:
        print("Warning: No Python files found in vllm/ir", file=sys.stderr)
        return 0

    all_violations = []
    for filepath in python_files:
        violations = check_file(filepath, vllm_root)
        if violations:
            all_violations.append((filepath, violations))

    # Report violations
    if all_violations:
        for filepath, violations in all_violations:
            rel_path = filepath.relative_to(vllm_root)
            for lineno, import_stmt in violations:
                print(
                    f"{rel_path}:{lineno}: Disallowed import: {import_stmt}",
                    file=sys.stderr,
                )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
