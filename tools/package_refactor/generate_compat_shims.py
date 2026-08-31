# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate or verify legacy import compatibility shims.

The mapping file describes source moves relative to the ``vllm`` package.
Single-file moves and package leaf modules use ``sys.modules`` aliases.
Package roots delegate attribute access lazily to avoid eager imports and
circular dependencies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict

HEADER = """# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
# COMPAT SHIM (auto-generated): old path -> canonical new path
"""


class Move(TypedDict):
    """One legacy-to-canonical package move."""

    old: str
    new: str
    file: bool


def _module_path(path: str) -> str:
    """Convert a package-relative source path to a Python module path."""
    if path.endswith(".py"):
        path = path[:-3]
    return path.replace("/", ".")


def _render_real_import(importer: str, new_module: str) -> str:
    """Render an import assignment using Ruff-compatible line wrapping."""
    statement = f'_real = {importer}.import_module("vllm.{new_module}")'
    if len(statement) <= 88:
        return statement + "\n"
    return f'_real = {importer}.import_module(\n    "vllm.{new_module}"\n)\n'


def _render_alias(old_path: str, new_module: str) -> str:
    """Render a shim that aliases a legacy module to its canonical module."""
    label = old_path[:-3] if old_path.endswith(".py") else old_path
    return (
        HEADER
        + "\n"
        + f'"""Compatibility shim: vllm.{label} -> vllm.{new_module} '
        + '(sys.modules alias)."""\n\n'
        + "import importlib\n"
        + "import sys\n\n"
        + _render_real_import("importlib", new_module)
        + "sys.modules[__name__] = _real\n"
    )


def _render_package(old_path: str, new_module: str) -> str:
    """Render a package shim with lazy attribute delegation."""
    label = old_path[:-11] if old_path.endswith("__init__.py") else old_path
    return (
        HEADER
        + "\n"
        + f'"""Compatibility shim: vllm.{label} -> vllm.{new_module} '
        + '(lazy __getattr__ delegation)."""\n\n'
        + "import importlib as _importlib\n\n"
        + _render_real_import("_importlib", new_module)
        + "\n\n"
        + "def __getattr__(name):\n"
        + "    return getattr(_real, name)\n\n\n"
        + "def __dir__():\n"
        + "    return dir(_real)\n\n\n"
        + '__all__ = getattr(_real, "__all__", [])\n'
    )


def _expected_shims(package_root: Path, moves: list[Move]) -> dict[Path, str]:
    """Build the expected legacy shim file set."""
    expected: dict[Path, str] = {}
    for move in moves:
        old_path = move["old"]
        new_path = move["new"]
        if move.get("file", False):
            expected[package_root / old_path] = _render_alias(
                old_path, _module_path(new_path)
            )
            continue

        canonical_root = package_root / new_path
        if not canonical_root.is_dir():
            raise FileNotFoundError(f"Canonical package not found: {canonical_root}")

        root_init = Path(old_path) / "__init__.py"
        expected[package_root / root_init] = _render_package(
            root_init.as_posix(), _module_path(new_path)
        )

        for source in sorted(canonical_root.rglob("*.py")):
            relative = source.relative_to(canonical_root)
            legacy = Path(old_path) / relative
            canonical = _module_path(
                f"{new_path}/{relative.with_suffix('').as_posix()}"
            )
            if source.name == "__init__.py":
                if relative == Path("__init__.py"):
                    continue
                expected[package_root / legacy] = _render_package(
                    legacy.as_posix(), canonical.removesuffix(".__init__")
                )
            else:
                expected[package_root / legacy] = _render_alias(
                    legacy.as_posix(), canonical
                )
    return expected


def _load_moves(mapping_path: Path) -> list[Move]:
    """Load and minimally validate the package move manifest."""
    moves = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(moves, list):
        raise TypeError("Mapping must contain a JSON list")
    for move in moves:
        if not isinstance(move, dict) or "old" not in move or "new" not in move:
            raise TypeError("Each mapping needs old and new paths")
    return moves


def main() -> int:
    """Generate shims, or verify that existing shims match the manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path("vllm"),
        help="Path to the vllm package root",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path(__file__).with_name("mapping.json"),
        help="Path to the move mapping JSON",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report stale or missing shims without writing files",
    )
    args = parser.parse_args()

    expected = _expected_shims(args.package_root, _load_moves(args.mapping))
    stale = [
        path
        for path, content in expected.items()
        if not path.is_file() or path.read_text(encoding="utf-8") != content
    ]
    if args.check:
        for path in stale:
            print(path)
        if stale:
            print(f"{len(stale)} compatibility shims need regeneration")
            return 1
        print(f"{len(expected)} compatibility shims are current")
        return 0

    for path in stale:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(expected[path], encoding="utf-8")
    print(f"updated {len(stale)} of {len(expected)} compatibility shims")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
