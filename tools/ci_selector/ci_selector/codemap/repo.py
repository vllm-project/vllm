# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Repo file catalog and Python module <-> path resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..handwritten import PACKAGE_ROOTS, SKIP_DIRS

INSTALLABLE_TEST_ROOT = "tests/vllm_test_utils"
INSTALLABLE_TEST_PARENT = "tests/plugins"
PROJECT_MARKERS = ("setup.py", "pyproject.toml")
TEST_FILE_GLOB = "test_*.py"

TESTS_ROOT = "tests"


def is_test_basename(path: str) -> bool:
    """True for a test-named basename anywhere in the tree. Call sites that
    must stay tests-only filter through this; leaf-edge attachment widens on
    purpose in factories._leaf_consumer."""
    return path.rsplit("/", 1)[-1].startswith("test_")


def is_test_file(path: str) -> bool:
    """The strict variant: under tests/ and test-named."""
    return path.startswith("tests/") and is_test_basename(path)


def test_file_catalog(repo: Path) -> list[str]:
    """All pytest-collectible test files under tests/, repo-relative."""
    return [
        path.relative_to(repo).as_posix()
        for path in sorted((repo / TESTS_ROOT).rglob(TEST_FILE_GLOB))
        if not SKIP_DIRS.intersection(path.parts)
    ]


@dataclass
class ModuleIndex:
    module_to_file: dict[str, str] = field(default_factory=dict)
    file_to_module: dict[str, str] = field(default_factory=dict)
    # file -> its pip-installable project dir; these load through entry points,
    # so no import edge reaches them
    installable_roots: dict[str, str] = field(default_factory=dict)

    def resolve(self, name: str) -> str | None:
        return self.module_to_file.get(name)

    def add(self, module: str, file: str) -> None:
        self.module_to_file[module] = file
        self.file_to_module[file] = module


def build_module_index(repo: Path) -> ModuleIndex:
    index = ModuleIndex()
    for root in PACKAGE_ROOTS:
        base = repo / root
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            if SKIP_DIRS.intersection(path.parts):
                continue
            rel = path.relative_to(repo)
            index.add(_module_name(rel), rel.as_posix())
    _add_installable_test_packages(repo, index)
    return index


def _module_name(rel: Path) -> str:
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][:-3]
    return ".".join(parts)


def _add_installable_test_packages(repo: Path, index: ModuleIndex) -> None:
    """Packages under tests/ that CI pip-installs and imports by their own
    top-level name. Map that name, not the tests.-prefixed path."""
    candidates = [repo / INSTALLABLE_TEST_ROOT]
    plugins = repo / INSTALLABLE_TEST_PARENT
    if plugins.is_dir():
        candidates.extend(p for p in plugins.iterdir() if p.is_dir())
    for project in candidates:
        if not any((project / marker).is_file() for marker in PROJECT_MARKERS):
            continue
        for path in project.glob("*.py"):
            index.installable_roots[path.relative_to(repo).as_posix()] = (
                project.relative_to(repo).as_posix()
            )
        for child in project.iterdir():
            if child.is_dir() and (child / "__init__.py").is_file():
                for path in child.rglob("*.py"):
                    rel = path.relative_to(repo)
                    pkg_rel = path.relative_to(project)
                    index.add(_module_name(pkg_rel), rel.as_posix())
                    index.installable_roots[rel.as_posix()] = project.relative_to(
                        repo
                    ).as_posix()
