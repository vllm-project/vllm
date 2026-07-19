# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]


def _requirements_file(path: Path) -> set[str]:
    return {
        str(Requirement(line))
        for raw_line in path.read_text().splitlines()
        if (line := raw_line.strip()) and not line.startswith("#")
    }


def _pep517_build_requirements() -> set[str]:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return {str(Requirement(item)) for item in pyproject["build-system"]["requires"]}


def test_empty_build_tools_mirror_pep517_without_torch():
    pep517_without_torch = {
        requirement
        for requirement in _pep517_build_requirements()
        if Requirement(requirement).name != "torch"
    }
    empty_build_tools = _requirements_file(ROOT / "requirements/build/empty.txt")

    assert empty_build_tools == pep517_without_torch


def test_pep517_torch_requirement_remains_platform_agnostic():
    torch_requirements = {
        requirement
        for requirement in _pep517_build_requirements()
        if Requirement(requirement).name == "torch"
    }

    assert torch_requirements == {"torch==2.11.0"}
