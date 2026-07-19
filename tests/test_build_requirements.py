# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement


def _torch_build_requirement(machine: str) -> Requirement:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    requirements = tomllib.loads(pyproject.read_text())["build-system"]["requires"]
    environment = default_environment() | {"platform_machine": machine}
    matches = [
        requirement
        for item in requirements
        if (requirement := Requirement(item)).name == "torch"
        and requirement.marker is not None
        and requirement.marker.evaluate(environment)
    ]

    assert len(matches) == 1
    return matches[0]


def test_aarch64_build_torch_matches_runtime_pin():
    assert str(_torch_build_requirement("aarch64").specifier) == "==2.10.0"


def test_other_platforms_keep_upstream_build_torch():
    assert str(_torch_build_requirement("x86_64").specifier) == "==2.11.0"
