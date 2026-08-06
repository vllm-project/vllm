# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_PY = REPO_ROOT / "setup.py"
MIN_FIXED_PYAV_VERSION = Version("17.1.0")


def _setup_audio_requirements() -> list[Requirement]:
    tree = ast.parse(SETUP_PY.read_text())

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "setup":
            continue

        for keyword in node.keywords:
            if keyword.arg != "extras_require":
                continue
            if not isinstance(keyword.value, ast.Dict):
                raise AssertionError("setup.py extras_require is not a dict literal")

            for key, value in zip(keyword.value.keys, keyword.value.values):
                if not isinstance(key, ast.Constant) or key.value != "audio":
                    continue
                requirements = ast.literal_eval(value)
                return [Requirement(requirement) for requirement in requirements]

    raise AssertionError("setup.py does not define an audio extra")


def _requirements_file_requirements(path: Path) -> list[Requirement]:
    requirements = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        requirements.append(Requirement(line))
    return requirements


def _pyav_requirement(requirements: list[Requirement]) -> Requirement:
    matches = [requirement for requirement in requirements if requirement.name == "av"]
    assert len(matches) == 1
    return matches[0]


def _has_fixed_pyav_floor(requirement: Requirement) -> bool:
    return any(
        specifier.operator in {">", ">="}
        and Version(specifier.version) >= MIN_FIXED_PYAV_VERSION
        for specifier in requirement.specifier
    )


def _pins_fixed_pyav(requirement: Requirement) -> bool:
    return any(
        specifier.operator == "=="
        and Version(specifier.version) >= MIN_FIXED_PYAV_VERSION
        for specifier in requirement.specifier
    )


@pytest.mark.parametrize(
    ("requirement", "expected"),
    [
        ("av>=17.1.0", True),
        ("av>17.0.1", False),
        ("av>=17.0.1", False),
        ("av", False),
    ],
)
def test_pyav_floor_semantics(requirement: str, expected: bool) -> None:
    assert _has_fixed_pyav_floor(Requirement(requirement)) is expected


def test_audio_extra_requires_fixed_pyav_floor() -> None:
    assert _has_fixed_pyav_floor(_pyav_requirement(_setup_audio_requirements()))


@pytest.mark.parametrize(
    "relative_path",
    [
        "requirements/test/cuda.in",
        "requirements/test/rocm.in",
    ],
)
def test_test_requirement_inputs_require_fixed_pyav_floor(relative_path: str) -> None:
    requirements = _requirements_file_requirements(REPO_ROOT / relative_path)
    assert _has_fixed_pyav_floor(_pyav_requirement(requirements))


@pytest.mark.parametrize(
    "relative_path",
    [
        "requirements/test/cpu.txt",
        "requirements/test/cuda.txt",
        "requirements/test/rocm.txt",
    ],
)
def test_test_requirement_locks_pin_fixed_pyav(relative_path: str) -> None:
    requirements = _requirements_file_requirements(REPO_ROOT / relative_path)
    assert _pins_fixed_pyav(_pyav_requirement(requirements))
