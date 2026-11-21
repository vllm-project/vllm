# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""tools/csrc_build_inputs.py keeps the native-extension Docker layers cached
across edits that setup.py does not read, and only those. The image builds run
it on the real files; these inputs mirror their shape."""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "tools" / "csrc_build_inputs.py"

SETUP = """
envs = load_module_from_path("envs", os.path.join(ROOT_DIR, "vllm", "envs.py"))
num_jobs = envs.MAX_JOBS
if envs.is_set("CMAKE_BUILD_TYPE"):
    cfg = envs.CMAKE_BUILD_TYPE
"""

ENVS = '''
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    MAX_JOBS: str | None = None


def env_with_choices(name, default, choices):
    """Validated getter."""
    return lambda: os.getenv(name, default)


def unrelated_helper():
    return 1


environment_variables = {
    # Maximum number of compilation jobs.
    "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
    "CMAKE_BUILD_TYPE": env_with_choices("CMAKE_BUILD_TYPE", None, ["Release"]),
    "VLLM_HOST_IP": lambda: os.getenv("VLLM_HOST_IP", ""),
}


def __getattr__(name):
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(name)


def is_set(name):
    """Whether it is set."""
    return name in os.environ
'''

PYPROJECT = """[build-system]
requires = ["setuptools"]

[project]
name = "vllm"

[tool.setuptools_scm]
version_file = "vllm/_version.py"

[tool.ruff.lint]
select = ["E"]
"""


def build_inputs(tmp_path: Path, envs=ENVS, pyproject=PYPROJECT) -> dict[str, str]:
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir(parents=True)
    (src / "setup.py").write_text(SETUP)
    (src / "envs.py").write_text(envs)
    (src / "pyproject.toml").write_text(pyproject)
    subprocess.run([sys.executable, SCRIPT, src, out], check=True)
    return {
        str(p.relative_to(out)): p.read_text() for p in out.rglob("*") if p.is_file()
    }


def test_output_has_only_what_setup_reads(tmp_path):
    files = build_inputs(tmp_path)
    assert sorted(files) == ["pyproject.toml", "vllm/__init__.py", "vllm/envs.py"]
    envs = files["vllm/envs.py"]
    for kept in ("'MAX_JOBS'", "'CMAKE_BUILD_TYPE'", "env_with_choices", "is_set"):
        assert kept in envs
    for dropped in ("VLLM_HOST_IP", "unrelated_helper", "TYPE_CHECKING:"):
        assert dropped not in envs
    assert "#" not in envs and '"""' not in envs
    assert files["pyproject.toml"] == (
        '[project]\nname = "vllm"\n\n[tool.setuptools_scm]\n'
        'version_file = "vllm/_version.py"\n\n'
    )


@pytest.mark.parametrize(
    "old, new, unchanged",
    [
        (
            '"VLLM_HOST_IP": lambda: os.getenv("VLLM_HOST_IP", "")',
            '"VLLM_X": None',
            True,
        ),
        ("# Maximum number of compilation jobs.", "# Edited comment.", True),
        ('"""Whether it is set."""', '"""Edited docstring."""', True),
        ('select = ["E"]', 'select = ["E", "F"]', True),
        ('os.getenv("MAX_JOBS", None)', 'os.getenv("MAX_JOBS", "1")', False),
        ('["Release"]', '["Release", "Debug"]', False),
        ('name = "vllm"', 'name = "vllm2"', False),
    ],
)
def test_output_changes_only_with_build_inputs(tmp_path, old, new, unchanged):
    base = build_inputs(tmp_path / "base")
    envs, pyproject = ENVS.replace(old, new), PYPROJECT.replace(old, new)
    assert (envs, pyproject) != (ENVS, PYPROJECT)
    assert (build_inputs(tmp_path / "edit", envs, pyproject) == base) is unchanged
