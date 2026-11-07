# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys

import pytest

from vllm import collect_env

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture
def uv_venv_without_pip(monkeypatch):
    """Simulate a uv-created venv, which does not ship pip."""
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    monkeypatch.setattr(collect_env, "is_uv_venv", lambda: True)


def test_uv_pip_list_targets_running_interpreter(uv_venv_without_pip):
    commands = []

    def run(command):
        commands.append(command)
        return 0, "torch==2.11.0\nnumpy==2.2.6\nrequests==2.32.0", ""

    _, out = collect_env.get_pip_packages(run)

    # Without --python, uv lists whichever environment it discovers from the
    # working directory, which is not necessarily the one running collect_env.
    assert commands == [
        ["uv", "pip", "list", "--format=freeze", "--python", sys.executable]
    ]
    assert out == "torch==2.11.0\nnumpy==2.2.6"


def test_pip_packages_not_collected_when_uv_fails(uv_venv_without_pip):
    def run(command):
        return 127, "", "Command not found: uv"

    pip_version, out = collect_env.get_pip_packages(run)

    assert pip_version == "pip3"
    assert out is None
