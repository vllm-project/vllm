# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

import vllm.collect_env as collect_env


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_get_pkg_version_ignores_non_linux_platforms(monkeypatch, platform):
    monkeypatch.setattr(collect_env, "get_platform", lambda: platform)

    def run_lambda(_command):
        raise AssertionError("Linux package manager commands must not run")

    assert collect_env.get_pkg_version(run_lambda, "igc") is None
