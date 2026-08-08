# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import collect_env

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize("platform", ["darwin", "win32", "cygwin"])
def test_get_pkg_version_skips_linux_package_managers_off_linux(
    monkeypatch, platform
):
    def unexpected_run(_command):
        raise AssertionError("Linux package manager must not run off Linux")

    monkeypatch.setattr(collect_env, "get_platform", lambda: platform)

    assert collect_env.get_pkg_version(unexpected_run, "igc") is None
