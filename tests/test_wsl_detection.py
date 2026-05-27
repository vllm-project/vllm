# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import platform
from unittest.mock import patch

from vllm.platforms.interface import in_wsl


def _uname(system="Linux", node="host", release="5.15.0", version="#1"):
    # platform.uname() returns a named tuple; build one here so in_wsl() can
    # read attributes (uname.release, uname.version) on the mock.
    return platform.uname_result(system, node, release, version, "x86_64")


def test_in_wsl_true_for_wsl2_release():
    fake = _uname(release="5.15.167.4-microsoft-standard-WSL2")
    with patch("vllm.platforms.interface.platform.uname", return_value=fake):
        assert in_wsl()


def test_in_wsl_true_for_wsl1_release():
    fake = _uname(release="4.4.0-19041-Microsoft", version="#1234-Microsoft")
    with patch("vllm.platforms.interface.platform.uname", return_value=fake):
        assert in_wsl()


def test_in_wsl_false_for_native_linux():
    fake = _uname(release="6.10.0-generic")
    with patch("vllm.platforms.interface.platform.uname", return_value=fake):
        assert not in_wsl()


def test_in_wsl_false_when_only_hostname_contains_microsoft():
    # Regression test for #41933: a pod or host named e.g. "microsoft-pod-0"
    # must not be detected as WSL.
    fake = _uname(node="microsoft-pod-0", release="6.10.0-generic")
    with patch("vllm.platforms.interface.platform.uname", return_value=fake):
        assert not in_wsl()
