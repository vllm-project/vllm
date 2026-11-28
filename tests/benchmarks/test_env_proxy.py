# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import urllib.request

import pytest

from vllm.benchmarks.serve import env_configured

PROXY_VARS = ["http_proxy", "https_proxy", "all_proxy", "no_proxy"]


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for name in PROXY_VARS:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.upper(), raising=False)
    # On macOS and Windows, getproxies() also reads the system proxy.
    monkeypatch.setattr(
        urllib.request, "getproxies", urllib.request.getproxies_environment
    )
    monkeypatch.setenv("NETRC", str(tmp_path / "netrc"))
    return monkeypatch


def test_nothing_configured(clean_env):
    assert not env_configured()


def test_proxy_for_another_scheme(clean_env):
    clean_env.setenv("HTTPS_PROXY", "http://proxy.internal:3128")
    assert env_configured()


def test_proxy_with_the_host_in_no_proxy(clean_env):
    # A redirect can leave the host in NO_PROXY for one that needs the proxy.
    clean_env.setenv("HTTP_PROXY", "http://proxy.internal:3128")
    clean_env.setenv("NO_PROXY", "localhost,127.0.0.1")
    assert env_configured()


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        # A redirect can go to a host other than the benchmark host.
        ("machine bench.internal login user password secret\n", True),
        ("default login user password secret\n", True),
        ("", False),
        ("machine", False),
    ],
)
def test_netrc(clean_env, tmp_path, content, expected):
    netrc_file = tmp_path / "netrc"
    netrc_file.write_text(content)
    netrc_file.chmod(0o600)
    assert env_configured() is expected
