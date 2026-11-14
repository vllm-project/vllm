# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.benchmarks.serve import env_proxy_applies

PROXY_VARS = ["http_proxy", "https_proxy", "all_proxy", "no_proxy"]


@pytest.fixture
def clean_env(monkeypatch):
    for name in PROXY_VARS:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.upper(), raising=False)
    return monkeypatch


def test_no_proxy_configured(clean_env):
    assert not env_proxy_applies("http://127.0.0.1:8000/v1/completions")


def test_proxy_for_the_scheme(clean_env):
    clean_env.setenv("HTTP_PROXY", "http://proxy.internal:3128")
    assert env_proxy_applies("http://10.0.0.5:8000/v1/completions")
    assert not env_proxy_applies("https://10.0.0.5:8000/v1/completions")


def test_host_excluded_by_no_proxy(clean_env):
    clean_env.setenv("HTTP_PROXY", "http://proxy.internal:3128")
    clean_env.setenv("NO_PROXY", "localhost,127.0.0.1")
    assert not env_proxy_applies("http://127.0.0.1:8000/v1/completions")
    assert env_proxy_applies("http://10.0.0.5:8000/v1/completions")
