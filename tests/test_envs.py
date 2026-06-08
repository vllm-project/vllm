# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from unittest.mock import patch

import pytest

import vllm.envs as envs
from vllm.envs import (
    disable_envs_cache,
    enable_envs_cache,
    environment_variables,
)


def test_getattr_without_cache(monkeypatch: pytest.MonkeyPatch):
    assert envs.VLLM_HOST_IP == ""
    assert envs.VLLM_PORT is None
    monkeypatch.setenv("VLLM_HOST_IP", "1.1.1.1")
    monkeypatch.setenv("VLLM_PORT", "1234")
    assert envs.VLLM_HOST_IP == "1.1.1.1"
    assert envs.VLLM_PORT == 1234
    # __getattr__ is not decorated with functools.cache
    assert not hasattr(envs.__getattr__, "cache_info")


def test_nixl_side_channel_host_is_not_compile_factor(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_NIXL_SIDE_CHANNEL_HOST", "10.0.0.15")

    assert "VLLM_NIXL_SIDE_CHANNEL_HOST" not in envs.compile_factors()


def test_getattr_with_cache(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_HOST_IP", "1.1.1.1")
    monkeypatch.setenv("VLLM_PORT", "1234")
    # __getattr__ is not decorated with functools.cache
    assert not hasattr(envs.__getattr__, "cache_info")

    # Enable envs cache and ignore ongoing environment changes
    enable_envs_cache()

    # __getattr__ is decorated with functools.cache
    assert hasattr(envs.__getattr__, "cache_info")
    start_hits = envs.__getattr__.cache_info().hits

    # 2 more hits due to VLLM_HOST_IP and VLLM_PORT accesses
    assert envs.VLLM_HOST_IP == "1.1.1.1"
    assert envs.VLLM_PORT == 1234
    assert envs.__getattr__.cache_info().hits == start_hits + 2

    # All environment variables are cached
    for environment_variable in environment_variables:
        envs.__getattr__(environment_variable)
    assert envs.__getattr__.cache_info().hits == start_hits + 2 + len(
        environment_variables
    )

    # Reset envs.__getattr__ back to none-cached version to
    # avoid affecting other tests
    envs.__getattr__ = envs.__getattr__.__wrapped__


def test_getattr_with_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_HOST_IP", "1.1.1.1")
    # __getattr__ is not decorated with functools.cache
    assert not hasattr(envs.__getattr__, "cache_info")

    # Enable envs cache and ignore ongoing environment changes
    enable_envs_cache()
    assert envs.VLLM_HOST_IP == "1.1.1.1"
    # With cache enabled, the environment variable value is cached and unchanged
    monkeypatch.setenv("VLLM_HOST_IP", "2.2.2.2")
    assert envs.VLLM_HOST_IP == "1.1.1.1"

    disable_envs_cache()
    assert envs.VLLM_HOST_IP == "2.2.2.2"
    # After cache disabled, the environment variable value would be synced
    # with os.environ
    monkeypatch.setenv("VLLM_HOST_IP", "3.3.3.3")
    assert envs.VLLM_HOST_IP == "3.3.3.3"


def test_precompiled_install_flags_are_orthogonal() -> None:
    with patch.dict(
        os.environ,
        {
            "VLLM_PRECOMPILED_WHEEL_LOCATION": "/tmp/vllm.whl",
            "VLLM_USE_PRECOMPILED_RUST": "1",
        },
        clear=False,
    ):
        assert environment_variables["VLLM_USE_PRECOMPILED"]() is False
        assert environment_variables["VLLM_USE_PRECOMPILED_RUST"]() is True


def test_is_envs_cache_enabled() -> None:
    assert not envs._is_envs_cache_enabled()
    enable_envs_cache()
    assert envs._is_envs_cache_enabled()

    # Only wrap one-layer of cache, so we only need to
    # call disable once to reset.
    enable_envs_cache()
    enable_envs_cache()
    enable_envs_cache()
    disable_envs_cache()
    assert not envs._is_envs_cache_enabled()

    disable_envs_cache()
    assert not envs._is_envs_cache_enabled()


class TestVllmMaxNSequences:
    def test_default_value(self):
        """Test that VLLM_MAX_N_SEQUENCES defaults to 64."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_MAX_N_SEQUENCES", None)
            if hasattr(envs.__getattr__, "cache_clear"):
                envs.__getattr__.cache_clear()

            assert envs.VLLM_MAX_N_SEQUENCES == 16384

    def test_custom_value(self, monkeypatch: pytest.MonkeyPatch):
        """Test that VLLM_MAX_N_SEQUENCES can be overridden."""
        monkeypatch.setenv("VLLM_MAX_N_SEQUENCES", "128")
        if hasattr(envs.__getattr__, "cache_clear"):
            envs.__getattr__.cache_clear()

        assert envs.VLLM_MAX_N_SEQUENCES == 128

    def test_sampling_params_respects_limit(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Test that SamplingParams rejects n above the limit."""
        from vllm.sampling_params import SamplingParams

        monkeypatch.delenv("VLLM_MAX_N_SEQUENCES", raising=False)
        if hasattr(envs.__getattr__, "cache_clear"):
            envs.__getattr__.cache_clear()

        max_n = envs.VLLM_MAX_N_SEQUENCES
        SamplingParams(n=max_n)

        with pytest.raises(ValueError, match="n must be at most"):
            SamplingParams(n=max_n + 1)

    def test_sampling_params_respects_custom_limit(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Test that SamplingParams uses the overridden env var limit."""
        from vllm.sampling_params import SamplingParams

        monkeypatch.setenv("VLLM_MAX_N_SEQUENCES", "128")
        if hasattr(envs.__getattr__, "cache_clear"):
            envs.__getattr__.cache_clear()

        SamplingParams(n=128)

        with pytest.raises(ValueError, match="n must be at most 128"):
            SamplingParams(n=129)
