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

# Frozen snapshot of the pre-refactor compile_factors() ignore-set (75
# entries). The declarative json_schema_extra markers must reproduce this
# exactly, minus VLLM_CPU_MOE_PREPACK (a dead entry with no backing field).
# Retired in the follow-up task once the literal is gone.
LEGACY_IGNORED_FACTORS = frozenset(
    {
        "MAX_JOBS",
        "VLLM_RPC_BASE_PATH",
        "VLLM_USE_MODELSCOPE",
        "VLLM_RINGBUFFER_WARNING_INTERVAL",
        "VLLM_DEBUG_DUMP_PATH",
        "VLLM_PORT",
        "VLLM_CACHE_ROOT",
        "VLLM_XLA_CACHE_PATH",
        "VLLM_CONFIG_ROOT",
        "VLLM_ENABLE_STARTUP_PLAN",
        "LD_LIBRARY_PATH",
        "VLLM_SERVER_DEV_MODE",
        "VLLM_USE_RUST_FRONTEND",
        "VLLM_RUST_FRONTEND_PATH",
        "VLLM_USE_PRECOMPILED_RUST",
        "VLLM_USE_FASTOKENS",
        "VLLM_DP_MASTER_IP",
        "VLLM_DP_MASTER_PORT",
        "VLLM_NIXL_SIDE_CHANNEL_HOST",
        "VLLM_RANDOMIZE_DP_DUMMY_INPUTS",
        "VLLM_MODEL_REDIRECT_PATH",
        "VLLM_HOST_IP",
        "VLLM_FORCE_AOT_LOAD",
        "S3_ACCESS_KEY_ID",
        "S3_SECRET_ACCESS_KEY",
        "S3_ENDPOINT_URL",
        "VLLM_USAGE_STATS_SERVER",
        "VLLM_NO_USAGE_STATS",
        "VLLM_DO_NOT_TRACK",
        "VLLM_LOGGING_LEVEL",
        "VLLM_LOGGING_PREFIX",
        "VLLM_LOGGING_STREAM",
        "VLLM_LOGGING_CONFIG_PATH",
        "VLLM_LOGGING_COLOR",
        "VLLM_LOG_STATS_INTERVAL",
        "VLLM_DEBUG_LOG_API_SERVER_RESPONSE",
        "VLLM_TUNED_CONFIG_FOLDER",
        "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR",
        "VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS",
        "VLLM_ENGINE_ITERATION_TIMEOUT_S",
        "VLLM_HTTP_TIMEOUT_KEEP_ALIVE",
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS",
        "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS",
        "VLLM_KEEP_ALIVE_ON_ENGINE_DEATH",
        "VLLM_IMAGE_FETCH_TIMEOUT",
        "VLLM_VIDEO_FETCH_TIMEOUT",
        "VLLM_AUDIO_FETCH_TIMEOUT",
        "VLLM_MEDIA_CACHE",
        "VLLM_MEDIA_CACHE_MAX_SIZE_MB",
        "VLLM_MEDIA_CACHE_TTL_HOURS",
        "VLLM_MEDIA_FETCH_MAX_RETRIES",
        "VLLM_MEDIA_URL_ALLOW_REDIRECTS",
        "VLLM_MEDIA_LOADING_THREAD_COUNT",
        "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB",
        "VLLM_MAX_AUDIO_DECODE_DURATION_S",
        "VLLM_MAX_AUDIO_PREPROCESS_WORKERS",
        "VLLM_MAX_IMAGE_PIXELS",
        "VLLM_VIDEO_LOADER_BACKEND",
        "VLLM_MEDIA_CONNECTOR",
        "VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME",
        "VLLM_ASSETS_CACHE",
        "VLLM_ASSETS_CACHE_MODEL_CLEAN",
        "VLLM_WORKER_MULTIPROC_METHOD",
        "VLLM_ENABLE_V1_MULTIPROCESSING",
        "VLLM_V1_OUTPUT_PROC_CHUNK_SIZE",
        "VLLM_CPU_KVCACHE_SPACE",
        "VLLM_CPU_MOE_PREPACK",
        "VLLM_ZENTORCH_WEIGHT_PREPACK",
        "VLLM_TEST_FORCE_LOAD_FORMAT",
        "VLLM_ENABLE_CUDA_COMPATIBILITY",
        "VLLM_CUDA_COMPATIBILITY_PATH",
        "VLLM_SKIP_MODEL_NAME_VALIDATION",
        "LOCAL_RANK",
        "CUDA_VISIBLE_DEVICES",
        "NO_COLOR",
    }
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


def test_p2p_side_channel_defaults_and_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VLLM_P2P_SIDE_CHANNEL_HOST", raising=False)
    monkeypatch.delenv("VLLM_P2P_SIDE_CHANNEL_PORT", raising=False)
    assert envs.VLLM_P2P_SIDE_CHANNEL_HOST == "localhost"
    assert envs.VLLM_P2P_SIDE_CHANNEL_PORT == 5710

    monkeypatch.setenv("VLLM_P2P_SIDE_CHANNEL_HOST", "10.0.0.20")
    monkeypatch.setenv("VLLM_P2P_SIDE_CHANNEL_PORT", "5799")
    assert envs.VLLM_P2P_SIDE_CHANNEL_HOST == "10.0.0.20"
    assert envs.VLLM_P2P_SIDE_CHANNEL_PORT == 5799


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
    # The Rust frontend flag is independent of the C-extension precompiled
    # flag: requesting the precompiled Rust frontend must not implicitly
    # enable the precompiled C extensions.
    with patch.dict(os.environ, {"VLLM_USE_PRECOMPILED_RUST": "1"}, clear=True):
        assert environment_variables["VLLM_USE_PRECOMPILED"]() is False
        assert environment_variables["VLLM_USE_PRECOMPILED_RUST"]() is True

    # ...and the reverse: requesting precompiled C extensions (here via a
    # wheel location, which enables VLLM_USE_PRECOMPILED) must not flip the
    # Rust frontend flag.
    with patch.dict(
        os.environ, {"VLLM_PRECOMPILED_WHEEL_LOCATION": "/tmp/vllm.whl"}, clear=True
    ):
        assert environment_variables["VLLM_USE_PRECOMPILED"]() is True
        assert environment_variables["VLLM_USE_PRECOMPILED_RUST"]() is False

    # ...and with both set together, each flag is still parsed independently.
    with patch.dict(
        os.environ,
        {
            "VLLM_PRECOMPILED_WHEEL_LOCATION": "/tmp/vllm.whl",
            "VLLM_USE_PRECOMPILED_RUST": "1",
        },
        clear=True,
    ):
        assert environment_variables["VLLM_USE_PRECOMPILED"]() is True
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


def test_non_compile_factors_matches_legacy_set():
    """The declarative markers reproduce the old literal exactly, minus the
    one dead entry (VLLM_CPU_MOE_PREPACK has no backing field)."""
    expected = LEGACY_IGNORED_FACTORS - {"VLLM_CPU_MOE_PREPACK"}
    assert expected == envs._NON_COMPILE_FACTORS
