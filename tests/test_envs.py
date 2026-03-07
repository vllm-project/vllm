# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

import vllm.envs as envs
import vllm.envs_impl as envs_impl
from vllm.envs import (
    disable_envs_cache,
    enable_envs_cache,
    env_list_with_choices,
    env_set_with_choices,
    env_with_choices,
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


class TestEnvWithChoices:
    """Test cases for env_with_choices function."""

    def test_default_value_returned_when_env_not_set(self):
        """Test default is returned when env var is not set."""
        env_func = env_with_choices(
            "NONEXISTENT_ENV", "default", ["option1", "option2"]
        )
        assert env_func() == "default"

    def test_none_default_returned_when_env_not_set(self):
        """Test that None is returned when env not set and default is None."""
        env_func = env_with_choices("NONEXISTENT_ENV", None, ["option1", "option2"])
        assert env_func() is None

    def test_valid_value_returned_case_sensitive(self):
        """Test that valid value is returned in case sensitive mode."""
        with patch.dict(os.environ, {"TEST_ENV": "option1"}):
            env_func = env_with_choices(
                "TEST_ENV", "default", ["option1", "option2"], case_sensitive=True
            )
            assert env_func() == "option1"

    def test_valid_lowercase_value_returned_case_insensitive(self):
        """Test that lowercase value is accepted in case insensitive mode."""
        with patch.dict(os.environ, {"TEST_ENV": "option1"}):
            env_func = env_with_choices(
                "TEST_ENV", "default", ["OPTION1", "OPTION2"], case_sensitive=False
            )
            assert env_func() == "option1"

    def test_valid_uppercase_value_returned_case_insensitive(self):
        """Test that uppercase value is accepted in case insensitive mode."""
        with patch.dict(os.environ, {"TEST_ENV": "OPTION1"}):
            env_func = env_with_choices(
                "TEST_ENV", "default", ["option1", "option2"], case_sensitive=False
            )
            assert env_func() == "OPTION1"

    def test_invalid_value_raises_error_case_sensitive(self):
        """Test that invalid value raises ValueError in case sensitive mode."""
        with patch.dict(os.environ, {"TEST_ENV": "invalid"}):
            env_func = env_with_choices(
                "TEST_ENV", "default", ["option1", "option2"], case_sensitive=True
            )
            with pytest.raises(
                ValueError, match="Invalid value 'invalid' for TEST_ENV"
            ):
                env_func()

    def test_case_mismatch_raises_error_case_sensitive(self):
        """Test that case mismatch raises ValueError in case sensitive mode."""
        with patch.dict(os.environ, {"TEST_ENV": "OPTION1"}):
            env_func = env_with_choices(
                "TEST_ENV", "default", ["option1", "option2"], case_sensitive=True
            )
            with pytest.raises(
                ValueError, match="Invalid value 'OPTION1' for TEST_ENV"
            ):
                env_func()

    def test_invalid_value_raises_error_case_insensitive(self):
        """Test that invalid value raises ValueError when case insensitive."""
        with patch.dict(os.environ, {"TEST_ENV": "invalid"}):
            env_func = env_with_choices(
                "TEST_ENV", "default", ["option1", "option2"], case_sensitive=False
            )
            with pytest.raises(
                ValueError, match="Invalid value 'invalid' for TEST_ENV"
            ):
                env_func()

    def test_callable_choices_resolved_correctly(self):
        """Test that callable choices are resolved correctly."""

        def get_choices():
            return ["dynamic1", "dynamic2"]

        with patch.dict(os.environ, {"TEST_ENV": "dynamic1"}):
            env_func = env_with_choices("TEST_ENV", "default", get_choices)
            assert env_func() == "dynamic1"

    def test_callable_choices_with_invalid_value(self):
        """Test that callable choices raise error for invalid values."""

        def get_choices():
            return ["dynamic1", "dynamic2"]

        with patch.dict(os.environ, {"TEST_ENV": "invalid"}):
            env_func = env_with_choices("TEST_ENV", "default", get_choices)
            with pytest.raises(
                ValueError, match="Invalid value 'invalid' for TEST_ENV"
            ):
                env_func()


class TestEnvListWithChoices:
    """Test cases for env_list_with_choices function."""

    def test_default_list_returned_when_env_not_set(self):
        """Test that default list is returned when env var is not set."""
        env_func = env_list_with_choices(
            "NONEXISTENT_ENV", ["default1", "default2"], ["option1", "option2"]
        )
        assert env_func() == ["default1", "default2"]

    def test_empty_default_list_returned_when_env_not_set(self):
        """Test that empty default list is returned when env not set."""
        env_func = env_list_with_choices("NONEXISTENT_ENV", [], ["option1", "option2"])
        assert env_func() == []

    def test_single_valid_value_parsed_correctly(self):
        """Test that single valid value is parsed correctly."""
        with patch.dict(os.environ, {"TEST_ENV": "option1"}):
            env_func = env_list_with_choices("TEST_ENV", [], ["option1", "option2"])
            assert env_func() == ["option1"]

    def test_multiple_valid_values_parsed_correctly(self):
        """Test that multiple valid values are parsed correctly."""
        with patch.dict(os.environ, {"TEST_ENV": "option1,option2"}):
            env_func = env_list_with_choices("TEST_ENV", [], ["option1", "option2"])
            assert env_func() == ["option1", "option2"]

    def test_values_with_whitespace_trimmed(self):
        """Test that values with whitespace are trimmed correctly."""
        with patch.dict(os.environ, {"TEST_ENV": " option1 , option2 "}):
            env_func = env_list_with_choices("TEST_ENV", [], ["option1", "option2"])
            assert env_func() == ["option1", "option2"]

    def test_empty_values_filtered_out(self):
        """Test that empty values are filtered out."""
        with patch.dict(os.environ, {"TEST_ENV": "option1,,option2,"}):
            env_func = env_list_with_choices("TEST_ENV", [], ["option1", "option2"])
            assert env_func() == ["option1", "option2"]

    def test_empty_string_returns_default(self):
        """Test that empty string returns default."""
        with patch.dict(os.environ, {"TEST_ENV": ""}):
            env_func = env_list_with_choices(
                "TEST_ENV", ["default"], ["option1", "option2"]
            )
            assert env_func() == ["default"]

    def test_only_commas_returns_default(self):
        """Test that string with only commas returns default."""
        with patch.dict(os.environ, {"TEST_ENV": ",,,"}):
            env_func = env_list_with_choices(
                "TEST_ENV", ["default"], ["option1", "option2"]
            )
            assert env_func() == ["default"]

    def test_case_sensitive_validation(self):
        """Test case sensitive validation."""
        with patch.dict(os.environ, {"TEST_ENV": "option1,OPTION2"}):
            env_func = env_list_with_choices(
                "TEST_ENV", [], ["option1", "option2"], case_sensitive=True
            )
            with pytest.raises(ValueError, match="Invalid value 'OPTION2' in TEST_ENV"):
                env_func()

    def test_case_insensitive_validation(self):
        """Test case insensitive validation."""
        with patch.dict(os.environ, {"TEST_ENV": "OPTION1,option2"}):
            env_func = env_list_with_choices(
                "TEST_ENV", [], ["option1", "option2"], case_sensitive=False
            )
            assert env_func() == ["OPTION1", "option2"]

    def test_invalid_value_in_list_raises_error(self):
        """Test that invalid value in list raises ValueError."""
        with patch.dict(os.environ, {"TEST_ENV": "option1,invalid,option2"}):
            env_func = env_list_with_choices("TEST_ENV", [], ["option1", "option2"])
            with pytest.raises(ValueError, match="Invalid value 'invalid' in TEST_ENV"):
                env_func()

    def test_callable_choices_resolved_correctly(self):
        """Test that callable choices are resolved correctly."""

        def get_choices():
            return ["dynamic1", "dynamic2"]

        with patch.dict(os.environ, {"TEST_ENV": "dynamic1,dynamic2"}):
            env_func = env_list_with_choices("TEST_ENV", [], get_choices)
            assert env_func() == ["dynamic1", "dynamic2"]

    def test_callable_choices_with_invalid_value(self):
        """Test that callable choices raise error for invalid values."""

        def get_choices():
            return ["dynamic1", "dynamic2"]

        with patch.dict(os.environ, {"TEST_ENV": "dynamic1,invalid"}):
            env_func = env_list_with_choices("TEST_ENV", [], get_choices)
            with pytest.raises(ValueError, match="Invalid value 'invalid' in TEST_ENV"):
                env_func()

    def test_duplicate_values_preserved(self):
        """Test that duplicate values in the list are preserved."""
        with patch.dict(os.environ, {"TEST_ENV": "option1,option1,option2"}):
            env_func = env_list_with_choices("TEST_ENV", [], ["option1", "option2"])
            assert env_func() == ["option1", "option1", "option2"]


class TestEnvSetWithChoices:
    """Test cases for env_set_with_choices function."""

    def test_default_list_returned_when_env_not_set(self):
        """Test that default list is returned when env var is not set."""
        env_func = env_set_with_choices(
            "NONEXISTENT_ENV", ["default1", "default2"], ["option1", "option2"]
        )
        assert env_func() == {"default1", "default2"}

    def test_empty_default_list_returned_when_env_not_set(self):
        """Test that empty default list is returned when env not set."""
        env_func = env_set_with_choices("NONEXISTENT_ENV", [], ["option1", "option2"])
        assert env_func() == set()

    def test_single_valid_value_parsed_correctly(self):
        """Test that single valid value is parsed correctly."""
        with patch.dict(os.environ, {"TEST_ENV": "option1"}):
            env_func = env_set_with_choices("TEST_ENV", [], ["option1", "option2"])
            assert env_func() == {"option1"}

    def test_multiple_valid_values_parsed_correctly(self):
        """Test that multiple valid values are parsed correctly."""
        with patch.dict(os.environ, {"TEST_ENV": "option1,option2"}):
            env_func = env_set_with_choices("TEST_ENV", [], ["option1", "option2"])
            assert env_func() == {"option1", "option2"}

    def test_values_with_whitespace_trimmed(self):
        """Test that values with whitespace are trimmed correctly."""
        with patch.dict(os.environ, {"TEST_ENV": " option1 , option2 "}):
            env_func = env_set_with_choices("TEST_ENV", [], ["option1", "option2"])
            assert env_func() == {"option1", "option2"}

    def test_empty_values_filtered_out(self):
        """Test that empty values are filtered out."""
        with patch.dict(os.environ, {"TEST_ENV": "option1,,option2,"}):
            env_func = env_set_with_choices("TEST_ENV", [], ["option1", "option2"])
            assert env_func() == {"option1", "option2"}

    def test_empty_string_returns_default(self):
        """Test that empty string returns default."""
        with patch.dict(os.environ, {"TEST_ENV": ""}):
            env_func = env_set_with_choices(
                "TEST_ENV", ["default"], ["option1", "option2"]
            )
            assert env_func() == {"default"}

    def test_only_commas_returns_default(self):
        """Test that string with only commas returns default."""
        with patch.dict(os.environ, {"TEST_ENV": ",,,"}):
            env_func = env_set_with_choices(
                "TEST_ENV", ["default"], ["option1", "option2"]
            )
            assert env_func() == {"default"}

    def test_case_sensitive_validation(self):
        """Test case sensitive validation."""
        with patch.dict(os.environ, {"TEST_ENV": "option1,OPTION2"}):
            env_func = env_set_with_choices(
                "TEST_ENV", [], ["option1", "option2"], case_sensitive=True
            )
            with pytest.raises(ValueError, match="Invalid value 'OPTION2' in TEST_ENV"):
                env_func()

    def test_case_insensitive_validation(self):
        """Test case insensitive validation."""
        with patch.dict(os.environ, {"TEST_ENV": "OPTION1,option2"}):
            env_func = env_set_with_choices(
                "TEST_ENV", [], ["option1", "option2"], case_sensitive=False
            )
            assert env_func() == {"OPTION1", "option2"}

    def test_invalid_value_in_list_raises_error(self):
        """Test that invalid value in list raises ValueError."""
        with patch.dict(os.environ, {"TEST_ENV": "option1,invalid,option2"}):
            env_func = env_set_with_choices("TEST_ENV", [], ["option1", "option2"])
            with pytest.raises(ValueError, match="Invalid value 'invalid' in TEST_ENV"):
                env_func()

    def test_callable_choices_resolved_correctly(self):
        """Test that callable choices are resolved correctly."""

        def get_choices():
            return ["dynamic1", "dynamic2"]

        with patch.dict(os.environ, {"TEST_ENV": "dynamic1,dynamic2"}):
            env_func = env_set_with_choices("TEST_ENV", [], get_choices)
            assert env_func() == {"dynamic1", "dynamic2"}

    def test_callable_choices_with_invalid_value(self):
        """Test that callable choices raise error for invalid values."""

        def get_choices():
            return ["dynamic1", "dynamic2"]

        with patch.dict(os.environ, {"TEST_ENV": "dynamic1,invalid"}):
            env_func = env_set_with_choices("TEST_ENV", [], get_choices)
            with pytest.raises(ValueError, match="Invalid value 'invalid' in TEST_ENV"):
                env_func()

    def test_duplicate_values_deduped(self):
        """Test that duplicate values in the list are deduped."""
        with patch.dict(os.environ, {"TEST_ENV": "option1,option1,option2"}):
            env_func = env_set_with_choices("TEST_ENV", [], ["option1", "option2"])
            assert env_func() == {"option1", "option2"}


# ── envs_impl-specific tests ────────────────────────────────────────────────────
# The tests below target vllm.envs_impl (the new implementation).
# envs_impl uses pydantic TypeAdapter for coercion, so error types differ from
# the old vllm.envs (which used bool(int(x))).


class TestVllmConfigureLogging:
    """Test VLLM_CONFIGURE_LOGGING via vllm.envs_impl."""

    def test_configure_logging_defaults_to_true(self):
        with patch.dict(os.environ, {}, clear=False):
            if "VLLM_CONFIGURE_LOGGING" in os.environ:
                del os.environ["VLLM_CONFIGURE_LOGGING"]
            if hasattr(envs_impl.__getattr__, "cache_clear"):
                envs_impl.__getattr__.cache_clear()
            result = envs_impl.VLLM_CONFIGURE_LOGGING
            assert result is True
            assert isinstance(result, bool)

    def test_configure_logging_with_zero_string(self):
        with patch.dict(os.environ, {"VLLM_CONFIGURE_LOGGING": "0"}):
            if hasattr(envs_impl.__getattr__, "cache_clear"):
                envs_impl.__getattr__.cache_clear()
            result = envs_impl.VLLM_CONFIGURE_LOGGING
            assert result is False
            assert isinstance(result, bool)

    def test_configure_logging_with_one_string(self):
        with patch.dict(os.environ, {"VLLM_CONFIGURE_LOGGING": "1"}):
            if hasattr(envs_impl.__getattr__, "cache_clear"):
                envs_impl.__getattr__.cache_clear()
            result = envs_impl.VLLM_CONFIGURE_LOGGING
            assert result is True
            assert isinstance(result, bool)

    def test_configure_logging_with_invalid_value_raises_error(self):
        # pydantic raises ValidationError (not ValueError) for invalid bool strings
        with patch.dict(os.environ, {"VLLM_CONFIGURE_LOGGING": "invalid"}):
            if hasattr(envs_impl.__getattr__, "cache_clear"):
                envs_impl.__getattr__.cache_clear()
            with pytest.raises(ValidationError):
                _ = envs_impl.VLLM_CONFIGURE_LOGGING


def test_basic_access_types(monkeypatch: pytest.MonkeyPatch):
    """Test default values and types via vllm.envs_impl."""
    monkeypatch.delenv("VLLM_HOST_IP", raising=False)
    monkeypatch.delenv("VLLM_PORT", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("VLLM_USE_MODELSCOPE", raising=False)

    assert isinstance(envs_impl.VLLM_HOST_IP, str)
    assert envs_impl.VLLM_HOST_IP == ""

    assert envs_impl.VLLM_PORT is None

    assert isinstance(envs_impl.LOCAL_RANK, int)
    assert envs_impl.LOCAL_RANK == 0

    assert isinstance(envs_impl.VLLM_USE_MODELSCOPE, bool)
    assert not envs_impl.VLLM_USE_MODELSCOPE


def test_env_var_parsing(monkeypatch: pytest.MonkeyPatch):
    """Test that vllm.envs_impl parses env vars to the correct types."""
    monkeypatch.setenv("VLLM_HOST_IP", "192.168.1.1")
    monkeypatch.setenv("VLLM_PORT", "8000")
    monkeypatch.setenv("LOCAL_RANK", "5")
    monkeypatch.setenv("VLLM_USE_MODELSCOPE", "1")

    assert isinstance(envs_impl.VLLM_HOST_IP, str)
    assert envs_impl.VLLM_HOST_IP == "192.168.1.1"

    assert isinstance(envs_impl.VLLM_PORT, int)
    assert envs_impl.VLLM_PORT == 8000

    assert isinstance(envs_impl.LOCAL_RANK, int)
    assert envs_impl.LOCAL_RANK == 5

    assert isinstance(envs_impl.VLLM_USE_MODELSCOPE, bool)
    assert envs_impl.VLLM_USE_MODELSCOPE


def test_lazy_defaults(monkeypatch: pytest.MonkeyPatch):
    """Test that callable default_factory values are resolved lazily."""
    monkeypatch.delenv("VLLM_CACHE_ROOT", raising=False)

    cache_root = envs_impl.VLLM_CACHE_ROOT
    assert isinstance(cache_root, str)
    assert "vllm" in cache_root


def test_is_set(monkeypatch: pytest.MonkeyPatch):
    """Test envs_impl.is_set() — works for any name, not just registered vars."""
    monkeypatch.delenv("VLLM_TEST_VAR_123", raising=False)
    assert not envs_impl.is_set("VLLM_TEST_VAR_123")

    monkeypatch.setenv("VLLM_TEST_VAR_123", "test")
    assert envs_impl.is_set("VLLM_TEST_VAR_123")
