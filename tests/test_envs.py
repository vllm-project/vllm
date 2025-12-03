# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from unittest.mock import patch

import pytest

import vllm.envs as envs
from vllm.envs import (
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


class TestVllmConfigureLogging:
    """Test cases for VLLM_CONFIGURE_LOGGING environment variable."""

    def test_configure_logging_defaults_to_true(self):
        """Test that VLLM_CONFIGURE_LOGGING defaults to True when not set."""
        # Ensure the env var is not set
        with patch.dict(os.environ, {}, clear=False):
            if "VLLM_CONFIGURE_LOGGING" in os.environ:
                del os.environ["VLLM_CONFIGURE_LOGGING"]

            # Clear cache if it exists
            if hasattr(envs.__getattr__, "cache_clear"):
                envs.__getattr__.cache_clear()

            result = envs.VLLM_CONFIGURE_LOGGING
            assert result is True
            assert isinstance(result, bool)

    def test_configure_logging_with_zero_string(self):
        """Test that VLLM_CONFIGURE_LOGGING='0' evaluates to False."""
        with patch.dict(os.environ, {"VLLM_CONFIGURE_LOGGING": "0"}):
            # Clear cache if it exists
            if hasattr(envs.__getattr__, "cache_clear"):
                envs.__getattr__.cache_clear()

            result = envs.VLLM_CONFIGURE_LOGGING
            assert result is False
            assert isinstance(result, bool)

    def test_configure_logging_with_one_string(self):
        """Test that VLLM_CONFIGURE_LOGGING='1' evaluates to True."""
        with patch.dict(os.environ, {"VLLM_CONFIGURE_LOGGING": "1"}):
            # Clear cache if it exists
            if hasattr(envs.__getattr__, "cache_clear"):
                envs.__getattr__.cache_clear()

            result = envs.VLLM_CONFIGURE_LOGGING
            assert result is True
            assert isinstance(result, bool)

    def test_configure_logging_with_invalid_value_raises_error(self):
        """Test that invalid VLLM_CONFIGURE_LOGGING value raises ValueError."""
        with patch.dict(os.environ, {"VLLM_CONFIGURE_LOGGING": "invalid"}):
            # Clear cache if it exists
            if hasattr(envs.__getattr__, "cache_clear"):
                envs.__getattr__.cache_clear()

            with pytest.raises(ValueError, match="invalid literal for int"):
                _ = envs.VLLM_CONFIGURE_LOGGING
