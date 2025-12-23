# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for sleep mode error handling.

This tests that when a request is sent while the engine is sleeping,
it returns a graceful error instead of crashing the engine.
"""

import pytest

from vllm import LLM, SamplingParams
from vllm.v1.engine.exceptions import EngineSleepingError


@pytest.fixture
def llm_with_sleep_mode():
    """Create an LLM with sleep mode enabled."""
    llm = LLM(
        model="facebook/opt-125m",
        enable_sleep_mode=True,
        enforce_eager=True,  # Faster for testing
        gpu_memory_utilization=0.3,  # Use less memory for testing
    )
    yield llm
    # Cleanup
    del llm


class TestSleepModeError:
    """Test suite for sleep mode error handling."""

    def test_normal_request_works(self, llm_with_sleep_mode):
        """Test that normal requests work when engine is not sleeping."""
        llm = llm_with_sleep_mode
        
        outputs = llm.generate("Hello, world!", SamplingParams(max_tokens=5))
        
        assert len(outputs) == 1
        assert len(outputs[0].outputs[0].token_ids) > 0

    def test_sleeping_request_raises_error(self, llm_with_sleep_mode):
        """Test that requests while sleeping raise EngineSleepingError, not crash."""
        llm = llm_with_sleep_mode
        
        # Put the model to sleep
        llm.sleep(level=1)
        
        # Request while sleeping should raise EngineSleepingError
        with pytest.raises(EngineSleepingError) as exc_info:
            llm.generate("Hello!", SamplingParams(max_tokens=5))
        
        # Verify error message is helpful
        assert "sleep" in str(exc_info.value).lower()
        assert "wake" in str(exc_info.value).lower()

    def test_wake_up_then_request_works(self, llm_with_sleep_mode):
        """Test that requests work after waking up."""
        llm = llm_with_sleep_mode
        
        # Put to sleep
        llm.sleep(level=1)
        
        # Wake up
        llm.wake_up()
        
        # Request should work again
        outputs = llm.generate("Hello!", SamplingParams(max_tokens=5))
        
        assert len(outputs) == 1
        assert len(outputs[0].outputs[0].token_ids) > 0

    def test_full_sleep_wake_cycle(self, llm_with_sleep_mode):
        """Test a full cycle: request -> sleep -> fail -> wake -> request."""
        llm = llm_with_sleep_mode
        
        # Step 1: Initial request works
        outputs = llm.generate("Step 1", SamplingParams(max_tokens=3))
        assert len(outputs) == 1
        
        # Step 2: Sleep
        llm.sleep(level=1)
        
        # Step 3: Request fails gracefully
        with pytest.raises(EngineSleepingError):
            llm.generate("Step 3", SamplingParams(max_tokens=3))
        
        # Step 4: Wake up
        llm.wake_up()
        
        # Step 5: Request works again
        outputs = llm.generate("Step 5", SamplingParams(max_tokens=3))
        assert len(outputs) == 1

    def test_exception_message_is_user_friendly(self, llm_with_sleep_mode):
        """Test that the exception message is helpful for users."""
        llm = llm_with_sleep_mode
        
        llm.sleep(level=1)
        
        try:
            llm.generate("Test", SamplingParams(max_tokens=5))
            pytest.fail("Expected EngineSleepingError")
        except EngineSleepingError as e:
            error_message = str(e)
            
            # Verify the message contains useful info
            assert "sleep" in error_message.lower(), \
                f"Error message should mention 'sleep': {error_message}"
            assert "wake" in error_message.lower(), \
                f"Error message should mention how to wake: {error_message}"


class TestEngineSleepingErrorClass:
    """Test the EngineSleepingError exception class itself."""

    def test_exception_has_message(self):
        """Test that EngineSleepingError has a default message."""
        error = EngineSleepingError()
        
        assert str(error) != ""
        assert "sleep" in str(error).lower()

    def test_exception_is_recoverable(self):
        """Test that EngineSleepingError inherits from Exception (not BaseException)."""
        error = EngineSleepingError()
        
        # Should be catchable with a normal except Exception
        assert isinstance(error, Exception)
        
        # Should NOT be a BaseException subclass that bypasses except Exception
        # (like SystemExit or KeyboardInterrupt)
        assert not isinstance(error, SystemExit)
        assert not isinstance(error, KeyboardInterrupt)
