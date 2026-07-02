# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import SamplingParams

@pytest.mark.skip_global_cleanup
def test_stop_token_ids_validation():
    # Valid stop_token_ids
    params = SamplingParams(stop_token_ids=[1, 2, 3])
    assert params.stop_token_ids == [1, 2, 3]

    # Invalid type
    with pytest.raises(TypeError, match="stop_token_ids must be a list"):
        SamplingParams(stop_token_ids="not_a_list")

    # Invalid element type
    with pytest.raises(ValueError, match="stop_token_ids must contain only integers"):
        SamplingParams(stop_token_ids=[1, "not_an_int", 3])

@pytest.mark.skip_global_cleanup
def test_stop_validation():
    # Valid stop strings
    params = SamplingParams(stop=["END", "STOP"])
    assert params.stop == ["END", "STOP"]

    # Invalid type
    with pytest.raises(TypeError, match="stop must be a list"):
        SamplingParams(stop=123)

    # Empty string in stop
    with pytest.raises(ValueError, match="stop cannot contain an empty string"):
        SamplingParams(stop=["END", ""])

@pytest.mark.skip_global_cleanup
def test_bad_words_validation():
    # Valid bad words
    params = SamplingParams(bad_words=["bad", "word"])
    assert params.bad_words == ["bad", "word"]

    # Invalid type
    with pytest.raises(TypeError, match="bad_words must be a list"):
        SamplingParams(bad_words={"bad": "word"})

    # Empty string in bad_words
    with pytest.raises(ValueError, match="bad_words cannot contain an empty string"):
        SamplingParams(bad_words=["bad", ""])

@pytest.mark.skip_global_cleanup
def test_update_from_generation_config():
    params = SamplingParams()
    params.stop_token_ids = None  # Force it to None to test the check
    with pytest.raises(ValueError, match="stop_token_ids must not be None"):
        params.update_from_generation_config({"eos_token_id": 45}, eos_token_id=50)


@pytest.mark.skip_global_cleanup
def test_validation_under_optimized_mode():
    import subprocess
    import sys
    code = """
from vllm import SamplingParams
try:
    SamplingParams(stop_token_ids="not_a_list")
    print("FAILED")
except TypeError:
    print("PASSED")
"""
    result = subprocess.run(
        [sys.executable, "-O", "-c", code],
        capture_output=True,
        text=True,
        check=True
    )
    assert result.stdout.strip().endswith("PASSED")

