# SPDX-License-Identifier: Apache-2.0
"""Tests for the SamplingParams class.
"""
from vllm import SamplingParams


def test_max_tokens_none():
    """max_tokens=None should be allowed"""
    SamplingParams(temperature=0.01, top_p=0.1, max_tokens=None)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
