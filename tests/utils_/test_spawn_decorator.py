# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for spawn_new_process_for_each_test decorator."""

import pytest

from tests.utils import spawn_new_process_for_each_test


@spawn_new_process_for_each_test
def test_spawn_decorator_passing():
    """Passing function should complete normally."""
    assert 1 + 1 == 2


@pytest.mark.xfail(raises=RuntimeError, strict=True)
@spawn_new_process_for_each_test
def test_spawn_decorator_failure_is_caught():
    """Failing function should raise RuntimeError, never silently pass."""
    raise ValueError("intentional failure")


@spawn_new_process_for_each_test
def test_spawn_decorator_skip():
    """pytest.skip inside subprocess should propagate correctly."""
    pytest.skip("intentional skip")


@spawn_new_process_for_each_test
@pytest.mark.parametrize("x,y,expected", [(1, 2, 3), (0, 0, 0)])
def test_spawn_decorator_parametrized(x, y, expected):
    """Args and kwargs must be forwarded correctly to subprocess."""
    assert x + y == expected
