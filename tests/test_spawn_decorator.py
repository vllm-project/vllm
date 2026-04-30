# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for spawn_new_process_for_each_test decorator."""

import pytest

from tests.utils import spawn_new_process_for_each_test


@spawn_new_process_for_each_test
def _passing_fn():
    assert 1 + 1 == 2


@spawn_new_process_for_each_test
def _failing_fn():
    raise ValueError("intentional failure")


@spawn_new_process_for_each_test
def _parametrized_fn(x, y, expected):
    assert x + y == expected


def test_spawn_decorator_passing():
    """Passing function should complete normally."""
    _passing_fn()


@pytest.mark.xfail(raises=RuntimeError, strict=True)
def test_spawn_decorator_failure_is_caught():
    """Failing function should raise RuntimeError, never silently pass."""
    _failing_fn()


def test_spawn_decorator_parametrized():
    """Args and kwargs must be forwarded correctly to subprocess."""
    _parametrized_fn(1, 2, expected=3)
