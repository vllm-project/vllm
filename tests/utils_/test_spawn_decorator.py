# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for spawn_new_process_for_each_test decorator."""

from pathlib import Path

import pytest

from tests.utils import spawn_new_process_for_each_test

pytest_plugins = ["pytester"]


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


def test_spawn_decorator_reports_skip(pytester):
    """A child skip must remain a skip in the parent pytest report."""
    result = pytester.runpytest_subprocess(
        f"{Path(__file__).resolve()}::test_spawn_decorator_skip", "-rs"
    )
    result.assert_outcomes(skipped=1)
    result.stdout.fnmatch_lines(["SKIPPED *intentional skip"])


def test_spawn_decorator_imports_octave_dispatch(pytester, monkeypatch):
    """The collected Octave test must reach its device check in a fresh process."""
    for variable in (
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
    ):
        monkeypatch.setenv(variable, "")

    test_file = (
        Path(__file__).resolve().parents[1] / "kernels/attention/test_rocm_octave.py"
    )
    items, recorder = pytester.inline_genitems(
        f"{test_file}::test_prefill_query_device_dispatch"
    )
    assert recorder.ret == pytest.ExitCode.OK
    (item,) = items
    # Call directly so module-level GPU marks cannot hide child import failures.
    with pytest.raises(
        pytest.skip.Exception,
        match="Requires an Octave-supported GPU and an unsupported GPU",
    ):
        item.obj()


@spawn_new_process_for_each_test
@pytest.mark.parametrize("x,y,expected", [(1, 2, 3), (0, 0, 0)])
def test_spawn_decorator_parametrized(x, y, expected):
    """Args and kwargs must be forwarded correctly to subprocess."""
    assert x + y == expected
