# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.collect_env import get_pip_packages


def test_get_pip_packages_handles_failed_command():
    """`get_pip_packages` must not crash when the list command fails.

    In a uv venv without `uv` on PATH, the command returns a nonzero code
    and `run_and_read_all` returns None. Guard against calling
    `.splitlines()` on None.
    """

    def failing_run_lambda(command):
        return 1, "", "command not found"

    _, out = get_pip_packages(failing_run_lambda)
    assert out is None
