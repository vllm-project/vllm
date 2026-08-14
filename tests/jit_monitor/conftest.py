# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.utils import jit_monitor


@pytest.fixture(autouse=True)
def _reset_monitor():
    """Reset global monitor state between tests.

    ``activate()`` installs process-global hooks and flips module globals, so
    without this every test would inherit the previous test's monitor state.
    """

    def reset():
        jit_monitor._active = False
        jit_monitor._mode = "warn"
        jit_monitor._verbose = False
        jit_monitor._cutedsl_hook_installed = False
        jit_monitor._tilelang_hook_installed = False
        jit_monitor._tilelang_jitimpl_compile_depth = 0

    reset()
    yield
    reset()
