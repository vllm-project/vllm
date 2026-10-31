# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shutdown test utils."""

from vllm.platforms import current_platform

# XPU forces the `spawn` multiprocessing start method (see
# `vllm.platforms.xpu`), which re-imports modules and re-initializes the
# device runtime in each child process instead of inheriting them via
# `fork`'s copy-on-write memory. That makes EngineCore/Worker subprocess
# startup noticeably slower and more variable, so give it extra headroom to
# avoid flaky timeouts.
SHUTDOWN_TEST_TIMEOUT_SEC = 240 if current_platform.is_xpu() else 120
SHUTDOWN_TEST_THRESHOLD_BYTES = 2 * 2**30
