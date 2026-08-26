# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shutdown test utils"""

from vllm.platforms import current_platform

# XPU tears down engine core processes and reclaims device memory more slowly,
# so the default budget expires before wait_for_gpu_memory_to_clear finishes.
SHUTDOWN_TEST_TIMEOUT_SEC = 240 if current_platform.is_xpu() else 120
SHUTDOWN_TEST_THRESHOLD_BYTES = 2 * 2**30
