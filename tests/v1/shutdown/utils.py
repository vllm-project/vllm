# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shutdown test utils"""

SHUTDOWN_TEST_TIMEOUT_SEC = 120
SHUTDOWN_TEST_THRESHOLD_BYTES = 2 * 2**30


def assert_mp_fork_context():
    """If we want a monkeypatch to apply to a child process, we need to ensure
    we are using the 'fork' method rather than 'spawn'. Without this assert,
    tests might fail unpredictably, and you would need to know to check for:

    > We must use the `spawn` multiprocessing start method. Overriding
    > VLLM_WORKER_MULTIPROC_METHOD to 'spawn'.
    > Reasons: CUDA is initialized
    """
    import vllm.envs as envs
    from vllm.utils.system_utils import get_mp_context

    _ = get_mp_context()
    assert envs.VLLM_WORKER_MULTIPROC_METHOD == "fork"
