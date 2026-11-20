# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only regression test for ExampleHiddenStatesConnector.shutdown().

Without a shutdown() override the connector inherits the no-op base
implementation, leaking the async-write ThreadPoolExecutor threads and any
open .lock fds (held with LOCK_EX). This test constructs the connector
without its GPU/config-dependent __init__ (via __new__) and checks that
shutdown() drains the executor and closes the lock fds.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (  # noqa: E501
    ExampleHiddenStatesConnector,
)


def test_shutdown_releases_executor_and_locks(tmp_path):
    # Build the connector without __init__ (which needs a VllmConfig + CUDA).
    conn = ExampleHiddenStatesConnector.__new__(ExampleHiddenStatesConnector)
    conn._executor = ThreadPoolExecutor(max_workers=1)
    lock_fd = os.open(str(tmp_path / "r0.lock"), os.O_CREAT | os.O_RDWR)
    conn._lock_fds = {"r0": lock_fd}

    conn.shutdown()

    # Executor is shut down: it rejects new work.
    with pytest.raises(RuntimeError):
        conn._executor.submit(lambda: None)

    # The lock fd was closed: fstat on it now fails.
    with pytest.raises(OSError):
        os.fstat(lock_fd)

    # Tracking dict is cleared.
    assert conn._lock_fds == {}
