# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test that MessageQueue uses the local node's IP for binding,
not a remote master_addr. This validates the fix for cross-node
data-parallel where each DP group leader must bind to its own IP.

The bug: multiproc_executor used `parallel_config.master_addr` as
`connect_ip` for every DP group's MessageQueue. For DP groups whose
leader is NOT on the master node, binding to master_addr fails with
"Cannot assign requested address".

The fix: use `get_ip()` (local node IP) instead of `master_addr`.
"""

import pytest
import zmq

from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.utils.network_utils import get_ip


def test_mq_bind_with_local_ip():
    """MessageQueue with remote readers should successfully bind
    when connect_ip is the local node's IP."""
    # n_reader=2, n_local_reader=1 means 1 remote reader,
    # which triggers the remote ZMQ socket bind.
    mq = MessageQueue(
        n_reader=2,
        n_local_reader=1,
        connect_ip=get_ip(),
    )
    handle = mq.export_handle()
    assert handle.remote_subscribe_addr is not None
    # The bound address should contain our local IP
    local_ip = get_ip()
    assert (
        local_ip in handle.remote_subscribe_addr
        or f"[{local_ip}]" in handle.remote_subscribe_addr
    )
    del mq


def test_mq_bind_with_non_local_ip_fails():
    """MessageQueue should fail to bind when connect_ip is a
    non-local IP address (simulating the bug where master_addr
    from a different node was used)."""
    # Use a non-local IP that we definitely can't bind to.
    # 198.51.100.1 is from TEST-NET-2 (RFC 5737), never locally assigned.
    non_local_ip = "198.51.100.1"
    with pytest.raises(zmq.error.ZMQError, match="Cannot assign requested address"):
        MessageQueue(
            n_reader=2,
            n_local_reader=1,
            connect_ip=non_local_ip,
        )


def test_mq_bind_defaults_to_local_ip():
    """When connect_ip is None, MessageQueue should auto-detect
    the local IP and bind successfully."""
    mq = MessageQueue(
        n_reader=2,
        n_local_reader=1,
        connect_ip=None,  # should fallback to get_ip()
    )
    handle = mq.export_handle()
    assert handle.remote_subscribe_addr is not None
    del mq


if __name__ == "__main__":
    test_mq_bind_with_local_ip()
    print("PASSED: test_mq_bind_with_local_ip")
    test_mq_bind_with_non_local_ip_fails()
    print("PASSED: test_mq_bind_with_non_local_ip_fails")
    test_mq_bind_defaults_to_local_ip()
    print("PASSED: test_mq_bind_defaults_to_local_ip")
    print("\nAll tests passed!")
