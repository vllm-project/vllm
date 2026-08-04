# SPDX-License-Identifier: Apache-2.0
"""
Integration tests for the nixl-backed transfer channel implementation.

Tests are written against the public contracts documented in
``transfer_channel/abstract.py`` and ``transfer_channel/api.py`` (the
``TransferChannelContext`` / ``TransferChannelServer`` / ``TransferChannelClient``
interfaces). They use only public methods and do not access private fields.

These tests require a working nixl runtime (with a UCX backend) and are skipped
when nixl is unavailable.

Most tests share a single module-scoped context-pair to avoid paying the (slow)
nixl agent initialization for every test. The reads are self-verifying -- each
reads fresh remote data into its target region and checks that region -- so a
reused buffer and connection do not affect their outcome. The one test that must
observe a pristine connection state (the 0 -> 1 connection-count transition)
uses its own function-scoped pair instead, since there is no public way to reset
an existing context's connected clients short of closing it.
"""

# Standard
from unittest.mock import MagicMock
import itertools
import time

# Third Party
import pytest
import torch

nixl = pytest.importorskip("nixl")

# First Party
from lmcache.v1.distributed.internal_api import L1MemoryDesc  # noqa: E402
from lmcache.v1.distributed.transfer_channel.impl.nixl_impl import (  # noqa: E402
    NixlTransferChannelContext,
)

_ALIGN = 256
_BUF_SIZE = 4096

# Each context must bind to a distinct port so multiple pairs can coexist in one
# process without clashing.
_port_counter = itertools.count(17900)


def _next_url() -> str:
    return f"127.0.0.1:{next(_port_counter)}"


def _make_context_pair():
    """Create two contexts backed by two distinct CPU buffers.

    ``buf_b`` holds a known byte pattern so transfers can be content-verified;
    ``buf_a`` starts zeroed. Returns ``(ctx_a, buf_a, ctx_b, buf_b)``.
    """
    buf_a = torch.zeros(_BUF_SIZE, dtype=torch.uint8)
    buf_b = torch.arange(0, 256, dtype=torch.uint8).repeat(_BUF_SIZE // 256)
    desc_a = L1MemoryDesc(ptr=buf_a.data_ptr(), size=buf_a.numel(), align_bytes=_ALIGN)
    desc_b = L1MemoryDesc(ptr=buf_b.data_ptr(), size=buf_b.numel(), align_bytes=_ALIGN)

    url_a, url_b = _next_url(), _next_url()
    ctx_a = NixlTransferChannelContext(desc_a, listen_url=url_a, advertise_url=url_a)
    ctx_b = NixlTransferChannelContext(desc_b, listen_url=url_b, advertise_url=url_b)
    return ctx_a, buf_a, ctx_b, buf_b


@pytest.fixture(scope="module")
def shared_contexts():
    """A single context-pair reused across connection-state-insensitive tests."""
    ctx_a, buf_a, ctx_b, buf_b = _make_context_pair()
    try:
        yield ctx_a, buf_a, ctx_b, buf_b
    finally:
        ctx_a.close()
        ctx_b.close()


@pytest.fixture
def fresh_contexts():
    """A brand-new context-pair for tests that need pristine connection state."""
    ctx_a, buf_a, ctx_b, buf_b = _make_context_pair()
    try:
        yield ctx_a, buf_a, ctx_b, buf_b
    finally:
        ctx_a.close()
        ctx_b.close()


def _wait_finished(client, task_id, timeout_s=5.0):
    """Poll query_read_status until the task reports finished (or timeout)."""
    deadline = time.monotonic() + timeout_s
    result = client.query_read_status(task_id)
    while not result.is_finished() and time.monotonic() < deadline:
        time.sleep(0.01)
        result = client.query_read_status(task_id)
    return result


# =========================================================
# Address translation (get_transfer_channel_address)
# =========================================================
def test_get_address_returns_matching_offsets_and_sizes(shared_contexts):
    ctx_a, _, _, _ = shared_contexts
    addrs = ctx_a.get_transfer_channel_address([(0, 512), (1024, 256)])
    assert len(addrs) == 2
    assert (addrs[0].offset, addrs[0].size) == (0, 512)
    assert (addrs[1].offset, addrs[1].size) == (1024, 256)


def test_get_address_out_of_region_raises_value_error(shared_contexts):
    ctx_a, _, _, _ = shared_contexts
    with pytest.raises(ValueError):
        ctx_a.get_transfer_channel_address([(_BUF_SIZE, 256)])
    with pytest.raises(ValueError):
        ctx_a.get_transfer_channel_address([(_BUF_SIZE - 128, 256)])


# =========================================================
# End-to-end read (submit_read / query_read_status)
# =========================================================
def test_read_copies_remote_data_into_local_buffer(shared_contexts):
    ctx_a, buf_a, ctx_b, buf_b = shared_contexts
    client = ctx_a.get_transfer_channel_client(ctx_b.advertise_url)

    local = ctx_a.get_transfer_channel_address([(0, 512)])
    remote = ctx_b.get_transfer_channel_address([(0, 512)])
    task_id = client.submit_read(local, remote)

    result = _wait_finished(client, task_id)
    assert result.is_finished() is True
    assert result.succeeded_mask == [True] * len(remote)
    assert torch.equal(buf_a[:512], buf_b[:512])


def test_read_into_offset_region(shared_contexts):
    ctx_a, buf_a, ctx_b, buf_b = shared_contexts
    client = ctx_a.get_transfer_channel_client(ctx_b.advertise_url)

    # Read remote [0, 256) into local [1024, 1280).
    local = ctx_a.get_transfer_channel_address([(1024, 256)])
    remote = ctx_b.get_transfer_channel_address([(0, 256)])
    task_id = client.submit_read(local, remote)

    result = _wait_finished(client, task_id)
    assert result.is_finished() is True
    assert torch.equal(buf_a[1024:1280], buf_b[0:256])


def test_submit_read_mismatched_lengths_raises_value_error(shared_contexts):
    ctx_a, _, ctx_b, _ = shared_contexts
    client = ctx_a.get_transfer_channel_client(ctx_b.advertise_url)
    local = ctx_a.get_transfer_channel_address([(0, 256)])
    remote = ctx_b.get_transfer_channel_address([(0, 256), (256, 256)])
    with pytest.raises(ValueError):
        client.submit_read(local, remote)


def test_query_unknown_task_id_raises_key_error(shared_contexts):
    ctx_a, _, ctx_b, _ = shared_contexts
    client = ctx_a.get_transfer_channel_client(ctx_b.advertise_url)
    with pytest.raises(KeyError):
        client.query_read_status(99999)


# =========================================================
# Client / server management
# =========================================================
def test_get_client_is_idempotent(shared_contexts):
    ctx_a, _, ctx_b, _ = shared_contexts
    first = ctx_a.get_transfer_channel_client(ctx_b.advertise_url)
    second = ctx_a.get_transfer_channel_client(ctx_b.advertise_url)
    assert first is second


def test_register_client_keeps_first_and_closes_duplicate(fresh_contexts):
    """A racing second registration keeps the first client and closes the dup."""
    ctx_a, _, _, _ = fresh_contexts
    key = "10.255.255.1:65000"
    first = MagicMock()
    second = MagicMock()

    assert ctx_a.register_client(key, first) is first
    # The peer's inbound connection registers a duplicate for the same key.
    assert ctx_a.register_client(key, second) is first
    second.close.assert_called_once()
    first.close.assert_not_called()
    assert ctx_a.get_num_connected_clients() == 1


def test_remove_client_closes_and_drops_it(fresh_contexts):
    """Removing a peer closes its client and drops it from the context."""
    ctx_a, _, _, _ = fresh_contexts
    key = "10.255.255.1:65000"
    client = MagicMock()
    ctx_a.register_client(key, client)
    assert ctx_a.get_num_connected_clients() == 1

    ctx_a.remove_transfer_channel_client(key)

    client.close.assert_called_once()
    assert ctx_a.get_num_connected_clients() == 0


def test_remove_unknown_client_is_noop(fresh_contexts):
    """Removing a peer with no registered client does nothing and does not raise."""
    ctx_a, _, _, _ = fresh_contexts
    ctx_a.remove_transfer_channel_client("10.255.255.1:65000")
    assert ctx_a.get_num_connected_clients() == 0


def test_remove_client_after_connect(fresh_contexts):
    """A live client obtained via get_transfer_channel_client can be removed."""
    ctx_a, _, ctx_b, _ = fresh_contexts
    ctx_a.get_transfer_channel_client(ctx_b.advertise_url)
    assert ctx_a.get_num_connected_clients() == 1

    ctx_a.remove_transfer_channel_client(ctx_b.advertise_url)
    assert ctx_a.get_num_connected_clients() == 0


def test_connecting_registers_clients_on_both_sides(fresh_contexts):
    ctx_a, _, ctx_b, _ = fresh_contexts
    assert ctx_a.get_num_connected_clients() == 0
    assert ctx_b.get_num_connected_clients() == 0

    ctx_a.get_transfer_channel_client(ctx_b.advertise_url)

    # A actively dialed B; B passively learned a reverse client for A.
    assert ctx_a.get_num_connected_clients() == 1
    assert ctx_b.get_num_connected_clients() == 1


def test_server_is_available_from_context(shared_contexts):
    ctx_a, _, _, _ = shared_contexts
    assert ctx_a.get_transfer_channel_server() is not None
