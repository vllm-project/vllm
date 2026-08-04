# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the global transfer channel context lifecycle:
``initialize_transfer_channel_context``, ``get_transfer_channel_context`` and
``delete_transfer_channel_context``.

Tests are written against the documented contracts in
``lmcache/v1/distributed/transfer_channel/__init__.py`` and use only the
public interface.
"""

# Standard
from collections.abc import Iterator
import itertools

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.transfer_channel import (
    TransferChannelContext,
    delete_transfer_channel_context,
    get_transfer_channel_context,
    initialize_transfer_channel_context,
)
from lmcache.v1.distributed.transfer_channel.factory import (
    register_transfer_channel_factory,
)

_name_counter = itertools.count()


def _unique_type_name() -> str:
    return f"test_lifecycle_{next(_name_counter)}"


class _FakeContext(TransferChannelContext):
    """Context double that records whether ``close`` was called."""

    def __init__(self, **kwargs) -> None:
        self.created_kwargs = kwargs
        self.closed = False

    def get_transfer_channel_server(self):
        raise NotImplementedError

    def get_transfer_channel_client(self, peer_advertise_url: str):
        raise NotImplementedError

    def remove_transfer_channel_client(self, peer_advertise_url: str) -> None:
        raise NotImplementedError

    def get_transfer_channel_address(self, lmcache_addresses):
        raise NotImplementedError

    def get_num_connected_clients(self) -> int:
        return 0

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_type() -> Iterator[str]:
    """Register a fake factory and guarantee the global context is cleared."""
    name = _unique_type_name()
    register_transfer_channel_factory(name, lambda **kw: _FakeContext(**kw))
    try:
        yield name
    finally:
        delete_transfer_channel_context()


def _l1_desc() -> L1MemoryDesc:
    return L1MemoryDesc(ptr=0, size=4096, align_bytes=256)


def test_get_before_initialize_raises_runtime_error(fake_type):
    with pytest.raises(RuntimeError):
        get_transfer_channel_context()


def test_initialize_returns_context_and_get_returns_same(fake_type):
    ctx = initialize_transfer_channel_context(
        transfer_channel_type=fake_type,
        l1_memory_desc=_l1_desc(),
        listen_url="0.0.0.0:7600",
        advertise_url="host:7600",
    )
    assert isinstance(ctx, TransferChannelContext)
    assert get_transfer_channel_context() is ctx


def test_double_initialize_raises_runtime_error(fake_type):
    initialize_transfer_channel_context(
        transfer_channel_type=fake_type,
        l1_memory_desc=_l1_desc(),
        listen_url="0.0.0.0:7600",
        advertise_url="host:7600",
    )
    with pytest.raises(RuntimeError):
        initialize_transfer_channel_context(
            transfer_channel_type=fake_type,
            l1_memory_desc=_l1_desc(),
            listen_url="0.0.0.0:7601",
            advertise_url="host:7601",
        )


def test_delete_closes_context_and_get_raises_again(fake_type):
    ctx = initialize_transfer_channel_context(
        transfer_channel_type=fake_type,
        l1_memory_desc=_l1_desc(),
        listen_url="0.0.0.0:7600",
        advertise_url="host:7600",
    )
    delete_transfer_channel_context()
    assert ctx.closed is True
    with pytest.raises(RuntimeError):
        get_transfer_channel_context()


def test_delete_without_initialize_is_noop(fake_type):
    # Should not raise even when no context exists.
    delete_transfer_channel_context()


def test_reinitialize_after_delete_succeeds(fake_type):
    first = initialize_transfer_channel_context(
        transfer_channel_type=fake_type,
        l1_memory_desc=_l1_desc(),
        listen_url="0.0.0.0:7600",
        advertise_url="host:7600",
    )
    delete_transfer_channel_context()
    second = initialize_transfer_channel_context(
        transfer_channel_type=fake_type,
        l1_memory_desc=_l1_desc(),
        listen_url="0.0.0.0:7600",
        advertise_url="host:7600",
    )
    assert second is not first
    assert get_transfer_channel_context() is second
