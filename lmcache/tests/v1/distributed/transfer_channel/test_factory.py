# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the transfer channel factory registry.

Tests are written against the documented contracts of
``register_transfer_channel_factory`` and ``create_transfer_channel_context``
in ``lmcache/v1/distributed/transfer_channel/factory.py`` and use only the
public interface.
"""

# Standard
import itertools

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.transfer_channel import (
    TransferChannelContext,
)
from lmcache.v1.distributed.transfer_channel.factory import (
    create_transfer_channel_context,
    register_transfer_channel_factory,
)

# Unique-name generator so each test registers a fresh type name (the registry
# is process-global and rejects duplicate registrations).
_name_counter = itertools.count()


def _unique_type_name() -> str:
    return f"test_fake_{next(_name_counter)}"


class _FakeContext(TransferChannelContext):
    """Minimal context double recording the kwargs it was created with."""

    def __init__(self, **kwargs) -> None:
        self.created_kwargs = kwargs

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
        pass


def _l1_desc() -> L1MemoryDesc:
    return L1MemoryDesc(ptr=0, size=4096, align_bytes=256)


# =========================================================
# register_transfer_channel_factory
# =========================================================
def test_create_invokes_registered_factory_with_kwargs():
    name = _unique_type_name()
    register_transfer_channel_factory(name, lambda **kw: _FakeContext(**kw))

    desc = _l1_desc()
    ctx = create_transfer_channel_context(
        transfer_channel_type=name,
        l1_memory_desc=desc,
        listen_url="0.0.0.0:7600",
        advertise_url="host:7600",
        backends=["UCX"],
    )

    assert isinstance(ctx, _FakeContext)
    assert ctx.created_kwargs["l1_memory_desc"] is desc
    assert ctx.created_kwargs["listen_url"] == "0.0.0.0:7600"
    assert ctx.created_kwargs["advertise_url"] == "host:7600"
    # Implementation-specific kwargs are forwarded as-is.
    assert ctx.created_kwargs["backends"] == ["UCX"]


def test_register_duplicate_type_raises_value_error():
    name = _unique_type_name()
    register_transfer_channel_factory(name, lambda **kw: _FakeContext(**kw))
    with pytest.raises(ValueError):
        register_transfer_channel_factory(name, lambda **kw: _FakeContext(**kw))


# =========================================================
# create_transfer_channel_context
# =========================================================
def test_create_unregistered_type_raises_value_error():
    with pytest.raises(ValueError):
        create_transfer_channel_context(
            transfer_channel_type=_unique_type_name(),
            l1_memory_desc=_l1_desc(),
            listen_url="0.0.0.0:7600",
            advertise_url="host:7600",
        )
