# SPDX-License-Identifier: Apache-2.0
"""Factory registry for transfer channel contexts.

Each implementation under ``impl/`` self-registers a factory callable via
``register_transfer_channel_factory`` at import time.
``create_transfer_channel_context`` looks the factory up by transfer-channel
type name and invokes it.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Callable

# First Party
from lmcache.logging import init_logger

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.internal_api import L1MemoryDesc
    from lmcache.v1.distributed.transfer_channel.abstract import (
        TransferChannelContext,
    )

logger = init_logger(__name__)

# A factory creates a ``TransferChannelContext``. It is always called with the
# keyword arguments ``l1_memory_desc``, ``listen_url`` and ``advertise_url``,
# plus any implementation-specific keyword arguments forwarded by the caller.
TransferChannelFactory = Callable[..., "TransferChannelContext"]

# Registry: transfer-channel type name -> factory callable.
_TRANSFER_CHANNEL_FACTORY_REGISTRY: dict[str, TransferChannelFactory] = {}


def register_transfer_channel_factory(
    transfer_channel_type: str,
    factory: TransferChannelFactory,
) -> None:
    """Register a factory that creates a context for ``transfer_channel_type``.

    Each implementation module should call this at import time.

    Args:
        transfer_channel_type: The type name to register the factory under
            (e.g. ``"nixl"``).
        factory: A callable invoked with the keyword arguments
            ``l1_memory_desc``, ``listen_url``, ``advertise_url`` and any
            implementation-specific keyword arguments, returning a
            ``TransferChannelContext``.

    Raises:
        ValueError: If a factory is already registered for this type name.
    """
    if transfer_channel_type in _TRANSFER_CHANNEL_FACTORY_REGISTRY:
        raise ValueError(
            f"Transfer channel factory already registered: {transfer_channel_type!r}"
        )
    _TRANSFER_CHANNEL_FACTORY_REGISTRY[transfer_channel_type] = factory


def create_transfer_channel_context(
    transfer_channel_type: str,
    l1_memory_desc: "L1MemoryDesc",
    listen_url: str,
    advertise_url: str,
    **kwargs,
) -> "TransferChannelContext":
    """Create a transfer channel context using the registered factory.

    Args:
        transfer_channel_type: The type name of the implementation to create
            (e.g. ``"nixl"``).
        l1_memory_desc: Describes the L1 memory region to register.
        listen_url: ``host:port`` this peer's server binds to.
        advertise_url: ``host:port`` this peer advertises as its identity.
        **kwargs: Implementation-specific keyword arguments forwarded to the
            factory.

    Returns:
        A new ``TransferChannelContext`` instance.

    Raises:
        ValueError: If no factory is registered for ``transfer_channel_type``.
    """
    factory = _TRANSFER_CHANNEL_FACTORY_REGISTRY.get(transfer_channel_type)
    if factory is None:
        known = sorted(_TRANSFER_CHANNEL_FACTORY_REGISTRY)
        raise ValueError(
            f"Unsupported transfer_channel_type: {transfer_channel_type!r}. "
            f"Registered types: {known}"
        )

    return factory(
        l1_memory_desc=l1_memory_desc,
        listen_url=listen_url,
        advertise_url=advertise_url,
        **kwargs,
    )
