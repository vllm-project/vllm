# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional

# First Party
from lmcache.v1.transfer_channel.abstract import BaseTransferChannel


# TODO(Jiayi): Refactor this function when we support more channels.
def CreateTransferChannel(
    channel_type: str,
    async_mode: bool,
    role: str,
    buffer_ptr: int,
    buffer_size: int,
    align_bytes: int,
    tp_rank: int,
    peer_init_url: str,
    device: Optional[str] = None,
    **kwargs,
) -> BaseTransferChannel:
    """
    Create a transfer channel based on the specified channel type.
    Supports "nixl" and "mock_memory" channel types.
    If nixl is not available, automatically falls back to mock_memory
    which is a mock implementation for testing purposes.

    :param channel_type: Type of the transfer channel (e.g., "nixl", "mock_memory").
    :param async_mode: Whether to operate in asynchronous mode.
    :param role: Role of the channel (e.g., "both", "sender" or "receiver").
    :param buffer_ptr: Pointer to the pre-allocated buffer.
    :param buffer_size: Size of the pre-allocated buffer in bytes.
    :param align_bytes: Alignment requirement in bytes.
    :param tp_rank: Tensor parallel rank of the current process.
    :param peer_init_url: Initialization URL for the peer.
    :kwargs: Additional keyword arguments specific to the channel type.

    :return: An instance of the specified transfer channel.
    """

    assert channel_type in ["nixl", "mock_memory"], (
        f"Unsupported channel type: {channel_type}"
    )

    if channel_type == "nixl":
        # First Party
        from lmcache.v1.transfer_channel.nixl_channel import NixlChannel

        assert "backends" in kwargs, (
            "`backends` must be provided to create nixl transfer channel."
        )
        transfer_channel = NixlChannel(
            async_mode=async_mode,
            role=role,
            buffer_ptr=buffer_ptr,
            buffer_size=buffer_size,
            align_bytes=align_bytes,
            tp_rank=tp_rank,
            peer_init_url=peer_init_url,
            device=device,
            **kwargs,
        )
        return transfer_channel

    if channel_type == "mock_memory":
        # First Party
        from lmcache.v1.transfer_channel.mock_memory_channel import (
            MockMemoryChannel,
        )

        mock_memory_channel: BaseTransferChannel = MockMemoryChannel(
            async_mode=async_mode,
            role=role,
            buffer_ptr=buffer_ptr,
            buffer_size=buffer_size,
            align_bytes=align_bytes,
            tp_rank=tp_rank,
            peer_init_url=peer_init_url,
            **kwargs,
        )
        return mock_memory_channel

    raise ValueError(f"Unsupported channel type: {channel_type}")
