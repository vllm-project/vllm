# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest
import torch

from vllm.distributed.ec_transfer.ec_connector.encoder_cache_transfer_buffer import (
    EncoderCacheTransferBuffer,
)


def make_cpu_buffer(buffer_size: int) -> EncoderCacheTransferBuffer:
    real_empty = torch.empty

    def empty_without_pin_memory(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return real_empty(*args, **kwargs)

    with patch(
        "vllm.distributed.ec_transfer.ec_connector."
        "encoder_cache_transfer_buffer.torch.empty",
        side_effect=empty_without_pin_memory,
    ):
        return EncoderCacheTransferBuffer(buffer_size=buffer_size, device="cpu")


def test_transfer_buffer_allocates_exact_byte_capacity():
    buffer = make_cpu_buffer(5)

    assert buffer.buffer_size == 5
    assert buffer.base_tensor.dtype == torch.uint8
    assert buffer.base_tensor.numel() == 5

    addr = buffer.allocate(5)
    assert addr == buffer.base_address
    assert buffer.free_size == 0


def test_pinned_slot_is_not_evicted():
    buffer = make_cpu_buffer(8)

    pinned_addr = buffer.allocate(4)
    buffer.allocate(4)
    buffer.pin(pinned_addr)

    with pytest.raises(BufferError, match="No unpinned"):
        buffer.allocate(5)

    assert buffer.is_pinned(pinned_addr)


def test_allocation_skips_pinned_slot_and_reuses_unpinned_space():
    buffer = make_cpu_buffer(12)

    pinned_addr = buffer.allocate(4)
    evictable_addr = buffer.allocate(4)
    buffer.allocate(4)
    buffer.pin(pinned_addr)

    addr = buffer.allocate(4)

    assert addr == evictable_addr
    assert buffer.is_pinned(pinned_addr)


def test_free_rejects_pinned_slot_until_unpinned():
    buffer = make_cpu_buffer(8)
    addr = buffer.allocate(4)

    buffer.pin(addr)
    with pytest.raises(ValueError, match="pinned"):
        buffer.free(addr)

    buffer.unpin(addr)
    buffer.free(addr)

    assert buffer.num_allocated == 0


def test_unpin_requires_active_pin():
    buffer = make_cpu_buffer(8)
    addr = buffer.allocate(4)

    with pytest.raises(ValueError, match="not pinned"):
        buffer.unpin(addr)
