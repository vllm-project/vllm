# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import torch

from vllm.distributed.ec_transfer.ec_connector.encoder_cache_transfer_buffer import (
    EncoderCacheTransferBuffer,
)


def test_transfer_buffer_allocates_exact_byte_capacity():
    real_empty = torch.empty

    def empty_without_pin_memory(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return real_empty(*args, **kwargs)

    with patch(
        "vllm.distributed.ec_transfer.ec_connector."
        "encoder_cache_transfer_buffer.torch.empty",
        side_effect=empty_without_pin_memory,
    ):
        buffer = EncoderCacheTransferBuffer(buffer_size=5, device="cpu")

    assert buffer.buffer_size == 5
    assert buffer.base_tensor.dtype == torch.uint8
    assert buffer.base_tensor.numel() == 5

    addr = buffer.allocate(5)
    assert addr == buffer.base_address
    assert buffer.free_size == 0
