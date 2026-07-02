# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.utils.extensible_tensor import ExtensibleTensor

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_extensible_tensor_grows_without_moving() -> None:
    buffer = ExtensibleTensor(4096, device="cuda")
    try:
        base_ptr = buffer.base_ptr
        first_view = buffer.resize_(1024)
        assert first_view.data_ptr() == base_ptr
        first_view.fill_(7)

        second_view = buffer.resize_(2048)
        assert second_view.data_ptr() == base_ptr
        assert torch.equal(second_view[:1024], torch.full_like(second_view[:1024], 7))

        second_view[1024:].fill_(3)
        assert torch.equal(buffer.tensor, second_view)

        full_view = buffer.full_view()
        assert full_view.data_ptr() == base_ptr
        assert full_view.numel() == 4096
    finally:
        buffer.free()


def test_extensible_tensor_rejects_shrink_and_overflow() -> None:
    buffer = ExtensibleTensor(1024, device="cuda")
    try:
        buffer.resize_(512)
        with pytest.raises(ValueError, match="grow-only"):
            buffer.resize_(256)
        with pytest.raises(ValueError, match="exceeds maximum size"):
            buffer.resize_(1025)
    finally:
        buffer.free()
