# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for CUDA host-memory registration helpers."""

import mmap
from unittest.mock import MagicMock, call

import pytest

from vllm.v1.simple_kv_offload.cuda_mem_ops import pin_tensor


class _CudaResult:
    def __init__(self, value: int):
        self.value = value


def test_pin_tensor_splits_large_driver_registration(monkeypatch):
    """A large host tensor must not require one giant driver allocation."""
    page = mmap.PAGESIZE
    tensor = MagicMock()
    tensor.data_ptr.return_value = 0x100000
    tensor.nbytes = 5 * page
    cudart = MagicMock()
    cudart.cudaHostRegister.return_value = _CudaResult(0)
    monkeypatch.setattr("torch.cuda.cudart", lambda: cudart)

    addresses = pin_tensor(tensor, max_chunk_bytes=2 * page)

    assert addresses == [0x100000, 0x102000, 0x104000]
    assert cudart.cudaHostRegister.call_args_list == [
        call(0x100000, 2 * page, 0),
        call(0x102000, 2 * page, 0),
        call(0x104000, page, 0),
    ]


def test_pin_tensor_rolls_back_registered_chunks_on_failure(monkeypatch):
    """A later registration failure must not leak earlier pinned chunks."""
    page = mmap.PAGESIZE
    tensor = MagicMock()
    tensor.data_ptr.return_value = 0x200000
    tensor.nbytes = 3 * page
    cudart = MagicMock()
    cudart.cudaHostRegister.side_effect = [_CudaResult(0), _CudaResult(1)]
    cudart.cudaHostUnregister.return_value = _CudaResult(0)
    monkeypatch.setattr("torch.cuda.cudart", lambda: cudart)

    with pytest.raises(RuntimeError, match="cudaHostRegister failed"):
        pin_tensor(tensor, max_chunk_bytes=2 * page)

    cudart.cudaHostUnregister.assert_called_once_with(0x200000)
