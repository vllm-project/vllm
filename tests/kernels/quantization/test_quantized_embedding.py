# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Triton dequant-gather kernel used by
``CompressedTensorsEmbeddingWNA16Int`` (quantized embedding lookup)."""

import pytest
import torch
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_embedding import (  # noqa: E501
    _dequant_gather_triton,
)
from vllm.platforms import current_platform


def _dequant_gather_torch(
    ids: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    hidden: int,
    num_bits: int,
) -> torch.Tensor:
    """Reference: gather packed rows by id, unpack int32-packed INT, dequant."""
    n = ids.shape[0]
    int8 = unpack_from_int32(weight_packed[ids], num_bits, torch.Size([n, hidden]))
    scale_rows = weight_scale[ids]
    w = int8.to(scale_rows.dtype)
    if scale_rows.shape[1] == 1:
        return w * scale_rows
    ng = scale_rows.shape[1]
    return (w.view(n, ng, hidden // ng) * scale_rows.unsqueeze(-1)).view(n, hidden)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Triton dequant kernel requires CUDA"
)
@pytest.mark.parametrize("num_bits", [2, 4, 8])
@pytest.mark.parametrize("group_size", [0, 256])  # 0 -> channel
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_ids", [1, 17, 4096])
def test_dequant_gather(num_bits, group_size, dtype, num_ids):
    torch.manual_seed(0)
    device = "cuda"
    vocab, hidden = 1000, 2048
    pack_factor = 32 // num_bits

    # Random full-range int32 packed weights (covers the sign bit -> exercises the
    # arithmetic-shift + mask unpack path).
    weight_packed = torch.randint(
        -(2**31),
        2**31,
        (vocab, hidden // pack_factor),
        dtype=torch.int32,
        device=device,
    )

    num_groups = 1 if group_size == 0 else hidden // group_size
    weight_scale = torch.rand(vocab, num_groups, dtype=dtype, device=device) + 0.01

    ids = torch.randint(0, vocab, (num_ids,), dtype=torch.long, device=device)

    out = _dequant_gather_triton(ids, weight_packed, weight_scale, hidden, num_bits)
    ref = _dequant_gather_torch(ids, weight_packed, weight_scale, hidden, num_bits)

    assert out.shape == (num_ids, hidden)
    assert out.dtype == dtype
    torch.testing.assert_close(out, ref)
