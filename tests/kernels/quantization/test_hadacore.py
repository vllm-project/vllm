# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch
from compressed_tensors.transform import deterministic_hadamard_matrix

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

if current_platform.is_rocm():
    pytest.skip(
        "These tests require hadacore_transform, not supported on ROCm.",
        allow_module_level=True,
    )


@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("hidden_dim", [2**n for n in range(10)])
def test_hadacore(batch_size, hidden_dim, dtype=torch.bfloat16, device="cuda"):
    # Rank 3 covers the reshape round trip the INT4 KV cache caller relies on.
    x = torch.eye(hidden_dim, dtype=dtype, device=device).repeat(batch_size, 1, 1)
    hadamard = deterministic_hadamard_matrix(
        hidden_dim, dtype=torch.float64, device="cuda"
    ) / math.sqrt(hidden_dim)

    y = ops.hadacore_transform(x.clone())
    y_true = (x.to(hadamard.dtype) @ hadamard.T).to(y.dtype)
    assert torch.allclose(y, y_true)

    y = ops.hadacore_transform(y)
    assert torch.allclose(y, x)


# Only an odd 256-element chunk count reaches the partial-chunk warp.  One
# case per hadamard size, since each is a separate kernel instantiation.
@pytest.mark.parametrize(
    "num_rows,hidden_dim",
    [
        (42, 64),  # 2688 -> padded 2816 -> 11 chunks, odd
        (24, 32),  # 768 -> 3 chunks, odd
        (6, 128),  # 768 -> 3 chunks, odd
        (3, 256),  # 768 -> 3 chunks, odd
        (8, 64),  # 512 -> 2 chunks, even (control)
    ],
)
def test_hadacore_partial_chunk(
    num_rows, hidden_dim, dtype=torch.bfloat16, device="cuda"
):
    """Row counts that leave a warp holding a partial chunk must still transform.

    hadacore splits work into 256-element chunks, 2 chunks per warp.  An odd
    chunk count leaves the last warp with one real chunk, which takes a
    separate masked code path.
    """
    x = torch.eye(hidden_dim, dtype=dtype, device=device)[:num_rows]
    hadamard = deterministic_hadamard_matrix(
        hidden_dim, dtype=torch.float64, device=device
    ) / math.sqrt(hidden_dim)

    y = ops.hadacore_transform(x.clone())
    y_true = (x.to(hadamard.dtype) @ hadamard.T).to(y.dtype)
    assert torch.allclose(y, y_true)

    y = ops.hadacore_transform(y)
    assert torch.allclose(y, x)
