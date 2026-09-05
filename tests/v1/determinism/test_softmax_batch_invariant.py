# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test the batch-invariant softmax override against torch.softmax.

`softmax_batch_invariant` is registered for `aten::_softmax(self, dim,
half_to_float)`, so it must honor `half_to_float` and a row's result must not
depend on how many other rows share the launch.
"""

import pytest
import torch
from utils import skip_unsupported

from vllm.model_executor.determinism.batch_invariant import (
    softmax_batch_invariant,
)
from vllm.platforms import current_platform

DEVICE_TYPE = current_platform.device_type


@skip_unsupported
@pytest.mark.parametrize("vocab_size", [1000, 4096, 151936])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_softmax_matches_torch(vocab_size: int, dtype: torch.dtype):
    torch.manual_seed(0)
    x = torch.randn(8, vocab_size, dtype=dtype, device=DEVICE_TYPE) * 4

    out = softmax_batch_invariant(x, -1, False)

    assert out.dtype == dtype
    torch.testing.assert_close(out, torch.softmax(x, dim=-1))


@skip_unsupported
def test_softmax_half_to_float():
    """`half_to_float=True` (fp16 input, `dtype=torch.float32`) returns fp32."""
    torch.manual_seed(0)
    x = torch.randn(8, 4096, dtype=torch.float16, device=DEVICE_TYPE) * 4

    out = softmax_batch_invariant(x, -1, True)

    assert out.dtype == torch.float32
    torch.testing.assert_close(out, torch.softmax(x, dim=-1, dtype=torch.float32))


@skip_unsupported
@pytest.mark.parametrize("vocab_size", [4097, 151936])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_softmax_batch_invariant(vocab_size: int, dtype: torch.dtype):
    """A row is bitwise identical whichever batch it is computed in."""
    torch.manual_seed(0)
    x = torch.randn(64, vocab_size, dtype=dtype, device=DEVICE_TYPE) * 4

    full = softmax_batch_invariant(x, -1, False)
    for batch_size in (1, 3, 32):
        part = softmax_batch_invariant(x[:batch_size], -1, False)
        assert torch.equal(part, full[:batch_size]), f"batch_size={batch_size}"


@skip_unsupported
@pytest.mark.parametrize("dim", [0, 1, -2])
def test_softmax_non_last_dim(dim: int):
    torch.manual_seed(0)
    x = torch.randn(4, 8, 16, dtype=torch.float32, device=DEVICE_TYPE)

    out = softmax_batch_invariant(x, dim, False)

    assert out.is_contiguous()
    torch.testing.assert_close(out, torch.softmax(x, dim=dim))


@skip_unsupported
@pytest.mark.parametrize("shape", [(0, 16), (3, 0)])
def test_softmax_empty(shape: tuple[int, ...]):
    x = torch.empty(shape, dtype=torch.float32, device=DEVICE_TYPE)

    out = softmax_batch_invariant(x, -1, False)

    assert out.shape == x.shape and out.dtype == x.dtype
