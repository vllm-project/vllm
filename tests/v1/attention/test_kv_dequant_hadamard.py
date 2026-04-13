# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.v1.attention.kv_dequant.hadamard import (
    _MAX_DENSE_FALLBACK_D,
    _get_hadamard_matrix,
    hadamard_transform,
)


def _normalized_hadamard_reference(d: int, device: torch.device) -> torch.Tensor:
    H = torch.ones((1, 1), dtype=torch.float32, device=device)
    while H.shape[0] < d:
        H = torch.cat(
            (
                torch.cat((H, H), dim=1),
                torch.cat((H, -H), dim=1),
            ),
            dim=0,
        )
    return H / math.sqrt(d)


def test_hadamard_transform_cpu_self_inverse():
    x = torch.randn(8, 4, 128, dtype=torch.float32, device="cpu")

    y = hadamard_transform(x)
    z = hadamard_transform(y)

    torch.testing.assert_close(z, x, atol=1e-4, rtol=1e-4)


def test_hadamard_transform_cpu_matches_normalized_reference():
    x = torch.randn(4, 128, dtype=torch.float32, device="cpu")
    H = _normalized_hadamard_reference(128, x.device)

    y = hadamard_transform(x)
    y_ref = x @ H

    torch.testing.assert_close(y, y_ref, atol=1e-5, rtol=1e-5)


def test_hadamard_transform_cpu_inplace():
    x = torch.randn(2, 64, dtype=torch.float32, device="cpu")
    x_before = x.clone()

    y = hadamard_transform(x, inplace=True)

    assert y.data_ptr() == x.data_ptr()
    assert not torch.equal(x, x_before)
    torch.testing.assert_close(hadamard_transform(x), x_before, atol=1e-4, rtol=1e-4)


def test_get_hadamard_matrix_rejects_large_d():
    with pytest.raises(ValueError, match=str(_MAX_DENSE_FALLBACK_D)):
        _get_hadamard_matrix(1 << 16, torch.float32, torch.device("cpu"))


@pytest.mark.skipif(
    not current_platform.is_cuda() or not hasattr(torch.ops._C, "hadacore_transform"),
    reason="CUDA hadacore path is unavailable in this environment.",
)
def test_hadamard_transform_cuda_handles_non_contiguous_input():
    x = torch.randn(4, 2, 128, dtype=torch.float16, device="cuda").transpose(0, 1)
    x_before = x.clone()

    y = hadamard_transform(x, inplace=False)

    ref = x.reshape(-1, 128).contiguous()
    ops.hadacore_transform(ref, inplace=True)
    ref = ref.view_as(x)

    torch.testing.assert_close(y, ref)
    torch.testing.assert_close(x, x_before)


@pytest.mark.skipif(
    not current_platform.is_cuda() or not hasattr(torch.ops._C, "hadacore_transform"),
    reason="CUDA hadacore path is unavailable in this environment.",
)
def test_hadamard_transform_dense_reference_matches_cuda_tier1():
    x = torch.randn(8, 128, dtype=torch.bfloat16, device="cuda")
    x_before = x.clone()

    y = hadamard_transform(x, inplace=False)
    H = _get_hadamard_matrix(128, torch.float32, x.device)
    y_ref = (x_before.float() @ H).to(x.dtype)

    torch.testing.assert_close(x, x_before)
    torch.testing.assert_close(y, y_ref, atol=2e-2, rtol=2e-2)
