# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the batched-weight RMS norm kernel (vllm._custom_ops.rms_norm).

``rms_norm`` can use the outermost input batch index to select the corresponding
weight row. The result must match that of looping ``rms_norm`` over that dimension.
"""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="rms_norm requires a CUDA/ROCm device",
)


@pytest.mark.parametrize(
    "shape",
    [
        (28, 17, 128),  # 3D: [num_rows, tokens, hidden]
        (1, 5, 2, 128),  # 4D: single row (edge case)
        (28, 13, 8, 128),  # 4D: [L, num_ctx, nkv, hd] (DFlash K-norm)
        (6, 3, 4, 769),  # 4D: non-power-of-two hidden size
    ],
)
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16, torch.float])
@pytest.mark.parametrize("seed", [42])
@torch.inference_mode()
def test_rms_norm_matches_loop(
    shape: tuple[int, ...], dtype: torch.dtype, seed: int
) -> None:
    set_random_seed(seed)
    torch.set_default_device("cuda")

    num_rows, hidden = shape[0], shape[-1]
    eps = 1e-6

    x = torch.randn(*shape, dtype=dtype) * 0.1
    # Distinct weight per row so that a wrong row index would be caught.
    weight = torch.randn(num_rows, hidden, dtype=dtype) * 0.1 + 1.0

    # Reference batched-weight rms norm.
    out_ref = torch.empty_like(x)
    for i in range(x.shape[0]):
        ops.rms_norm(out_ref[i], x[i], weight[i], eps)

    out = torch.empty_like(x)
    ops.rms_norm(out, x, weight, eps)

    # Expect bitwise-identical results.
    torch.testing.assert_close(out, out_ref, atol=0, rtol=0)


@torch.inference_mode()
def test_rms_norm_validates_shapes() -> None:
    torch.set_default_device("cuda")

    x = torch.randn(4, 8, 128, dtype=torch.float)
    out = torch.empty_like(x)
    # Expect num rows mismatch.
    with pytest.raises(RuntimeError):
        ops.rms_norm(out, x, torch.randn(3, 128), 1e-6)
    # Expect hidden size mismatch.
    with pytest.raises(RuntimeError):
        ops.rms_norm(out, x, torch.randn(4, 64), 1e-6)
