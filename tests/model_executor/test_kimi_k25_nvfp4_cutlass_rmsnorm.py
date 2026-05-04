# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform

if not torch.cuda.is_available() or not current_platform.is_device_capability_family(
    100
):
    pytest.skip(
        "This test only runs on Blackwell GPUs (SM10x).", allow_module_level=True
    )

cutlass_torch = pytest.importorskip("cutlass.torch")
from_dlpack = pytest.importorskip("cutlass.cute.runtime").from_dlpack

from vllm.model_executor.specialized_models.kimi_k2_5_nvfp4.model import (  # noqa: E402
    kimik25_rmsnorm,
)


def _pytorch_rmsnorm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    x_fp32 = x.float()
    inv_rms = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + eps)
    # Match the kernel's cast order: normalize in fp32, cast to bf16, then scale.
    return (x_fp32 * inv_rms).to(torch.bfloat16) * weight


@pytest.mark.parametrize(
    ("num_tokens", "hidden_size", "k"),
    [
        (7, 512, 4),
        (64, 1536, 3),
    ],
)
def test_kimik25_cutlass_rmsnorm_matches_pytorch(
    num_tokens: int,
    hidden_size: int,
    k: int,
) -> None:
    torch.manual_seed(0)

    x = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)
    eps = 1e-5

    expected = _pytorch_rmsnorm_reference(x, weight, eps)

    actual = x.clone()
    kimik25_rmsnorm(
        from_dlpack(actual).mark_layout_dynamic(),
        from_dlpack(weight),
        hidden_size,
        eps,
        k,
        cutlass_torch.current_stream(),
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)
