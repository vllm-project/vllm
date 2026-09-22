# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Large-shape guard for the batch-invariant fused SiLU*up + FP8 quant kernel.

Inputs with ``numel >= 2**31`` used to overflow the int32 group count, so the
launch failed with ``invalid argument`` and the error surfaced only at the next
CUDA call. The kernel must accept such inputs and stay bitwise-identical to
processing the same rows in smaller chunks.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.utils import fp8_utils
from vllm.platforms import current_platform

GROUP_SIZE = 128


def _quant(x: torch.Tensor, use_ue8m0: bool) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.empty(
        x.shape[0], x.shape[1] // 2, device=x.device, dtype=torch.float8_e4m3fn
    )
    q, s = fp8_utils.fused_silu_mul_per_token_group_quant_fp8(
        x,
        output_q=q,
        use_ue8m0=use_ue8m0,
        round_scale=use_ue8m0,
        clamp_limit=7.0,
        masked_m=None,
        group_size=GROUP_SIZE,
    )
    torch.accelerator.synchronize()
    return q, s


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.skipif(
    not fp8_utils.is_batch_invariant_quant_kernel_enabled(),
    reason="batch-invariant kernel library not available",
)
@pytest.mark.parametrize("use_ue8m0", [True, False])
def test_fused_silu_quant_accepts_inputs_beyond_int32(use_ue8m0: bool):
    rows, width = 524288 + 128, 4096  # rows * width > 2**31 elements
    if torch.cuda.get_device_properties(0).total_memory < 24 * 1024**3:
        pytest.skip("needs >= 24 GiB of device memory for the >2**31-element input")
    gen = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16, generator=gen) * 3

    q_full, s_full = _quant(x, use_ue8m0)

    half = rows // 2
    q_a, s_a = _quant(x[:half], use_ue8m0)
    q_b, s_b = _quant(x[half:], use_ue8m0)
    q_ref = torch.cat([q_a, q_b], dim=0)
    s_ref = (
        torch.cat([s_a, s_b], dim=0)
        if s_a.shape[0] == half
        else torch.cat([s_a, s_b], dim=1)
    )

    assert torch.equal(q_full.view(torch.uint8), q_ref.view(torch.uint8))
    assert torch.equal(s_full.contiguous(), s_ref.contiguous())
