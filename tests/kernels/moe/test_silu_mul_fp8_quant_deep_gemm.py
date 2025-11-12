# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe import (
    persistent_masked_m_silu_mul_quant,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv

fp8_dtype = torch.float8_e4m3fn

CASES = [
    (1, 1, 128, fp8_dtype),
    (1, 4, 128, fp8_dtype),
    (2, 4, 256, fp8_dtype),
    (32, 64, 256, fp8_dtype),
    (17, 31, 768, fp8_dtype),
    (1, 1, 128 * 1, fp8_dtype),
    (1, 1, 128 * 3, fp8_dtype),
    (1, 1, 128 * 4, fp8_dtype),
    (8, 16, 128 * 1, fp8_dtype),
    (8, 16, 128 * 2, fp8_dtype),
    (8, 16, 128 * 3, fp8_dtype),
    (8, 64, 7168, fp8_dtype),
    (8, 128, 128 * 33, fp8_dtype),
    (8, 128, 7168, fp8_dtype),
    (8, 512, 7168, fp8_dtype),
    (8, 1024, 7168, fp8_dtype),
    (256, 8, 7168, fp8_dtype),
    (256, 32, 7168, fp8_dtype),
    (256, 64, 7168, fp8_dtype),
    # Only add a few fnuz tests to help with long CI times.
    (8, 512, 7168, torch.float8_e4m3fnuz),
    (8, 1024, 7168, torch.float8_e4m3fnuz),
]


@pytest.mark.parametrize("E,T,H,fp8_type", CASES)
@torch.inference_mode()
def test_silu_mul_fp8_quant_deep_gemm(E, T, H, fp8_type):
    group_size = 128
    current_platform.seed_everything(42)

    # Input tensor of shape (E, T, 2*H)
    y = torch.randn((E, T, 2 * H), dtype=torch.bfloat16, device="cuda")
    tokens_per_expert = torch.randint(
        low=0,
        high=T,
        size=(E,),
        dtype=torch.int32,
        device="cuda",
    )

    # Run the SiLU V2 kernel
    # TODO (varun): use_e8m0 is set to false as the reference impl does
    # not handle that case.
    y_q, y_s = persistent_masked_m_silu_mul_quant(
        y, tokens_per_expert, group_size=group_size, use_ue8m0=False
    )

    torch.cuda.synchronize()
    fp8_info = torch.finfo(fp8_dtype)
    fp8_max = fp8_info.max
    fp8_min = fp8_info.min
    eps = 1e-10

    y1 = y[..., :H].float()
    y2 = y[..., H:]
    silu_x = y1 * torch.sigmoid(y1)
    merged = silu_x * y2

    for e in range(E):
        nt = tokens_per_expert[e].item()
        ref_s = torch.empty(
            (T, cdiv(H, group_size)), dtype=torch.float32, device="cuda"
        )
        ref_q = torch.empty((T, H), dtype=fp8_dtype, device="cuda")

        for t in range(nt):
            data = merged[e, t].float()
            ref_q_row = torch.empty_like(data)

            # process full groups
            n_full_groups = H // group_size
            if n_full_groups > 0:
                data_grp = data[: n_full_groups * group_size].view(
                    n_full_groups, group_size
                )
                amax = data_grp.abs().amax(dim=1).clamp(min=eps)
                scale = amax / fp8_max
                scaled = data[: n_full_groups * group_size] / scale.repeat_interleave(
                    group_size
                )
                ref_q_row[: n_full_groups * group_size] = scaled.clamp(
                    fp8_min, fp8_max
                ).to(fp8_dtype)
                ref_s[t, :n_full_groups] = scale

            # process remainder group
            rem = H % group_size
            if rem > 0:
                data_rem = data[-rem:]
                amax = data_rem.abs().amax().clamp(min=eps)
                scale = amax / fp8_max
                scaled = data_rem / scale
                ref_q_row[-rem:] = scaled.clamp(fp8_min, fp8_max).to(fp8_dtype)
                ref_s[t, -1] = scale

            ref_q[t] = ref_q_row

        y_se = y_s[e].float()
        y_qe = y_q[e].float()

        torch.testing.assert_close(
            y_qe[:nt].to(torch.float32),
            ref_q[:nt].to(torch.float32),
            atol=2,
            rtol=2e-1,
        )

        torch.testing.assert_close(y_se[:nt], ref_s[:nt], atol=1e-4, rtol=1e-2)
