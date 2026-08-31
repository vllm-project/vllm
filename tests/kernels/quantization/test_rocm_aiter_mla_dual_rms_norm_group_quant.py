# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fused_mla_dual_rms_norm_group_quant (AITER fused_qk_rmsnorm_group_quant) vs the
un-fused pair: rocm_aiter_rmsnorm_fp8_group_quant (Triton) for q + RMSNorm for kv."""
import pytest
import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.platforms import current_platform


@pytest.mark.skipif(
    not current_platform.is_rocm() or not rocm_aiter_ops.is_enabled(),
    reason="ROCm AITER only",
)
@pytest.mark.parametrize("transpose_scale", [False, True])
def test_fused_mla_dual_rms_norm_group_quant_matches_unfused(transpose_scale):
    with set_current_vllm_config(VllmConfig()):
        torch.manual_seed(0)
        dev = "cuda"
        M, QD, KD, G = 64, 1536, 512, 128
        proj = torch.randn(M, QD + KD + 64, device=dev, dtype=torch.bfloat16) * 0.7
        q_c = proj[:, :QD].contiguous()
        kv_c = proj[:, QD : QD + KD].contiguous()
        qw = torch.rand(QD, device=dev, dtype=torch.bfloat16) + 0.5
        kw = torch.rand(KD, device=dev, dtype=torch.bfloat16) + 0.5
        eps = 1e-6

        quant_op = torch.ops.vllm.rocm_aiter_rmsnorm_fp8_group_quant
        op_has_transpose = any(
            a.name == "transpose_scale" for a in quant_op.default._schema.arguments
        )
        if transpose_scale and not op_has_transpose:
            pytest.skip(
                "rocm_aiter_rmsnorm_fp8_group_quant has no transpose_scale"
            )
        if op_has_transpose:
            q_ref, s_ref = quant_op(q_c, qw, eps, G, transpose_scale)
        else:
            q_ref, s_ref = quant_op(q_c, qw, eps, G)
        xf = kv_c.float()
        kv_ref = (
            xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        ).to(torch.bfloat16) * kw

        q_f, s_f, kv_f = torch.ops.vllm.fused_mla_dual_rms_norm_group_quant(
            q_c, qw, kv_c, kw, eps, eps, G, transpose_scale
        )
        torch.cuda.synchronize()

        # scales must agree exactly up to fp32 rounding
        assert ((s_f - s_ref).abs() / s_ref.abs().clamp(min=1e-8)).max() < 1e-5
        # q: HIP vs Triton may round differently by one fp8 ulp on a tiny fraction
        dq = (q_f.float() - q_ref.float()).abs()
        assert (dq > 0).float().mean() < 0.001
        # kv: bf16 rounding between fp32 accumulations
        dk = (kv_f.float() - kv_ref.float()).abs()
        assert dk.mean() < 0.005
