# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The column-major (M-major scale) variants of the AITER RMSNorm + group-quant
and SiLU-mul + group-quant fusions.

On gfx1250 QuantFP8.forward_hip asks aiter's group quant for M-major scales
(``rocm_aiter_group_fp8_quant(x, 128, True)``) and re-views them for the
Gluon preshuffled block-scale GEMM. These patterns must still fuse that graph,
and the fused op's scales must equal the unfused ones after the same re-view.
"""

import pytest
import torch

import vllm.config
from tests.compile.backend import TestBackend
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    ModelConfig,
    PassConfig,
    VllmConfig,
)
from vllm.platforms import current_platform

EPS = 1e-6
HIDDEN_SIZE = 512
GROUP_SIZE = 128


def _cdna_version() -> int:
    if not current_platform.is_rocm():
        return 0
    from vllm.platforms.rocm import get_cdna_version

    return get_cdna_version()


pytestmark = pytest.mark.skipif(
    _cdna_version() < 5, reason="M-major group-quant scales are emitted on CDNA >= 5"
)


def _m_major_view(scale: torch.Tensor) -> torch.Tensor:
    """QuantFP8.forward_hip's re-view of an M-major (M, G) scale."""
    m, groups = scale.shape
    return scale.view(groups, m).transpose(-2, -1)


class _RMSQuantModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(HIDDEN_SIZE, dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(x)
        rms = torch.ops.vllm_ir.rms_norm(x, self.weight, EPS)
        q, s = torch.ops.vllm.rocm_aiter_group_fp8_quant.default(rms, GROUP_SIZE, True)
        return q, _m_major_view(s)


class _FusedAddRMSQuantModel(_RMSQuantModel):
    def forward(  # type: ignore[override]
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.relu(x)
        rms, residual = torch.ops.vllm_ir.fused_add_rms_norm(
            x, residual, self.weight, EPS
        )
        q, s = torch.ops.vllm.rocm_aiter_group_fp8_quant.default(rms, GROUP_SIZE, True)
        return q, _m_major_view(s), residual


def _config() -> VllmConfig:
    return VllmConfig(
        model_config=ModelConfig(dtype=torch.bfloat16),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["+rms_norm", "+quant_fp8"],
            pass_config=PassConfig(fuse_norm_quant=True, eliminate_noops=True),
        ),
    )


@pytest.mark.parametrize("fused_add", [False, True])
@pytest.mark.parametrize("num_tokens", [7, 64])
def test_col_major_rms_group_quant_fusion(
    fused_add: bool, num_tokens: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch._dynamo.reset()
    vllm_config = _config()
    with vllm.config.set_current_vllm_config(vllm_config), monkeypatch.context() as m:
        from vllm.compilation.passes.fusion.rocm_aiter_fusion import (
            RocmAiterRMSNormQuantFusionPass,
        )

        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.bfloat16)
        torch.manual_seed(0)
        m.setenv("VLLM_ROCM_USE_AITER", "1")
        rocm_aiter_ops.refresh_env_variables()

        fusion_pass = RocmAiterRMSNormQuantFusionPass(vllm_config)
        backend = TestBackend(
            NoOpEliminationPass(vllm_config),
            fusion_pass,
            PostCleanupPass(vllm_config),
        )
        model = _FusedAddRMSQuantModel() if fused_add else _RMSQuantModel()
        x = torch.randn(num_tokens, HIDDEN_SIZE)
        inputs = (x, torch.randn(num_tokens, HIDDEN_SIZE)) if fused_add else (x,)
        for t in inputs:
            torch._dynamo.mark_dynamic(t, 0)

        unfused = model(*inputs)
        fused = torch.compile(model, backend=backend)(*inputs)

        assert fusion_pass.matched_count == 1
        fused_op = (
            rocm_aiter_ops.get_rmsnorm_group_add_fused_quant_op()
            if fused_add
            else rocm_aiter_ops.get_rmsnorm_group_fused_quant_op()
        )
        backend.check_after_ops([fused_op])
        (q, scale), (q_ref, scale_ref) = fused[:2], unfused[:2]
        # The scales reach the GEMM in the same M-major presentation.
        assert scale.shape == scale_ref.shape
        assert scale.stride() == scale_ref.stride()
        torch.testing.assert_close(scale, scale_ref, atol=1e-6, rtol=1e-2)
        # The fused kernel quantizes from the fp32 norm, the unfused path from
        # its bf16 output: FP8 values may differ by one e4m3 step on a few
        # elements, and no more.
        diff = (q.float() - q_ref.float()).abs()
        assert bool((diff <= q_ref.float().abs() * 0.125 + 1e-6).all())
        assert (diff > 0).float().mean().item() < 0.05
        if fused_add:
            torch.testing.assert_close(fused[2], unfused[2])
