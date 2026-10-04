# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for the MXFP8 patterns registered by
``RocmAiterRMSNormQuantFusionPass``.

The MXFP8 linear path (``_mxfp8_dot_scaled_linear``) quantizes its activation
in a separate launch immediately before the ``tl.dot_scaled`` GEMM. AITER's
fused RMSNorm can emit the same (FP8 E4M3, per-32 E8M0 scale) pair itself, so
these patterns fold the quant into the norm:

* ``AiterRMSNormMxfp8QuantPattern`` -- ``rms_norm -> mxfp8_quantize``
* ``AiterFusedAddRMSNormMxfp8QuantPattern`` -- the residual-add shape, which is
  the one that dominates the DeepSeek-V4.1 decode step
* ``AiterFusedAddRMSNormMxfp8QuantViewPattern`` -- view-tolerant sibling, for
  the 2D flatten that ``Mxfp8LinearKernel.apply_weights`` inserts

Requires CDNA4 (gfx950) for native ``dot_scaled`` MX support, and an aiter
build containing https://github.com/ROCm/aiter/pull/5480 -- without it,
``add_rmsnorm_quant`` aborts for a group size of 32 at hidden sizes in
(4096, 6144].
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

EPS = 1e-5
# DeepSeek-V4.1-Flash's hidden size, and the reason the aiter fix is needed:
# it lands in the (4096, 6144] dispatch band.
HIDDEN_SIZE = 5120


class _RMSNormMxfp8Model(torch.nn.Module):
    """``rms_norm -> mxfp8_quantize``."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(HIDDEN_SIZE, dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # keep the graph input off a matched pattern node
        x = torch.relu(x)
        rms = torch.ops.vllm_ir.rms_norm(x, self.weight, EPS)
        return torch.ops.vllm.mxfp8_quantize.default(rms, False, 0)


class _FusedAddRMSNormMxfp8Model(torch.nn.Module):
    """``fused_add_rms_norm -> mxfp8_quantize``."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(HIDDEN_SIZE, dtype=torch.bfloat16))

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.relu(x)
        rms, residual_out = torch.ops.vllm_ir.fused_add_rms_norm(
            x, residual, self.weight, EPS
        )
        q, s = torch.ops.vllm.mxfp8_quantize.default(rms, False, 0)
        return q, residual_out, s


class _FusedAddRMSNormMxfp8ViewModel(_FusedAddRMSNormMxfp8Model):
    """The graph the MXFP8 linear actually produces.

    ``RocmDotScaledMxfp8LinearKernel.apply_weights`` flattens to 2D
    (``x.reshape(-1, x.shape[-1])``) before quantizing. Hidden states are
    already 2D, so that reshape is a no-op at runtime, but with a dynamic token
    dimension it is not provably one and survives into the graph the fusion
    pass sees.
    """

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.relu(x)
        rms, residual_out = torch.ops.vllm_ir.fused_add_rms_norm(
            x, residual, self.weight, EPS
        )
        rms_2d = rms.reshape(-1, rms.shape[-1])
        q, s = torch.ops.vllm.mxfp8_quantize.default(rms_2d, False, 0)
        return q, residual_out, s


@pytest.mark.skipif(
    not current_platform.is_rocm() or not current_platform.supports_mx(),
    reason="fused MXFP8 norm quant requires CDNA4 (gfx95x)",
)
@pytest.mark.parametrize(
    "model_cls, n_inputs, expect_op",
    [
        (_RMSNormMxfp8Model, 1, "get_rmsnorm_mxfp8_quant_op"),
        (_FusedAddRMSNormMxfp8Model, 2, "get_rmsnorm_with_add_mxfp8_quant_op"),
        (_FusedAddRMSNormMxfp8ViewModel, 2, "get_rmsnorm_with_add_mxfp8_quant_op"),
    ],
    ids=["rms_norm", "fused_add_rms_norm", "fused_add_rms_norm_view"],
)
def test_aiter_rms_mxfp8_quant_fusion(
    model_cls: type[torch.nn.Module],
    n_inputs: int,
    expect_op: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The norm + MXFP8 quant pair must collapse into one AITER op, and the
    fused result must agree with the unfused pair."""
    torch._dynamo.reset()

    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=torch.bfloat16),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["+rms_norm"],
            pass_config=PassConfig(
                fuse_norm_quant=True,
                eliminate_noops=True,
            ),
        ),
    )

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
        model = model_cls()

        args = [torch.randn(8, HIDDEN_SIZE) * 0.5 for _ in range(n_inputs)]
        torch._dynamo.mark_dynamic(args[0], 0)

        # both norms write their residual in place, so each call needs its own
        outputs_unfused = model(*[a.clone() for a in args])
        outputs_fused = torch.compile(model, backend=backend)(
            *[a.clone() for a in args]
        )

        assert fusion_pass.matched_count == 1, (
            f"Expected {model_cls.__name__} to fuse into the AITER MXFP8 norm "
            f"op (matched_count == 1), got {fusion_pass.matched_count}"
        )
        backend.check_after_ops([getattr(rocm_aiter_ops, expect_op)()])
        # the standalone quant launch must be gone, not merely reduced
        backend.check_before_ops([torch.ops.vllm.mxfp8_quantize.default])

        # AITER and vLLM break ties differently when rounding a block amax to
        # its E8M0 scale, so a few blocks out of a thousand land one power of
        # two apart and no elementwise tolerance holds. What has to be true is
        # that the fused kernel is no less faithful than the quant it replaces,
        # so compare both against an fp32 reference of the same math.
        ref = torch.relu(args[0]).float()
        if n_inputs == 2:
            ref = ref + args[1].float()
        ref = (
            ref
            * torch.rsqrt(ref.pow(2).mean(-1, keepdim=True) + EPS)
            * model.weight.float()
        )

        def rel_l1(q: torch.Tensor, s: torch.Tensor) -> float:
            scales = torch.exp2(s.float() - 127.0)[:, :, None]
            deq = (q.float().view(q.shape[0], -1, 32) * scales).view(q.shape)
            return ((deq - ref).abs().sum() / ref.abs().sum()).item()

        err_fused = rel_l1(outputs_fused[0], outputs_fused[-1])
        err_unfused = rel_l1(outputs_unfused[0], outputs_unfused[-1])

        # E4M3 with a shared per-32 exponent lands near 2%; the absolute bound
        # keeps a degenerate output (e.g. all-zero scales) from passing the
        # relative check below.
        assert err_fused < 0.05, f"fused MXFP8 error too large: {err_fused}"
        assert err_fused <= err_unfused * 1.05, (
            f"fused MXFP8 quant is less accurate than the unfused pair: "
            f"{err_fused} vs {err_unfused}"
        )

        if n_inputs == 2:
            torch.testing.assert_close(outputs_fused[1], outputs_unfused[1])
