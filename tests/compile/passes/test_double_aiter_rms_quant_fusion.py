# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for the DoubleQuant fan-out variants registered by
``RocmAiterRMSNormQuantFusionPass``.

Both variants target a 1-to-2 fan-out where one ``rms_norm`` output feeds
two distinct ``rocm_aiter_group_fp8_quant`` consumers and rewrite it into
two independent fused ``rms_norm + group_fp8_quant`` ops:

* ``DoubleAiterRMSFp8GroupQuantPattern`` matches the un-viewed shape
  (e.g. Kimi-K2.5 / DSR1).
* ``DoubleAiterRMSFp8GroupQuantViewPattern`` (this PR) is the view-tolerant
  sibling that additionally matches the
  ``rms_norm -> view -> group_fp8_quant`` shape that DSv3.2's MLA indexer
  q_c norm exposes through ``Fp8BlockScaledMMLinearKernel.apply_weights``'s
  2D-flatten boilerplate.
"""

import pytest
import torch

import vllm.config
from tests.compile.backend import TestBackend
from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    ModelConfig,
    PassConfig,
    VllmConfig,
)

EPS = 1e-5
HIDDEN_SIZE = 256
GROUP_SIZE = 128


class _NoViewDoubleQuantModel(torch.nn.Module):
    """``rms_norm -> 2x group_fp8_quant`` fan-out (Kimi-K2.5 / DSR1 shape)."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(HIDDEN_SIZE, dtype=torch.bfloat16))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # avoid graph input being a direct arg to a matched pattern node
        x = torch.relu(x)
        rms = torch.ops.vllm_ir.rms_norm(x, self.weight, EPS)
        q1, s1 = torch.ops.vllm.rocm_aiter_group_fp8_quant.default(rms, GROUP_SIZE)
        q2, s2 = torch.ops.vllm.rocm_aiter_group_fp8_quant.default(rms, GROUP_SIZE)
        return q1, s1, q2, s2


class _ViewDoubleQuantModel(torch.nn.Module):
    """``rms_norm -> view -> 2x group_fp8_quant`` fan-out (DSv3.2 shape).

    Reproduces the FX-graph shape produced by ``Fp8BlockScaledMMLinearKernel``'s
    2D-flatten before the FP8 group quant op.
    """

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(HIDDEN_SIZE, dtype=torch.bfloat16))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.relu(x)
        rms = torch.ops.vllm_ir.rms_norm(x, self.weight, EPS)
        view = rms.view(-1, rms.shape[-1])
        q1, s1 = torch.ops.vllm.rocm_aiter_group_fp8_quant.default(view, GROUP_SIZE)
        q2, s2 = torch.ops.vllm.rocm_aiter_group_fp8_quant.default(view, GROUP_SIZE)
        return q1, s1, q2, s2


@pytest.mark.parametrize(
    "model_cls",
    [_NoViewDoubleQuantModel, _ViewDoubleQuantModel],
    ids=["no_view", "with_view"],
)
@pytest.mark.skipif(
    not is_aiter_found_and_supported(),
    reason="Only test on ROCm with AITER installed and supported",
)
def test_double_aiter_rms_fp8_group_quant_fusion(
    model_cls: type[torch.nn.Module],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Both fan-out shapes (with and without an intermediate view) must fuse
    into ``rocm_aiter_rmsnorm_fp8_group_quant``: the no-view shape via
    ``DoubleAiterRMSFp8GroupQuantPattern`` and the viewed shape via the
    new ``DoubleAiterRMSFp8GroupQuantViewPattern`` sibling.

    A failure on the ``with_view`` parametrization is a regression on the
    DSv3.2 q_c norm path that this PR's view-tolerant pattern is intended
    to cover.
    """
    torch._dynamo.reset()

    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=torch.bfloat16),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["+rms_norm", "+quant_fp8"],
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
        passes = [
            NoOpEliminationPass(vllm_config),
            fusion_pass,
            PostCleanupPass(vllm_config),
        ]
        backend = TestBackend(*passes)
        model = model_cls()

        x = torch.randn(8, HIDDEN_SIZE)
        torch._dynamo.mark_dynamic(x, 0)

        outputs_unfused = model(x)
        model_fused = torch.compile(model, backend=backend)
        outputs_fused = model_fused(x)

        # Both consumers must be rewritten into the fused op (one
        # ``register_replacement`` rewrite covers the whole 1-to-2 fan-out).
        assert fusion_pass.matched_count == 1, (
            f"Expected the {model_cls.__name__} fan-out to fuse via the "
            f"DoubleQuant pattern (matched_count == 1), got "
            f"{fusion_pass.matched_count}"
        )

        fused_op = rocm_aiter_ops.get_rmsnorm_group_fused_quant_op()
        backend.check_after_ops([fused_op])

        # Numerical parity sanity-check: the fused pair must match the
        # unfused pair on FP8 outputs (exact byte-equality is the goal,
        # but allow a tiny tolerance for any residual numeric noise).
        for fused_t, unfused_t in zip(outputs_fused, outputs_unfused):
            torch.testing.assert_close(
                fused_t.to(torch.float32),
                unfused_t.to(torch.float32),
                atol=1e-2,
                rtol=1e-2,
            )
