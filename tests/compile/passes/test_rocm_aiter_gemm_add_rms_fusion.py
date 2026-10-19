# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.config
from tests.compile.backend import TestBackend
from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
from vllm.compilation.passes.fx_utils import find_op_nodes
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    ModelConfig,
    PassConfig,
    VllmConfig,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.model_executor.layers.utils import rocm_unquantized_gemm


class TestModel(torch.nn.Module):
    def __init__(self, num_layers: int, hidden_size: int, k: int, gemma_norm: bool):
        super().__init__()
        norm_cls = GemmaRMSNorm if gemma_norm else RMSNorm
        self.norm = [norm_cls(hidden_size, eps=1e-6) for _ in range(num_layers)]
        for norm in self.norm:
            torch.nn.init.normal_(norm.weight, std=0.1)
        self.weight = [torch.randn(hidden_size, k) / k**0.5 for _ in range(num_layers)]

    def forward(self, x, residual):
        # avoid having graph inputs be pattern args directly
        x = torch.relu(x)
        residual = torch.relu(residual)
        outs = []
        for norm, weight in zip(self.norm, self.weight):
            h = rocm_unquantized_gemm(self, x, weight)
            out, residual = norm(h, residual)
            outs.append(out)
        return residual, *outs

    def ops_in_model_before(self):
        return [
            torch.ops.vllm.rocm_unquantized_gemm.default,
            torch.ops.vllm_ir.fused_add_rms_norm,
        ]

    def ops_in_model_after(self):
        return [rocm_aiter_ops.get_gemm_add_rmsnorm_op()]


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_layers", [3])
@pytest.mark.parametrize("gemma_norm", [False, True])
@pytest.mark.parametrize("fused_op_available", [True, False])
@pytest.mark.parametrize("num_tokens", [16])
# AITER's gfx950 Triton config for N=256, K=7168 splits K at M <= 16.
@pytest.mark.parametrize("hidden_size, k", [(256, 7168)])
@pytest.mark.skipif(
    not is_aiter_found_and_supported(),
    reason="Only test on ROCm with AITER installed and supported",
)
def test_fuse_gemm_add_rms(
    dtype: torch.dtype,
    num_layers: int,
    gemma_norm: bool,
    fused_op_available: bool,
    num_tokens: int,
    hidden_size: int,
    k: int,
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        pytest.skip("The AITER tuned GEMM dispatch is gfx950-only")

    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=dtype),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["+rms_norm"],
            pass_config=PassConfig(fuse_gemm_add_rms=True, eliminate_noops=True),
        ),
    )

    with vllm.config.set_current_vllm_config(vllm_config), monkeypatch.context() as m:
        from aiter.ops.triton.normalization import fused_reduce_add_rmsnorm as mod

        from vllm.compilation.passes.fusion.rocm_aiter_fusion import (
            RocmAiterGemmAddRMSNormFusionPass,
        )

        torch.set_default_device("cuda")
        torch.set_default_dtype(dtype)
        torch.manual_seed(1)

        m.setenv("VLLM_ROCM_USE_AITER", "1")
        m.setenv("VLLM_ROCM_USE_SKINNY_GEMM", "0")
        rocm_aiter_ops.refresh_env_variables()
        if not fused_op_available:
            m.setattr(
                rocm_aiter_ops, "is_fused_reduce_add_rmsnorm_supported", lambda: False
            )
        # Route the GEMM through the AITER Triton kernel whatever the tuned CSV says.
        m.setattr(
            "vllm.model_executor.layers.utils._aiter_tuned_gemm_libtype",
            lambda *args: "triton",
        )
        input_ranks = []
        fused_reduce_add_rmsnorm = mod.fused_reduce_add_rmsnorm

        def spy(x, *args, **kwargs):
            input_ranks.append(x.dim())
            return fused_reduce_add_rmsnorm(x, *args, **kwargs)

        m.setattr(mod, "fused_reduce_add_rmsnorm", spy)

        fusion_pass = RocmAiterGemmAddRMSNormFusionPass(vllm_config)
        passes = [
            NoOpEliminationPass(vllm_config),
            fusion_pass,
            PostCleanupPass(vllm_config),
        ]
        backend = TestBackend(*passes)
        model = TestModel(num_layers, hidden_size, k, gemma_norm)

        x = torch.randn(num_tokens, k)
        residual = torch.randn(num_tokens, hidden_size)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(residual, 0)
        outputs_unfused = model(x, residual)

        model_fused = torch.compile(model, backend=backend)
        outputs_fused = model_fused(x, residual)

        # atol: one bf16 ulp at |residual| 4-8; only unfused rounds before the add.
        torch.testing.assert_close(outputs_unfused, outputs_fused, atol=6e-2, rtol=2e-2)

        if not fused_op_available:
            # Without the AITER op the pass registers nothing; the graph stays unfused.
            assert fusion_pass.matched_count == 0 and not input_ranks
            backend.check_after_ops(model.ops_in_model_before())
            return

        assert fusion_pass.matched_count == num_layers
        assert input_ranks and all(rank == 3 for rank in input_ranks)

        backend.check_before_ops(model.ops_in_model_before())
        backend.check_after_ops(model.ops_in_model_after())
        # Gemma norms must take the Gemma form, which folds (1 + w) into the kernel.
        fused_op = model.ops_in_model_after()[0]
        names = [a.name for a in fused_op._schema.arguments]
        fused = list(find_op_nodes(fused_op, backend.graph_post_pass))
        assert fused and all(
            dict(zip(names, n.args), **n.kwargs)["gemma_norm"] == gemma_norm
            for n in fused
        )
