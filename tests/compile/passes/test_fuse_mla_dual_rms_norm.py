# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit test for the MLADualRMSNormFusionPass.

The pass fuses paired q/kv RMS norms in MLA attention into a single
fused_mla_dual_rms_norm op backed by AITER's fused_qk_rmsnorm kernel.
"""

import pytest
import torch

import vllm.config
from tests.compile.backend import TestBackend
from vllm._aiter_ops import (
    check_aiter_fused_qk_rmsnorm_per_token_quant,
    is_aiter_found_and_supported,
    rocm_aiter_ops,
)
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    ModelConfig,
    PassConfig,
    VllmConfig,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.platforms import current_platform

# MLA attention geometry for DeepSeek-V3 / Kimi-K2
Q_DIM = 1536
KV_C_DIM = 512
K_PE_DIM = 64
EPS = 1e-6

FP8_DTYPE = current_platform.fp8_dtype()


class MLADualRMSNormTestModel(torch.nn.Module):
    """
    Minimal model reproducing the MLA dual RMS norm pattern:
        linear -> split([q_dim, kv_dim])
            +-- q_c (getitem 0) -> rms_norm(q_w, eps) -> linear
            +-- kv_lora (getitem 1) -> split([kv_c_dim, k_pe_dim])
                    +-- kv_c (getitem 0) -> rms_norm(kv_w, eps)
                    +-- k_pe
    """

    def __init__(
        self,
        hidden_size: int,
        q_dim: int = Q_DIM,
        kv_c_dim: int = KV_C_DIM,
        k_pe_dim: int = K_PE_DIM,
        eps: float = EPS,
    ):
        super().__init__()
        self.q_dim = q_dim
        self.kv_dim = kv_c_dim + k_pe_dim
        self.kv_c_dim = kv_c_dim
        self.k_pe_dim = k_pe_dim

        self.proj = torch.nn.Linear(hidden_size, q_dim + self.kv_dim, bias=False)
        self.q_norm = RMSNorm(q_dim, eps=eps)
        self.kv_norm = RMSNorm(kv_c_dim, eps=eps)
        self.q_b_proj = torch.nn.Linear(q_dim, hidden_size, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Avoid graph input being a direct arg to a matched pattern node
        x = torch.relu(x)

        projected = self.proj(x)

        q_c, kv_lora = projected.split([self.q_dim, self.kv_dim], dim=-1)
        kv_c, k_pe = kv_lora.split([self.kv_c_dim, self.k_pe_dim], dim=-1)

        q_normed = self.q_norm(q_c)
        kv_normed = self.kv_norm(kv_c)

        q_out = self.q_b_proj(q_normed)
        return q_out, kv_normed, k_pe

    def ops_in_model_before(self):
        return [torch.ops.vllm_ir.rms_norm.default]

    def ops_in_model_after(self):
        return [torch.ops.vllm.fused_mla_dual_rms_norm.default]


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.skipif(
    not is_aiter_found_and_supported(),
    reason="Only test on ROCm with AITER installed and supported",
)
def test_fuse_mla_dual_rms_norm(
    dtype: torch.dtype,
    hidden_size: int,
    monkeypatch: pytest.MonkeyPatch,
):
    torch._dynamo.reset()

    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=dtype),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["+rms_norm"],
            pass_config=PassConfig(
                fuse_mla_dual_rms_norm=True,
                eliminate_noops=True,
            ),
        ),
    )

    with vllm.config.set_current_vllm_config(vllm_config), monkeypatch.context() as m:
        from vllm.compilation.passes.fusion.rocm_aiter_fusion import (
            MLADualRMSNormFusionPass,
        )

        torch.set_default_device("cuda")
        torch.set_default_dtype(dtype)
        torch.manual_seed(42)

        m.setenv("VLLM_ROCM_USE_AITER", "1")
        rocm_aiter_ops.refresh_env_variables()

        fusion_pass = MLADualRMSNormFusionPass(vllm_config)
        passes = [
            NoOpEliminationPass(vllm_config),
            fusion_pass,
            PostCleanupPass(vllm_config),
        ]
        backend = TestBackend(*passes)
        model = MLADualRMSNormTestModel(hidden_size)

        x = torch.randn(1, hidden_size)
        torch._dynamo.mark_dynamic(x, 0)

        outputs_unfused = model(x)

        model_fused = torch.compile(model, backend=backend)
        outputs_fused = model_fused(x)

        torch.testing.assert_close(outputs_unfused, outputs_fused, atol=1e-2, rtol=1e-2)

        assert fusion_pass.matched_count == 1, (
            f"Expected 1 fused pair, got {fusion_pass.matched_count}"
        )

        backend.check_before_ops(model.ops_in_model_before())
        backend.check_after_ops(model.ops_in_model_after())


class MLADualRMSNormFp8PerTokenTestModel(torch.nn.Module):
    """
    Minimal model reproducing the FP8 MLA attention path with *per-token* quant.

    In this path only the *q* latent is FP8 per-token-quantized (it feeds the FP8
    ``q_b_proj`` GEMM); the *kv* latent is RMS-normed and consumed by attention
    as bf16. So ``RocmAiterRMSNormQuantFusionPass`` (which runs earlier) folds
    only the q-side ``rms_norm -> fp8 per-token quant`` into a single
    ``rocm_aiter_rmsnorm_fused_dynamic_quant`` op (a single ``(M, 1)`` scale) and
    leaves the kv side a plain ``vllm_ir.rms_norm``. This model mirrors that
    asymmetric graph:

        linear -> split([q_dim, kv_dim])
            +-- q_c (getitem 0) -> rocm_aiter_rmsnorm_fused_dynamic_quant -> dequant
            +-- kv_lora (getitem 1) -> split([kv_c_dim, k_pe_dim])
                    +-- kv_c (getitem 0) -> rms_norm (bf16)
                    +-- k_pe

    ``forward`` dequantizes the q fp8 output in-graph and returns it as bf16.
    Returning the fp8 tensor directly graph-breaks dynamo (the quant op then
    escapes to eager and the fusion never matches), so the dequant keeps the
    quant op inside the compiled FX graph. The unfused path runs aiter's
    ``rmsnorm2d_fwd_with_dynamicquant`` (q) plus ``vllm_ir.rms_norm`` (kv); the
    fused path runs both through the single HIP ``fused_qk_rmsnorm_per_token_quant``
    kernel. With identical per-token scales the dequantized q values are bit-exact
    except on fp8 rounding ties (at most one e4m3 step), and the bf16 kv norm
    matches to bf16 tolerance.
    """

    def __init__(
        self,
        hidden_size: int,
        q_dim: int = Q_DIM,
        kv_c_dim: int = KV_C_DIM,
        k_pe_dim: int = K_PE_DIM,
        eps: float = EPS,
    ):
        super().__init__()
        self.q_dim = q_dim
        self.kv_dim = kv_c_dim + k_pe_dim
        self.kv_c_dim = kv_c_dim
        self.k_pe_dim = k_pe_dim
        self.eps = eps

        self.proj = torch.nn.Linear(hidden_size, q_dim + self.kv_dim, bias=False)
        self.q_weight = torch.nn.Parameter(torch.ones(q_dim))
        # kv latent is RMS-normed only (no quant); RMSNorm with custom_ops
        # ["+rms_norm"] lowers to vllm_ir.rms_norm, matching the FP8 graph.
        self.kv_norm = RMSNorm(kv_c_dim, eps=eps)

    def _dequant(self, x_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        # Per-token: a single (M, 1) scale broadcast across the row.
        return (x_fp8.to(torch.float32) * scale).to(torch.bfloat16)

    def forward(self, x: torch.Tensor):
        # Avoid graph input being a direct arg to a matched pattern node
        x = torch.relu(x)

        projected = self.proj(x)

        q_c, kv_lora = projected.split([self.q_dim, self.kv_dim], dim=-1)
        kv_c, k_pe = kv_lora.split([self.kv_c_dim, self.k_pe_dim], dim=-1)

        q_fp8, q_scale = torch.ops.vllm.rocm_aiter_rmsnorm_fused_dynamic_quant(
            q_c, self.q_weight, self.eps, FP8_DTYPE
        )
        kv_normed = self.kv_norm(kv_c)

        # Dequant q in-graph keeps the quant op inside the compiled graph so the
        # fusion pass can match it (returning fp8 directly graph-breaks dynamo);
        # kv_normed is already bf16 and is returned (and matched) directly.
        return self._dequant(q_fp8, q_scale), kv_normed, k_pe

    def ops_in_model_before(self):
        return [
            torch.ops.vllm.rocm_aiter_rmsnorm_fused_dynamic_quant.default,
            torch.ops.vllm_ir.rms_norm.default,
        ]

    def ops_in_model_after(self):
        return [torch.ops.vllm.fused_mla_dual_rms_norm_per_token_quant.default]


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.skipif(
    not is_aiter_found_and_supported()
    or not check_aiter_fused_qk_rmsnorm_per_token_quant(),
    reason=(
        "Only test on ROCm with AITER (incl. fused_qk_rmsnorm_per_token_quant) "
        "installed and supported"
    ),
)
def test_fuse_mla_dual_rms_norm_fp8_per_token(
    dtype: torch.dtype,
    hidden_size: int,
    monkeypatch: pytest.MonkeyPatch,
):
    torch._dynamo.reset()

    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=dtype),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["+rms_norm"],
            pass_config=PassConfig(
                fuse_mla_dual_rms_norm=True,
                eliminate_noops=True,
            ),
        ),
    )

    with vllm.config.set_current_vllm_config(vllm_config), monkeypatch.context() as m:
        from vllm.compilation.passes.fusion.rocm_aiter_fusion import (
            MLADualRMSNormFusionPass,
        )

        torch.set_default_device("cuda")
        torch.set_default_dtype(dtype)
        torch.manual_seed(42)

        m.setenv("VLLM_ROCM_USE_AITER", "1")
        rocm_aiter_ops.refresh_env_variables()

        fusion_pass = MLADualRMSNormFusionPass(vllm_config)
        passes = [
            NoOpEliminationPass(vllm_config),
            fusion_pass,
            PostCleanupPass(vllm_config),
        ]
        backend = TestBackend(*passes)
        model = MLADualRMSNormFp8PerTokenTestModel(hidden_size)

        x = torch.randn(4, hidden_size)
        torch._dynamo.mark_dynamic(x, 0)

        # Decode runs under inference_mode; the fp8 quant op has no autograd
        # kernel, so without this AOTAutograd builds a joint forward/backward
        # graph that graph-breaks and the fusion never fires.
        with torch.inference_mode():
            outputs_unfused = model(x)

            model_fused = torch.compile(model, backend=backend)
            outputs_fused = model_fused(x)

        q_deq_u, kv_normed_u, k_pe_u = outputs_unfused
        q_deq_f, kv_normed_f, k_pe_f = outputs_fused

        # k_pe is a pure passthrough and must be bit-identical.
        torch.testing.assert_close(k_pe_u, k_pe_f, atol=0, rtol=0)

        # kv latent is RMS-normed (not quantized) on both paths; the Triton and
        # HIP rms norms agree to bf16 tolerance.
        torch.testing.assert_close(kv_normed_u, kv_normed_f, atol=1e-2, rtol=1e-2)

        # With matching per-token scales the dequantized q values are bit-exact
        # except where the unfused and fused quant kernels round an fp8 tie
        # differently. Require the bulk to be exact (this catches scale bugs,
        # which shift whole rows) and bound the rare ties to one e4m3 step
        # (relative gap 2**-3 = 0.125).
        E4M3_STEP = 0.125
        exact_frac = (q_deq_u == q_deq_f).float().mean().item()
        assert exact_frac > 0.99, (
            f"q: only {exact_frac:.4f} of elements bit-exact; scales likely differ"
        )
        torch.testing.assert_close(q_deq_u, q_deq_f, atol=1e-2, rtol=E4M3_STEP)

        assert fusion_pass.matched_count == 1, (
            f"Expected 1 fused pair, got {fusion_pass.matched_count}"
        )

        backend.check_before_ops(model.ops_in_model_before())
        backend.check_after_ops(model.ops_in_model_after())
