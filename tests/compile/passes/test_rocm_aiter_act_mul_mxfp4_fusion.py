# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from tests.compile.backend import TestBackend
from vllm._aiter_ops import IS_AITER_FOUND, rocm_aiter_ops
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    PassConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.activation import GeluAndMul, SiluAndMul
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and IS_AITER_FOUND),
    reason="ROCm with AITER only",
)


def _is_gfx950() -> bool:
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


class ActMulMxfp4Model(torch.nn.Module):
    """act_and_mul feeding an MXFP4 weight through the AITER ASM GEMM op."""

    def __init__(
        self,
        act: torch.nn.Module,
        hidden_size: int,
        dtype: torch.dtype,
        use_asm_gemm: bool,
    ):
        super().__init__()
        import aiter
        from aiter.ops.shuffle import shuffle_weight

        self.act = act
        self.dtype = dtype
        self.use_asm_gemm = use_asm_gemm
        w = torch.randn(hidden_size, hidden_size, dtype=dtype) * 0.02
        self.w_ref = w
        w_q, w_s = aiter.per_1x32_f4_quant_hip(w, shuffle=False)
        w_q, w_s = w_q.view(torch.uint8), w_s.view(torch.uint8)
        # The layouts AiterMxfp4LinearKernel.process_weights_after_loading
        # produces for each GEMM path.
        if use_asm_gemm:
            sm, sn = w_s.shape
            w_s = (
                w_s.view(sm // 32, 2, 16, sn // 8, 2, 4, 1)
                .permute(0, 3, 5, 2, 4, 1, 6)
                .contiguous()
                .view(sm, sn)
            )
            w_q = shuffle_weight(w_q, layout=(16, 16))
        else:
            w_s = w_s.T.contiguous()
        self.weight = torch.nn.Parameter(w_q, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(w_s, requires_grad=False)

    def exact(self, x: torch.Tensor, act_name: str) -> torch.Tensor:
        """act(gate) * up in fp32 against the unquantized weight."""
        d = x.shape[-1] // 2
        gate, up = x[..., :d].float(), x[..., d:].float()
        if act_name == "silu":
            h = torch.nn.functional.silu(gate) * up
        else:
            approximate = "tanh" if act_name == "gelu_tanh" else "none"
            h = torch.nn.functional.gelu(gate, approximate=approximate) * up
        return h @ self.w_ref.float().T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(x)
        return torch.ops.vllm.gemm_with_dynamic_quant(
            y, self.weight, self.weight_scale, self.use_asm_gemm, self.dtype
        )


@pytest.mark.parametrize("use_asm_gemm", [True, False])
@pytest.mark.parametrize("num_tokens", [16, 256])
@pytest.mark.parametrize("act_name", ["silu", "gelu", "gelu_tanh"])
@pytest.mark.parametrize("enable_act_custom_op", [True, False])
def test_rocm_aiter_act_mul_mxfp4_gemm_fusion(
    use_asm_gemm: bool,
    num_tokens: int,
    act_name: str,
    enable_act_custom_op: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    if not _is_gfx950():
        pytest.skip("AITER MXFP4 GEMM needs gfx950")
    dtype = torch.bfloat16
    hidden_size = 1024
    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)
    torch.manual_seed(0)

    custom_op = "gelu_and_mul" if act_name.startswith("gelu") else "silu_and_mul"
    custom_ops = ["none"] + ([f"+{custom_op}"] if enable_act_custom_op else [])
    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=custom_ops,
            backend="eager",
            pass_config=PassConfig(fuse_act_quant=True, eliminate_noops=True),
        ),
    )

    with set_current_vllm_config(config), monkeypatch.context() as m:
        m.setenv("VLLM_ROCM_USE_AITER", "1")
        rocm_aiter_ops.refresh_env_variables()
        from vllm.compilation.passes.fusion.rocm_aiter_fusion import (
            RocmAiterActMulMxfp4GemmFusionPass,
        )

        fusion_pass = RocmAiterActMulMxfp4GemmFusionPass(config)
        backend = TestBackend(
            NoOpEliminationPass(config), fusion_pass, PostCleanupPass(config)
        )
        if act_name == "silu":
            act = SiluAndMul()
        else:
            act = GeluAndMul("tanh" if act_name == "gelu_tanh" else "none")
        model = ActMulMxfp4Model(act, hidden_size, dtype, use_asm_gemm)

        x = torch.randn(num_tokens, 2 * hidden_size)
        torch._dynamo.mark_dynamic(x, 0)
        ref = model(x)
        out = torch.compile(model, backend=backend)(x)

        if not use_asm_gemm:
            # The Triton GEMM path keeps Inductor's act_and_mul: measured on
            # MI355X, aiter's fused act + quant was slower there.
            assert fusion_pass.matched_count == 0
            return
        assert fusion_pass.matched_count == 1
        gemm_nodes = [
            n
            for n in backend.graph_post_pass.nodes
            if n.target == torch.ops.vllm.gemm_with_dynamic_quant.default
        ]
        assert len(gemm_nodes) == 1
        # The fused op must compute what the matched graph computed: on ROCm the
        # native GeluAndMul drops the tanh approximation (see forward_native).
        if act_name == "silu":
            expected = "silu"
        elif act_name == "gelu_tanh" and (
            enable_act_custom_op or not current_platform.is_rocm()
        ):
            expected = "gelu_tanh"
        else:
            expected = "gelu"
        assert gemm_nodes[0].args[-1] == expected

        # The fused path quantizes act(gate) * up with aiter's Triton kernel
        # instead of the HIP quant, so individual MXFP4 roundings differ. Hold it
        # to the unfused path's accuracy against the exact product.
        exact = model.exact(x, act_name)

        def rel(y: torch.Tensor) -> float:
            return ((y.float() - exact).norm() / exact.norm()).item()

        assert rel(out) <= rel(ref) * 1.05 + 1e-3, (rel(out), rel(ref))
