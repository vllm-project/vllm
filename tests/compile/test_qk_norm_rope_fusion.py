# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest  # type: ignore
import torch

from tests.compile.backend import LazyInitPass, TestBackend
from vllm.compilation.noop_elimination import NoOpEliminationPass
from vllm.compilation.post_cleanup import PostCleanupPass
from vllm.compilation.qk_norm_rope_fusion import (
    FUSED_QK_ROPE_OP,
    QKNormRoPEFusionPass,
    RMS_OP,
)
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    PassConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.attention import Attention, AttentionType
from vllm.platforms import current_platform

class QKNormRoPETestModel(torch.nn.Module):
    """A minimal model that exercises the unfused Q/K RMSNorm + RoPE pattern.

    It also instantiates an Attention layer to register itself into
    vllm_config.compilation_config.static_forward_context, which the fusion
    pass uses to infer per-layer sizes.
    """

    def __init__(
        self,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        vllm_config: VllmConfig,
        prefix: str = "model.layers.0.self_attn.attn",
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim

        # Register an Attention layer so the pass can read sizes from config
        self.attn = Attention(
            num_heads=num_heads,
            head_size=head_dim,
            scale=1.0 / (head_dim**0.5),
            num_kv_heads=num_kv_heads,
            cache_config=vllm_config.cache_config,
            prefix=prefix,
            attn_type=AttentionType.DECODER,
        )

        # Use the same RMSNorm and RoPE components as models do
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=4096,
            base=10000,
        )

    def forward(self, qkv: torch.Tensor, positions: torch.Tensor):
        # Unfused baseline: split, per-head RMS, then RoPE
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q_by_head = q.view(
            *q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim
        )
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)

        k_by_head = k.view(
            *k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim
        )
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)

        q, k = self.rotary_emb(positions, q, k)
        return q, k, v


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("T", [17])
@pytest.mark.parametrize("num_heads, num_kv_heads, head_dim", [(16, 2, 128)])
@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Only test on CUDA and ROCm",
)
def test_qk_norm_rope_fusion(dtype, T, num_heads, num_kv_heads, head_dim):
    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)
    torch.manual_seed(0)

    # Enable the fusion pass and necessary custom ops
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["+rms_norm", "+rotary_embedding"],
            pass_config=PassConfig(
                enable_qk_norm_rope_fusion=True,
                enable_noop=True,
            ),
        )
    )

    # Build model; creating Attention during init registers it into
    # static_forward_context for the pass to query sizes
    with set_current_vllm_config(vllm_config):
        model = QKNormRoPETestModel(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            vllm_config=vllm_config,
        )

    qkv = torch.randn(T, num_heads * head_dim + 2 * num_kv_heads * head_dim)
    positions = torch.arange(T, dtype=torch.long, device=qkv.device)

    # Eager baseline
    with set_current_vllm_config(vllm_config):
        q_eager, k_eager, v_eager = model(qkv, positions)

    # Set up backend with our fusion pass
    noop_pass = NoOpEliminationPass(vllm_config)
    fusion_pass = LazyInitPass(QKNormRoPEFusionPass, vllm_config)
    cleanup_pass = PostCleanupPass(vllm_config)
    backend = TestBackend(noop_pass, fusion_pass, cleanup_pass)

    # Compile and run
    with set_current_vllm_config(vllm_config):
        model_compiled = torch.compile(model, backend=backend, fullgraph=True)
        q_comp, k_comp, v_comp = model_compiled(qkv, positions)

    # Numerical check
    atol = 1e-2 if dtype == torch.bfloat16 else 2e-3
    rtol = atol
    torch.testing.assert_close(q_eager, q_comp, atol=atol, rtol=rtol)
    torch.testing.assert_close(k_eager, k_comp, atol=atol, rtol=rtol)
    torch.testing.assert_close(v_eager, v_comp, atol=atol, rtol=rtol)

    # Ensure the pass matched at least one site
    assert fusion_pass.pass_.matched_count > 0

    # Pre graph should contain unfused RMS, post graph should contain fused op
    backend.check_before_ops([RMS_OP])
    # At least the fused op should be present after
    backend.check_after_ops([FUSED_QK_ROPE_OP])
