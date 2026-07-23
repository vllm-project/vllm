# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch._ops import OpOverload, OpOverloadPacket

import vllm.compilation.passes.fusion.qk_norm_rope_fusion as mrope_fusion_mod
from tests.compile.backend import TestBackend
from vllm.compilation.passes.fusion.matcher_utils import MROPE_OP
from vllm.compilation.passes.fusion.qk_norm_rope_fusion import (
    FUSED_QK_MROPE_OP,
    QKNormMRoPEFusionPass,
)
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.compilation.passes.utility.split_coalescing import SplitCoalescingPass
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    ModelConfig,
    PassConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionType


class QKNormMRoPETestModel(torch.nn.Module):
    def __init__(
        self,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        eps: float,
        mrope_section: list[int],
        vllm_config: VllmConfig,
        dtype: torch.dtype,
        prefix: str = "model.layers.0.self_attn.attn",
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.eps = eps
        self.dtype = dtype

        # Register layer geometry for the fusion pass via Attention.
        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=1.0 / self.head_dim**0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            prefix=prefix,
            attn_type=AttentionType.DECODER,
        )

        self.q_norm = RMSNorm(self.head_dim, eps=self.eps)
        self.k_norm = RMSNorm(self.head_dim, eps=self.eps)
        # mRoPE rotation is neox-style; full rotary (rotary_dim == head_dim).
        self.rotary_emb = MRotaryEmbedding(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position_embeddings=4096,
            base=10000,
            is_neox_style=True,
            dtype=self.dtype,
            mrope_section=mrope_section,
            mrope_interleaved=False,
        )

    def forward(self, qkv: torch.Tensor, positions: torch.Tensor):
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        return q, k, v

    def ops_in_model_before(self) -> list[OpOverload | OpOverloadPacket]:
        return [torch.ops.vllm_ir.rms_norm, MROPE_OP]

    def ops_in_model_after(self) -> list[OpOverload | OpOverloadPacket]:
        return [FUSED_QK_MROPE_OP]


@pytest.mark.parametrize("eps", [1e-5, 1e-6])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Only test on cuda and rocm platform",
)
def test_qk_norm_mrope_fusion(monkeypatch, eps, dtype):
    if not hasattr(torch.ops._C, "fused_qk_norm_mrope"):
        pytest.skip("fused_qk_norm_mrope custom op not available")

    torch.set_default_device(current_platform.device_type)
    torch.set_default_dtype(dtype)
    torch.manual_seed(0)

    num_heads, num_kv_heads, head_dim = 16, 4, 128
    mrope_section = [16, 24, 24]  # sums to head_dim // 2
    T = 5

    # The pass reads mrope_section from the model's HF config; a synthetic
    # ModelConfig has none, so inject it to isolate the pattern-matching logic.
    monkeypatch.setattr(
        mrope_fusion_mod,
        "_discover_mrope_configs",
        lambda config: ((tuple(mrope_section), False),),
    )

    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=dtype),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["+rms_norm", "+rotary_embedding"],
            pass_config=PassConfig(
                enable_qk_norm_rope_fusion=True,
                eliminate_noops=True,
            ),
        ),
    )

    with (
        set_current_vllm_config(vllm_config),
        vllm_config.kernel_config.ir_op_priority.set_priority(),
    ):
        model = QKNormMRoPETestModel(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            eps=eps,
            mrope_section=mrope_section,
            vllm_config=vllm_config,
            dtype=dtype,
        )

        noop_pass = NoOpEliminationPass(vllm_config)
        coalesce_pass = SplitCoalescingPass(vllm_config)
        fusion_pass = QKNormMRoPEFusionPass(vllm_config)
        cleanup_pass = PostCleanupPass(vllm_config)

        backend = TestBackend(noop_pass, coalesce_pass, fusion_pass, cleanup_pass)
        backend_baseline = TestBackend(noop_pass, cleanup_pass)

        qkv = torch.randn(T, model.q_size + 2 * model.kv_size)
        # mRoPE positions: [3, num_tokens].
        pos = torch.randint(0, 1000, (3, T), dtype=torch.long, device=qkv.device)
        qkv_unfused = qkv.clone()
        pos_unfused = pos.clone()

        torch._dynamo.mark_dynamic(qkv, 0)
        torch._dynamo.mark_dynamic(pos, 1)
        model_fused = torch.compile(model, backend=backend)
        q_fused, k_fused, v_fused = model_fused(qkv, pos)

        torch._dynamo.mark_dynamic(qkv_unfused, 0)
        torch._dynamo.mark_dynamic(pos_unfused, 1)
        model_unfused = torch.compile(model, backend=backend_baseline)
        q_unfused, k_unfused, v_unfused = model_unfused(qkv_unfused, pos_unfused)

        if dtype == torch.float16:
            ATOL, RTOL = (2e-3, 2e-3)
        else:
            ATOL, RTOL = (1e-2, 1e-2)

        torch.testing.assert_close(q_unfused, q_fused, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(k_unfused, k_fused, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(v_unfused, v_fused, atol=ATOL, rtol=RTOL)

        assert fusion_pass.matched_count == 1

        backend.check_before_ops(model.ops_in_model_before())
        backend.check_after_ops(model.ops_in_model_after())
