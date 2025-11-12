# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.compile.backend import TestBackend
from vllm.attention import Attention, AttentionType
from vllm.compilation.matcher_utils import FLASHINFER_ROTARY_OP, RMS_OP, ROTARY_OP
from vllm.compilation.noop_elimination import NoOpEliminationPass
from vllm.compilation.post_cleanup import PostCleanupPass
from vllm.compilation.qk_norm_rope_fusion import (
    FUSED_QK_ROPE_OP,
    QKNormRoPEFusionPass,
)
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    ModelConfig,
    PassConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.platforms import current_platform

RSQRT_OP = torch.ops.aten.rsqrt.default
INDEX_SELECT_OP = torch.ops.aten.index.Tensor


class QKNormRoPETestModel(torch.nn.Module):
    def __init__(
        self,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        eps: float,
        is_neox: bool,
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
        self.rotary_dim = head_dim
        self.eps = eps
        self.dtype = dtype

        # Register layer metadata for the fusion pass via Attention.
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
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position_embeddings=4096,
            base=10000,
            is_neox_style=is_neox,
            dtype=self.dtype,
        )
        self.enable_rms_norm_custom_op = self.q_norm.enabled()
        self.enable_rope_custom_op = self.rotary_emb.enabled()

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

    def ops_in_model_before(self) -> list[torch._ops.OpOverload]:
        ops = []
        if self.enable_rms_norm_custom_op:
            ops.append(RMS_OP)
        else:
            ops.append(RSQRT_OP)

        if self.enable_rope_custom_op:
            if self.rotary_emb.use_flashinfer:
                ops.append(FLASHINFER_ROTARY_OP)
            else:
                ops.append(ROTARY_OP)
        else:
            ops.append(INDEX_SELECT_OP)
        return ops

    def ops_in_model_after(self) -> list[torch._ops.OpOverload]:
        return [FUSED_QK_ROPE_OP]


@pytest.mark.parametrize("eps", [1e-5, 1e-6])
@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize("enable_rms_norm_custom_op", [True, False])
@pytest.mark.parametrize("enable_rope_custom_op", [True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Only test on cuda and rocm platform",
)
def test_qk_norm_rope_fusion(
    eps, is_neox, enable_rms_norm_custom_op, enable_rope_custom_op, dtype
):
    if not hasattr(torch.ops._C, "fused_qk_norm_rope"):
        pytest.skip("fused_qk_norm_rope custom op not available")

    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)
    torch.manual_seed(0)

    custom_ops: list[str] = []
    if enable_rms_norm_custom_op:
        custom_ops.append("+rms_norm")
    if enable_rope_custom_op:
        custom_ops.append("+rotary_embedding")

    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=dtype),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=custom_ops,
            pass_config=PassConfig(
                enable_qk_norm_rope_fusion=True,
                enable_noop=True,
            ),
        ),
    )

    num_heads, num_kv_heads, head_dim = 16, 4, 128
    T = 5

    with set_current_vllm_config(vllm_config):
        model = QKNormRoPETestModel(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            eps=eps,
            is_neox=is_neox,
            vllm_config=vllm_config,
            dtype=dtype,
        )

        noop_pass = NoOpEliminationPass(vllm_config)
        fusion_pass = QKNormRoPEFusionPass(vllm_config)
        cleanup_pass = PostCleanupPass(vllm_config)

        backend = TestBackend(noop_pass, fusion_pass, cleanup_pass)
        backend_baseline = TestBackend(noop_pass, cleanup_pass)

        qkv = torch.randn(T, model.q_size + 2 * model.kv_size)
        pos = torch.arange(T, dtype=torch.long, device=qkv.device)
        qkv_unfused = qkv.clone()
        pos_unfused = pos.clone()

        torch._dynamo.mark_dynamic(qkv, 0)
        torch._dynamo.mark_dynamic(pos, 0)
        model_fused = torch.compile(model, backend=backend)
        q_fused, k_fused, v_fused = model_fused(qkv, pos)

        torch._dynamo.mark_dynamic(qkv_unfused, 0)
        torch._dynamo.mark_dynamic(pos_unfused, 0)
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
