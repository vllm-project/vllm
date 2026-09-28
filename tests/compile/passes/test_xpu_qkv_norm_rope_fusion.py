# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import pytest
import torch

from tests.compile.backend import TestBackend
from vllm.compilation.passes.fusion.xpu_qkv_norm_rope_fusion import (
    XpuQkvNormRopeFusionPass,
)
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    ModelConfig,
    PassConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionType

FUSED_OP = "_xpu_C.qkv_split_norm_rope"

ROPE_PARAMETERS = {
    "mrope_interleaved": True,
    "mrope_section": [11, 11, 10],
    "partial_rotary_factor": 0.25,
    "rope_theta": 10000000,
    "rope_type": "default",
}


# Only config.json is read (for rms_norm_eps and rope_parameters).
MODEL = os.environ.get("VLLM_TEST_QWEN36_MOE_MODEL", "Qwen/Qwen3.6-35B-A3B")


class GatedQkvModel(torch.nn.Module):
    """The unfused gated QKV post-processing of Qwen3NextAttention."""

    def __init__(self, num_heads, num_kv_heads, head_dim, vllm_config, dtype):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = (
            num_heads,
            num_kv_heads,
            head_dim,
        )
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.attn = Attention(
            num_heads=num_heads,
            head_size=head_dim,
            scale=head_dim**-0.5,
            num_kv_heads=num_kv_heads,
            cache_config=vllm_config.cache_config,
            prefix="model.layers.3.self_attn.attn",
            attn_type=AttentionType.DECODER,
        )
        self.q_norm = GemmaRMSNorm(head_dim, eps=1e-6)
        self.k_norm = GemmaRMSNorm(head_dim, eps=1e-6)
        with torch.no_grad():
            self.q_norm.weight.normal_(0, 0.1)
            self.k_norm.weight.normal_(0, 0.1)
        self.rotary_emb = get_rope(
            head_size=head_dim,
            max_position=4096,
            rope_parameters=ROPE_PARAMETERS,
            dtype=dtype,
        )

    def forward(self, qkv, positions):
        q_gate, k, v = qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
        orig_shape = q_gate.shape[:-1]
        q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
        q, gate = torch.chunk(q_gate, 2, dim=-1)
        q = q.reshape(*orig_shape, -1)
        gate = gate.reshape(*orig_shape, -1)
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(
            -1, self.num_heads * self.head_dim
        )
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(
            -1, self.num_kv_heads * self.head_dim
        )
        q, k = self.rotary_emb(positions, q, k)
        # As Attention.forward does before the attention op.
        return (
            q.view(-1, self.num_heads, self.head_dim),
            k.view(-1, self.num_kv_heads, self.head_dim),
            v.view(-1, self.num_kv_heads, self.head_dim),
            torch.sigmoid(gate),
        )


def _has_op(graph, name):
    # The fused op sits inside auto_functionalized(op, ...).
    return any(
        n.op == "call_function"
        and (name in str(n.target) or (n.args and name in str(n.args[0])))
        for n in graph.nodes
    )


@pytest.mark.skipif(not current_platform.is_xpu(), reason="XPU only")
@pytest.mark.parametrize("mrope", [True, False])
@pytest.mark.parametrize("num_tokens", [1, 5])
def test_xpu_qkv_norm_rope_fusion(mrope, num_tokens):
    if not hasattr(torch.ops._xpu_C, "qkv_split_norm_rope"):
        pytest.skip("qkv_split_norm_rope not available")
    dtype = torch.float16
    torch.set_default_device("xpu")
    torch.set_default_dtype(dtype)
    torch.manual_seed(0)
    vllm_config = VllmConfig(
        model_config=ModelConfig(model=MODEL, dtype=dtype, trust_remote_code=True),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            pass_config=PassConfig(fuse_xpu_qkv_norm_rope=True, eliminate_noops=True),
        ),
    )
    with (
        set_current_vllm_config(vllm_config),
        vllm_config.kernel_config.ir_op_priority.set_priority(),
    ):
        model = GatedQkvModel(8, 1, 256, vllm_config, dtype)
        fusion = XpuQkvNormRopeFusionPass(vllm_config)
        noop, cleanup = NoOpEliminationPass(vllm_config), PostCleanupPass(vllm_config)
        backend = TestBackend(noop, fusion, cleanup)
        backend_ref = TestBackend(noop, cleanup)
        T = num_tokens
        qkv = torch.randn(T, 2 * model.q_size + 2 * model.kv_size) * 3
        pos = torch.randint(0, 4096, (3, T) if mrope else (T,))
        args, args_ref = (qkv.clone(), pos.clone()), (qkv.clone(), pos.clone())
        for q_in, p_in in (args, args_ref):
            # Dynamic token count, as in vLLM's compiled graphs.
            torch._dynamo.mark_dynamic(q_in, 0)
            torch._dynamo.mark_dynamic(p_in, p_in.dim() - 1)
        ref = torch.compile(model, backend=backend_ref)(*args_ref)
        out = torch.compile(model, backend=backend)(*args)
        assert fusion.matched_count == 1
        assert _has_op(backend.graph_post_pass, "qkv_split_norm_rope")
        for a, b in zip(out, ref):
            torch.testing.assert_close(a, b, atol=2e-2, rtol=2e-2)
