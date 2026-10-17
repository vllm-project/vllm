# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity and correctness test suite for Gemma 4 fused Triton operations.

Validates:
1. Entrypoints:
   - fused_qkv_norm_rope (standard & KV-shared)
   - fused_post_attn_add_pre_ff_norm
   - fused_mlp_ple_epilogue (with & without PLE)
   - fused_mlp_norm_and_moe_prenorm & fused_moe_combine_norm_ple_epilogue
2. Invariants:
   - Weightless V-norm
   - KV-sharing layer behavior
   - Long-context position stability (p = 131,072, theta = 1,000,000)
3. End-to-end Gemma4DecoderLayer parity (PyTorch eager reference vs fused ops)
4. TorchDynamo zero-graph-break fullgraph compilation (@torch.compile(fullgraph=True))
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config.vllm import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_gemma4_ops import (
    fused_mlp_norm_and_moe_prenorm,
    fused_mlp_ple_epilogue,
    fused_moe_combine_norm_ple_epilogue,
    fused_ple_model_proj_norm_combine,
    fused_post_attn_add_moe_prenorms,
    fused_post_attn_add_pre_ff_norm,
    fused_qkv_norm_rope,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import get_rope


def compute_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


@pytest.fixture(autouse=True)
def setup_vllm_config():
    with set_current_vllm_config(VllmConfig()):
        yield


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for Triton tests"
)
class TestGemma4FusedOps:
    @classmethod
    def setup_class(cls):
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

    def test_fused_qkv_norm_rope_standard(self):
        """Test fused QKV split + RMSNorm + Neox RoPE for non-shared layers."""
        M = 4
        num_heads = 8
        num_kv_heads = 2
        head_dim = 256
        eps = 1e-6
        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim
        total_dim = q_size + 2 * kv_size

        qkv = torch.randn(M, total_dim, dtype=torch.bfloat16, device="cuda")
        positions = torch.tensor([0, 10, 50, 120], dtype=torch.int64, device="cuda")

        q_norm = RMSNorm(head_dim, eps=eps, dtype=torch.bfloat16).to("cuda")
        k_norm = RMSNorm(head_dim, eps=eps, dtype=torch.bfloat16).to("cuda")
        v_norm = RMSNorm(head_dim, eps=eps, has_weight=False, dtype=torch.bfloat16).to(
            "cuda"
        )
        rope = get_rope(
            head_dim,
            max_position=8192,
            is_neox_style=True,
            rope_parameters={"rope_theta": 10000.0},
            dtype=torch.bfloat16,
        )

        # Reference PyTorch implementation
        q_ref, k_ref, v_ref = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q_ref = q_norm(q_ref.unflatten(-1, (num_heads, head_dim))).flatten(-2, -1)
        k_ref = k_norm(k_ref.unflatten(-1, (num_kv_heads, head_dim))).flatten(-2, -1)
        q_ref, k_ref = rope(positions, q_ref, k_ref)
        v_ref = v_norm(v_ref.unflatten(-1, (num_kv_heads, head_dim))).flatten(-2, -1)

        # Fused Triton implementation
        out_q, out_k, out_v = fused_qkv_norm_rope(
            qkv,
            positions,
            rope.cos_sin_cache,
            q_norm.weight,
            k_norm.weight,
            num_heads,
            num_kv_heads,
            head_dim,
            eps=eps,
            is_kv_shared_layer=False,
        )

        cos_sim_q = compute_cosine_similarity(q_ref, out_q)
        cos_sim_k = compute_cosine_similarity(k_ref, out_k)
        cos_sim_v = compute_cosine_similarity(v_ref, out_v)

        assert cos_sim_q >= 0.99999, f"Q cosine similarity {cos_sim_q} < 0.99999"
        assert cos_sim_k >= 0.99999, f"K cosine similarity {cos_sim_k} < 0.99999"
        assert cos_sim_v >= 0.99999, f"V cosine similarity {cos_sim_v} < 0.99999"

        # Verify max absolute diff is bounded within BF16 numerical precision
        assert (q_ref - out_q).abs().max().item() < 0.05
        assert (k_ref - out_k).abs().max().item() < 0.05
        assert (v_ref - out_v).abs().max().item() < 0.05

    def test_fused_qkv_norm_rope_kv_shared(self):
        """Test fused QKV for KV-shared layers (skips K/V norm & RoPE)."""
        M = 4
        num_heads = 8
        num_kv_heads = 2
        head_dim = 256
        eps = 1e-6
        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim
        total_dim = q_size + 2 * kv_size

        qkv = torch.randn(M, total_dim, dtype=torch.bfloat16, device="cuda")
        positions = torch.tensor([5, 15, 25, 35], dtype=torch.int64, device="cuda")

        q_norm = RMSNorm(head_dim, eps=eps, dtype=torch.bfloat16).to("cuda")
        k_norm = RMSNorm(head_dim, eps=eps, dtype=torch.bfloat16).to("cuda")
        rope = get_rope(
            head_dim,
            max_position=8192,
            is_neox_style=True,
            rope_parameters={"rope_theta": 10000.0},
            dtype=torch.bfloat16,
        )

        # Reference PyTorch implementation for KV-shared
        q_ref, k_ref, v_ref = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q_ref = q_norm(q_ref.unflatten(-1, (num_heads, head_dim))).flatten(-2, -1)
        q_ref = rope(positions, q_ref, k_ref)[0]

        out_q, out_k, out_v = fused_qkv_norm_rope(
            qkv,
            positions,
            rope.cos_sin_cache,
            q_norm.weight,
            k_norm.weight,
            num_heads,
            num_kv_heads,
            head_dim,
            eps=eps,
            is_kv_shared_layer=True,
        )

        cos_sim_q = compute_cosine_similarity(q_ref, out_q)
        assert cos_sim_q >= 0.99999
        # K and V must match input slices exactly
        torch.testing.assert_close(out_k, k_ref, atol=0.0, rtol=0.0)
        torch.testing.assert_close(out_v, v_ref, atol=0.0, rtol=0.0)

    def test_weightless_v_norm_invariant(self):
        """Assert V-norm is strictly weightless (has_weight=False)."""
        M = 2
        num_heads = 4
        num_kv_heads = 2
        head_dim = 128
        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim
        total_dim = q_size + 2 * kv_size

        qkv = torch.randn(M, total_dim, dtype=torch.bfloat16, device="cuda")
        positions = torch.zeros(M, dtype=torch.int64, device="cuda")
        cos_sin_cache = torch.ones(16, head_dim, dtype=torch.bfloat16, device="cuda")
        wq = torch.randn(head_dim, dtype=torch.bfloat16, device="cuda")
        wk = torch.randn(head_dim, dtype=torch.bfloat16, device="cuda")

        _, _, out_v = fused_qkv_norm_rope(
            qkv,
            positions,
            cos_sin_cache,
            wq,
            wk,
            num_heads,
            num_kv_heads,
            head_dim,
            is_kv_shared_layer=False,
        )

        v_raw = qkv[:, q_size + kv_size : q_size + 2 * kv_size]
        v_norm_ref = RMSNorm(
            head_dim, eps=1e-6, has_weight=False, dtype=torch.bfloat16
        ).to("cuda")
        v_ref = v_norm_ref(v_raw.unflatten(-1, (num_kv_heads, head_dim))).flatten(
            -2, -1
        )

        diff = (v_ref - out_v).abs().max().item()
        cos_sim = compute_cosine_similarity(v_ref, out_v)
        assert cos_sim >= 0.99999
        assert diff < 0.01

    def test_fused_post_attn_add_pre_ff_norm(self):
        """Test Kernel 2: post_attn_norm + residual add + pre_ff_norm."""
        M = 8
        H = 2048
        eps = 1e-6

        attn_out = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
        residual = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
        post_attn_norm = RMSNorm(H, eps=eps, dtype=torch.bfloat16).to("cuda")
        pre_ff_norm = RMSNorm(H, eps=eps, dtype=torch.bfloat16).to("cuda")

        # Reference
        ref_normed_attn = post_attn_norm(attn_out)
        ref_res = ref_normed_attn + residual
        ref_pre_ff = pre_ff_norm(ref_res)

        # Fused
        out_pre_ff, out_res = fused_post_attn_add_pre_ff_norm(
            attn_out, residual, post_attn_norm.weight, pre_ff_norm.weight, eps=eps
        )

        cos_sim_res = compute_cosine_similarity(ref_res, out_res)
        cos_sim_pre = compute_cosine_similarity(ref_pre_ff, out_pre_ff)

        assert cos_sim_res >= 0.99999
        assert cos_sim_pre >= 0.99999
        assert (ref_res - out_res).abs().max().item() < 0.05
        assert (ref_pre_ff - out_pre_ff).abs().max().item() < 0.05

    def test_fused_mlp_ple_epilogue_without_ple(self):
        """Test Kernel 3: Dense path without PLE."""
        M = 4
        H = 1536
        eps = 1e-6

        mlp_out = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
        residual = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
        post_ff_norm = RMSNorm(H, eps=eps, dtype=torch.bfloat16).to("cuda")
        layer_scalar = torch.tensor([1.25], dtype=torch.float32, device="cuda")

        # Reference
        ref_h = post_ff_norm(mlp_out) + residual
        ref_out = ref_h * layer_scalar

        # Fused
        fused_out = fused_mlp_ple_epilogue(
            mlp_out,
            residual,
            post_ff_norm.weight,
            per_layer_input=None,
            layer_scalar=layer_scalar,
            eps=eps,
        )

        cos_sim = compute_cosine_similarity(ref_out, fused_out)
        assert cos_sim >= 0.99999
        assert (ref_out - fused_out).abs().max().item() < 0.05

    def test_fused_mlp_ple_epilogue_with_ple(self):
        """Test Kernel 3: Dense path with PLE multi-SM pipeline."""
        M = 4
        H = 1536
        ple_dim = 256
        eps = 1e-6

        mlp_out = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
        residual = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
        post_ff_norm = RMSNorm(H, eps=eps, dtype=torch.bfloat16).to("cuda")
        post_ple_norm = RMSNorm(H, eps=eps, dtype=torch.bfloat16).to("cuda")
        layer_scalar = torch.tensor([0.95], dtype=torch.float32, device="cuda")

        per_layer_input = torch.randn(M, ple_dim, dtype=torch.bfloat16, device="cuda")
        gate_linear = nn.Linear(
            H, ple_dim, bias=False, dtype=torch.bfloat16, device="cuda"
        )
        proj_linear = nn.Linear(
            ple_dim, H, bias=False, dtype=torch.bfloat16, device="cuda"
        )

        # Reference
        ref_h = post_ff_norm(mlp_out) + residual
        ref_gate = F.gelu(gate_linear(ref_h), approximate="tanh") * per_layer_input
        ref_proj = proj_linear(ref_gate)
        ref_ple_norm = post_ple_norm(ref_proj)
        ref_out = (ref_h + ref_ple_norm) * layer_scalar

        # Fused
        fused_out = fused_mlp_ple_epilogue(
            mlp_out,
            residual,
            post_ff_norm.weight,
            per_layer_input=per_layer_input,
            per_layer_input_gate=gate_linear,
            per_layer_projection=proj_linear,
            post_ple_weight=post_ple_norm.weight,
            layer_scalar=layer_scalar,
            eps=eps,
        )

        cos_sim = compute_cosine_similarity(ref_out, fused_out)
        assert cos_sim >= 0.99999
        assert (ref_out - fused_out).abs().max().item() < 0.05

    def test_fused_moe_kernels(self):
        """Test Kernel 3A & 3B: MoE split and combine epilogue."""
        M = 4
        H = 2048
        eps = 1e-6

        mlp_out = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
        residual = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
        post_ff_1_norm = RMSNorm(H, eps=eps, dtype=torch.bfloat16).to("cuda")
        pre_ff_2_norm = RMSNorm(H, eps=eps, dtype=torch.bfloat16).to("cuda")
        post_ff_2_norm = RMSNorm(H, eps=eps, dtype=torch.bfloat16).to("cuda")
        post_ff_norm = RMSNorm(H, eps=eps, dtype=torch.bfloat16).to("cuda")
        layer_scalar = torch.tensor([1.1], dtype=torch.float32, device="cuda")

        # Kernel 3A: Prenorm
        ref_h1 = post_ff_1_norm(mlp_out)
        ref_moe_in = pre_ff_2_norm(residual)

        fused_h1, fused_moe_in = fused_mlp_norm_and_moe_prenorm(
            mlp_out, residual, post_ff_1_norm.weight, pre_ff_2_norm.weight, eps=eps
        )

        assert compute_cosine_similarity(ref_h1, fused_h1) >= 0.99999
        assert compute_cosine_similarity(ref_moe_in, fused_moe_in) >= 0.99999

        # Simulated MoE output
        moe_out = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")

        # Kernel 3B: Combine
        ref_post_2 = post_ff_2_norm(moe_out)
        ref_comb = ref_h1 + ref_post_2
        ref_out = (post_ff_norm(ref_comb) + residual) * layer_scalar

        fused_out = fused_moe_combine_norm_ple_epilogue(
            fused_h1,
            moe_out,
            residual,
            post_ff_2_norm.weight,
            post_ff_norm.weight,
            per_layer_input=None,
            layer_scalar=layer_scalar,
            eps=eps,
        )

        cos_sim_comb = compute_cosine_similarity(ref_out, fused_out)
        assert cos_sim_comb >= 0.99999
        assert (ref_out - fused_out).abs().max().item() < 0.05

    def test_long_context_rope(self):
        """Assert zero NaN/inf and precision at p = 131,072 with theta = 1,000,000."""
        head_dim = 256
        num_heads = 8
        num_kv_heads = 2
        p = 131072
        theta = 1000000.0

        rope = get_rope(
            head_dim,
            max_position=p + 16,
            is_neox_style=True,
            rope_parameters={"rope_theta": theta},
            dtype=torch.bfloat16,
        )

        qkv = torch.randn(
            1,
            (num_heads + 2 * num_kv_heads) * head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        positions = torch.tensor([p], dtype=torch.int64, device="cuda")
        wq = torch.ones(head_dim, dtype=torch.bfloat16, device="cuda")
        wk = torch.ones(head_dim, dtype=torch.bfloat16, device="cuda")

        out_q, out_k, _ = fused_qkv_norm_rope(
            qkv,
            positions,
            rope.cos_sin_cache,
            wq,
            wk,
            num_heads,
            num_kv_heads,
            head_dim,
            is_kv_shared_layer=False,
        )

        assert not torch.isnan(out_q).any()
        assert not torch.isinf(out_q).any()
        assert not torch.isnan(out_k).any()
        assert not torch.isinf(out_k).any()

        # Compare with reference
        q_raw = qkv[:, : num_heads * head_dim]
        k_raw = qkv[:, num_heads * head_dim : (num_heads + num_kv_heads) * head_dim]
        q_norm = RMSNorm(head_dim, eps=1e-6, dtype=torch.bfloat16).to("cuda")
        k_norm = RMSNorm(head_dim, eps=1e-6, dtype=torch.bfloat16).to("cuda")
        q_ref = q_norm(q_raw.unflatten(-1, (num_heads, head_dim))).flatten(-2, -1)
        k_ref = k_norm(k_raw.unflatten(-1, (num_kv_heads, head_dim))).flatten(-2, -1)
        q_ref, k_ref = rope(positions, q_ref, k_ref)

        assert compute_cosine_similarity(q_ref, out_q) >= 0.99999
        assert compute_cosine_similarity(k_ref, out_k) >= 0.99999

    def test_gemma4_decoder_layer_parity(self, dist_init):
        """End-to-end parity test for Gemma4DecoderLayer (PyTorch eager
        reference vs fused ops)."""
        from types import SimpleNamespace

        from vllm.model_executor.models.gemma4 import Gemma4DecoderLayer

        config = SimpleNamespace(
            hidden_size=512,
            head_dim=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=1024,
            rms_norm_eps=1e-6,
            max_position_embeddings=4096,
            hidden_activation="gelu_pytorch_tanh",
            attention_bias=False,
            layer_types=["full_attention"],
            rope_parameters={"full_attention": {"rope_theta": 10000.0}},
            hidden_size_per_layer_input=128,
            enable_moe_block=False,
            num_hidden_layers=1,
            num_kv_shared_layers=0,
        )

        class DummyAttn(nn.Module):
            def forward(self, q, k, v, *args, **kwargs):
                return q

        layer = (
            Gemma4DecoderLayer(config, prefix="model.layers.0")
            .to("cuda")
            .to(torch.bfloat16)
        )
        layer.self_attn.attn = DummyAttn()

        # Initialize uninitialized parameter weights to avoid NaNs
        for p in layer.parameters():
            p.data.normal_(0.0, 0.02)
        layer.input_layernorm.weight.data.fill_(1.0)
        layer.post_attention_layernorm.weight.data.fill_(1.0)
        layer.pre_feedforward_layernorm.weight.data.fill_(1.0)
        layer.post_feedforward_layernorm.weight.data.fill_(1.0)
        layer.post_per_layer_input_norm.weight.data.fill_(1.0)
        layer.layer_scalar.data.fill_(1.0)

        M = 2
        positions = torch.tensor([2, 5], dtype=torch.int64, device="cuda")
        hidden_states = torch.randn(
            M, config.hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        per_layer_input = torch.randn(
            M, config.hidden_size_per_layer_input, dtype=torch.bfloat16, device="cuda"
        )

        # Reference unfused computation
        res = hidden_states
        hidden_states_norm = layer.input_layernorm(res)
        attn_out = layer.self_attn(positions, hidden_states_norm)
        post_attn = layer.post_attention_layernorm(attn_out)
        res = post_attn + res
        pre_ff = layer.pre_feedforward_layernorm(res)
        mlp_out = layer.mlp(pre_ff)
        post_ff = layer.post_feedforward_layernorm(mlp_out)
        res = post_ff + res
        if per_layer_input is not None and layer.per_layer_input_gate is not None:
            gate = layer.per_layer_input_gate(res)
            gate = torch.nn.functional.gelu(gate, approximate="tanh")
            gated = gate * per_layer_input
            ple = layer.post_per_layer_input_norm(layer.per_layer_projection(gated))
            res = res + ple
        out_unfused = res * layer.layer_scalar

        # Run fused production decoder layer
        out_fused, _ = layer(
            positions, hidden_states, residual=None, per_layer_input=per_layer_input
        )

        cos_sim = compute_cosine_similarity(out_unfused, out_fused)
        assert cos_sim >= 0.99999, (
            f"Decoder layer cosine similarity {cos_sim} < 0.99999"
        )
        assert (out_unfused - out_fused).abs().max().item() < 0.08

    def test_torch_compile_fullgraph(self):
        """Verify fused custom ops run under torch.compile(fullgraph=True)
        with zero graph breaks."""
        M = 4
        H = 512
        eps = 1e-6

        @torch.compile(fullgraph=True)
        def compiled_fused_block(attn_out, residual, post_w, pre_w, mlp_w, scalar):
            pre_ff, new_res = torch.ops.vllm.gemma4_fused_post_attn_add_pre_ff_norm(
                attn_out, residual, post_w, pre_w, eps
            )
            # Simulated MLP
            mlp_out = pre_ff * 0.5
            out = torch.ops.vllm.gemma4_fused_post_ff_norm_add_scalar(
                mlp_out, new_res, mlp_w, scalar, eps
            )
            return out

        attn_out = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
        residual = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
        post_w = torch.ones(H, dtype=torch.bfloat16, device="cuda")
        pre_w = torch.ones(H, dtype=torch.bfloat16, device="cuda")
        mlp_w = torch.ones(H, dtype=torch.bfloat16, device="cuda")
        scalar = torch.tensor([1.0], dtype=torch.float32, device="cuda")

        compiled_out = compiled_fused_block(
            attn_out, residual, post_w, pre_w, mlp_w, scalar
        )
        assert compiled_out.shape == (M, H)
        assert not torch.isnan(compiled_out).any()

    def test_fused_qkv_norm_rope_q_only(self):
        """Test fused Q-only norm + RoPE for KV-shared layers with sliced GEMM."""
        M = 4
        num_heads = 8
        num_kv_heads = 2
        head_dim = 256
        eps = 1e-6
        q_size = num_heads * head_dim

        q = torch.randn(M, q_size, dtype=torch.bfloat16, device="cuda")
        positions = torch.tensor([1, 11, 21, 31], dtype=torch.int64, device="cuda")

        q_norm = RMSNorm(head_dim, eps=eps, dtype=torch.bfloat16).to("cuda")
        k_norm = RMSNorm(head_dim, eps=eps, dtype=torch.bfloat16).to("cuda")
        rope = get_rope(
            head_dim,
            max_position=8192,
            is_neox_style=True,
            rope_parameters={"rope_theta": 10000.0},
            dtype=torch.bfloat16,
        )

        dummy_k = torch.zeros(
            M, num_kv_heads * head_dim, dtype=torch.bfloat16, device="cuda"
        )
        q_ref = q_norm(q.unflatten(-1, (num_heads, head_dim))).flatten(-2, -1)
        q_ref = rope(positions, q_ref, dummy_k)[0]

        out_q, out_k, out_v = fused_qkv_norm_rope(
            q,
            positions,
            rope.cos_sin_cache,
            q_norm.weight,
            k_norm.weight,
            num_heads,
            num_kv_heads,
            head_dim,
            eps=eps,
            is_kv_shared_layer=True,
        )

        cos_sim_q = compute_cosine_similarity(q_ref, out_q)
        assert cos_sim_q >= 0.99999, f"Q cosine similarity {cos_sim_q} < 0.99999"
        assert (q_ref - out_q).abs().max().item() < 0.05
        assert out_k.numel() == 0 or out_k.shape[-1] == 0
        assert out_v.numel() == 0 or out_v.shape[-1] == 0

    def test_fused_qkv_norm_rope_k_eq_v(self):
        """Test fused QK projection with k_eq_v=True (deduplicated KV)."""
        M = 4
        num_heads = 8
        num_kv_heads = 2
        head_dim = 256
        eps = 1e-6
        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim

        qk = torch.randn(M, q_size + kv_size, dtype=torch.bfloat16, device="cuda")
        positions = torch.tensor([3, 13, 23, 33], dtype=torch.int64, device="cuda")

        q_norm = RMSNorm(head_dim, eps=eps, dtype=torch.bfloat16).to("cuda")
        k_norm = RMSNorm(head_dim, eps=eps, dtype=torch.bfloat16).to("cuda")
        v_norm = RMSNorm(head_dim, eps=eps, has_weight=False, dtype=torch.bfloat16).to(
            "cuda"
        )
        rope = get_rope(
            head_dim,
            max_position=8192,
            is_neox_style=True,
            rope_parameters={"rope_theta": 10000.0},
            dtype=torch.bfloat16,
        )

        q_raw, k_raw = qk.split([q_size, kv_size], dim=-1)
        q_ref = q_norm(q_raw.unflatten(-1, (num_heads, head_dim))).flatten(-2, -1)
        k_ref = k_norm(k_raw.unflatten(-1, (num_kv_heads, head_dim))).flatten(-2, -1)
        q_ref, k_ref = rope(positions, q_ref, k_ref)
        v_ref = v_norm(k_raw.unflatten(-1, (num_kv_heads, head_dim))).flatten(-2, -1)

        out_q, out_k, out_v = fused_qkv_norm_rope(
            qk,
            positions,
            rope.cos_sin_cache,
            q_norm.weight,
            k_norm.weight,
            num_heads,
            num_kv_heads,
            head_dim,
            eps=eps,
            is_kv_shared_layer=False,
            is_k_eq_v=True,
        )

        cos_sim_q = compute_cosine_similarity(q_ref, out_q)
        cos_sim_k = compute_cosine_similarity(k_ref, out_k)
        cos_sim_v = compute_cosine_similarity(v_ref, out_v)

        assert cos_sim_q >= 0.99999, f"Q cosine similarity {cos_sim_q} < 0.99999"
        assert cos_sim_k >= 0.99999, f"K cosine similarity {cos_sim_k} < 0.99999"
        assert cos_sim_v >= 0.99999, f"V cosine similarity {cos_sim_v} < 0.99999"
        assert (q_ref - out_q).abs().max().item() < 0.05
        assert (k_ref - out_k).abs().max().item() < 0.05
        assert (v_ref - out_v).abs().max().item() < 0.05

    def test_fused_post_attn_add_moe_prenorms(self):
        """Test Kernel 4: post_attn_norm, residual add, pre_ff_norm,
        pre_moe_norm, and router_norm & scale."""
        M = 4
        H = 2048
        eps = 1e-6

        attn_out = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
        residual = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
        post_attn_norm = RMSNorm(H, eps=eps, dtype=torch.bfloat16).to("cuda")
        pre_ff_norm = RMSNorm(H, eps=eps, dtype=torch.bfloat16).to("cuda")
        pre_moe_norm = RMSNorm(H, eps=eps, dtype=torch.bfloat16).to("cuda")
        router_scale = torch.randn(H, dtype=torch.bfloat16, device="cuda")

        # Reference
        ref_res = post_attn_norm(attn_out) + residual
        ref_pre_ff = pre_ff_norm(ref_res)
        ref_pre_moe = pre_moe_norm(ref_res)
        ref_rstd = torch.rsqrt(ref_res.float().pow(2).mean(dim=-1, keepdim=True) + eps)
        ref_router = (ref_res.float() * ref_rstd * router_scale).to(torch.bfloat16)

        out_pre_ff, out_pre_moe, out_router, out_res = fused_post_attn_add_moe_prenorms(
            attn_out,
            residual,
            post_attn_norm.weight,
            pre_ff_norm.weight,
            pre_moe_norm.weight,
            router_scale,
            eps=eps,
        )

        assert compute_cosine_similarity(ref_res, out_res) >= 0.99999
        assert compute_cosine_similarity(ref_pre_ff, out_pre_ff) >= 0.99999
        assert compute_cosine_similarity(ref_pre_moe, out_pre_moe) >= 0.99999
        assert compute_cosine_similarity(ref_router, out_router) >= 0.99999
        assert (ref_res - out_res).abs().max().item() < 0.05
        assert (ref_pre_ff - out_pre_ff).abs().max().item() < 0.05
        assert (ref_pre_moe - out_pre_moe).abs().max().item() < 0.05
        assert (ref_router - out_router).abs().max().item() < 0.05

    def test_cross_layer_norm_fusion(self):
        """Test cross-layer norm fusion: post_ff_norm + add + scalar +
        next_input_norm."""
        M = 4
        H = 2048
        eps = 1e-6

        mlp_out = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
        residual = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
        post_ff_norm = RMSNorm(H, eps=eps, dtype=torch.bfloat16).to("cuda")
        next_input_norm = RMSNorm(H, eps=eps, dtype=torch.bfloat16).to("cuda")
        layer_scalar = torch.tensor([1.15], dtype=torch.float32, device="cuda")

        # Reference PyTorch implementation:
        ref_h = post_ff_norm(mlp_out) + residual
        ref_res_next = (ref_h.float() * layer_scalar).to(torch.bfloat16)
        ref_normed_next = next_input_norm(ref_res_next)

        # Fused cross-layer epilogue:
        normed_next, res_next = fused_mlp_ple_epilogue(
            mlp_out,
            residual,
            post_ff_norm.weight,
            layer_scalar=layer_scalar,
            next_norm_weight=next_input_norm.weight,
            eps=eps,
        )

        assert compute_cosine_similarity(ref_res_next, res_next) >= 0.99999
        assert compute_cosine_similarity(ref_normed_next, normed_next) >= 0.99999
        assert (ref_res_next - res_next).abs().max().item() < 0.05
        assert (ref_normed_next - normed_next).abs().max().item() < 0.05

    @pytest.mark.parametrize("M", [1, 2, 4, 8, 16])
    def test_2kernel_fused_ple_pipeline(self, M: int):
        """Test 2-kernel L2-resident fused PLE pipeline for M <= 16, H == 1536,
        PLE_DIM == 256."""
        H = 1536
        ple_dim = 256
        eps = 1e-6

        mlp_out = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
        residual = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
        ple_in = torch.randn(M, ple_dim, dtype=torch.bfloat16, device="cuda")

        post_ff_norm = RMSNorm(H, eps=eps, dtype=torch.bfloat16).to("cuda")
        gate_linear = nn.Linear(H, ple_dim, bias=False).to(
            device="cuda", dtype=torch.bfloat16
        )
        proj_linear = nn.Linear(ple_dim, H, bias=False).to(
            device="cuda", dtype=torch.bfloat16
        )
        post_ple_norm = RMSNorm(H, eps=eps, dtype=torch.bfloat16).to("cuda")
        next_input_norm = RMSNorm(H, eps=eps, dtype=torch.bfloat16).to("cuda")
        layer_scalar = torch.tensor([1.25], dtype=torch.float32, device="cuda")

        # Reference PyTorch execution
        ref_h = post_ff_norm(mlp_out) + residual
        ref_gate = F.gelu(gate_linear(ref_h), approximate="tanh") * ple_in
        ref_proj = proj_linear(ref_gate)
        ref_res = (post_ple_norm(ref_proj) + ref_h) * layer_scalar
        ref_normed = next_input_norm(ref_res)

        # Fused execution with next norm
        normed_out, res_out = fused_mlp_ple_epilogue(
            mlp_out,
            residual,
            post_ff_norm,
            per_layer_input=ple_in,
            per_layer_input_gate=gate_linear,
            per_layer_projection=proj_linear,
            post_ple_weight=post_ple_norm,
            layer_scalar=layer_scalar,
            next_norm_weight=next_input_norm,
            eps=eps,
        )

        assert compute_cosine_similarity(ref_res, res_out) >= 0.99999
        assert compute_cosine_similarity(ref_normed, normed_out) >= 0.99998
        assert (ref_res - res_out).abs().max().item() < 0.08
        assert (ref_normed - normed_out).abs().max().item() < 0.08

        # Fused execution without next norm
        res_standalone = fused_mlp_ple_epilogue(
            mlp_out,
            residual,
            post_ff_norm,
            per_layer_input=ple_in,
            per_layer_input_gate=gate_linear,
            per_layer_projection=proj_linear,
            post_ple_weight=post_ple_norm,
            layer_scalar=layer_scalar,
            eps=eps,
        )
        assert compute_cosine_similarity(ref_res, res_standalone) >= 0.99999
        assert (ref_res - res_standalone).abs().max().item() < 0.08

    def test_fused_ple_model_proj_norm_combine(self):
        """Test fused top-level PLE model projection norm + combine."""
        M = 4
        num_layers = 26
        P = 256
        H = 1536
        eps = 1e-6

        model_proj = torch.randn(M, num_layers * P, dtype=torch.bfloat16, device="cuda")
        embed_ple = torch.randn(M, num_layers, P, dtype=torch.bfloat16, device="cuda")
        norm = RMSNorm(P, eps=eps, dtype=torch.bfloat16).to("cuda")
        proj_scale = H**-0.5
        input_scale = 2.0**-0.5

        # Reference with embed_ple
        proj_3d = model_proj.reshape(M, num_layers, P)
        ref_scaled = proj_3d * proj_scale
        ref_norm = norm(ref_scaled)
        ref_combined = ((ref_norm.float() + embed_ple.float()) * input_scale).to(
            torch.bfloat16
        )

        out_combined = fused_ple_model_proj_norm_combine(
            model_proj,
            norm,
            embed_ple=embed_ple,
            num_layers=num_layers,
            proj_scale=proj_scale,
            input_scale=input_scale,
            eps=eps,
        )

        assert compute_cosine_similarity(ref_combined, out_combined) >= 0.99999
        assert (ref_combined - out_combined).abs().max().item() < 0.05

        # Reference without embed_ple
        out_no_embed = fused_ple_model_proj_norm_combine(
            model_proj,
            norm,
            embed_ple=None,
            num_layers=num_layers,
            proj_scale=proj_scale,
            input_scale=input_scale,
            eps=eps,
        )
        assert compute_cosine_similarity(ref_norm, out_no_embed) >= 0.99999
        assert (ref_norm - out_no_embed).abs().max().item() < 0.05

    @pytest.mark.parametrize("H", [2560, 3840, 5120])
    def test_large_hidden_warps_and_e4b_ple(self, H: int):
        """Test num_warps=16 for H >= 3840 and H=2560 2-kernel PLE pipeline."""
        M = 4
        eps = 1e-6
        mlp_out = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
        residual = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
        post_ff_norm = RMSNorm(H, eps=eps, dtype=torch.bfloat16).to("cuda")
        next_norm = RMSNorm(H, eps=eps, dtype=torch.bfloat16).to("cuda")
        layer_scalar = torch.tensor([1.1], dtype=torch.float32, device="cuda")

        if H == 2560:
            ple_dim = 256
            ple_in = torch.randn(M, ple_dim, dtype=torch.bfloat16, device="cuda")
            gate_linear = nn.Linear(H, ple_dim, bias=False).to(
                device="cuda", dtype=torch.bfloat16
            )
            proj_linear = nn.Linear(ple_dim, H, bias=False).to(
                device="cuda", dtype=torch.bfloat16
            )
            post_ple_norm = RMSNorm(H, eps=eps, dtype=torch.bfloat16).to("cuda")

            ref_h = post_ff_norm(mlp_out) + residual
            ref_gate = F.gelu(gate_linear(ref_h), approximate="tanh") * ple_in
            ref_proj = proj_linear(ref_gate)
            ref_res = (post_ple_norm(ref_proj) + ref_h) * layer_scalar
            ref_normed = next_norm(ref_res)

            normed_out, res_out = fused_mlp_ple_epilogue(
                mlp_out,
                residual,
                post_ff_norm,
                per_layer_input=ple_in,
                per_layer_input_gate=gate_linear,
                per_layer_projection=proj_linear,
                post_ple_weight=post_ple_norm,
                layer_scalar=layer_scalar,
                next_norm_weight=next_norm,
                eps=eps,
            )
        else:
            ref_h = post_ff_norm(mlp_out) + residual
            ref_res = (ref_h.float() * layer_scalar).to(torch.bfloat16)
            ref_normed = next_norm(ref_res)
            normed_out, res_out = fused_mlp_ple_epilogue(
                mlp_out,
                residual,
                post_ff_norm,
                layer_scalar=layer_scalar,
                next_norm_weight=next_norm,
                eps=eps,
            )

        assert compute_cosine_similarity(ref_res, res_out) >= 0.99999
        assert compute_cosine_similarity(ref_normed, normed_out) >= 0.99999

    def test_moe_router_custom_op_and_diffusion_self_conditioning(self):
        """Test MoE router custom op under torch.compile and
        DiffusionGemmaSelfConditioning."""
        from vllm.model_executor.models.diffusion_gemma import (
            DiffusionGemmaSelfConditioning,
        )
        from vllm.model_executor.models.gemma4 import (
            gemma4_fused_routing_kernel_triton,
            gemma4_routing_function_torch,
        )

        T, E, K = 8, 128, 8
        gating = torch.randn(T, E, dtype=torch.float32, device="cuda")
        scale = torch.rand(E, dtype=torch.bfloat16, device="cuda") + 0.5

        ref_w, ref_ids = gemma4_routing_function_torch(gating, K, scale)

        @torch.compile(fullgraph=True)
        def compiled_route(g, s):
            return gemma4_fused_routing_kernel_triton(g, 8, s)

        out_w, out_ids = compiled_route(gating, scale)
        assert torch.equal(ref_ids, out_ids)
        assert torch.allclose(ref_w, out_w, atol=1e-4, rtol=1e-4)

        # Test DiffusionGemmaSelfConditioning merged gate_up parity
        sc = DiffusionGemmaSelfConditioning(
            hidden_size=1536, self_conditioning_size=6144
        ).to(device="cuda", dtype=torch.bfloat16)
        inp = torch.randn(64, 1536, dtype=torch.bfloat16, device="cuda")
        soft = torch.randn(64, 1536, dtype=torch.bfloat16, device="cuda")

        x = sc.pre_norm(soft)
        ref_sc = sc.post_norm(
            inp
            + sc.down_proj(F.gelu(sc.gate_proj(x), approximate="tanh") * sc.up_proj(x))
        )
        out_sc = sc(inp, soft)
        assert compute_cosine_similarity(ref_sc, out_sc) >= 0.99999
        assert (ref_sc - out_sc).abs().max().item() < 0.02

    def test_empty_tensor_guards(self):
        """Test M == 0 empty tensor handling across all fused operations."""
        device = "cuda"
        dtype = torch.bfloat16

        # 1. Fused QKV Norm + RoPE empty guards
        head_dim = 256
        num_heads = 6
        num_kv_heads = 1
        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim
        qkv_empty = torch.empty(0, q_size + 2 * kv_size, device=device, dtype=dtype)
        positions_empty = torch.empty(0, device=device, dtype=torch.int64)
        cos_sin_cache = torch.randn(1024, head_dim, device=device, dtype=dtype)
        q_w = torch.randn(head_dim, device=device, dtype=dtype)
        k_w = torch.randn(head_dim, device=device, dtype=dtype)

        q, k, v = fused_qkv_norm_rope(
            qkv_empty,
            positions_empty,
            cos_sin_cache,
            q_w,
            k_w,
            num_heads,
            num_kv_heads,
            head_dim,
        )
        assert q.shape == (0, q_size)
        assert k.shape == (0, kv_size)
        assert v.shape == (0, kv_size)

        # KV-shared empty guard
        q_only_empty = torch.empty(0, q_size, device=device, dtype=dtype)
        q_shared, k_shared, v_shared = fused_qkv_norm_rope(
            q_only_empty,
            positions_empty,
            cos_sin_cache,
            q_w,
            k_w,
            num_heads,
            num_kv_heads,
            head_dim,
            is_kv_shared_layer=True,
        )
        assert q_shared.shape == (0, q_size)
        assert k_shared.shape == (0, kv_size)
        assert v_shared.shape == (0, kv_size)

        # 2. Fused Post-Attn Add Pre-FF Norm empty guard
        H = 1536
        attn_empty = torch.empty(0, H, device=device, dtype=dtype)
        res_empty = torch.empty(0, H, device=device, dtype=dtype)
        post_w = torch.randn(H, device=device, dtype=dtype)
        pre_w = torch.randn(H, device=device, dtype=dtype)
        out_pre_ff, out_res = fused_post_attn_add_pre_ff_norm(
            attn_empty, res_empty, post_w, pre_w
        )
        assert out_pre_ff.shape == (0, H)
        assert out_res.shape == (0, H)

        # 3. Fused MLP PLE Epilogue empty guard
        mlp_empty = torch.empty(0, H, device=device, dtype=dtype)
        post_ff_w = torch.randn(H, device=device, dtype=dtype)
        next_w = torch.randn(H, device=device, dtype=dtype)
        epilogue_res = fused_mlp_ple_epilogue(mlp_empty, res_empty, post_ff_w)
        assert isinstance(epilogue_res, torch.Tensor)
        assert epilogue_res.shape == (0, H)

        normed, res = fused_mlp_ple_epilogue(  # type: ignore[misc]
            mlp_empty, res_empty, post_ff_w, next_norm_weight=next_w
        )
        assert normed.shape == (0, H)
        assert res.shape == (0, H)

        # 4. Fused Post-Attn Add MoE Prenorms empty guard
        H_moe = 2816
        attn_moe_empty = torch.empty(0, H_moe, device=device, dtype=dtype)
        res_moe_empty = torch.empty(0, H_moe, device=device, dtype=dtype)
        post_moe_w = torch.randn(H_moe, device=device, dtype=dtype)
        pre_ff_moe_w = torch.randn(H_moe, device=device, dtype=dtype)
        pre_moe_w = torch.randn(H_moe, device=device, dtype=dtype)
        router_scale = torch.tensor(1.0, device=device, dtype=torch.float32)
        p_ff, m_in, r_in, r_out = fused_post_attn_add_moe_prenorms(
            attn_moe_empty,
            res_moe_empty,
            post_moe_w,
            pre_ff_moe_w,
            pre_moe_w,
            router_scale,
        )
        assert p_ff.shape == (0, H_moe)
        assert m_in.shape == (0, H_moe)
        assert r_in.shape == (0, H_moe)
        assert r_out.shape == (0, H_moe)

        # 5. Fused MoE Combine Norm PLE Epilogue empty guard
        h1_empty = torch.empty(0, H_moe, device=device, dtype=dtype)
        moe_empty = torch.empty(0, H_moe, device=device, dtype=dtype)
        w_post2 = torch.randn(H_moe, device=device, dtype=dtype)
        w_post_ff = torch.randn(H_moe, device=device, dtype=dtype)
        moe_epilogue_res = fused_moe_combine_norm_ple_epilogue(
            h1_empty, moe_empty, res_moe_empty, w_post2, w_post_ff
        )
        assert isinstance(moe_epilogue_res, torch.Tensor)
        assert moe_epilogue_res.shape == (0, H_moe)

        # 6. Fused PLE Model Proj Norm Combine empty guard
        P = 256
        L = 35
        model_proj_empty = torch.empty(0, L, P, device=device, dtype=dtype)
        norm_w = torch.randn(P, device=device, dtype=dtype)
        embed_empty = torch.empty(0, device=device, dtype=dtype)
        scale = torch.tensor(1.0, device=device, dtype=torch.float32)
        ple_comb = fused_ple_model_proj_norm_combine(
            model_proj_empty, norm_w, embed_empty, scale, scale
        )
        assert ple_comb.shape == (0, L, P)
