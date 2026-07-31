# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for K3 math, vLLM attention registration, and TP partitioning.

The portable model maps vLLM's flattened token tensors to hidden states. These
tests guard the KDA equation and attention-residual equation at unit level, then
use a two-rank model construction test to catch registration or sharding drift.
"""

from __future__ import annotations

import inspect
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

import vllm.models.kimi_k3.portable as portable_kimi
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.models.interfaces import supports_pp
from vllm.models.kimi_k3.portable import (
    KimiDeltaAttention,
    KimiK3ForCausalLM,
    KimiMoE,
    MultiHeadLatentAttention,
)
from vllm.models.kimi_k3.portable.attention import CausalDepthwiseConv1d
from vllm.models.kimi_k3.portable.layers import AttentionResidual, RMSNorm, situ
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.v1.attention.backends.linear_attn import LinearAttentionBackend


def _tiny_config() -> KimiLinearConfig:
    return KimiLinearConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        hidden_act="situ",
        rms_norm_eps=1e-5,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        v_head_dim=4,
        mla_use_nope=True,
        mla_use_output_gate=True,
        num_experts=4,
        num_experts_per_token=2,
        num_shared_experts=1,
        moe_intermediate_size=16,
        routed_expert_hidden_size=12,
        first_k_dense_replace=1,
        latent_moe_use_norm=True,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        attn_res_block_size=1,
        linear_attn_config={
            "kda_layers": [1],
            "full_attn_layers": [2],
            "head_dim": 4,
            "num_heads": 4,
            "short_conv_kernel_size": 3,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
    )


def test_kda_recurrence_matches_direct_equation() -> None:
    torch.manual_seed(1)
    attention = KimiDeltaAttention.__new__(KimiDeltaAttention)
    nn.Module.__init__(attention)
    attention.head_dim = 4

    sequence_length, num_heads, head_dim = 3, 2, attention.head_dim
    query, key, value = (
        torch.randn(sequence_length, num_heads, head_dim) for _ in range(3)
    )
    gate = -torch.rand(sequence_length, num_heads, head_dim)
    beta = torch.rand(sequence_length, num_heads)
    initial_state = torch.randn(num_heads, head_dim, head_dim)

    actual, actual_state = attention._recurrent_kda(
        query,
        key,
        value,
        gate,
        beta,
        initial_state,
    )

    query = query.float()
    query = query * torch.rsqrt(query.square().sum(dim=-1, keepdim=True) + 1e-6)
    query = query * head_dim**-0.5
    key = key.float()
    key = key * torch.rsqrt(key.square().sum(dim=-1, keepdim=True) + 1e-6)
    state = initial_state.float()
    expected = []
    for token_idx in range(sequence_length):
        state = state * gate[token_idx].exp().unsqueeze(-1)
        prediction = torch.einsum("hk,hkv->hv", key[token_idx], state)
        delta = beta[token_idx, :, None] * (value[token_idx] - prediction)
        state = state + torch.einsum("hk,hv->hkv", key[token_idx], delta)
        expected.append(torch.einsum("hk,hkv->hv", query[token_idx], state))

    torch.testing.assert_close(actual, torch.stack(expected))
    torch.testing.assert_close(actual_state, state)


class _TupleLinear(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, hidden_size))

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, None]:
        return torch.nn.functional.linear(inputs, self.weight), None


def test_attention_residual_matches_direct_equation() -> None:
    torch.manual_seed(5)
    residual = AttentionResidual.__new__(AttentionResidual)
    nn.Module.__init__(residual)
    residual.norm = RMSNorm(4, 1e-5)
    residual.proj = _TupleLinear(4)
    prefix_sum = torch.randn(3, 4)
    block_residuals = torch.randn(3, 2, 4)

    actual = residual(prefix_sum, block_residuals)

    values = torch.cat((block_residuals, prefix_sum.unsqueeze(-2)), dim=-2)
    normalized = values.float() * torch.rsqrt(
        values.float().square().mean(dim=-1, keepdim=True) + residual.norm.eps
    )
    normalized = normalized * residual.norm.weight.float()
    scores = torch.nn.functional.linear(normalized, residual.proj.weight)
    expected = (scores.softmax(dim=-2) * values).sum(dim=-2)
    torch.testing.assert_close(actual, expected)


def test_situ_matches_direct_equation() -> None:
    gate = torch.tensor([-3.0, 0.5, 7.0])
    up = torch.tensor([-30.0, 2.0, 40.0])

    actual = situ(gate, up, beta=4.0, linear_beta=25.0)

    expected_gate = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
    expected_up = 25.0 * torch.tanh(up / 25.0)
    torch.testing.assert_close(actual, expected_gate * expected_up)


def test_short_convolution_carries_its_own_state() -> None:
    convolution = CausalDepthwiseConv1d.__new__(CausalDepthwiseConv1d)
    nn.Module.__init__(convolution)
    convolution.channels = 1
    convolution.kernel_size = 3
    convolution.weight = nn.Parameter(torch.tensor([[[1.0, 2.0, 3.0]]]))

    first_output, sconv_state = convolution(
        torch.tensor([[4.0], [5.0]]),
        torch.tensor([[1.0, 2.0]]),
    )
    second_output, sconv_state = convolution(
        torch.tensor([[6.0]]),
        sconv_state,
    )

    torch.testing.assert_close(
        first_output, torch.nn.functional.silu(torch.tensor([[17.0], [25.0]]))
    )
    torch.testing.assert_close(
        second_output,
        torch.nn.functional.silu(torch.tensor([[32.0]])),
    )
    torch.testing.assert_close(sconv_state, torch.tensor([[5.0, 6.0]]))


def test_model_has_no_public_cache_api() -> None:
    assert "cache" not in inspect.signature(KimiK3ForCausalLM.forward).parameters
    assert "use_cache" not in inspect.signature(KimiK3ForCausalLM.forward).parameters
    assert not supports_pp(KimiK3ForCausalLM)
    assert not hasattr(KimiK3ForCausalLM, "make_empty_intermediate_tensors")
    assert not hasattr(portable_kimi, "KimiK3Cache")
    assert not hasattr(portable_kimi, "MLACache")


def test_k3_checkpoint_names_map_to_model_parameters() -> None:
    mapped = KimiK3ForCausalLM.hf_to_vllm_mapper.apply_list(
        [
            "language_model.layers.0.self_attention_res_norm.weight",
            "language_model.model.embed_tokens.weight",
        ]
    )

    assert mapped == [
        "model.layers.0.self_attention_res.norm.weight",
        "model.embed_tokens.weight",
    ]


def _run_vllm_registration_test(
    rank: int,
    world_size: int,
    init_file: str,
) -> None:
    config = _tiny_config()
    vllm_config = VllmConfig()
    vllm_config.model_config = SimpleNamespace(
        dtype=torch.float32,
        head_dtype=None,
        hf_text_config=config,
        is_mm_prefix_lm=False,
        is_moe=True,
        max_model_len=128,
    )
    try:
        with set_current_vllm_config(vllm_config):
            init_distributed_environment(
                world_size=world_size,
                rank=rank,
                distributed_init_method=f"file://{init_file}",
                local_rank=rank,
                backend="gloo",
            )
            initialize_model_parallel(
                tensor_model_parallel_size=world_size,
                backend="gloo",
            )
            model = KimiK3ForCausalLM(vllm_config=vllm_config)
            kda = model.model.layers[0].self_attn
            mla = model.model.layers[1].self_attn

            assert isinstance(kda, KimiDeltaAttention)
            assert isinstance(mla, MultiHeadLatentAttention)
            assert isinstance(mla, Attention)
            assert mla.head_size_v == mla.qk_head_dim
            assert kda.get_attn_backend() is LinearAttentionBackend
            assert (
                vllm_config.compilation_config.static_forward_context[
                    "model.layers.0.self_attn"
                ]
                is kda
            )
            assert (
                vllm_config.compilation_config.static_forward_context[
                    "model.layers.1.self_attn"
                ]
                is mla
            )

            assert kda.q_proj.weight.shape == (8, 16)
            assert mla.q_b_proj.weight.shape == (12, 8)
            assert model.lm_head.weight.shape == (32, 16)
            parameter_names = set(dict(model.named_parameters()))
            assert "model.layers.1.block_sparse_moe.gate.weight" in parameter_names
            assert (
                "model.layers.1.block_sparse_moe.experts.0.w1.weight" in parameter_names
            )
            assert not model.load_weights(
                [
                    ("vision_tower.unused", torch.empty(0)),
                    ("mm_projector.unused", torch.empty(0)),
                ]
            )

            linear_config = _tiny_config()
            linear_config.hidden_act = "silu"
            linear_config.attn_res_block_size = None
            linear_config.routed_expert_hidden_size = None
            linear_config.mla_use_output_gate = False
            linear_config.linear_attn_config = {
                "kda_layers": [1],
                "full_attn_layers": [2],
                "head_dim": 4,
                "num_heads": 4,
                "short_conv_kernel_size": 3,
            }
            vllm_config.model_config.hf_text_config = linear_config
            linear_model = KimiK3ForCausalLM(
                vllm_config=vllm_config,
                prefix="linear",
            )
            linear_kda = linear_model.model.layers[0].self_attn
            linear_moe = linear_model.model.layers[1].block_sparse_moe

            assert isinstance(linear_moe, KimiMoE)
            assert linear_moe.routed_expert_down_proj is None
            assert linear_moe.routed_expert_up_proj is None
            assert not linear_kda.use_full_rank_gate
            assert not hasattr(linear_kda, "g_proj")
            assert linear_kda.g_a_proj.weight.shape == (4, 16)
            assert linear_kda.g_b_proj.weight.shape == (8, 4)
            sconv_shape, _ = linear_kda.get_state_shape()
            assert sconv_shape[0] * sconv_shape[1] == 3 * 8 * 2
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="The TP registration test requires torch.distributed with Gloo",
)
def test_vllm_attention_registration_and_tp_partitioning(tmp_path) -> None:
    init_file = str(tmp_path / "portable_kimi_tp_init")
    mp.spawn(
        _run_vllm_registration_test,
        args=(2, init_file),
        nprocs=2,
        join=True,
    )
    if os.path.exists(init_file):
        os.unlink(init_file)
