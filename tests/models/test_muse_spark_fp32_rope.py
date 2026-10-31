# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark rotates keys and queries with its FP32 RoPE tables.

Neither the DSpark drafter's context K/V precompute nor the unfused attention
path may replace a table with a lower-precision copy or round the rotation to
one.
"""

import json

import pytest
import torch

from vllm.config import (
    AttentionConfig,
    CompilationConfig,
    ModelConfig,
    ParallelConfig,
    SpeculativeConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.models.muse_spark.common.dspark import MuseSparkDSparkModel
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.attention.backends.registry import AttentionBackendEnum

HIDDEN, NUM_LAYERS, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM = 512, 3, 4, 2, 64
MAX_POSITION = 32768
NUM_TOKENS = 4096

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="GPU RoPE kernel"
)


@pytest.fixture
def drafter(tmp_path, dist_init):
    target_path, draft_path = tmp_path / "target", tmp_path / "draft"
    target_path.mkdir()
    draft_path.mkdir()
    target = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "vocab_size": 128,
        "torch_dtype": "bfloat16",
    }
    draft = {
        "model_type": "muse_spark_dspark",
        "hidden_size": HIDDEN,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": NUM_HEADS,
        "num_key_value_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": 128,
        "max_position_embeddings": MAX_POSITION,
        "norm_affine": False,
        "use_attn_o_gate": True,
        "use_attn_o_norm": True,
        "moe": {"is_enabled": False},
        "global_attn_cfg": [2048],
        "layer_types": ["sliding_attention"] * NUM_LAYERS,
        "target_hidden_size": 64,
        "target_layer_ids": [1, 3],
        "num_target_layers": 2,
        "mask_token_id": 127,
        "markov_rank": 8,
        "markov_head_type": "vanilla",
        "enable_confidence_head": False,
        "dflash_config": {
            "causal": True,
            "mask_token_id": 127,
            "target_layer_ids": [1, 3],
        },
        "torch_dtype": "bfloat16",
    }
    (target_path / "config.json").write_text(json.dumps(target))
    (draft_path / "config.json").write_text(json.dumps(draft))
    target_config = ModelConfig(model=str(target_path), tokenizer_mode="skip")
    vllm_config = VllmConfig(
        model_config=target_config,
        speculative_config=SpeculativeConfig(
            model=str(draft_path),
            method="dspark",
            num_speculative_tokens=7,
            target_model_config=target_config,
            target_parallel_config=ParallelConfig(),
        ),
        # Serving enables the RoPE custom op, whose eager forward casts the table.
        compilation_config=CompilationConfig(custom_ops=["all"]),
        # Writes K/V in the backend, so attention takes the unfused RoPE path.
        attention_config=AttentionConfig(backend=AttentionBackendEnum.TRITON_ATTN),
    )
    with (
        set_current_vllm_config(vllm_config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cuda"),
        torch.no_grad(),
    ):
        model = MuseSparkDSparkModel(vllm_config=vllm_config, prefix="model")
        yield model


def _assert_rotated(got, x, table, positions, what) -> None:
    """``got`` is ``x`` ([tokens, heads, head_dim]) GPT-J-rotated off the table."""
    cos, sin = table.double()[positions].chunk(2, dim=-1)
    cos, sin = cos[:, None], sin[:, None]
    even, odd = x.double()[..., 0::2], x.double()[..., 1::2]
    ref = torch.stack([even * cos - odd * sin, odd * cos + even * sin], -1)
    ref = ref.flatten(-2)
    # One bf16 rounding of the result, plus the FP32 rounding of the products,
    # which dominates where the rotated value nearly cancels. A BF16 table is
    # off by up to 2^-9 of the inputs and fails this.
    ulp = torch.ldexp(
        torch.ones_like(ref), torch.frexp(ref.abs().float())[1].double() - 8
    )
    slack = 2**-22 * (even.abs() + odd.abs()).repeat_interleave(2, dim=-1)
    err = ((got.double() - ref).abs() / (ulp + slack)).max().item()
    assert err <= 1.0, f"{what}: {err:.2f} bf16 ulp"


def test_dspark_context_keys_use_the_fp32_rope_table(drafter, monkeypatch):
    torch.manual_seed(0)
    kv_size = NUM_KV_HEADS * HEAD_DIM
    for layer in drafter.layers:
        attn = layer.self_attn
        # One-hot rows keep the projection exact whatever the GEMM's order.
        weight = torch.zeros_like(attn.qkv_proj.weight)
        rows = torch.arange(attn.q_size, attn.q_size + 2 * kv_size)
        weight[rows, torch.randperm(HIDDEN)[: rows.numel()]] = 1
        attn.qkv_proj.weight.copy_(weight)
    stored: list[torch.Tensor] = []
    for layer in drafter.layers:
        impl = layer.self_attn.attn.impl
        monkeypatch.setattr(
            impl,
            "do_kv_cache_update",
            lambda _layer, key, *args: stored.append(key.clone()),
        )
    rope = drafter.layers[0].self_attn.rotary_emb
    table = rope.cos_sin_cache
    assert table.dtype == torch.float32

    context = torch.randn(NUM_TOKENS, HIDDEN)
    positions = torch.randint(0, MAX_POSITION, (NUM_TOKENS,))
    drafter.precompute_and_store_context_kv(
        context, positions, torch.arange(NUM_TOKENS)
    )

    assert rope.cos_sin_cache is table
    assert len(stored) == NUM_LAYERS
    # K-norm as the drafter rounds it, then the rotation.
    kv = drafter._project_muse_context_kv(context)
    k32 = kv[..., :kv_size].unflatten(-1, (NUM_KV_HEADS, HEAD_DIM)).float()
    eps = drafter.layers[0].self_attn.rms_norm_eps
    keys = (k32 * torch.rsqrt(k32.pow(2).mean(-1, keepdim=True) + eps)).to(kv.dtype)
    for layer_idx, (got, k) in enumerate(zip(stored, keys)):
        _assert_rotated(got, k, table, positions, f"context key, layer {layer_idx}")


def test_unfused_attention_uses_the_fp32_rope_table(drafter):
    torch.manual_seed(0)
    attn = drafter.layers[0].self_attn
    assert not attn.fused_qk_norm_rope_kv_insert
    table = attn.rotary_emb.cos_sin_cache
    assert table.dtype == torch.float32

    qkv = torch.randn(NUM_TOKENS, attn.qkv_proj.weight.shape[0])
    positions = torch.randint(0, MAX_POSITION, (NUM_TOKENS,))
    q, k, _, _ = attn._qk_norm_rope_kv_insert(qkv.clone(), positions)

    assert attn.rotary_emb.cos_sin_cache is table
    for name, got, start, size in (
        ("query", q, 0, attn.q_size),
        ("key", k, attn.q_size, attn.kv_size),
    ):
        x = qkv[:, start : start + size].reshape(-1, HEAD_DIM)
        x = attn.qk_norm(x, out_dtype=torch.float32).to(qkv.dtype)
        x = x.view(NUM_TOKENS, -1, HEAD_DIM)
        _assert_rotated(got.unflatten(-1, (-1, HEAD_DIM)), x, table, positions, name)
