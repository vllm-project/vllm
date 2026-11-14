# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the ZAQERIS AXIOM native vLLM implementation.

These are component-level tests (no 29 GiB checkpoint required). They exercise
the M3 weight container, the quant method, the linear/embedding/attention/MoE
modules and the full `ZaqerisAxiomForCausalLM` forward path with synthetic
buffers.
"""

import json

import pytest
import torch

import vllm.model_executor.models.zaqeris_axiom as zaqeris_axiom_mod
from vllm.config import ModelConfig, VllmConfig
from vllm.model_executor.models import registry as me_models
from vllm.model_executor.models.zaqeris_axiom import (
    ZaqerisAxiomAttention,
    ZaqerisAxiomDecoderLayer,
    ZaqerisAxiomEmbedding,
    ZaqerisAxiomExpert,
    ZaqerisAxiomForCausalLM,
    ZaqerisAxiomLinear,
    ZaqerisAxiomM3QuantMethod,
    ZaqerisAxiomM3Weight,
    ZaqerisAxiomSparseMoeBlock,
)

# Register the architecture so ModelConfig can resolve it in tests.
me_models.ModelRegistry.register_model(
    "ZaqerisAxiomForCausalLM", ZaqerisAxiomForCausalLM
)

ROPE = {
    "rope_type": "yarn",
    "factor": 1.0,
    "original_max_position_embeddings": 64,
    "extended_max_position_embeddings": 64,
}


def _synthetic_config(**overrides):
    cfg = {
        "architectures": ["ZaqerisAxiomForCausalLM"],
        "model_type": "qwen2",
        "vocab_size": 256,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "hidden_act": "silu",
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "rope_parameters": ROPE,
        "tie_word_embeddings": False,
    }
    cfg.update(overrides)
    return cfg


def _fill_m3(mod, *, block_size=8):
    """Populate every M3 bundle in `mod` with random codebooks/indices."""
    for parent in mod.modules():
        mw = getattr(parent, "m3_weight", None)
        if not isinstance(mw, ZaqerisAxiomM3Weight):
            continue
        out, inn, bs = mw.out_features, mw.in_features, mw.block_size
        is_emb = isinstance(parent, ZaqerisAxiomEmbedding)
        if is_emb:
            n_vec = out * (inn // bs)
            mw.m3_cb1 = torch.nn.Parameter(torch.randn(8, bs, dtype=torch.float16))
            mw.m3_cb2 = torch.nn.Parameter(torch.randn(8, bs, dtype=torch.float16))
            mw.m3_cb3 = torch.nn.Parameter(torch.randn(8, bs, dtype=torch.float16))
        else:
            n_vec = out * inn
            mw.m3_cb1 = torch.nn.Parameter(torch.randn(8, dtype=torch.float16))
            mw.m3_cb2 = torch.nn.Parameter(torch.randn(8, dtype=torch.float16))
            mw.m3_cb3 = torch.nn.Parameter(torch.randn(8, dtype=torch.float16))
        mw.m3_idx1 = torch.randint(0, 8, (n_vec,), dtype=torch.uint8)
        mw.m3_idx2 = torch.randint(0, 8, (n_vec,), dtype=torch.uint8)
        mw.m3_idx3 = torch.randint(0, 8, (n_vec,), dtype=torch.uint8)
        mw.m3_outliers = torch.empty(0, dtype=torch.float16)
        mw.m3_outlier_idx = torch.empty((0, 2), dtype=torch.int32)


class _StubAttn(torch.nn.Module):
    """Drop-in for vLLM's Attention that needs a model-runner forward context."""

    def __init__(self, num_heads, num_kv_heads, head_dim, scaling):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scaling = scaling

    def forward(self, q, k, v, **kwargs):
        bsz, q_len, _ = q.shape
        nh, nkh, hd = self.num_heads, self.num_kv_heads, self.head_dim
        q = q.view(bsz, q_len, nh, hd).transpose(1, 2)
        k = k.view(bsz, q_len, nkh, hd).transpose(1, 2)
        v = v.view(bsz, q_len, nkh, hd).transpose(1, 2)
        if nkh != nh:
            rep = nh // nkh
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        o = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=self.scaling
        )
        return o.transpose(1, 2).contiguous().view(bsz, q_len, nh * hd)


def _stub_attn(mod):
    for sub in mod.modules():
        if isinstance(sub, ZaqerisAxiomAttention):
            sub.attn = _StubAttn(
                sub.num_heads, sub.num_kv_heads, sub.head_dim, sub.scaling
            )


def test_m3_weight_reconstruct_shape():
    w = ZaqerisAxiomM3Weight(48, 32, block_size=8)
    n_vec = 48 * 32
    w.m3_cb1 = torch.nn.Parameter(torch.randn(8, dtype=torch.float16))
    w.m3_cb2 = torch.nn.Parameter(torch.randn(8, dtype=torch.float16))
    w.m3_cb3 = torch.nn.Parameter(torch.randn(8, dtype=torch.float16))
    w.m3_idx1 = torch.randint(0, 8, (n_vec,), dtype=torch.uint8)
    w.m3_idx2 = torch.randint(0, 8, (n_vec,), dtype=torch.uint8)
    w.m3_idx3 = torch.randint(0, 8, (n_vec,), dtype=torch.uint8)
    out = w.reconstruct()
    assert tuple(out.shape) == (48, 32)


def test_m3_quant_method_creates_m3_weight_bundle():
    lin = ZaqerisAxiomLinear(32, 48, bias=False, quant_config=None, prefix="w")
    assert isinstance(lin.quant_method, ZaqerisAxiomM3QuantMethod)
    assert isinstance(lin.m3_weight, ZaqerisAxiomM3Weight)
    assert lin.m3_weight.out_features == 48
    assert lin.m3_weight.in_features == 32
    assert lin.m3_weight.block_size == 8


def test_linear_forward():
    lin = ZaqerisAxiomLinear(32, 48, bias=False, quant_config=None, prefix="w")
    _fill_m3(lin)
    x = torch.randn(2, 8, 32, dtype=torch.float16)
    y = lin(x)
    assert tuple(y.shape) == (2, 8, 48)


def test_linear_with_bias():
    lin = ZaqerisAxiomLinear(32, 16, bias=True, quant_config=None, prefix="wb")
    _fill_m3(lin)
    x = torch.randn(2, 8, 32, dtype=torch.float16)
    y = lin(x)
    assert tuple(y.shape) == (2, 8, 16)
    assert lin.has_bias


def test_embedding_forward():
    emb = ZaqerisAxiomEmbedding(256, 32, block_size=8)
    _fill_m3(emb)
    e = emb(torch.randint(0, 256, (2, 8)))
    assert tuple(e.shape) == (2, 8, 32)


def test_attention_forward(default_vllm_config):
    attn = ZaqerisAxiomAttention(
        32,
        4,
        2,
        8,
        ROPE,
        64,
        1e-6,
        cache_config=None,
        quant_config=None,
        prefix="a",
    )
    _fill_m3(attn)
    _stub_attn(attn)
    x = torch.randn(2, 8, 32, dtype=torch.float16)
    pos = torch.arange(8, dtype=torch.long).unsqueeze(0).expand(2, -1)
    a = attn(pos, x)
    assert tuple(a.shape) == (2, 8, 32)


def test_expert_forward():
    expert = ZaqerisAxiomExpert(32, 64, quant_config=None, prefix="e")
    _fill_m3(expert)
    x = torch.randn(2, 8, 32, dtype=torch.float16)
    y = expert(x)
    # An expert maps hidden -> intermediate -> hidden.
    assert tuple(y.shape) == (2, 8, 32)


def test_moe_block_forward():
    moe = ZaqerisAxiomSparseMoeBlock(
        32,
        64,
        4,
        2,
        "silu",
        quant_config=None,
        prefix="moe",
    )
    _fill_m3(moe)
    x = torch.randn(2, 8, 32, dtype=torch.float16)
    y = moe(x)
    assert tuple(y.shape) == (2, 8, 32)


def test_decoder_layer_forward(default_vllm_config):
    layer = ZaqerisAxiomDecoderLayer(
        32,
        4,
        2,
        8,
        64,
        4,
        2,
        "silu",
        ROPE,
        64,
        1e-6,
        cache_config=None,
        quant_config=None,
        prefix="l",
    )
    _fill_m3(layer)
    _stub_attn(layer)
    x = torch.randn(2, 8, 32, dtype=torch.float16)
    pos = torch.arange(8, dtype=torch.long).unsqueeze(0).expand(2, -1)
    h, r = layer(pos, x, None)
    assert tuple(h.shape) == (2, 8, 32)
    assert tuple(r.shape) == (2, 8, 32)


@pytest.fixture
def zaqeris_vllm_config(monkeypatch, tmp_path):
    """Single-rank, CPU, no-compile config for direct model construction."""
    from vllm.config import (
        CompilationConfig,
        DeviceConfig,
        set_current_vllm_config,
    )

    # The model uses get_pp_group().is_first_rank / is_last_rank to decide
    # whether to build the embedding / lm_head. Stub it for single-rank.
    class _FakePP:
        is_first_rank = True
        is_last_rank = True
        rank_in_group = 0
        world_size = 1

    monkeypatch.setattr(
        zaqeris_axiom_mod,
        "get_pp_group",
        lambda: _FakePP(),
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_pp_group",
        lambda: _FakePP(),
    )

    # Write a minimal AXIOM config so ModelConfig can resolve the architecture.
    # head_dtype is pinned to the model dtype because the LogitsProcessor
    # refuses a non-model head_dtype for a quantized lm_head.
    cfg = dict(_synthetic_config())
    cfg["torch_dtype"] = "float16"
    cfg["head_dtype"] = "model"
    cfg_dir = tmp_path / "axiom_cfg"
    cfg_dir.mkdir()
    (cfg_dir / "config.json").write_text(json.dumps(cfg))

    vc = VllmConfig(
        model_config=ModelConfig(model=str(cfg_dir)),
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(mode=0, custom_ops=["none"]),
    )
    with set_current_vllm_config(vc):
        yield vc


def test_head_dtype_matches_model_dtype(zaqeris_vllm_config):
    # head_dtype defaults to float32 for generation models, which the
    # LogitsProcessor refuses for a quantized lm_head. The test config pins
    # it to the model dtype so the normal (non-fp32-head) path is exercised.
    cfg = zaqeris_vllm_config.model_config
    assert cfg.head_dtype == cfg.dtype


def test_for_causal_lm_forward(zaqeris_vllm_config):
    model = ZaqerisAxiomForCausalLM(vllm_config=zaqeris_vllm_config)
    _fill_m3(model)
    _stub_attn(model)
    pos = torch.arange(8, dtype=torch.long).unsqueeze(0).expand(2, -1)
    # The test config pins head_dtype to the model dtype, so the M3 lm_head
    # projection runs in the same dtype as the hidden states.
    h = model(input_ids=torch.randint(0, 256, (2, 8)), positions=pos)
    assert tuple(h.shape) == (2, 8, 32)
    logits = model.compute_logits(h)
    assert tuple(logits.shape) == (2, 8, 256)


def test_for_causal_lm_expert_metadata(zaqeris_vllm_config):
    model = ZaqerisAxiomForCausalLM(vllm_config=zaqeris_vllm_config)
    # ZaqerisAxiomForCausalLM implements the MixtureOfExperts protocol.
    assert model.num_logical_experts == 4
    assert model.num_physical_experts == 4
    assert model.num_local_physical_experts == 4
    assert model.num_routed_experts == 4
    assert model.num_shared_experts == 0
    assert model.num_redundant_experts == 0
    assert model.num_expert_groups == 1
    assert model.num_moe_layers == 2
    assert len(model.moe_layers) == 2
