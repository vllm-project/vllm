# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of Dory math and loading, with a reference attention backend."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.config import (
    CompilationConfig,
    DeviceConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import parallel_state
from vllm.model_executor.models import dory
from vllm.model_executor.models.utils import extract_layer_index
from vllm.transformers_utils.configs.dory import DoryConfig


class ReferenceAttention(nn.Module):
    """Single-request causal GQA, including sliding-window and decode caching."""

    def __init__(self, num_heads, head_size, scale, num_kv_heads, **kwargs):
        super().__init__()
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_size, self.scale = head_size, scale
        self.prefix = kwargs["prefix"]
        self.window = kwargs["per_layer_sliding_window"]
        self.keys = self.values = None

    def forward(self, q, k, v):
        q = q.view(-1, self.num_heads, self.head_size).transpose(0, 1)
        k = k.view(-1, self.num_kv_heads, self.head_size).transpose(0, 1)
        v = v.view(-1, self.num_kv_heads, self.head_size).transpose(0, 1)
        offset = self.start_pos
        self.keys = (
            k if self.keys is None else torch.cat((self.keys[:, :offset], k), dim=1)
        )
        self.values = (
            v if self.values is None else torch.cat((self.values[:, :offset], v), dim=1)
        )
        key_pos = torch.arange(self.keys.shape[1])
        query_pos = torch.arange(offset, offset + q.shape[1])[:, None]
        mask = key_pos <= query_pos
        if self.window is not None:
            mask &= key_pos > query_pos - self.window
        out = torch.nn.functional.scaled_dot_product_attention(
            q, self.keys, self.values, attn_mask=mask, scale=self.scale, enable_gqa=True
        )
        return out.transpose(0, 1).reshape(-1, self.num_heads * self.head_size)


@pytest.fixture
def model_factory(monkeypatch):
    # Keep the actual vLLM linear/embedding modules and their weight loaders.
    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(world_size=1, rank_in_group=0)
    )
    monkeypatch.setattr(dory, "Attention", ReferenceAttention)
    vc = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(mode=0, custom_ops=["none"]),
    )

    def build(**overrides):
        kwargs = dict(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_attention_heads=2,
            num_key_value_heads=1,
            attention_head_dim=8,
            num_hidden_layers=8,
            n_input_layers=2,
            n_recurrent_layers=4,
            n_output_layers=2,
            n_recurrent_loops=3,
            max_position_embeddings=32,
            swa_window_size=3,
            output_transform="replace",
            rope_profile_layers=[2, 1, 1, 1, 2, 1, 1, 1],
            rope_theta_2=100,
        )
        kwargs.update(overrides)
        cfg = DoryConfig(**kwargs)
        vc.model_config = SimpleNamespace(
            hf_config=cfg, head_dtype=None, dtype=torch.float32
        )
        with set_current_vllm_config(vc):
            model = dory.DoryForCausalLM(vllm_config=vc)

        def set_cache_positions(module, args):
            # Emulate the runner's slot mapping for contiguous request tokens.
            for cache in module.modules():
                if isinstance(cache, ReferenceAttention):
                    cache.start_pos = int(args[1][0])

        model.register_forward_pre_hook(set_cache_positions)
        return model

    with set_current_vllm_config(vc):
        yield build


@pytest.mark.parametrize("num_loops", [3, 27])
@torch.inference_mode()
def test_chunked_prefill_and_decode_keep_loop_caches_separate(model_factory, num_loops):
    model = model_factory(n_recurrent_loops=num_loops)
    torch.manual_seed(0)
    for param in model.parameters():
        param.normal_(std=0.02)
    tokens, positions = torch.arange(7), torch.arange(7)
    expected = model(tokens, positions)
    caches = [m for m in model.modules() if isinstance(m, ReferenceAttention)]
    # Two outer attention layers + two shared recurrent layers per loop.
    assert (
        len(caches)
        == len({extract_layer_index(m.prefix) for m in caches})
        == (2 + 2 * num_loops)
    )
    for cache in caches:
        assert cache.keys is not None
        assert cache.keys.shape[1] == len(tokens)
        cache.keys = cache.values = None
    actual = torch.cat(
        [model(tokens[:3], positions[:3])]
        + [model(tokens[i : i + 1], positions[i : i + 1]) for i in range(3, 7)]
    )
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("cache_mode", ["per_loop", "last_loop"])
def test_loop_count_does_not_duplicate_projection_parameters(model_factory, cache_mode):
    single_loop = model_factory(n_recurrent_loops=1)
    many_loops = model_factory(n_recurrent_loops=27, recurrent_kv_cache_mode=cache_mode)
    assert {name: p.shape for name, p in single_loop.named_parameters()} == {
        name: p.shape for name, p in many_loops.named_parameters()
    }


@pytest.mark.parametrize("num_loops", [1, 3])
@torch.inference_mode()
def test_last_loop_kv_matches_final_loop_history(model_factory, num_loops):
    """Reuse final-loop history while preserving current-loop KV within a chunk."""
    shared = model_factory(
        n_recurrent_loops=num_loops, recurrent_kv_cache_mode="last_loop"
    )
    reference = model_factory(n_recurrent_loops=num_loops)
    torch.manual_seed(0)
    for param in shared.parameters():
        param.normal_(std=0.02)
    reference.load_state_dict(shared.state_dict())
    assert sum(isinstance(m, ReferenceAttention) for m in shared.modules()) == 4
    tokens = positions = torch.arange(7)
    # Include prefill chunks, decode, and history beyond the sliding window.
    for start, end in ((0, 3), (3, 4), (4, 7)):
        expected = reference(tokens[start:end], positions[start:end])
        actual = shared(tokens[start:end], positions[start:end])
        torch.testing.assert_close(actual, expected)
        for layer, ref_layer in zip(shared.backbone.layers, reference.backbone.layers):
            if layer.char == "-":
                continue
            cache, final = layer.mixer.attn[0], ref_layer.mixer.attn[-1]
            assert cache.keys.shape[1] == end
            torch.testing.assert_close(cache.keys, final.keys)
            torch.testing.assert_close(cache.values, final.values)
            # Next step, every reference loop starts from final-loop history.
            for ref_cache in ref_layer.mixer.attn:
                ref_cache.keys = final.keys.clone()
                ref_cache.values = final.values.clone()


@pytest.mark.parametrize("reverse", [False, True])
@torch.inference_mode()
def test_split_scales_load_independently_in_either_order(model_factory, reverse):
    model = model_factory()
    prefix = "backbone.layers.1.mixer"
    gate, up = torch.arange(24.0), torch.arange(24.0) + 100
    head = torch.arange(32 * 16.0).reshape(32, 16) / 512
    scale = torch.linspace(0.01, 0.03, 32)
    weights = [
        (f"{prefix}.suv_gate", gate),
        ("lm_head.weight", head),
        (f"{prefix}.suv_up", up),
        ("lm_head.logit_scale", scale),
    ]
    if reverse:
        weights.reverse()
    loaded = set()
    for name, tensor in weights:
        loaded_now = model.load_weights([(name, tensor)])
        assert loaded_now == {name}
        loaded |= loaded_now
    assert loaded == {name for name, _ in weights}
    torch.testing.assert_close(model.backbone.layers[1].mixer.suv_gate, gate)
    torch.testing.assert_close(model.backbone.layers[1].mixer.suv_up, up)
    torch.testing.assert_close(model.lm_head.weight[:32], head)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_logits_apply_vocab_scale_after_projection(model_factory, monkeypatch, dtype):
    model = model_factory().to(dtype)
    monkeypatch.setattr(model.logits_processor, "_gather_logits", lambda logits: logits)
    torch.manual_seed(0)
    head = torch.randn(32, 16, dtype=dtype)
    scale = torch.linspace(0.01, 0.03, 32).to(dtype)
    hidden = torch.randn(3, 16, dtype=dtype)
    model.load_weights([("lm_head.weight", head), ("lm_head.logit_scale", scale)])
    expected = torch.nn.functional.linear(hidden, head).float() * (
        scale.float() / model.config.init_method_std
    )
    torch.testing.assert_close(model.compute_logits(hidden), expected)


@torch.inference_mode()
def test_exported_qkv_and_mlp_weights_load_in_correct_order(model_factory):
    model = model_factory()
    attn = "backbone.layers.0.mixer"
    mlp = "backbone.layers.1.mixer"
    pieces = [
        (f"{attn}.q_proj.weight", torch.full((16, 16), 1.0)),
        (f"{attn}.k_proj.weight", torch.full((8, 16), 2.0)),
        (f"{attn}.v_proj.weight", torch.full((8, 16), 3.0)),
        (f"{mlp}.gate_proj.weight", torch.full((24, 16), 4.0)),
        (f"{mlp}.up_proj.weight", torch.full((24, 16), 5.0)),
    ]
    loaded = model.load_weights(reversed(pieces))
    assert loaded == {f"{attn}.qkv_proj.weight", f"{mlp}.gate_up_proj.weight"}
    torch.testing.assert_close(
        model.backbone.layers[0].mixer.qkv_proj.weight,
        torch.cat([w for _, w in pieces[:3]]),
    )
    torch.testing.assert_close(
        model.backbone.layers[1].mixer.gate_up_proj.weight,
        torch.cat([w for _, w in pieces[3:]]),
    )


@pytest.mark.parametrize(
    "name,shape",
    [
        ("backbone.layers.0.mixer.q_proj.weight", (32, 16)),
        ("backbone.layers.1.mixer.gate_proj.weight", (32, 16)),
        ("backbone.layers.1.mixer.down_proj.weight", (16, 32)),
        ("backbone.layers.1.mixer.suv_gate", (32,)),
    ],
)
def test_weight_loader_rejects_silent_truncation(model_factory, name, shape):
    model = model_factory()
    with pytest.raises(ValueError, match="expected shape"):
        model.load_weights([(name, torch.zeros(shape))])
