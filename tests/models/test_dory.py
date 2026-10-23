# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of Dory math and loading, with a reference attention backend."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from huggingface_hub.errors import StrictDataclassClassValidationError
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
        self.prefix = self.layer_name = kwargs["prefix"]
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

    def build(
        *,
        hf_overrides=None,
        quant_config=None,
        num_draft_loops=0,
        num_shared_loops=0,
        **overrides,
    ):
        kwargs = dict(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            num_hidden_layers=8,
            n_input_layers=2,
            n_recurrent_layers=4,
            n_output_layers=2,
            n_recurrent_loops=3,
            max_position_embeddings=32,
            layer_types=["sliding_attention", "mlp"] * 3 + ["full_attention", "mlp"],
            sliding_window=3,
            rope_parameters={
                kind: {
                    "rope_type": "default",
                    "rope_theta": theta,
                    "partial_rotary_factor": 1.0,
                }
                for kind, theta in (
                    ("sliding_attention", 100),
                    ("full_attention", 10000),
                )
            },
        )
        kwargs.update(overrides)
        cfg = DoryConfig(**kwargs)
        cfg.update(hf_overrides or {})
        vc.model_config = SimpleNamespace(
            hf_config=cfg, head_dtype=None, dtype=torch.float32
        )
        vc.quant_config = quant_config
        vc.speculative_config = (
            SimpleNamespace(
                method="looped_self_draft",
                num_draft_loops=num_draft_loops,
                num_shared_loops=num_shared_loops,
            )
            if num_draft_loops
            else None
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


def test_invalid_loop_override_is_rejected_before_model_creation(model_factory):
    with pytest.raises(StrictDataclassClassValidationError, match="n_recurrent_loops"):
        model_factory(hf_overrides={"n_recurrent_loops": 0})


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
            if not layer.is_attention:
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
        scale.float() / model.config.initializer_range
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


@pytest.fixture(params=["mxfp8", "nvfp4"])
def quantized_model(request, monkeypatch, model_factory):
    from vllm.model_executor.layers.quantization import modelopt
    from vllm.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_w8a8_mxfp8,
    )

    # Exercise real parameter allocation and loaders without requiring GPU kernels.
    kernel = SimpleNamespace(input_quant_key=lambda: None)
    monkeypatch.setattr(modelopt, "select_linear_kernel", lambda *a, **kw: kernel)
    monkeypatch.setattr(
        compressed_tensors_w8a8_mxfp8,
        "init_mxfp8_linear_kernel",
        lambda **kw: kernel,
    )
    CompressedTensorsConfig = compressed_tensors.CompressedTensorsConfig
    monkeypatch.setattr(
        CompressedTensorsConfig, "_check_scheme_supported", lambda *a: None
    )
    if request.param == "nvfp4":
        quant = modelopt.ModelOptNvFp4Config(
            quant_method="W4A16_NVFP4", is_checkpoint_nvfp4_serialized=True
        )
    else:
        quant = CompressedTensorsConfig.from_config(
            {
                "format": "mxfp8-quantized",
                "config_groups": {
                    "weights": {
                        "targets": ["Linear"],
                        "weights": {
                            "num_bits": 8,
                            "type": "float",
                            "strategy": "group",
                            "group_size": 32,
                            "symmetric": True,
                            "dynamic": False,
                            "scale_dtype": "torch.uint8",
                        },
                    }
                },
            }
        )
    quant.packed_modules_mapping = dory.DoryForCausalLM.packed_modules_mapping
    return model_factory(
        quant_config=quant, hidden_size=64, intermediate_size=96, head_dim=32
    )


@torch.inference_mode()
def test_quantized_projection_weights_and_scales_load(quantized_model):
    """Packed columns, block scales and scalar scales retain every shard's values."""
    model = quantized_model
    cases = [
        (0, "qkv_proj", ["q_proj", "k_proj", "v_proj"], [64, 32, 32]),
        (1, "gate_up_proj", ["gate_proj", "up_proj"], [96, 96]),
        (0, "g_proj", ["g_proj"], [64]),
        (0, "o_proj", ["o_proj"], [64]),
        (1, "down_proj", ["down_proj"], [64]),
    ]
    for index, target, sources, rows in cases:
        prefix = f"backbone.layers.{index}.mixer"
        layer = model.get_submodule(f"{prefix}.{target}")
        for suffix in ("weight", "weight_scale", "weight_scale_2"):
            param = getattr(layer, suffix, None)
            if param is None:
                continue
            scalar = suffix == "weight_scale_2"
            pieces = [
                torch.full(
                    () if scalar else (n, param.shape[1]), i + 1, dtype=param.dtype
                )
                for i, n in enumerate(rows)
            ]
            loaded = model.load_weights(
                reversed(
                    list(
                        zip(
                            [f"{prefix}.{source}.{suffix}" for source in sources],
                            pieces,
                        )
                    )
                )
            )
            assert loaded == {f"{prefix}.{target}.{suffix}"}
            expected = torch.stack(pieces) if scalar else torch.cat(pieces)
            torch.testing.assert_close(param.float(), expected.float())


@pytest.mark.parametrize("suffix", ["weight", "weight_scale"])
def test_quantized_loader_rejects_oversized_shards(quantized_model, suffix):
    param = getattr(quantized_model.backbone.layers[0].mixer.qkv_proj, suffix)
    oversized = torch.zeros((65, param.shape[1]), dtype=param.dtype)
    with pytest.raises(ValueError, match="expected shape"):
        quantized_model.load_weights(
            [(f"backbone.layers.0.mixer.q_proj.{suffix}", oversized)]
        )


@pytest.mark.parametrize(
    "num_draft_loops,num_shared_loops", [(1, 1), (1, 0), (2, 2), (2, 0), (2, 1)]
)
@torch.inference_mode()
def test_self_speculation_reuses_prefix_and_preserves_kv_after_rejection(
    model_factory, num_draft_loops, num_shared_loops
):
    """Reuse must match recomputation across full, partial and zero acceptance."""
    torch.manual_seed(42)
    kwargs = dict(num_draft_loops=num_draft_loops, num_shared_loops=num_shared_loops)
    model = model_factory(**kwargs)
    reference = model_factory(**kwargs)
    target_reference = model_factory()
    draft_reference = model_factory(n_recurrent_loops=num_draft_loops)
    for param in model.parameters():
        param.normal_(std=0.02)
    reference.load_state_dict(model.state_dict())
    target_reference.load_state_dict(model.state_dict())
    draft_reference.load_state_dict(model.state_dict())
    from vllm.model_executor.models.interfaces import SupportsLoopedSelfDraft

    assert isinstance(model, SupportsLoopedSelfDraft)
    draft = model.get_looped_self_draft_model()
    assert list(draft.parameters()) == []
    assert draft.target is model
    caches = [m for m in model.modules() if isinstance(m, ReferenceAttention)]
    assert len({extract_layer_index(m.prefix) for m in caches}) == len(caches)
    output_cache = model.backbone.layers[6].mixer
    assert output_cache.attn[1] is not output_cache.attn[0]
    shared = model.backbone.num_shared_loops
    num_loops = model.config.n_recurrent_loops
    draft_names = set(model.get_draft_attn_layer_names())
    assert model.backbone.layers[0].mixer.attn[0].layer_name in draft_names
    private_target_caches = []
    for layer in model.backbone.layers[2:6]:
        if not layer.is_attention:
            continue
        assert len(layer.mixer.attn) == num_loops + num_draft_loops - shared
        for i, attn in enumerate(layer.mixer.attn):
            assert (attn.layer_name in draft_names) == (i < shared or i >= num_loops)
        private_target_caches.extend(layer.mixer.attn[shared:num_loops])

    # CPU analogue of the activation buffer; GPU row mapping is tested separately.
    class ActivationCache:
        num_speculative_tokens = 3
        verify = False

        def __init__(self):
            self.inputs, self.states = [], []
            self.start_pos = 0

        def record(self, input_hidden_states, hidden_states):
            self.inputs.append(input_hidden_states.clone())
            self.states.append(hidden_states.clone())

        def merge(self, tail_input, tail_state):
            return torch.cat([*self.inputs, tail_input]), torch.cat(
                [*self.states, tail_state]
            )

        @contextmanager
        def prefix_context(self):
            # Emulate compact slot mappings for the missing final prefix row.
            for cache in caches:
                cache.start_pos = self.start_pos + self.num_speculative_tokens
            yield
            for cache in caches:
                cache.start_pos = self.start_pos

    cache = ActivationCache()
    model.draft_activation_cache = cache
    tokens = torch.arange(4)
    actual = model(tokens, tokens)
    torch.testing.assert_close(actual, reference(tokens, tokens))
    torch.testing.assert_close(actual, target_reference(tokens, tokens))
    draft_reference(tokens, tokens)
    start = 4
    for accepted in [3, 0, 2, 1, 3]:
        cache.verify = False
        cache.inputs.clear()
        cache.states.clear()
        tokens = torch.randint(0, 32, (4,))
        positions = torch.arange(start, start + 4)
        target_keys = [attn.keys.clone() for attn in private_target_caches]
        for step in range(3):
            for target in (model, reference):
                for attn in target.modules():
                    if isinstance(attn, ReferenceAttention):
                        attn.start_pos = start + step
                actual_draft = target.backbone.draft_forward(
                    tokens[step : step + 1], positions[step : step + 1]
                )
            expected_draft = draft_reference(
                tokens[step : step + 1], positions[step : step + 1]
            )
            torch.testing.assert_close(
                actual_draft, expected_draft, atol=2e-6, rtol=2e-5
            )
        for attn, keys in zip(private_target_caches, target_keys):
            torch.testing.assert_close(attn.keys, keys, atol=0, rtol=0)
        cache.start_pos = start
        cache.verify = True
        actual = model(tokens, positions)
        expected = reference(tokens, positions)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(
            actual, target_reference(tokens, positions), atol=2e-6, rtol=2e-5
        )
        draft_reference(tokens, positions)
        reference_caches = [
            m for m in reference.modules() if isinstance(m, ReferenceAttention)
        ]
        for actual_cache, expected_cache in zip(caches, reference_caches):
            torch.testing.assert_close(
                actual_cache.keys, expected_cache.keys, atol=2e-6, rtol=2e-5
            )
            torch.testing.assert_close(
                actual_cache.values, expected_cache.values, atol=2e-6, rtol=2e-5
            )
        start += accepted + 1


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"num_draft_loops": 3}, "below n_recurrent_loops"),
        ({"num_draft_loops": 1, "recurrent_kv_cache_mode": "last_loop"}, "per_loop"),
    ],
)
def test_self_draft_rejects_incompatible_model_layout(model_factory, kwargs, match):
    with pytest.raises(ValueError, match=match):
        model_factory(**kwargs)
