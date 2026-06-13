# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for AnyModel (NAS heterogeneous architecture support).

No GPU or model weights are required.
"""

import types
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from vllm.model_executor.models.anymodel import (
    _ARCH_REGISTRY,
    ArchInfo,
    NoOpAttention,
    NoOpMLP,
    NoOpNorm,
    _apply_no_ops,
    _arch_info_from_config,
    _create_layer_config,
    _has_overrides,
    _resolve_layer_class,
    _unregister_layer,
)


def _ns(**kwargs):
    return types.SimpleNamespace(**kwargs)


def _block(
    *,
    attn_no_op: bool = False,
    kv_heads: int | None = None,
    ffn_no_op: bool = False,
    intermediate_size: int | None = None,
    hidden_act: str | None = None,
    extra: dict | None = None,
) -> dict:
    """Build a per_layer_config entry in the flat HF heterogeneity schema."""
    entry: dict = {}
    skip: list[str] = []
    if attn_no_op:
        skip.append("attention")
    if ffn_no_op:
        skip.append("mlp")
    if skip:
        entry["skip"] = skip
    if kv_heads is not None:
        entry["num_key_value_heads"] = kv_heads
    if intermediate_size is not None:
        entry["intermediate_size"] = intermediate_size
    if hidden_act is not None:
        entry["hidden_act"] = hidden_act
    if extra:
        entry.update(extra)
    return entry


def _base_config(**overrides):
    defaults = dict(
        num_key_value_heads=8,
        intermediate_size=14336,
        hidden_act="silu",
        num_experts=None,
        moe_intermediate_size=None,
        num_local_experts=None,
        n_routed_experts=None,
    )
    defaults.update(overrides)
    return _ns(**defaults)


_LLAMA_INFO = ArchInfo(
    decoder_layer_module=".llama", decoder_layer_class="LlamaDecoderLayer"
)


class TestArchRegistry:
    _ALL_ARCHS = [
        "LlamaForCausalLM",
        "MistralForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
        "Qwen2MoeForCausalLM",
        "MixtralForCausalLM",
        "GptOssForCausalLM",
        "NemotronHForCausalLM",
        "NemotronHPuzzleForCausalLM",
        "Qwen3VLForConditionalGeneration",
    ]

    @pytest.mark.parametrize("arch", _ALL_ARCHS)
    def test_arch_registered(self, arch):
        assert arch in _ARCH_REGISTRY

    def test_nemotronh_puzzle_is_alias(self):
        assert (
            _ARCH_REGISTRY["NemotronHPuzzleForCausalLM"]
            is _ARCH_REGISTRY["NemotronHForCausalLM"]
        )

    @pytest.mark.parametrize(
        "arch, field, expected",
        [
            ("LlamaForCausalLM", "decoder_layer_module", ".llama"),
            ("LlamaForCausalLM", "decoder_layer_class", "LlamaDecoderLayer"),
            ("MistralForCausalLM", "decoder_layer_class", "MistralDecoderLayer"),
            ("Qwen2ForCausalLM", "decoder_layer_module", ".qwen2"),
            ("Qwen3ForCausalLM", "decoder_layer_class", "Qwen3DecoderLayer"),
            ("MixtralForCausalLM", "ffn_module", "block_sparse_moe"),
            ("GptOssForCausalLM", "attn_module", "attn"),
            ("GptOssForCausalLM", "decoder_layer_class", "TransformerBlock"),
            (
                "NemotronHForCausalLM",
                "hybrid_pattern_field",
                "hybrid_override_pattern",
            ),
            ("NemotronHForCausalLM", "attn_module", "mixer"),
            ("NemotronHForCausalLM", "ffn_module", "mixer"),
            (
                "Qwen3VLForConditionalGeneration",
                "layers_path",
                "language_model.model.layers",
            ),
            ("Qwen3VLForConditionalGeneration", "init_prefix", "model"),
            ("Qwen3VLForConditionalGeneration", "layer_hf_config", "text_config"),
            ("Qwen3VLForConditionalGeneration", "base_model_module", ".qwen3_vl"),
        ],
    )
    def test_arch_field(self, arch, field, expected):
        assert getattr(_ARCH_REGISTRY[arch], field) == expected

    def test_nemotronh_has_all_hybrid_codes(self):
        info = _ARCH_REGISTRY["NemotronHForCausalLM"]
        assert set(info.decoder_layer_class_map) >= {"*", "-", "E", "M"}

    def test_defaults(self):
        info = _LLAMA_INFO
        assert info.attn_module == "self_attn"
        assert info.ffn_module == "mlp"
        assert info.attn_norm_module == "input_layernorm"
        assert info.ffn_norm_module == "post_attention_layernorm"
        assert info.layers_path == "model.layers"
        assert info.decoder_layer_class_map is None
        assert info.init_prefix is None
        assert info.layer_hf_config is None

    def test_only_vl_sets_base_model_module(self):
        for arch, info in _ARCH_REGISTRY.items():
            if arch == "Qwen3VLForConditionalGeneration":
                continue
            assert info.base_model_module is None, arch


class TestResolveLayerClass:
    @pytest.fixture()
    def hybrid_info(self):
        return ArchInfo(
            decoder_layer_module=".nemotron_h",
            decoder_layer_class="NemotronHAttentionDecoderLayer",
            decoder_layer_class_map={
                "*": "NemotronHAttentionDecoderLayer",
                "-": "NemotronHMLPDecoderLayer",
                "E": "NemotronHMoEDecoderLayer",
                "M": "NemotronHMambaDecoderLayer",
            },
            hybrid_pattern_field="hybrid_override_pattern",
        )

    @pytest.mark.parametrize(
        "pattern, idx, expected",
        [
            ("*-EM", 0, "NemotronHAttentionDecoderLayer"),
            ("*-EM", 1, "NemotronHMLPDecoderLayer"),
            ("*-EM", 2, "NemotronHMoEDecoderLayer"),
            ("*-EM", 3, "NemotronHMambaDecoderLayer"),
        ],
    )
    def test_hybrid_dispatch(self, hybrid_info, pattern, idx, expected):
        config = _ns(hybrid_override_pattern=pattern)
        assert _resolve_layer_class(hybrid_info, config, idx).__name__ == expected

    def test_no_map_falls_back_to_default(self):
        cls = _resolve_layer_class(_LLAMA_INFO, _ns(), layer_idx=0)
        assert cls.__name__ == "LlamaDecoderLayer"

    def test_unknown_char_falls_back(self, hybrid_info):
        config = _ns(hybrid_override_pattern="X")
        cls = _resolve_layer_class(hybrid_info, config, layer_idx=0)
        assert cls.__name__ == "NemotronHAttentionDecoderLayer"

    def test_idx_out_of_pattern_falls_back(self, hybrid_info):
        config = _ns(hybrid_override_pattern="*")
        cls = _resolve_layer_class(hybrid_info, config, layer_idx=5)
        assert cls.__name__ == "NemotronHAttentionDecoderLayer"


class TestCreateLayerConfig:
    def test_no_overrides_copies_global(self):
        cfg = _base_config()
        result = _create_layer_config(cfg, _block(), _LLAMA_INFO)
        assert result.num_key_value_heads == cfg.num_key_value_heads
        assert result.intermediate_size == cfg.intermediate_size

    def test_kv_heads_override(self):
        cfg = _base_config(num_key_value_heads=8)
        result = _create_layer_config(cfg, _block(kv_heads=4), _LLAMA_INFO)
        assert result.num_key_value_heads == 4
        assert cfg.num_key_value_heads == 8

    def test_intermediate_size_override(self):
        cfg = _base_config(intermediate_size=14336)
        result = _create_layer_config(cfg, _block(intermediate_size=8192), _LLAMA_INFO)
        assert result.intermediate_size == 8192

    def test_hidden_act_override(self):
        cfg = _base_config(hidden_act="silu")
        result = _create_layer_config(cfg, _block(hidden_act="gelu"), _LLAMA_INFO)
        assert result.hidden_act == "gelu"

    def test_skip_does_not_set_skip_attr_on_config(self):
        # The "skip" key is consumed by _apply_no_ops; _create_layer_config
        # must not leak it onto the returned config.
        cfg = _base_config()
        result = _create_layer_config(cfg, _block(attn_no_op=True), _LLAMA_INFO)
        assert not hasattr(result, "skip")

    def test_arbitrary_top_level_override(self):
        # Under the flat HF schema, any top-level parent-config key may be
        # overridden per-layer.  _create_layer_config applies them verbatim.
        cfg = _base_config(hidden_size=4096)
        result = _create_layer_config(
            cfg, _block(extra={"hidden_size": 1024}), _LLAMA_INFO
        )
        assert result.hidden_size == 1024
        assert cfg.hidden_size == 4096

    def test_moe_expert_count_override(self):
        # MoE experts are now a flat top-level key (matching the parent
        # config's canonical name, e.g. "num_experts" or "num_local_experts").
        cfg = _base_config(num_experts=8, moe_intermediate_size=1024)
        info = ArchInfo(
            decoder_layer_module=".qwen2_moe",
            decoder_layer_class="Qwen2MoeDecoderLayer",
        )
        result = _create_layer_config(
            cfg,
            _block(extra={"num_experts": 4, "moe_intermediate_size": 512}),
            info,
        )
        assert result.num_experts == 4
        assert result.moe_intermediate_size == 512


class TestApplyNoOps:
    def _make_layer(self, **name_overrides):
        names = {
            "self_attn": None,
            "mlp": None,
            "input_layernorm": None,
            "post_attention_layernorm": None,
        }
        names.update(name_overrides)
        layer = MagicMock(spec=nn.Module)
        for attr in names:
            setattr(layer, attr, MagicMock(spec=nn.Module))
        return layer

    def test_no_noop_leaves_intact(self):
        layer = self._make_layer()
        orig = layer.self_attn
        _apply_no_ops(layer, _block(), _LLAMA_INFO)
        assert layer.self_attn is orig

    def test_attn_noop(self):
        layer = self._make_layer()
        _apply_no_ops(layer, _block(attn_no_op=True), _LLAMA_INFO)
        assert isinstance(layer.self_attn, NoOpAttention)
        assert isinstance(layer.input_layernorm, NoOpNorm)
        assert not isinstance(layer.mlp, NoOpMLP)

    def test_ffn_noop(self):
        layer = self._make_layer()
        _apply_no_ops(layer, _block(ffn_no_op=True), _LLAMA_INFO)
        assert isinstance(layer.mlp, NoOpMLP)
        assert isinstance(layer.post_attention_layernorm, NoOpNorm)
        assert not isinstance(layer.self_attn, NoOpAttention)

    def test_both_noop(self):
        layer = self._make_layer()
        _apply_no_ops(layer, _block(attn_no_op=True, ffn_no_op=True), _LLAMA_INFO)
        assert isinstance(layer.self_attn, NoOpAttention)
        assert isinstance(layer.mlp, NoOpMLP)

    def test_mixtral_ffn_noop(self):
        info = _ARCH_REGISTRY["MixtralForCausalLM"]
        layer = self._make_layer(block_sparse_moe=None)
        _apply_no_ops(layer, _block(ffn_no_op=True), info)
        assert isinstance(layer.block_sparse_moe, NoOpMLP)

    def test_gptoss_attn_noop(self):
        info = _ARCH_REGISTRY["GptOssForCausalLM"]
        layer = self._make_layer(attn=None)
        _apply_no_ops(layer, _block(attn_no_op=True), info)
        assert isinstance(layer.attn, NoOpAttention)

    def test_nemotronh_shared_mixer_only_replaced_when_both_noop(self):
        # NemotronH's attn and ffn share `mixer` (and `norm`), so replacing
        # the shared module on a single-sided no-op would break active
        # Mamba/MoE/Attention layers. `_apply_no_ops` must only replace
        # when BOTH sides are no-op.
        info = _ARCH_REGISTRY["NemotronHForCausalLM"]

        # Single-sided no-op must leave the shared mixer/norm intact.
        layer = MagicMock(spec=nn.Module)
        layer.mixer = MagicMock(spec=nn.Module)
        layer.norm = MagicMock(spec=nn.Module)
        orig_mixer, orig_norm = layer.mixer, layer.norm
        _apply_no_ops(layer, _block(attn_no_op=True, ffn_no_op=False), info)
        assert layer.mixer is orig_mixer
        assert layer.norm is orig_norm

        # Both-sided no-op replaces the shared module and norm.
        layer = MagicMock(spec=nn.Module)
        layer.mixer = MagicMock(spec=nn.Module)
        layer.norm = MagicMock(spec=nn.Module)
        _apply_no_ops(layer, _block(attn_no_op=True, ffn_no_op=True), info)
        assert isinstance(layer.mixer, NoOpAttention)
        assert isinstance(layer.norm, NoOpNorm)


class TestNoOpModules:
    def test_noop_attention_positional(self):
        x = torch.randn(4, 16)
        assert NoOpAttention()(x) is x

    def test_noop_attention_keyword(self):
        x = torch.randn(4, 16)
        assert NoOpAttention()(hidden_states=x) is x

    def test_noop_mlp_identity(self):
        x = torch.randn(4, 16)
        assert NoOpMLP()(x) is x

    def test_noop_norm_no_residual_returns_zeros(self):
        x = torch.randn(4, 16)
        out = NoOpNorm()(x)
        assert out.shape == x.shape
        assert out.eq(0).all()

    def test_noop_norm_with_residual_passes_through(self):
        x, r = torch.randn(4, 16), torch.randn(4, 16)
        out_x, out_r = NoOpNorm()(x, r)
        assert out_x is x
        assert out_r is r


class TestHasOverrides:
    @pytest.mark.parametrize(
        "entry, expected",
        [
            (_block(), False),
            (_block(kv_heads=2), True),
            (_block(intermediate_size=4096), True),
            (_block(hidden_act="gelu"), True),
            (_block(extra={"num_local_experts": 4}), True),
            (_block(attn_no_op=True), False),
            (_block(ffn_no_op=True), False),
            (_block(attn_no_op=True, ffn_no_op=True), False),
            (_block(kv_heads=2, attn_no_op=True), True),
        ],
    )
    def test_has_overrides(self, entry, expected):
        assert _has_overrides(entry) is expected


class TestUnregisterLayer:
    def _mock_vllm_config(self, keys):
        ctx = {k: "dummy" for k in keys}
        cc = _ns(static_forward_context=ctx, static_all_moe_layers=[])
        return _ns(compilation_config=cc)

    def test_removes_only_target_layer(self):
        vc = self._mock_vllm_config(
            [
                "model.layers.1.self_attn",
                "model.layers.1.mlp",
                "model.layers.10.self_attn",
                "model.layers.10.mlp",
            ]
        )
        _unregister_layer("model.layers.1", vc)
        ctx = vc.compilation_config.static_forward_context
        assert "model.layers.1.self_attn" not in ctx
        assert "model.layers.1.mlp" not in ctx
        assert "model.layers.10.self_attn" in ctx
        assert "model.layers.10.mlp" in ctx

    def test_empty_context(self):
        vc = self._mock_vllm_config([])
        _unregister_layer("model.layers.0", vc)
        assert len(vc.compilation_config.static_forward_context) == 0

    def test_deeply_nested_keys(self):
        vc = self._mock_vllm_config(
            [
                "model.layers.5.self_attn.o_proj",
                "model.layers.5.self_attn.qkv_proj",
                "model.layers.6.self_attn",
            ]
        )
        _unregister_layer("model.layers.5", vc)
        ctx = vc.compilation_config.static_forward_context
        assert len(ctx) == 1
        assert "model.layers.6.self_attn" in ctx


class TestAnyModelRegistryFlow:
    def test_anymodel_in_registry(self):
        from vllm.model_executor.models.registry import ModelRegistry

        assert "AnyModel" in ModelRegistry.models

    def test_resolve_wrapper_cls(self):
        from vllm.model_executor.models.anymodel import AnyModel

        mc = _ns(
            hf_config=_ns(
                base_architecture="LlamaForCausalLM",
                anymodel_arch_info=None,
            ),
        )
        wrapper = AnyModel.resolve_wrapper_cls(mc)
        assert issubclass(wrapper, AnyModel)
        assert "Llama" in wrapper.__name__

    def test_resolve_arch_raises_without_base_architecture(self):
        from vllm.model_executor.models.anymodel import AnyModel

        with pytest.raises(ValueError, match="base_architecture"):
            AnyModel._resolve_arch(_ns())


class TestArchInfoFromConfig:
    _MINIMAL = {
        "decoder_layer_module": ".llama",
        "decoder_layer_class": "LlamaDecoderLayer",
    }

    def test_returns_none_when_absent(self):
        assert _arch_info_from_config(_ns()) is None
        assert _arch_info_from_config(_ns(anymodel_arch_info=None)) is None

    def test_loads_from_dict(self):
        info = _arch_info_from_config(_ns(anymodel_arch_info=self._MINIMAL))
        assert info.decoder_layer_class == "LlamaDecoderLayer"
        assert info.attn_module == "self_attn"

    def test_loads_from_namespace(self):
        info = _arch_info_from_config(_ns(anymodel_arch_info=_ns(**self._MINIMAL)))
        assert info.decoder_layer_class == "LlamaDecoderLayer"

    def test_overrides_applied(self):
        d = {**self._MINIMAL, "attn_module": "attn", "ffn_module": "block_sparse_moe"}
        info = _arch_info_from_config(_ns(anymodel_arch_info=d))
        assert info.attn_module == "attn"
        assert info.ffn_module == "block_sparse_moe"

    def test_unknown_keys_ignored(self):
        d = {**self._MINIMAL, "future_field": 42}
        info = _arch_info_from_config(_ns(anymodel_arch_info=d))
        assert info is not None
        assert not hasattr(info, "future_field")

    def test_config_driven_takes_priority(self):
        custom = {**self._MINIMAL, "attn_module": "mixer"}
        info = _arch_info_from_config(_ns(anymodel_arch_info=custom))
        assert info.attn_module == "mixer"
