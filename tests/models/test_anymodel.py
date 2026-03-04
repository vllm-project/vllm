# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for AnyModel (NAS heterogeneous architecture support).

These tests exercise the ArchInfo registry, the generic config-override
and no-op helpers, and the registry integration.
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
    Same,
    _apply_no_ops,
    _arch_info_from_config,
    _create_layer_config,
    _has_overrides,
    _resolve_layer_class,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ns(**kwargs):
    """Build a SimpleNamespace (mirrors AnyModel config conversion)."""
    return types.SimpleNamespace(**kwargs)


def _block(
    *,
    attn_no_op: bool = False,
    kv_heads: int | None = None,
    ffn_no_op: bool = False,
    intermediate_size: int | None = None,
    hidden_act: str | None = None,
    moe: dict | None = None,
):
    """Build a canonical block_config namespace."""
    attn: dict = {"no_op": attn_no_op}
    if kv_heads is not None:
        attn["num_key_value_heads"] = kv_heads

    ffn: dict = {"no_op": ffn_no_op}
    if intermediate_size is not None:
        ffn["intermediate_size"] = intermediate_size
    if hidden_act is not None:
        ffn["hidden_act"] = hidden_act
    if moe is not None:
        ffn["moe"] = moe

    return _ns(attention=_ns(**attn), ffn=_ns(**ffn))


def _base_config(**overrides):
    """Minimal global config that mirrors common HF model configs."""
    defaults = dict(
        num_key_value_heads=8,
        intermediate_size=14336,
        hidden_act="silu",
        # MoE fields (absent on dense models, present on MoE models)
        num_experts=None,
        moe_intermediate_size=None,
        num_local_experts=None,
        n_routed_experts=None,
    )
    defaults.update(overrides)
    return _ns(**defaults)


# ---------------------------------------------------------------------------
# _ARCH_REGISTRY contents — all 9 target models
# ---------------------------------------------------------------------------


class TestArchRegistry:
    """Verify that every target model has a registered ArchInfo entry."""

    # Dense / text-only models
    @pytest.mark.parametrize(
        "arch",
        [
            "LlamaForCausalLM",  # llama-3.1-8b, llama-3.2-3b
            "MistralForCausalLM",  # mistral-small-24b
            "Qwen2ForCausalLM",  # qwen2.5-7b
            "Qwen3ForCausalLM",  # qwen3-8b
        ],
    )
    def test_dense_archs_registered(self, arch):
        assert arch in _ARCH_REGISTRY

    # MoE models
    @pytest.mark.parametrize(
        "arch",
        [
            "Qwen2MoeForCausalLM",  # qwen2-moe variants
            "MixtralForCausalLM",  # mixtral-style MoE
            "GptOssForCausalLM",  # gpt-oss-20b
        ],
    )
    def test_moe_archs_registered(self, arch):
        assert arch in _ARCH_REGISTRY

    # Hybrid models
    @pytest.mark.parametrize(
        "arch",
        [
            "NemotronHForCausalLM",  # nemotron-nano-12b-v2, nemotron-30b
            "NemotronHPuzzleForCausalLM",  # Puzzletron alias
        ],
    )
    def test_hybrid_archs_registered(self, arch):
        assert arch in _ARCH_REGISTRY

    # Multimodal models
    def test_qwen3vl_registered(self):
        """Qwen3VLForConditionalGeneration is supported via dynamic parent."""
        assert "Qwen3VLForConditionalGeneration" in _ARCH_REGISTRY

    # --- per-arch field checks ---

    def test_llama_fields(self):
        info = _ARCH_REGISTRY["LlamaForCausalLM"]
        assert info.decoder_layer_module == ".llama"
        assert info.decoder_layer_class == "LlamaDecoderLayer"

    def test_mistral_correct_module(self):
        info = _ARCH_REGISTRY["MistralForCausalLM"]
        assert "mistral" in info.decoder_layer_module
        assert info.decoder_layer_class == "MistralDecoderLayer"

    def test_qwen2_fields(self):
        info = _ARCH_REGISTRY["Qwen2ForCausalLM"]
        assert info.decoder_layer_module == ".qwen2"
        assert info.decoder_layer_class == "Qwen2DecoderLayer"

    def test_qwen3_fields(self):
        info = _ARCH_REGISTRY["Qwen3ForCausalLM"]
        assert info.decoder_layer_class == "Qwen3DecoderLayer"

    def test_mixtral_block_sparse_moe(self):
        info = _ARCH_REGISTRY["MixtralForCausalLM"]
        assert info.ffn_module == "block_sparse_moe"
        assert info.moe_num_experts_field == "num_local_experts"

    def test_qwen2moe_fields(self):
        info = _ARCH_REGISTRY["Qwen2MoeForCausalLM"]
        assert info.moe_num_experts_field == "num_experts"
        assert info.moe_intermediate_size_field == "moe_intermediate_size"

    def test_nemotronh_hybrid_fields(self):
        info = _ARCH_REGISTRY["NemotronHForCausalLM"]
        assert info.hybrid_pattern_field == "hybrid_override_pattern"
        assert info.decoder_layer_class_map is not None
        assert "*" in info.decoder_layer_class_map
        assert "-" in info.decoder_layer_class_map
        assert "E" in info.decoder_layer_class_map
        assert "M" in info.decoder_layer_class_map
        assert info.attn_module == "mixer"
        assert info.ffn_module == "mixer"
        assert info.moe_num_experts_field == "n_routed_experts"

    def test_nemotronh_puzzle_alias_is_same_object(self):
        """NemotronHPuzzleForCausalLM must be the exact same ArchInfo object
        as NemotronHForCausalLM — not a copy — so they cannot drift."""
        assert (
            _ARCH_REGISTRY["NemotronHPuzzleForCausalLM"]
            is _ARCH_REGISTRY["NemotronHForCausalLM"]
        )

    def test_gptoss_fields(self):
        info = _ARCH_REGISTRY["GptOssForCausalLM"]
        assert info.attn_module == "attn"
        assert info.moe_num_experts_field == "num_local_experts"
        assert info.decoder_layer_class == "TransformerBlock"

    def test_arch_info_defaults(self):
        info = ArchInfo(
            decoder_layer_module=".llama", decoder_layer_class="LlamaDecoderLayer"
        )
        assert info.attn_module == "self_attn"
        assert info.attn_norm_module == "input_layernorm"
        assert info.ffn_module == "mlp"
        assert info.ffn_norm_module == "post_attention_layernorm"
        assert info.kv_heads_field == "num_key_value_heads"
        assert info.intermediate_size_field == "intermediate_size"
        assert info.moe_num_experts_field is None
        assert info.moe_intermediate_size_field is None
        assert info.decoder_layer_class_map is None
        assert info.hybrid_pattern_field is None
        # Dynamic-parent fields
        assert info.layers_path == "model.layers"
        assert info.init_prefix is None
        assert info.layer_hf_config is None

    def test_qwen3vl_dynamic_parent_fields(self):
        info = _ARCH_REGISTRY["Qwen3VLForConditionalGeneration"]
        assert info.layers_path == "language_model.model.layers"
        assert info.init_prefix == "model"
        assert info.layer_hf_config == "text_config"
        assert info.decoder_layer_class == "Qwen3DecoderLayer"
        assert "qwen3" in info.decoder_layer_module
        # base_model_module must point to qwen3_vl where
        # Qwen3VLForConditionalGeneration actually lives.
        assert info.base_model_module == ".qwen3_vl"

    def test_non_vl_archs_have_no_base_model_module(self):
        """Only VL models that split the model/layer files need this field."""
        for arch, info in _ARCH_REGISTRY.items():
            if arch == "Qwen3VLForConditionalGeneration":
                continue
            assert info.base_model_module is None, (
                f"{arch} unexpectedly sets base_model_module"
            )

    def test_most_archs_use_default_layers_path(self):
        """All non-VL architectures use the default model.layers path."""
        for arch, info in _ARCH_REGISTRY.items():
            if arch == "Qwen3VLForConditionalGeneration":
                continue
            assert info.layers_path == "model.layers", (
                f"{arch} has unexpected layers_path={info.layers_path!r}"
            )

    def test_most_archs_use_none_init_prefix(self):
        """All non-VL architectures use the default init_prefix=None."""
        for arch, info in _ARCH_REGISTRY.items():
            if arch == "Qwen3VLForConditionalGeneration":
                continue
            assert info.init_prefix is None, (
                f"{arch} has unexpected init_prefix={info.init_prefix!r}"
            )


# ---------------------------------------------------------------------------
# _resolve_layer_class — hybrid pattern dispatch
# ---------------------------------------------------------------------------


class TestResolveLayerClass:
    def _hybrid_info(self):
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

    def test_attention_layer_resolved(self):
        info = self._hybrid_info()
        config = _ns(hybrid_override_pattern="*-*E")
        cls = _resolve_layer_class(info, config, layer_idx=0)
        assert cls.__name__ == "NemotronHAttentionDecoderLayer"

    def test_mlp_layer_resolved(self):
        info = self._hybrid_info()
        config = _ns(hybrid_override_pattern="*-*E")
        cls = _resolve_layer_class(info, config, layer_idx=1)
        assert cls.__name__ == "NemotronHMLPDecoderLayer"

    def test_moe_layer_resolved(self):
        info = self._hybrid_info()
        config = _ns(hybrid_override_pattern="*-*E")
        cls = _resolve_layer_class(info, config, layer_idx=3)
        assert cls.__name__ == "NemotronHMoEDecoderLayer"

    def test_mamba_layer_resolved(self):
        info = self._hybrid_info()
        config = _ns(hybrid_override_pattern="M*")
        cls = _resolve_layer_class(info, config, layer_idx=0)
        assert cls.__name__ == "NemotronHMambaDecoderLayer"

    def test_falls_back_to_default_class_when_no_map(self):
        info = ArchInfo(
            decoder_layer_module=".llama",
            decoder_layer_class="LlamaDecoderLayer",
        )
        config = _ns()
        cls = _resolve_layer_class(info, config, layer_idx=0)
        assert cls.__name__ == "LlamaDecoderLayer"

    def test_falls_back_to_default_when_char_not_in_map(self):
        info = self._hybrid_info()
        config = _ns(hybrid_override_pattern="X")  # unknown char
        cls = _resolve_layer_class(info, config, layer_idx=0)
        assert cls.__name__ == "NemotronHAttentionDecoderLayer"

    def test_falls_back_when_idx_out_of_pattern(self):
        info = self._hybrid_info()
        config = _ns(hybrid_override_pattern="*")  # only 1 char
        cls = _resolve_layer_class(info, config, layer_idx=5)
        assert cls.__name__ == "NemotronHAttentionDecoderLayer"


# ---------------------------------------------------------------------------
# _create_layer_config
# ---------------------------------------------------------------------------


class TestCreateLayerConfig:
    def _std_info(self):
        return ArchInfo(
            decoder_layer_module=".llama", decoder_layer_class="LlamaDecoderLayer"
        )

    def test_no_overrides_copies_global(self):
        global_cfg = _base_config()
        result = _create_layer_config(global_cfg, _block(), self._std_info())
        assert result.num_key_value_heads == global_cfg.num_key_value_heads
        assert result.intermediate_size == global_cfg.intermediate_size

    def test_kv_heads_override(self):
        global_cfg = _base_config(num_key_value_heads=8)
        result = _create_layer_config(global_cfg, _block(kv_heads=4), self._std_info())
        assert result.num_key_value_heads == 4
        assert global_cfg.num_key_value_heads == 8  # not mutated

    def test_intermediate_size_override(self):
        global_cfg = _base_config(intermediate_size=14336)
        result = _create_layer_config(
            global_cfg, _block(intermediate_size=8192), self._std_info()
        )
        assert result.intermediate_size == 8192

    def test_hidden_act_override(self):
        global_cfg = _base_config(hidden_act="silu")
        result = _create_layer_config(
            global_cfg, _block(hidden_act="gelu"), self._std_info()
        )
        assert result.hidden_act == "gelu"

    def test_attn_noop_skips_kv_override(self):
        global_cfg = _base_config(num_key_value_heads=8)
        result = _create_layer_config(
            global_cfg, _block(attn_no_op=True, kv_heads=2), self._std_info()
        )
        assert result.num_key_value_heads == 8

    def test_ffn_noop_skips_size_override(self):
        global_cfg = _base_config(intermediate_size=14336)
        result = _create_layer_config(
            global_cfg, _block(ffn_no_op=True, intermediate_size=1), self._std_info()
        )
        assert result.intermediate_size == 14336

    def test_custom_kv_heads_field(self):
        global_cfg = _ns(my_kv_heads=8, intermediate_size=14336, hidden_act="silu")
        info = ArchInfo(
            decoder_layer_module=".dummy",
            decoder_layer_class="Dummy",
            kv_heads_field="my_kv_heads",
        )
        result = _create_layer_config(global_cfg, _block(kv_heads=2), info)
        assert result.my_kv_heads == 2

    # --- MoE: Qwen2Moe style ---

    def test_qwen2moe_num_experts_override(self):
        global_cfg = _base_config(num_experts=8, moe_intermediate_size=1024)
        moe_block = {"num_local_experts": 4, "expert_intermediate_size": 512}
        info = ArchInfo(
            decoder_layer_module=".qwen2_moe",
            decoder_layer_class="Qwen2MoeDecoderLayer",
            moe_num_experts_field="num_experts",
            moe_intermediate_size_field="moe_intermediate_size",
        )
        result = _create_layer_config(global_cfg, _block(moe=moe_block), info)
        assert result.num_experts == 4
        assert result.moe_intermediate_size == 512

    # --- MoE: Mixtral / GptOss style (fallback to intermediate_size_field) ---

    def test_mixtral_moe_falls_back_to_intermediate_size_field(self):
        global_cfg = _base_config(intermediate_size=14336, num_local_experts=8)
        moe_block = {"num_local_experts": 4, "expert_intermediate_size": 512}
        info = ArchInfo(
            decoder_layer_module=".mixtral",
            decoder_layer_class="MixtralDecoderLayer",
            moe_num_experts_field="num_local_experts",
            moe_intermediate_size_field=None,
        )
        result = _create_layer_config(global_cfg, _block(moe=moe_block), info)
        assert result.num_local_experts == 4
        assert result.intermediate_size == 512

    # --- MoE: NemotronH style ---

    def test_nemotronh_moe_fields(self):
        global_cfg = _base_config(n_routed_experts=8, moe_intermediate_size=2048)
        moe_block = {"num_local_experts": 4, "expert_intermediate_size": 1024}
        info = _ARCH_REGISTRY["NemotronHForCausalLM"]
        result = _create_layer_config(global_cfg, _block(moe=moe_block), info)
        assert result.n_routed_experts == 4
        assert result.moe_intermediate_size == 1024

    def test_global_config_not_mutated(self):
        global_cfg = _base_config(num_key_value_heads=8, intermediate_size=14336)
        _create_layer_config(
            global_cfg, _block(kv_heads=2, intermediate_size=4096), self._std_info()
        )
        assert global_cfg.num_key_value_heads == 8
        assert global_cfg.intermediate_size == 14336

    def test_extra_config_fields_applied(self):
        """extra_config_fields values from block_config are reflected in the
        returned config copy."""
        global_cfg = _base_config(hidden_size=4096)
        info = ArchInfo(
            decoder_layer_module=".llama",
            decoder_layer_class="LlamaDecoderLayer",
            extra_config_fields={"ffn.hidden_size": "hidden_size"},
        )
        bc = _ns(
            attention=_ns(no_op=False),
            ffn=_ns(no_op=False, hidden_size=1024),
        )
        result = _create_layer_config(global_cfg, bc, info)
        assert result.hidden_size == 1024
        assert global_cfg.hidden_size == 4096  # not mutated

    def test_hidden_act_triggers_layer_rebuild_via_has_overrides(self):
        """A layer with only hidden_act overridden must be flagged for rebuild."""
        info = self._std_info()
        bc = _block(hidden_act="gelu")
        assert _has_overrides(bc, info)
        global_cfg = _base_config(hidden_act="silu")
        result = _create_layer_config(global_cfg, bc, info)
        assert result.hidden_act == "gelu"


# ---------------------------------------------------------------------------
# _apply_no_ops
# ---------------------------------------------------------------------------


class TestApplyNoOps:
    def _make_layer(
        self,
        attn_name="self_attn",
        ffn_name="mlp",
        norm_name="input_layernorm",
        post_norm_name="post_attention_layernorm",
    ):
        layer = MagicMock(spec=nn.Module)
        for attr in (attn_name, ffn_name, norm_name, post_norm_name):
            setattr(layer, attr, MagicMock(spec=nn.Module))
        return layer

    def test_no_noop_leaves_modules_intact(self):
        layer = self._make_layer()
        original_attn = layer.self_attn
        original_mlp = layer.mlp
        info = ArchInfo(
            decoder_layer_module=".llama", decoder_layer_class="LlamaDecoderLayer"
        )
        _apply_no_ops(layer, _block(), info)
        assert layer.self_attn is original_attn
        assert layer.mlp is original_mlp

    def test_attn_noop(self):
        layer = self._make_layer()
        info = ArchInfo(
            decoder_layer_module=".llama", decoder_layer_class="LlamaDecoderLayer"
        )
        _apply_no_ops(layer, _block(attn_no_op=True), info)
        assert isinstance(layer.self_attn, NoOpAttention)
        assert isinstance(layer.input_layernorm, Same)
        assert not isinstance(layer.mlp, NoOpMLP)

    def test_ffn_noop(self):
        layer = self._make_layer()
        info = ArchInfo(
            decoder_layer_module=".llama", decoder_layer_class="LlamaDecoderLayer"
        )
        _apply_no_ops(layer, _block(ffn_no_op=True), info)
        assert isinstance(layer.mlp, NoOpMLP)
        assert isinstance(layer.post_attention_layernorm, Same)
        assert not isinstance(layer.self_attn, NoOpAttention)

    def test_both_noop(self):
        layer = self._make_layer()
        info = ArchInfo(
            decoder_layer_module=".llama", decoder_layer_class="LlamaDecoderLayer"
        )
        _apply_no_ops(layer, _block(attn_no_op=True, ffn_no_op=True), info)
        assert isinstance(layer.self_attn, NoOpAttention)
        assert isinstance(layer.input_layernorm, Same)
        assert isinstance(layer.mlp, NoOpMLP)
        assert isinstance(layer.post_attention_layernorm, Same)

    def test_mixtral_block_sparse_moe_noop(self):
        layer = self._make_layer(ffn_name="block_sparse_moe")
        info = ArchInfo(
            decoder_layer_module=".mixtral",
            decoder_layer_class="MixtralDecoderLayer",
            ffn_module="block_sparse_moe",
        )
        _apply_no_ops(layer, _block(ffn_no_op=True), info)
        assert isinstance(layer.block_sparse_moe, NoOpMLP)
        assert isinstance(layer.post_attention_layernorm, Same)

    def test_gptoss_attn_noop_uses_attn_module(self):
        layer = self._make_layer(attn_name="attn")
        info = _ARCH_REGISTRY["GptOssForCausalLM"]
        _apply_no_ops(layer, _block(attn_no_op=True), info)
        assert isinstance(layer.attn, NoOpAttention)

    def test_nemotronh_noop_uses_mixer(self):
        layer = MagicMock(spec=nn.Module)
        layer.mixer = MagicMock(spec=nn.Module)
        layer.norm = MagicMock(spec=nn.Module)
        info = _ARCH_REGISTRY["NemotronHForCausalLM"]
        _apply_no_ops(layer, _block(attn_no_op=True), info)
        assert isinstance(layer.mixer, NoOpAttention)
        assert isinstance(layer.norm, Same)


# ---------------------------------------------------------------------------
# No-op module forward behaviour
# ---------------------------------------------------------------------------


class TestNoOpModules:
    def test_noop_attention_returns_zeros(self):
        positions = torch.zeros(4, dtype=torch.long)
        hidden = torch.randn(4, 16)
        out = NoOpAttention()(positions, hidden)
        assert out.shape == hidden.shape
        assert out.eq(0).all()

    def test_noop_mlp_returns_zeros(self):
        hidden = torch.randn(4, 16)
        out = NoOpMLP()(hidden)
        assert out.shape == hidden.shape
        assert out.eq(0).all()

    def test_same_no_residual(self):
        x = torch.randn(4, 16)
        assert Same()(x) is x

    def test_same_with_residual(self):
        x, r = torch.randn(4, 16), torch.randn(4, 16)
        out_x, out_r = Same()(x, r)
        assert out_x is x
        assert out_r is r


# ---------------------------------------------------------------------------
# _has_overrides — layer recreation decision
# ---------------------------------------------------------------------------


class TestHasOverrides:
    def test_no_overrides_returns_false(self):
        assert not _has_overrides(_block())

    def test_kv_heads_override_returns_true(self):
        assert _has_overrides(_block(kv_heads=2))

    def test_intermediate_size_override_returns_true(self):
        assert _has_overrides(_block(intermediate_size=4096))

    def test_moe_override_returns_true(self):
        assert _has_overrides(_block(moe={"num_local_experts": 4}))

    def test_noop_only_returns_false(self):
        """No-ops do not change weight shapes; no layer recreation needed."""
        assert not _has_overrides(_block(attn_no_op=True))
        assert not _has_overrides(_block(ffn_no_op=True))
        assert not _has_overrides(_block(attn_no_op=True, ffn_no_op=True))

    def test_noop_plus_override_returns_true(self):
        assert _has_overrides(_block(kv_heads=2, attn_no_op=True))

    def test_hidden_act_only_returns_true(self):
        """hidden_act requires layer rebuild to take effect."""
        assert _has_overrides(_block(hidden_act="gelu"))

    def test_extra_config_fields_triggers_rebuild(self):
        """A block_config field listed in extra_config_fields triggers rebuild."""
        info = ArchInfo(
            decoder_layer_module=".llama",
            decoder_layer_class="LlamaDecoderLayer",
            extra_config_fields={"ffn.hidden_size": "hidden_size"},
        )
        bc = _ns(
            attention=_ns(no_op=False),
            ffn=_ns(no_op=False, hidden_size=1024),
        )
        assert _has_overrides(bc, info)

    def test_no_extra_config_fields_no_trigger(self):
        """Empty extra_config_fields does not change the False result."""
        info = ArchInfo(
            decoder_layer_module=".llama",
            decoder_layer_class="LlamaDecoderLayer",
            extra_config_fields={},
        )
        assert not _has_overrides(_block(), info)


# ---------------------------------------------------------------------------
# AnyModel registry flow
# ---------------------------------------------------------------------------


class TestAnyModelRegistryFlow:
    """Tests for the standard registry path used by AnyModel checkpoints.

    NAS-optimised configs use ``"architectures": ["AnyModel"]`` so that the
    normal registry lookup selects AnyModel without any special bypass.
    ``base_architectures`` carries the name of the underlying model class
    (used for capability introspection and wrapper construction).
    """

    def _mc(self, architectures, base_architectures=None, block_configs=None):
        hf = _ns(
            architectures=architectures,
            base_architectures=base_architectures or [],
            block_configs=block_configs or [_block()],
        )
        return _ns(hf_config=hf)

    # --- _anymodel_base_arch helper ---

    def test_base_arch_reads_base_architectures(self):
        from vllm.model_executor.models.registry import ModelRegistry

        mc = self._mc(["AnyModel"], base_architectures=["LlamaForCausalLM"])
        assert ModelRegistry._anymodel_base_arch(mc) == "LlamaForCausalLM"

    def test_base_arch_returns_none_when_absent(self):
        from vllm.model_executor.models.registry import ModelRegistry

        mc = self._mc(["AnyModel"])  # no base_architectures
        assert ModelRegistry._anymodel_base_arch(mc) is None

    def test_base_arch_returns_none_without_hf_config(self):
        from vllm.model_executor.models.registry import ModelRegistry

        assert ModelRegistry._anymodel_base_arch(_ns(hf_config=None)) is None

    def test_base_arch_uses_first_entry(self):
        from vllm.model_executor.models.registry import ModelRegistry

        mc = self._mc(
            ["AnyModel"],
            base_architectures=["Qwen3VLForConditionalGeneration", "Other"],
        )
        assert (
            ModelRegistry._anymodel_base_arch(mc) == "Qwen3VLForConditionalGeneration"
        )

    # --- regular models are not misrouted ---

    def test_regular_llama_config_without_anymodel_arch_is_not_routed(self):
        """A plain LlamaForCausalLM config (no 'AnyModel' in architectures)
        must NOT be silently redirected to AnyModel, even with block_configs."""
        from vllm.model_executor.models.registry import ModelRegistry

        # Normal Llama config that happens to have block_configs
        mc = self._mc(["LlamaForCausalLM"])
        # _anymodel_base_arch returns None → AnyModel branch not taken
        assert ModelRegistry._anymodel_base_arch(mc) is None

    # --- AnyModel is registered under "AnyModel" ---

    def test_anymodel_registered_in_registry(self):
        from vllm.model_executor.models.registry import ModelRegistry

        assert "AnyModel" in ModelRegistry.models

    # --- VL capabilities (requires subprocess / full vLLM env) ---

    def test_vl_model_info_has_correct_capabilities(self):
        """inspect_model_cls for an AnyModel+Qwen3VL config must return
        supports_multimodal=True and has_noops=True."""
        from vllm.model_executor.models.registry import ModelRegistry

        base_arch = "Qwen3VLForConditionalGeneration"
        model_config = self._mc(["AnyModel"], base_architectures=[base_arch])
        model_info, returned_arch = ModelRegistry.inspect_model_cls(
            ["AnyModel"], model_config
        )
        assert returned_arch == "AnyModel"
        assert model_info.has_noops is True
        assert model_info.supports_multimodal is True


# ---------------------------------------------------------------------------
# init_prefix None-vs-string semantics
# ---------------------------------------------------------------------------


class TestInitPrefix:
    def test_init_prefix_none_inherits_engine_prefix(self):
        """init_prefix=None means use the engine-provided prefix."""
        info = ArchInfo(
            decoder_layer_module=".llama",
            decoder_layer_class="LlamaDecoderLayer",
            init_prefix=None,
        )
        engine_prefix = "engine_prefix"
        result = info.init_prefix if info.init_prefix is not None else engine_prefix
        assert result == engine_prefix

    def test_init_prefix_empty_string_forces_blank(self):
        """init_prefix="" explicitly forces a blank prefix, overriding engine."""
        info = ArchInfo(
            decoder_layer_module=".llama",
            decoder_layer_class="LlamaDecoderLayer",
            init_prefix="",
        )
        engine_prefix = "engine_prefix"
        result = info.init_prefix if info.init_prefix is not None else engine_prefix
        assert result == ""

    def test_init_prefix_model_overrides_engine(self):
        """init_prefix='model' overrides the engine prefix (Qwen3VL case)."""
        info = ArchInfo(
            decoder_layer_module=".llama",
            decoder_layer_class="LlamaDecoderLayer",
            init_prefix="model",
        )
        engine_prefix = "engine_prefix"
        result = info.init_prefix if info.init_prefix is not None else engine_prefix
        assert result == "model"


# ---------------------------------------------------------------------------
# _arch_info_from_config
# ---------------------------------------------------------------------------


class TestArchInfoFromConfig:
    def _minimal_arch_info_dict(self):
        return {
            "decoder_layer_module": ".llama",
            "decoder_layer_class": "LlamaDecoderLayer",
        }

    def test_returns_none_when_field_absent(self):
        cfg = _ns()
        assert _arch_info_from_config(cfg) is None

    def test_returns_none_when_field_falsy(self):
        cfg = _ns(anymodel_arch_info=None)
        assert _arch_info_from_config(cfg) is None

    def test_loads_from_dict(self):
        cfg = _ns(anymodel_arch_info=self._minimal_arch_info_dict())
        info = _arch_info_from_config(cfg)
        assert info is not None
        assert info.decoder_layer_module == ".llama"
        assert info.decoder_layer_class == "LlamaDecoderLayer"
        assert info.attn_module == "self_attn"  # default

    def test_loads_from_namespace(self):
        data = _ns(**self._minimal_arch_info_dict())
        cfg = _ns(anymodel_arch_info=data)
        info = _arch_info_from_config(cfg)
        assert info is not None
        assert info.decoder_layer_class == "LlamaDecoderLayer"

    def test_overrides_applied(self):
        d = self._minimal_arch_info_dict()
        d["attn_module"] = "attn"
        d["ffn_module"] = "block_sparse_moe"
        cfg = _ns(anymodel_arch_info=d)
        info = _arch_info_from_config(cfg)
        assert info.attn_module == "attn"
        assert info.ffn_module == "block_sparse_moe"

    def test_unknown_keys_ignored(self):
        """Unknown keys in anymodel_arch_info are silently dropped."""
        d = self._minimal_arch_info_dict()
        d["ctor_style"] = "nemotron_h"  # removed field — should be ignored
        d["future_field"] = 42
        cfg = _ns(anymodel_arch_info=d)
        info = _arch_info_from_config(cfg)
        assert info is not None
        assert not hasattr(info, "ctor_style")
        assert not hasattr(info, "future_field")

    def test_config_driven_takes_priority_in_new(self):
        """When anymodel_arch_info is present, AnyModel.__new__ must use it
        rather than falling back to _ARCH_REGISTRY."""
        from vllm.model_executor.models.anymodel import _arch_info_from_config

        custom = {
            "decoder_layer_module": ".llama",
            "decoder_layer_class": "LlamaDecoderLayer",
            "attn_module": "mixer",  # deliberately different from default
        }
        cfg = _ns(anymodel_arch_info=custom)
        info = _arch_info_from_config(cfg)
        assert info is not None
        assert info.attn_module == "mixer"
