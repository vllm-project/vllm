# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for AnyModel (NAS heterogeneous architecture support).

These tests exercise the ArchInfo registry, the generic config-override
and no-op helpers, and the _should_use_anymodel detection logic.
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
    _create_layer_config,
    _resolve_layer_class,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ns(**kwargs):
    """Build a SimpleNamespace (mirrors AnyModelForCausalLMConfig conversion)."""
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

    def test_qwen3vl_not_registered_with_explanation(self):
        """Qwen3VL is multimodal — needs AnyModelForConditionalGeneration."""
        assert "Qwen3VLForConditionalGeneration" not in _ARCH_REGISTRY

    # --- per-arch field checks ---

    def test_llama_vllm_config_ctor(self):
        assert _ARCH_REGISTRY["LlamaForCausalLM"].ctor_style == "vllm_config"

    def test_mistral_vllm_config_ctor_and_correct_module(self):
        info = _ARCH_REGISTRY["MistralForCausalLM"]
        assert info.ctor_style == "vllm_config"
        assert "mistral" in info.decoder_layer_module
        assert info.decoder_layer_class == "MistralDecoderLayer"

    def test_qwen2_standard_ctor(self):
        assert _ARCH_REGISTRY["Qwen2ForCausalLM"].ctor_style == "standard"

    def test_qwen3_standard_ctor(self):
        info = _ARCH_REGISTRY["Qwen3ForCausalLM"]
        assert info.ctor_style == "standard"
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
        assert info.ctor_style == "nemotron_h"
        assert info.hybrid_pattern_field == "hybrid_override_pattern"
        assert info.decoder_layer_class_map is not None
        assert "*" in info.decoder_layer_class_map
        assert "-" in info.decoder_layer_class_map
        assert "E" in info.decoder_layer_class_map
        assert "M" in info.decoder_layer_class_map
        assert info.attn_module == "mixer"
        assert info.ffn_module == "mixer"
        assert info.moe_num_experts_field == "n_routed_experts"

    def test_nemotronh_puzzle_alias_matches_nemotronh(self):
        """NemotronHPuzzleForCausalLM should be structurally identical."""
        a = _ARCH_REGISTRY["NemotronHForCausalLM"]
        b = _ARCH_REGISTRY["NemotronHPuzzleForCausalLM"]
        assert a.decoder_layer_class_map == b.decoder_layer_class_map
        assert a.hybrid_pattern_field == b.hybrid_pattern_field
        assert a.ctor_style == b.ctor_style

    def test_gptoss_fields(self):
        info = _ARCH_REGISTRY["GptOssForCausalLM"]
        assert info.ctor_style == "gpt_oss"
        assert info.attn_module == "attn"
        assert info.moe_num_experts_field == "num_local_experts"
        assert info.decoder_layer_class == "TransformerBlock"

    def test_arch_info_defaults(self):
        info = ArchInfo(
            decoder_layer_module=".llama", decoder_layer_class="LlamaDecoderLayer"
        )
        assert info.ctor_style == "standard"
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


# ---------------------------------------------------------------------------
# _resolve_layer_class — hybrid pattern dispatch
# ---------------------------------------------------------------------------


class TestResolveLayerClass:
    def _hybrid_info(self):
        return ArchInfo(
            decoder_layer_module=".nemotron_h",
            decoder_layer_class="NemotronHAttentionDecoderLayer",
            ctor_style="nemotron_h",
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
# _should_use_anymodel — registry integration
# ---------------------------------------------------------------------------


class TestShouldUseAnymodel:
    def _mc(self, architectures, block_configs):
        hf = _ns(architectures=architectures, block_configs=block_configs)
        return _ns(hf_config=hf)

    def test_triggers_for_known_arch(self):
        from vllm.model_executor.models.registry import ModelRegistry

        assert ModelRegistry._should_use_anymodel(
            self._mc(["LlamaForCausalLM"], [_block()] * 2)
        )

    def test_no_block_configs_no_trigger(self):
        from vllm.model_executor.models.registry import ModelRegistry

        mc = _ns(hf_config=_ns(architectures=["LlamaForCausalLM"], block_configs=None))
        assert not ModelRegistry._should_use_anymodel(mc)

    def test_unknown_arch_no_trigger(self):
        from vllm.model_executor.models.registry import ModelRegistry

        assert not ModelRegistry._should_use_anymodel(
            self._mc(["UnknownModelForCausalLM"], [_block()])
        )

    def test_empty_architectures_no_trigger(self):
        from vllm.model_executor.models.registry import ModelRegistry

        mc = _ns(hf_config=_ns(architectures=[], block_configs=[_block()]))
        assert not ModelRegistry._should_use_anymodel(mc)

    def test_no_hf_config_no_trigger(self):
        from vllm.model_executor.models.registry import ModelRegistry

        assert not ModelRegistry._should_use_anymodel(_ns(hf_config=None))

    @pytest.mark.parametrize(
        "arch",
        [
            # All 9 target model architectures (except Qwen3VL — multimodal)
            "LlamaForCausalLM",  # llama-3.1-8b, llama-3.2-3b
            "MistralForCausalLM",  # mistral-small-24b
            "Qwen2ForCausalLM",  # qwen2.5-7b
            "Qwen3ForCausalLM",  # qwen3-8b
            "Qwen2MoeForCausalLM",
            "MixtralForCausalLM",
            "NemotronHForCausalLM",  # nemotron-nano-12b-v2, nemotron-30b
            "NemotronHPuzzleForCausalLM",
            "GptOssForCausalLM",  # gpt-oss-20b
        ],
    )
    def test_triggers_for_all_registered_archs(self, arch):
        from vllm.model_executor.models.registry import ModelRegistry

        assert ModelRegistry._should_use_anymodel(self._mc([arch], [_block()]))

    def test_qwen3vl_does_not_trigger(self):
        """Qwen3VL is multimodal and not yet supported by AnyModelForCausalLM."""
        from vllm.model_executor.models.registry import ModelRegistry

        assert not ModelRegistry._should_use_anymodel(
            self._mc(["Qwen3VLForConditionalGeneration"], [_block()])
        )
