# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that GPTQ quantized layers with fused/packed modules are correctly
detected when modules_in_block_to_quantize contains the fused name.

Run `pytest tests/quantization/test_gptq_packed_modules.py -v`.
"""

from vllm.model_executor.layers.quantization.utils.gptq_utils import (
    is_layer_gptq_quantized,
)
from vllm.model_executor.model_loader.utils import (
    _expand_packed_modules_in_block_to_quantize,
)


class TestPackedModulesQuantizedDetection:
    """Regression tests for issue #43967.

    When a GPTQ model's quantize_config.json lists fused module names
    (e.g. ``mlp.gate_up_proj``) in ``modules_in_block_to_quantize`` and
    vLLM unpacks them via ``packed_modules_mapping``, the quantization
    check must still recognise the fused layer as quantized.
    """

    def test_fused_gate_up_proj_detected_with_expanded_layers(self):
        quantized_layers = [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
        ]
        fused_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}

        result = is_layer_gptq_quantized(
            prefix="model.layers.0.mlp.gate_up_proj",
            quantized_layers=quantized_layers,
            fused_mapping=fused_mapping,
        )
        assert result is True

    def test_fused_gate_up_proj_not_detected_with_unexpanded_layers(self):
        quantized_layers = [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "mlp.gate_up_proj",
        ]
        fused_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}

        # After _expand_packed_modules_in_block_to_quantize runs,
        # "mlp.gate_up_proj" should become "mlp.gate_proj" + "mlp.up_proj"
        class _FakeConfig:
            modules_in_block_to_quantize = list(quantized_layers)

        config = _FakeConfig()
        _expand_packed_modules_in_block_to_quantize(config, fused_mapping)

        result = is_layer_gptq_quantized(
            prefix="model.layers.0.mlp.gate_up_proj",
            quantized_layers=config.modules_in_block_to_quantize,
            fused_mapping=fused_mapping,
        )
        assert result is True

    def test_unfused_layer_still_detected(self):
        quantized_layers = ["self_attn.q_proj", "mlp.gate_proj"]
        fused_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}

        result = is_layer_gptq_quantized(
            prefix="model.layers.0.self_attn.q_proj",
            quantized_layers=quantized_layers,
            fused_mapping=fused_mapping,
        )
        assert result is True

    def test_unquantized_layer_not_detected(self):
        quantized_layers = ["self_attn.q_proj"]
        fused_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}

        result = is_layer_gptq_quantized(
            prefix="model.layers.0.mlp.down_proj",
            quantized_layers=quantized_layers,
            fused_mapping=fused_mapping,
        )
        assert result is False


class TestExpandPackedModulesInBlockToQuantize:
    def test_expands_fused_names(self):
        class _Cfg:
            modules_in_block_to_quantize = [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "mlp.gate_up_proj",
            ]

        packed = {"gate_up_proj": ["gate_proj", "up_proj"]}
        _expand_packed_modules_in_block_to_quantize(_Cfg, packed)

        assert _Cfg.modules_in_block_to_quantize == [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
        ]

    def test_no_match_unchanged(self):
        class _Cfg:
            modules_in_block_to_quantize = [
                "self_attn.q_proj",
                "mlp.gate_proj",
            ]

        packed = {"gate_up_proj": ["gate_proj", "up_proj"]}
        _expand_packed_modules_in_block_to_quantize(_Cfg, packed)

        assert _Cfg.modules_in_block_to_quantize == [
            "self_attn.q_proj",
            "mlp.gate_proj",
        ]

    def test_empty_modules_is_noop(self):
        class _Cfg:
            modules_in_block_to_quantize = []

        packed = {"gate_up_proj": ["gate_proj", "up_proj"]}
        _expand_packed_modules_in_block_to_quantize(_Cfg, packed)
        assert _Cfg.modules_in_block_to_quantize == []

    def test_missing_attr_is_noop(self):
        class _Cfg:
            pass

        packed = {"gate_up_proj": ["gate_proj", "up_proj"]}
        _expand_packed_modules_in_block_to_quantize(_Cfg, packed)

    def test_qkv_proj_expansion(self):
        class _Cfg:
            modules_in_block_to_quantize = [
                "self_attn.qkv_proj",
                "mlp.gate_up_proj",
            ]

        packed = {
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
            "gate_up_proj": ["gate_proj", "up_proj"],
        }
        _expand_packed_modules_in_block_to_quantize(_Cfg, packed)

        assert _Cfg.modules_in_block_to_quantize == [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
        ]
