# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import MagicMock
from vllm.config import DeviceConfig, ModelConfig, VllmConfig
from vllm.config.quantization import QuantizationConfigArgs, QuantSpec, resolve_quantization_config
from vllm.config.vllm import set_current_vllm_config
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig
from vllm.model_executor.layers.quantization.online.fp8 import Fp8PerTensorOnlineLinearMethod


class TestPerLayerOnlineQuantization(unittest.TestCase):

    def test_quantization_config_args_online_coercion(self):
        config = QuantizationConfigArgs(
            online={
                "re:.*self_attn.*": "fp8_per_tensor",
                "lm_head": "mxfp8",
            }
        )
        self.assertIn("re:.*self_attn.*", config.online)
        self.assertIsInstance(config.online["re:.*self_attn.*"], QuantSpec)
        self.assertIn("lm_head", config.online)
        self.assertIsInstance(config.online["lm_head"], QuantSpec)

    def test_quantization_config_args_invalid_regex(self):
        with self.assertRaises(ValueError):
            QuantizationConfigArgs(
                online={
                    "re:[invalid_regex": "fp8_per_tensor",
                }
            )

    def test_quantization_config_args_online_ignore_collision(self):
        with self.assertRaises(ValueError):
            QuantizationConfigArgs(
                online={
                    "lm_head": "fp8_per_tensor",
                },
                ignore=["lm_head"],
            )

    def test_resolve_quantization_config_with_online(self):
        resolved = resolve_quantization_config(
            quantization=None,
            quantization_config={
                "online": {
                    "re:.*self_attn.*": "fp8_per_tensor",
                }
            },
        )
        self.assertIsNotNone(resolved)
        self.assertIn("re:.*self_attn.*", resolved.online)

    def test_online_quantization_config_layer_matching(self):
        args = QuantizationConfigArgs(
            online={
                "re:.*self_attn.*": "fp8_per_tensor",
            }
        )
        quant_config = OnlineQuantizationConfig(args)
        linear_layer = MagicMock(spec=LinearBase)

        model_config = ModelConfig(
            model="facebook/opt-125m",
            tokenizer="facebook/opt-125m",
            dtype="float16",
        )
        device_config = DeviceConfig(device="cpu")
        vllm_config = VllmConfig(model_config=model_config, device_config=device_config)

        with set_current_vllm_config(vllm_config):
            # Self-attention layer matching regex -> should return FP8 method
            attn_method = quant_config.get_quant_method(linear_layer, "model.layers.0.self_attn.q_proj")
            self.assertIsInstance(attn_method, Fp8PerTensorOnlineLinearMethod)

            # MLP layer not matching regex -> should fall back to UnquantizedLinearMethod
            mlp_method = quant_config.get_quant_method(linear_layer, "model.layers.0.mlp.gate_proj")
            self.assertIsInstance(mlp_method, UnquantizedLinearMethod)


if __name__ == "__main__":
    unittest.main()
