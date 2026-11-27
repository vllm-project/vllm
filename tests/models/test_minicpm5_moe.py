# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from vllm.models.minicpm5_moe.model import MiniCPM5Gate
from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs.minicpm5_moe import MiniCPM5MoEConfig


class TestMiniCPM5MoEConfig(unittest.TestCase):
    def test_checkpoint_defaults_and_loading_without_remote_code(self):
        """The 15B checkpoint has head_dim != hidden_size / num_attention_heads."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            data = {
                "model_type": "minicpm5_moe",
                "architectures": ["MiniCPM5MoEForCausalLM"],
                "vocab_size": 130560,
                "hidden_size": 2048,
                "intermediate_size": 8192,
                "num_hidden_layers": 28,
                "num_attention_heads": 32,
                "num_key_value_heads": 2,
                "head_dim": 128,
                "max_position_embeddings": 524288,
                "pad_token_id": 1,
                "bos_token_id": 0,
                "num_experts_per_tok": 16,
                "n_routed_experts": 160,
                "n_shared_experts": 1,
                "moe_intermediate_size": 512,
                "first_k_dense_replace": 1,
                "routed_scaling_factor": 3.66,
                "use_gated_attention": True,
                "rope_theta": 5000000,
                "eos_token_id": [1, 130073],
                "auto_map": {"AutoConfig": "must_not_execute.Config"},
            }
            (path / "config.json").write_text(json.dumps(data))
            (path / "must_not_execute.py").write_text(
                "raise RuntimeError('remote code executed')"
            )
            config = get_config(str(path), trust_remote_code=False)
            self.assertIsInstance(config, MiniCPM5MoEConfig)
            defaults = MiniCPM5MoEConfig()
            for key, value in data.items():
                if key != "auto_map":
                    self.assertEqual(getattr(config, key), value, key)
                if key not in {"architectures", "auto_map"}:
                    self.assertEqual(getattr(defaults, key), value, key)
            self.assertEqual(config.rope_parameters["rope_theta"], 5000000)
            self.assertEqual(config.rope_parameters["rope_type"], "default")

    def test_legacy_rope_type_is_normalized(self):
        scaling = {"type": "linear", "factor": 2.0}
        config = MiniCPM5MoEConfig(rope_theta=12345, rope_parameters=scaling)
        self.assertEqual(
            config.rope_parameters,
            {
                "rope_type": "linear",
                "factor": 2.0,
                "rope_theta": 12345,
            },
        )
        self.assertEqual(scaling, {"type": "linear", "factor": 2.0})

    def test_explicit_v5_rope_wins_and_round_trips(self):
        parameters = {"rope_type": "linear", "factor": 3.0, "rope_theta": 45678}
        config = MiniCPM5MoEConfig(rope_theta=12345, rope_parameters=parameters)
        self.assertEqual(config.rope_parameters, parameters)
        self.assertEqual(config.rope_theta, 45678)
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            reloaded = MiniCPM5MoEConfig.from_pretrained(directory)
        self.assertEqual(reloaded.rope_parameters, parameters)
        self.assertEqual(
            parameters, {"rope_type": "linear", "factor": 3.0, "rope_theta": 45678}
        )

    def test_explicit_none_fallbacks_and_forwarded_kwargs(self):
        config = MiniCPM5MoEConfig(
            hidden_size=96,
            num_attention_heads=3,
            num_key_value_heads=None,
            head_dim=None,
            num_experts_per_tok=None,
            use_cache=False,
            custom_field=7,
        )
        self.assertEqual(config.head_dim, 32)
        self.assertEqual(config.num_key_value_heads, 3)
        self.assertIsNone(config.num_experts_per_tok)
        self.assertFalse(config.use_cache)
        self.assertEqual(config.custom_field, 7)


class TestMiniCPM5Gate(unittest.TestCase):
    def test_bf16_parameters_produce_fp32_logits_without_early_rounding(self):
        weight = nn.Parameter(torch.ones(1, 2, dtype=torch.bfloat16))
        correction_bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        gate = MiniCPM5Gate(weight, correction_bias)
        hidden = torch.tensor([[1.0, 1.0 / 256]], dtype=torch.bfloat16)

        logits, bias = gate(hidden)

        self.assertEqual(logits.dtype, torch.float32)
        self.assertIsNone(bias)
        self.assertEqual(logits.item(), 1.0 + 1.0 / 256)
        self.assertNotEqual(logits.item(), logits.to(torch.bfloat16).float().item())
        self.assertIs(gate.weight, weight)
        self.assertIs(gate.e_score_correction_bias, correction_bias)

    def test_weight_reload_updates_fp32_routing(self):
        weight = nn.Parameter(torch.ones(1, 2, dtype=torch.bfloat16))
        correction_bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        gate = MiniCPM5Gate(weight, correction_bias)
        with torch.no_grad():
            weight.copy_(torch.tensor([[2.0, 3.0]], dtype=torch.bfloat16))
        logits, _ = gate(torch.tensor([[1.0, 0.5]], dtype=torch.bfloat16))
        self.assertEqual(logits.dtype, torch.float32)
        self.assertEqual(logits.item(), 3.5)


if __name__ == "__main__":
    unittest.main()
