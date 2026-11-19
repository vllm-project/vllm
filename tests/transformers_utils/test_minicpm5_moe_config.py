# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import tempfile
import unittest
from pathlib import Path

from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs.minicpm5_moe import MiniCPM5MoEConfig


class TestMiniCPM5MoEConfig(unittest.TestCase):
    def test_checkpoint_names_and_explicit_dimensions_without_remote_code(self):
        """The 15B checkpoint has head_dim != hidden_size / num_attention_heads."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            data = {
                "model_type": "minicpm5_moe",
                "architectures": ["MiniCPM5MoEForCausalLM"],
                "hidden_size": 2048,
                "num_attention_heads": 32,
                "num_key_value_heads": 2,
                "head_dim": 128,
                "num_experts_per_tok": 16,
                "n_routed_experts": 160,
                "n_shared_experts": 1,
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
            for key, value in data.items():
                if key != "auto_map":
                    self.assertEqual(getattr(config, key), value, key)
            self.assertEqual(config.rope_parameters["rope_theta"], 5000000)
            self.assertEqual(config.rope_parameters["rope_type"], "default")

    def test_legacy_scaled_rope_preserves_scaling(self):
        scaling = {"type": "linear", "factor": 2.0}
        config = MiniCPM5MoEConfig(rope_theta=12345, rope_scaling=scaling)
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
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            reloaded = MiniCPM5MoEConfig.from_pretrained(directory)
        self.assertEqual(reloaded.rope_parameters, parameters)
        self.assertEqual(
            parameters, {"rope_type": "linear", "factor": 3.0, "rope_theta": 45678}
        )

    def test_reference_defaults_and_forwarded_kwargs(self):
        config = MiniCPM5MoEConfig(
            hidden_size=96, num_attention_heads=3, use_cache=False, custom_field=7
        )
        self.assertEqual(config.head_dim, 32)
        self.assertEqual(config.num_key_value_heads, 3)
        self.assertIsNone(config.num_experts_per_tok)
        self.assertFalse(config.use_cache)
        self.assertEqual(config.custom_field, 7)


if __name__ == "__main__":
    unittest.main()
