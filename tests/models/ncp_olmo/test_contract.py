# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the fail-closed NCP-ArchPreview export contract."""

from __future__ import annotations

import copy
import unittest

from vllm.model_executor.models.ncp_olmo.contract import (
    BackendContractError,
    NCPOlmo3BackendConfig,
)


def valid_config() -> dict[str, object]:
    """Return a minimal NCP-ArchPreview inference contract."""

    return {
        "architectures": ["NCPOlmo3ForCausalLM"],
        "model_type": "ncp_olmo3",
        "weight_key_format": "huggingface_state_dict",
        "conceptlm_backbone": "olmo3",
        "hidden_size": 4096,
        "num_layers": 32,
        "num_attention_heads": 32,
        "num_query_groups": 32,
        "qk_layernorm": True,
        "ffn_hidden_size": 11008,
        "vocab_size": 100278,
        "max_sequence_length": 65536,
        "conceptlm_encoder_layers": 16,
        "conceptlm_decoder_layers": 16,
        "conceptlm_special_layers": 8,
        "conceptlm_chunk_size": 4,
        "conceptlm_shift_feature": True,
        "conceptlm_chunk_merge_method": "meanpooling",
        "conceptlm_layer_norm_option": "normed_add",
        "conceptlm_hlm_attention_mode": "backbone_window",
        "conceptlm_hlm_ffn_hidden_size": None,
        "window_size": [4096, 0],
        "window_attn_skip_freq": 4,
        "position_embedding_type": "rope",
        "rotary_interleaved": False,
        "conceptlm_v22_vq_codebook_size": 128,
        "conceptlm_v22_vq_num_codebooks": 32,
        "conceptlm_v22_vq_merge_mode": "raw_logits",
        "conceptlm_v21_dd_self_dd_mode": "dd",
        "conceptlm_v21_dd_encoder_self_dd": True,
        "conceptlm_v21_dd_concept_self_dd": True,
        "conceptlm_v21_dd_two_route_add": True,
        "conceptlm_v21_enable_concept_read_encoder": True,
        "conceptlm_v21_enable_decoder_read_encoder": True,
        "conceptlm_v21_enable_decoder_read_concept": True,
        "conceptlm_v21_final_read_concept_gate": True,
    }


class TestNCPOlmo3BackendContract(unittest.TestCase):
    def test_valid_contract(self) -> None:
        normalized = NCPOlmo3BackendConfig.from_mapping(valid_config())
        self.assertEqual(normalized.encoder_layers, 16)
        self.assertEqual(normalized.decoder_layers, 16)
        self.assertEqual(normalized.hlm_layers, 8)
        self.assertEqual(normalized.chunk_size, 4)

    def test_unsupported_config_variants_fail(self) -> None:
        unsupported = {
            "conceptlm_chunk_merge_method": "unsupported",
            "conceptlm_layer_norm_option": "unsupported",
            "conceptlm_hlm_attention_mode": "unsupported",
            "conceptlm_v22_vq_merge_mode": "unsupported",
            "conceptlm_v21_dd_self_dd_mode": "unsupported",
        }
        for name, value in unsupported.items():
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(BackendContractError, f"{name} must be"),
            ):
                broken = copy.deepcopy(valid_config())
                broken[name] = value
                NCPOlmo3BackendConfig.from_mapping(broken)

    def test_unsupported_architecture_values_fail(self) -> None:
        unsupported = {
            "position_embedding_type": "unsupported",
            "rotary_interleaved": True,
            "conceptlm_shift_feature": False,
            "conceptlm_hlm_ffn_hidden_size": 11008,
        }
        for name, value in unsupported.items():
            with self.subTest(name=name), self.assertRaises(BackendContractError):
                broken = copy.deepcopy(valid_config())
                broken[name] = value
                NCPOlmo3BackendConfig.from_mapping(broken)

    def test_required_boolean_flags_reject_integer_values(self) -> None:
        broken = copy.deepcopy(valid_config())
        broken["conceptlm_v21_dd_encoder_self_dd"] = 1
        with self.assertRaisesRegex(
            BackendContractError,
            "conceptlm_v21_dd_encoder_self_dd must be True",
        ):
            NCPOlmo3BackendConfig.from_mapping(broken)

    def test_native_megatron_model_identity_fails(self) -> None:
        broken = copy.deepcopy(valid_config())
        broken["architectures"] = ["UnsupportedNCPForCausalLM"]
        broken["model_type"] = "unsupported_ncp"
        with self.assertRaisesRegex(BackendContractError, "model identity"):
            NCPOlmo3BackendConfig.from_mapping(broken)

    def test_native_megatron_weight_key_format_fails(self) -> None:
        broken = copy.deepcopy(valid_config())
        broken["weight_key_format"] = "native_megatron_state_dict"
        with self.assertRaisesRegex(BackendContractError, "native Megatron"):
            NCPOlmo3BackendConfig.from_mapping(broken)

    def test_missing_concept_layout_fails(self) -> None:
        broken = copy.deepcopy(valid_config())
        del broken["conceptlm_encoder_layers"]
        with self.assertRaisesRegex(
            BackendContractError,
            "missing required config field: conceptlm_encoder_layers",
        ):
            NCPOlmo3BackendConfig.from_mapping(broken)

    def test_token_layer_sum_must_match(self) -> None:
        broken = copy.deepcopy(valid_config())
        broken["conceptlm_decoder_layers"] = 15
        with self.assertRaisesRegex(BackendContractError, "num_layers must equal"):
            NCPOlmo3BackendConfig.from_mapping(broken)

    def test_first_olmo3_backend_requires_full_mha(self) -> None:
        broken = copy.deepcopy(valid_config())
        broken["num_query_groups"] = 8
        with self.assertRaisesRegex(BackendContractError, "full multi-head attention"):
            NCPOlmo3BackendConfig.from_mapping(broken)


if __name__ == "__main__":
    unittest.main()
