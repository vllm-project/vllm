# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the fail-closed ConceptLM export contract."""

from __future__ import annotations

import copy
import unittest

from vllm.model_executor.models.ncp_olmo.contract import (
    BackendContractError,
    ConceptLMBackendConfig,
)


def valid_config() -> dict[str, object]:
    """Return a minimal Stage3-like inference contract."""

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
        "qk_norm_mode": "full_hidden",
        "qk_norm_weight_size": 4096,
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


class TestConceptLMBackendContract(unittest.TestCase):
    def test_valid_contract(self) -> None:
        normalized = ConceptLMBackendConfig.from_mapping(valid_config())
        self.assertEqual(normalized.encoder_layers, 16)
        self.assertEqual(normalized.decoder_layers, 16)
        self.assertEqual(normalized.hlm_layers, 8)
        self.assertEqual(normalized.chunk_size, 4)
        self.assertEqual(normalized.qk_norm_mode, "full_hidden")
        self.assertEqual(normalized.qk_norm_weight_size, 4096)

    def test_per_head_qk_norm_contract(self) -> None:
        per_head = copy.deepcopy(valid_config())
        per_head["qk_norm_mode"] = "per_head"
        per_head["qk_norm_weight_size"] = 128

        normalized = ConceptLMBackendConfig.from_mapping(per_head)

        self.assertEqual(normalized.qk_norm_mode, "per_head")
        self.assertEqual(normalized.qk_norm_weight_size, 128)

    def test_per_head_qk_norm_requires_head_dim_weight(self) -> None:
        broken = copy.deepcopy(valid_config())
        broken["qk_norm_mode"] = "per_head"
        broken["qk_norm_weight_size"] = 4096

        with self.assertRaisesRegex(
            BackendContractError,
            "requires qk_norm_weight_size=128",
        ):
            ConceptLMBackendConfig.from_mapping(broken)

    def test_legacy_contract_defaults_to_full_hidden_qk_norm(self) -> None:
        legacy = copy.deepcopy(valid_config())
        del legacy["qk_norm_mode"]
        del legacy["qk_norm_weight_size"]

        normalized = ConceptLMBackendConfig.from_mapping(legacy)

        self.assertEqual(normalized.qk_norm_mode, "full_hidden")
        self.assertEqual(normalized.qk_norm_weight_size, 4096)

    def test_native_megatron_model_identity_fails(self) -> None:
        broken = copy.deepcopy(valid_config())
        broken["architectures"] = ["ConceptLMV22VQForCausalLM"]
        broken["model_type"] = "conceptlm_v22_vq"
        with self.assertRaisesRegex(BackendContractError, "model identity"):
            ConceptLMBackendConfig.from_mapping(broken)

    def test_native_megatron_weight_key_format_fails(self) -> None:
        broken = copy.deepcopy(valid_config())
        broken["weight_key_format"] = "native_megatron_state_dict"
        with self.assertRaisesRegex(BackendContractError, "native Megatron"):
            ConceptLMBackendConfig.from_mapping(broken)

    def test_missing_concept_layout_fails(self) -> None:
        broken = copy.deepcopy(valid_config())
        del broken["conceptlm_encoder_layers"]
        with self.assertRaisesRegex(
            BackendContractError,
            "missing required config field: conceptlm_encoder_layers",
        ):
            ConceptLMBackendConfig.from_mapping(broken)

    def test_token_layer_sum_must_match(self) -> None:
        broken = copy.deepcopy(valid_config())
        broken["conceptlm_decoder_layers"] = 15
        with self.assertRaisesRegex(BackendContractError, "num_layers must equal"):
            ConceptLMBackendConfig.from_mapping(broken)

    def test_first_olmo3_backend_requires_full_mha(self) -> None:
        broken = copy.deepcopy(valid_config())
        broken["num_query_groups"] = 8
        with self.assertRaisesRegex(BackendContractError, "full multi-head attention"):
            ConceptLMBackendConfig.from_mapping(broken)


if __name__ == "__main__":
    unittest.main()
