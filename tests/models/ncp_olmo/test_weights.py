# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for pure-HF NCP-ArchPreview weight mapping."""

from __future__ import annotations

import unittest
from collections.abc import Mapping

import pytest
import torch

from vllm.model_executor.models.ncp_olmo.weights import (
    NCPOlmo3WeightConfig,
    TokenWeightConfig,
    audit_ncp_olmo3_weight_shapes,
    audit_token_weight_shapes,
    expected_ncp_olmo3_weight_shapes,
    expected_token_weight_shapes,
    resolve_checkpoint_weight,
)


def token_config() -> TokenWeightConfig:
    """Return a small full-MHA token-tower shape contract."""
    return TokenWeightConfig(
        hidden_size=8,
        intermediate_size=12,
        vocab_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        encoder_layers=2,
        decoder_layers=2,
    )


def pure_hf_manifest(
    expected: Mapping[str, tuple[int, ...]],
    config: TokenWeightConfig,
) -> dict[str, tuple[int, ...]]:
    """Convert internal fused parameter shapes to the public pure-HF schema."""
    hidden_size = config.hidden_size
    head_dim = hidden_size // config.num_attention_heads
    query_size = config.num_attention_heads * head_dim
    key_value_size = config.num_key_value_heads * head_dim
    manifest: dict[str, tuple[int, ...]] = {}
    for name, shape in expected.items():
        hf_name = f"model.{name}"
        if name == "embedding.word_embeddings.weight":
            manifest["model.embed_tokens.weight"] = shape
        elif name == "output_layer.weight":
            manifest["lm_head.weight"] = shape
        elif name.endswith(".self_attention.linear_qkv.weight"):
            prefix = hf_name.removesuffix(".self_attention.linear_qkv.weight")
            manifest[f"{prefix}.self_attn.q_proj.weight"] = (
                query_size,
                hidden_size,
            )
            manifest[f"{prefix}.self_attn.k_proj.weight"] = (
                key_value_size,
                hidden_size,
            )
            manifest[f"{prefix}.self_attn.v_proj.weight"] = (
                key_value_size,
                hidden_size,
            )
        elif name.endswith(".self_attention.linear_proj.weight"):
            manifest[
                hf_name.replace(".self_attention.linear_proj.", ".self_attn.o_proj.")
            ] = shape
        elif name.endswith(".self_attention.q_layernorm.weight"):
            manifest[
                hf_name.replace(".self_attention.q_layernorm.", ".self_attn.q_norm.")
            ] = shape
        elif name.endswith(".self_attention.k_layernorm.weight"):
            manifest[
                hf_name.replace(".self_attention.k_layernorm.", ".self_attn.k_norm.")
            ] = shape
        elif name.endswith(".mlp.linear_fc1.weight"):
            prefix = hf_name.removesuffix(".mlp.linear_fc1.weight")
            split_shape = (config.intermediate_size, hidden_size)
            manifest[f"{prefix}.mlp.gate_proj.weight"] = split_shape
            manifest[f"{prefix}.mlp.up_proj.weight"] = split_shape
        elif name.endswith(".mlp.linear_fc2.weight"):
            manifest[hf_name.replace(".mlp.linear_fc2.", ".mlp.down_proj.")] = shape
        elif hf_name.endswith(".final_layernorm.weight"):
            prefix = hf_name.removesuffix(".final_layernorm.weight")
            manifest[f"{prefix}.norm.weight"] = shape
        else:
            manifest[hf_name] = shape
    return manifest


class TestTokenWeightAudit(unittest.TestCase):
    def test_pure_hf_manifest_has_full_coverage(self) -> None:
        config = token_config()
        expected = expected_token_weight_shapes(config)
        manifest = pure_hf_manifest(expected, config)
        manifest["model.decoder.norm._extra_state"] = (0,)
        manifest["model.concept_predictor.hlm_block.layers.0.mlp.gate_proj.weight"] = (
            24,
            8,
        )

        report = audit_token_weight_shapes(manifest, config)

        self.assertTrue(report.ok)
        self.assertEqual(report.expected_parameter_count, 35)
        self.assertEqual(report.matched_parameter_count, 35)
        self.assertEqual(
            report.ignored_token_metadata,
            ("model.decoder.norm._extra_state",),
        )
        self.assertEqual(report.non_token_tensor_count, 1)

    def test_missing_and_shape_mismatch_fail(self) -> None:
        config = token_config()
        expected = expected_token_weight_shapes(config)
        manifest = pure_hf_manifest(expected, config)
        del manifest["model.encoder.layers.1.mlp.down_proj.weight"]
        manifest["lm_head.weight"] = (31, 8)

        report = audit_token_weight_shapes(manifest, token_config())

        self.assertFalse(report.ok)
        self.assertEqual(
            report.missing_parameters, ("encoder.layers.1.mlp.linear_fc2.weight",)
        )
        self.assertEqual(
            report.shape_mismatches,
            ("lm_head.weight -> output_layer.weight: expected (32, 8), found (31, 8)",),
        )

    def test_hf_wrapper_prefix_has_full_coverage(self) -> None:
        config = token_config()
        expected = expected_token_weight_shapes(config)
        manifest = {
            f"wrapper.{name}": shape
            for name, shape in pure_hf_manifest(expected, config).items()
        }
        manifest["wrapper.model.decoder.norm._extra_state"] = (0,)

        report = audit_token_weight_shapes(manifest, token_config())

        self.assertTrue(report.ok)
        self.assertEqual(report.matched_parameter_count, len(expected))

    def test_standard_hf_split_projections_have_full_coverage(self) -> None:
        config = token_config()
        expected = expected_token_weight_shapes(config)
        manifest = pure_hf_manifest(expected, config)

        report = audit_token_weight_shapes(manifest, config)

        self.assertTrue(report.ok, report.to_dict())
        self.assertEqual(report.matched_parameter_count, len(expected))

    def test_incomplete_hf_qkv_is_reported_missing(self) -> None:
        config = token_config()
        expected = expected_token_weight_shapes(config)
        name = "encoder.layers.0.self_attention.linear_qkv.weight"
        manifest = pure_hf_manifest(expected, config)
        del manifest["model.encoder.layers.0.self_attn.v_proj.weight"]

        report = audit_token_weight_shapes(manifest, config)

        self.assertFalse(report.ok)
        self.assertIn(name, report.missing_parameters)


class TestCheckpointKeyResolution(unittest.TestCase):
    def test_arbitrary_hf_wrapper_prefix_is_stripped(self) -> None:
        parameter = "decoder_read_encoder_routes.0.w1.weight"
        target = resolve_checkpoint_weight(f"module.model.{parameter}", {parameter})

        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(target.parameter_name, parameter)
        self.assertIsNone(target.shard_id)

    def test_hf_q_projection_maps_to_qkv_shard(self) -> None:
        parameter = "encoder.layers.0.self_attention.linear_qkv.weight"
        target = resolve_checkpoint_weight(
            "model.encoder.layers.0.self_attn.q_proj.weight", {parameter}
        )

        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(target.parameter_name, parameter)
        self.assertEqual(target.shard_id, "q")

    def test_native_megatron_fused_qkv_is_rejected(self) -> None:
        parameter = "encoder.layers.0.self_attention.linear_qkv.weight"
        target = resolve_checkpoint_weight(parameter, {parameter})

        self.assertIsNone(target)

    def test_native_megatron_embedding_is_rejected(self) -> None:
        parameter = "embedding.word_embeddings.weight"
        target = resolve_checkpoint_weight(parameter, {parameter})

        self.assertIsNone(target)

    def test_hf_tower_norm_maps_to_final_layernorm(self) -> None:
        parameter = "decoder.final_layernorm.weight"
        target = resolve_checkpoint_weight("model.decoder.norm.weight", {parameter})

        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(target.parameter_name, parameter)


class TestNCPOlmo3WeightAudit(unittest.TestCase):
    def test_ncp_olmo3_dimensions_define_722_parameters(self) -> None:
        config = NCPOlmo3WeightConfig(
            token=TokenWeightConfig(
                hidden_size=4096,
                intermediate_size=11008,
                vocab_size=100278,
                num_attention_heads=32,
                num_key_value_heads=32,
                encoder_layers=16,
                decoder_layers=16,
            ),
            hlm_layers=8,
            codebook_size=128,
            num_codebooks=32,
        )

        self.assertEqual(len(expected_ncp_olmo3_weight_shapes(config)), 722)

    def test_pure_hf_full_manifest_has_722_parameters(self) -> None:
        config = NCPOlmo3WeightConfig(
            token=token_config(),
            hlm_layers=2,
            codebook_size=4,
            num_codebooks=2,
        )
        expected = expected_ncp_olmo3_weight_shapes(config)
        manifest = pure_hf_manifest(expected, config.token)
        manifest["model.decoder.norm._extra_state"] = (0,)
        manifest["model.concept_predictor.hlm_block.norm._extra_state"] = (0,)

        report = audit_ncp_olmo3_weight_shapes(manifest, config)

        self.assertTrue(report.ok)
        self.assertEqual(report.expected_parameter_count, 114)
        self.assertEqual(report.matched_parameter_count, 114)
        self.assertEqual(len(report.ignored_metadata), 2)

    def test_hf_wrapped_full_manifest_has_complete_route_coverage(self) -> None:
        config = NCPOlmo3WeightConfig(
            token=token_config(), hlm_layers=2, codebook_size=4, num_codebooks=2
        )
        expected = expected_ncp_olmo3_weight_shapes(config)
        manifest = {
            f"wrapper.{name}": shape
            for name, shape in pure_hf_manifest(expected, config.token).items()
        }

        report = audit_ncp_olmo3_weight_shapes(manifest, config)

        self.assertTrue(report.ok, report.to_dict())
        self.assertEqual(report.matched_parameter_count, 114)
        self.assertEqual(report.ignored_metadata, ())


def test_pure_hf_tensors_load_real_model(ncp_model) -> None:
    config = ncp_model.ncp_weight_config
    shapes = expected_ncp_olmo3_weight_shapes(config)
    manifest = pure_hf_manifest(shapes, config.token)
    weights = {
        name: torch.full(shape, (index + 1) / 100, dtype=torch.bfloat16, device="cuda")
        for index, (name, shape) in enumerate(manifest.items())
    }

    loaded = ncp_model.load_weights(weights.items())

    assert loaded == {name for name, _ in ncp_model.named_parameters()}
    attention = ncp_model.token_backbone.encoder.layers[0].self_attention
    prefix = "model.encoder.layers.0"
    torch.testing.assert_close(
        attention.linear_qkv.weight,
        torch.cat(
            [
                weights[f"{prefix}.self_attn.{part}_proj.weight"]
                for part in ("q", "k", "v")
            ]
        ),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        ncp_model.token_backbone.encoder.layers[0].mlp.linear_fc1.weight,
        torch.cat(
            [weights[f"{prefix}.mlp.{part}_proj.weight"] for part in ("gate", "up")]
        ),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        ncp_model.token_backbone.embedding.word_embeddings.weight[
            : config.token.vocab_size
        ],
        weights["model.embed_tokens.weight"],
        rtol=0,
        atol=0,
    )


def test_native_megatron_checkpoint_keys_are_rejected(ncp_model) -> None:
    with pytest.raises(ValueError, match="unexpected ConceptLM checkpoint"):
        ncp_model.load_weights([("embedding.word_embeddings.weight", torch.zeros(1))])


if __name__ == "__main__":
    unittest.main()
