# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for pure-HF NCP-ArchPreview weight mapping."""

from __future__ import annotations

import unittest
from collections.abc import Mapping

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    try:
        from vllm.model_executor.models.ncp_olmo.model import (
            NCPOlmo3ForCausalLM,
        )
    except ImportError:
        NCPOlmo3ForCausalLM = None
else:
    NCPOlmo3ForCausalLM = None

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


@unittest.skipIf(
    torch is None or NCPOlmo3ForCausalLM is None,
    "torch and the pinned vLLM runtime are required",
)
class TestModelHFWeightLoading(unittest.TestCase):
    class _Parameter:
        def __init__(self) -> None:
            self.shards: list[str | int | None] = []
            self.loaded: dict[str | int | None, object] = {}

        def weight_loader(
            self,
            parameter: object,
            loaded_weight: object,
            shard_id: str | int | None = None,
        ) -> None:
            if parameter is not self:
                raise AssertionError("weight loader received the wrong parameter")
            self.shards.append(shard_id)
            self.loaded[shard_id] = loaded_weight.clone()  # type: ignore[attr-defined]

        def canonical_weight(self):
            if None in self.loaded:
                return self.loaded[None]
            if set(self.loaded) == {"q", "k", "v"}:
                return torch.cat(
                    (self.loaded["q"], self.loaded["k"], self.loaded["v"]),
                    dim=0,
                )
            if set(self.loaded) == {0, 1}:
                return torch.cat((self.loaded[0], self.loaded[1]), dim=0)
            raise AssertionError(
                f"incomplete parameter shards: {sorted(map(str, self.loaded))}"
            )

    class _Module:
        def __init__(self, params: Mapping[str, object]) -> None:
            self.params = dict(params)

        def named_parameters(self, remove_duplicate: bool = False):
            del remove_duplicate
            return iter(self.params.items())

    class _BackendConfig:
        num_attention_heads = 2
        num_key_value_heads = 2

    class _Model:
        token_backbone: object
        highlevel: object
        routes: object
        backend_config: object

        def named_parameters(self):
            for root in ("token_backbone", "highlevel", "routes"):
                module = getattr(self, root)
                for name, parameter in module.named_parameters():
                    yield f"{root}.{name}", parameter

    def _build_fake_model(self, names: dict[str, tuple[int, ...]]):
        params = {name: self._Parameter() for name in names}
        token = {
            name: parameter
            for name, parameter in params.items()
            if name.startswith(("embedding.", "encoder.", "decoder.", "output_layer."))
        }
        highlevel = {
            name: parameter
            for name, parameter in params.items()
            if name.startswith(
                ("concept_vq_input_norm.", "concept_quantizer.", "concept_predictor.")
            )
        }
        routes = {
            name: parameter
            for name, parameter in params.items()
            if name not in token and name not in highlevel
        }
        model = self._Model()
        model.token_backbone = self._Module(token)
        model.highlevel = self._Module(highlevel)
        model.routes = self._Module(routes)
        model.backend_config = self._BackendConfig()
        return model, params

    @staticmethod
    def _pure_hf_weights(
        shapes: Mapping[str, tuple[int, ...]],
        config: TokenWeightConfig,
    ):
        return (
            (name, torch.zeros(shape))
            for name, shape in pure_hf_manifest(shapes, config).items()
        )

    @staticmethod
    def _standard_hf_name(name: str) -> str:
        if name == "embedding.word_embeddings.weight":
            return "model.embed_tokens.weight"
        if name == "output_layer.weight":
            return "lm_head.weight"
        hf_name = f"model.{name}"
        for native_part, hf_part in (
            (".self_attention.linear_proj.", ".self_attn.o_proj."),
            (".self_attention.q_layernorm.", ".self_attn.q_norm."),
            (".self_attention.k_layernorm.", ".self_attn.k_norm."),
            (".mlp.linear_fc2.", ".mlp.down_proj."),
        ):
            hf_name = hf_name.replace(native_part, hf_part)
        if hf_name.endswith(".final_layernorm.weight"):
            hf_name = hf_name.removesuffix(".final_layernorm.weight") + ".norm.weight"
        return hf_name

    def test_full_standard_hf_manifest_loads_all_722_parameters(self) -> None:
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
        names = expected_ncp_olmo3_weight_shapes(config)
        model, params = self._build_fake_model(names)

        weights = []
        for name in names:
            if name == "embedding.word_embeddings.weight":
                weights.append(("model.embed_tokens.weight", torch.zeros(1)))
            elif name == "output_layer.weight":
                weights.append(("lm_head.weight", torch.zeros(1)))
            elif name.endswith(".self_attention.linear_qkv.weight"):
                prefix = f"model.{name}".removesuffix(
                    ".self_attention.linear_qkv.weight"
                )
                for projection in ("q", "k", "v"):
                    weights.append(
                        (f"{prefix}.self_attn.{projection}_proj.weight", torch.zeros(1))
                    )
            elif name.endswith(".mlp.linear_fc1.weight"):
                prefix = f"model.{name}".removesuffix(".mlp.linear_fc1.weight")
                weights.extend(
                    (
                        (f"{prefix}.mlp.gate_proj.weight", torch.zeros(1)),
                        (f"{prefix}.mlp.up_proj.weight", torch.zeros(1)),
                    )
                )
            else:
                weights.append((self._standard_hf_name(name), torch.zeros(1)))

        loaded = NCPOlmo3ForCausalLM.load_weights(model, weights)

        self.assertEqual(len(loaded), 722)
        qkv = params["encoder.layers.0.self_attention.linear_qkv.weight"]
        fc1 = params["encoder.layers.0.mlp.linear_fc1.weight"]
        self.assertEqual(qkv.shards, ["q", "k", "v"])
        self.assertEqual(fc1.shards, [0, 1])

    def test_native_megatron_checkpoint_keys_are_rejected(self) -> None:
        config = NCPOlmo3WeightConfig(
            token=TokenWeightConfig(
                hidden_size=32,
                intermediate_size=12,
                vocab_size=32,
                num_attention_heads=2,
                num_key_value_heads=2,
                encoder_layers=16,
                decoder_layers=16,
            ),
            hlm_layers=8,
            codebook_size=4,
            num_codebooks=32,
        )
        shapes = expected_ncp_olmo3_weight_shapes(config)
        model, _ = self._build_fake_model(shapes)

        with self.assertRaisesRegex(ValueError, "unexpected ConceptLM checkpoint"):
            NCPOlmo3ForCausalLM.load_weights(
                model,
                ((name, torch.zeros(shape)) for name, shape in shapes.items()),
            )


if __name__ == "__main__":
    unittest.main()
