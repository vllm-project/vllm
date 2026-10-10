# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-checkpoint admission, independent of Torch/device construction.

The contract is HF full shapes in, complete coverage or a rejection out. These
checks catch missing fused inputs and aliases before expensive model execution.
Collective and actual IPC behavior are tested separately through EngineCore.
"""

import unittest
from types import SimpleNamespace

from vllm.v1.worker.layout_checkpoint import LayoutCheckpoint

CONFIG = dict(
    model_type="llama",
    hidden_size=8,
    num_attention_heads=4,
    num_key_value_heads=1,
    intermediate_size=16,
    vocab_size=32,
    num_hidden_layers=1,
    tie_word_embeddings=False,
)
WIRE_SHAPES = {
    "model.embed_tokens.weight": (32, 8),
    "model.norm.weight": (8,),
    "lm_head.weight": (32, 8),
    "model.layers.0.input_layernorm.weight": (8,),
    "model.layers.0.post_attention_layernorm.weight": (8,),
    "model.layers.0.self_attn.q_proj.weight": (8, 8),
    "model.layers.0.self_attn.k_proj.weight": (2, 8),
    "model.layers.0.self_attn.v_proj.weight": (2, 8),
    "model.layers.0.self_attn.o_proj.weight": (8, 8),
    "model.layers.0.mlp.gate_proj.weight": (16, 8),
    "model.layers.0.mlp.up_proj.weight": (16, 8),
    "model.layers.0.mlp.down_proj.weight": (8, 16),
}


def chunk(names):
    return {
        "names": list(names),
        "dtype_names": ["bfloat16"] * len(names),
        "shapes": [list(WIRE_SHAPES[name]) for name in names],
        "ipc_handles": [{} for _ in names],
    }


class TestLayoutCheckpoint(unittest.TestCase):
    def test_unknown_metadata_is_rejected_without_consuming_coverage(self):
        checkpoint = LayoutCheckpoint(SimpleNamespace(**CONFIG))
        payload = chunk(list(WIRE_SHAPES))
        with self.assertRaisesRegex(ValueError, "Unexpected IPC"):
            checkpoint.validate_chunk({**payload, "extra": 1})
        self.assertEqual(checkpoint.received, set())
        checkpoint.commit_chunk(checkpoint.validate_chunk(payload))
        checkpoint.require_complete()

    def test_unused_native_optional_fields_remain_accepted(self):
        checkpoint = LayoutCheckpoint(SimpleNamespace(**CONFIG))
        payload = {
            **chunk(list(WIRE_SHAPES)),
            "ipc_handles_pickled": None,
            "tensor_sizes": [],
        }
        checkpoint.commit_chunk(checkpoint.validate_chunk(payload))
        checkpoint.require_complete()

    def test_all_fused_inputs_must_arrive_before_completion(self):
        for missing in (
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
        ):
            with self.subTest(missing=missing):
                checkpoint = LayoutCheckpoint(SimpleNamespace(**CONFIG))
                payload = checkpoint.validate_chunk(
                    chunk([name for name in WIRE_SHAPES if name != missing])
                )
                checkpoint.commit_chunk(payload)
                with self.assertRaisesRegex(ValueError, "Incomplete checkpoint"):
                    checkpoint.require_complete()
                checkpoint.commit_chunk(checkpoint.validate_chunk(chunk([missing])))
                checkpoint.require_complete()

    def test_metadata_receipt_alone_does_not_prove_loading(self):
        checkpoint = LayoutCheckpoint(SimpleNamespace(**CONFIG))
        checkpoint.validate_chunk(chunk(list(WIRE_SHAPES)))
        with self.assertRaisesRegex(ValueError, "Incomplete checkpoint"):
            checkpoint.require_complete()

    def test_shape_dtype_unknown_and_duplicate_are_rejected(self):
        checkpoint = LayoutCheckpoint(SimpleNamespace(**CONFIG))
        name = "model.layers.0.self_attn.k_proj.weight"
        for change in (
            {"shapes": [[8, 8]]},
            {"dtype_names": ["float32"]},
            {"names": ["not.a.parameter"]},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                checkpoint.validate_chunk({**chunk([name]), **change})
        checkpoint.commit_chunk(checkpoint.validate_chunk(chunk([name])))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            checkpoint.validate_chunk(chunk([name]))

    def test_tied_alias_counts_once_and_is_loaded_as_embedding(self):
        checkpoint = LayoutCheckpoint(
            SimpleNamespace(**{**CONFIG, "tie_word_embeddings": True})
        )
        payload = checkpoint.validate_chunk(
            chunk([name for name in WIRE_SHAPES if name != "model.embed_tokens.weight"])
        )
        self.assertNotIn("lm_head.weight", payload["names"])
        self.assertIn("model.embed_tokens.weight", payload["names"])
        checkpoint.commit_chunk(payload)
        checkpoint.require_complete()
        with self.assertRaisesRegex(ValueError, "tied alias"):
            checkpoint.validate_chunk(chunk(["model.embed_tokens.weight"]))

    def test_two_aliases_in_one_chunk_cannot_double_count_coverage(self):
        checkpoint = LayoutCheckpoint(
            SimpleNamespace(**{**CONFIG, "tie_word_embeddings": True})
        )
        with self.assertRaisesRegex(ValueError, "tied alias"):
            checkpoint.validate_chunk(
                chunk(["model.embed_tokens.weight", "lm_head.weight"])
            )

    def test_unrecognized_runtime_parameter_refuses_the_schema(self):
        checkpoint = LayoutCheckpoint(SimpleNamespace(**CONFIG))
        with self.assertRaisesRegex(ValueError, "Model parameters do not match"):
            checkpoint.validate_model_parameters({"other.weight"})
