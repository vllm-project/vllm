# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for last-token selection and FP32 decision head loading."""

import tempfile
import unittest

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3_5Config

from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.models.startlux_decision import (
    StartLuxDecisionForSequenceClassification,
    StartLuxDecisionPooler,
)
from vllm.pooling_params import PoolingParams
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates


class TestStartLuxDecisionPooler(unittest.TestCase):
    def test_candidate_ids_must_be_nonempty_distinct_and_nonnegative(self):
        for ids in ([], [1, 1], [-1, 2]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                StartLuxDecisionPooler(ids, 4)

    def test_candidate_ids_must_fit_the_loaded_vocabulary(self):
        pooler = StartLuxDecisionPooler([1, 8], 4)
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            pooler.load_head(torch.zeros(8, 4))

    def test_prompt_attention_is_applied_to_full_attention_config(self):
        for mode, expected in [
            (None, True),
            ("causal", True),
            ("bidirectional", False),
            ("unknown", None),
        ]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as model_dir:
                hf = Qwen3_5Config(
                    architectures=["StartLuxDecisionForSequenceClassification"],
                    text_config={
                        "num_hidden_layers": 2,
                        "layer_types": ["linear_attention", "full_attention"],
                    },
                )
                if mode is not None:
                    hf.jev_prompt_attention = mode
                hf.save_pretrained(model_dir)
                kwargs = dict(
                    model=model_dir,
                    runner="pooling",
                    skip_tokenizer_init=True,
                    language_model_only=True,
                    enforce_eager=True,
                    max_model_len=128,
                )
                if expected is None:
                    with self.assertRaisesRegex(ValueError, "Unsupported StartLux"):
                        ModelConfig(**kwargs)
                    continue
                config = ModelConfig(**kwargs)
                self.assertIs(config.hf_config.is_causal, expected)
                self.assertIs(config.hf_text_config.is_causal, expected)
                self.assertEqual(config.runner_type, "pooling")
                self.assertEqual(
                    config.hf_text_config.layer_types,
                    ["linear_attention", "full_attention"],
                )
                if not expected:
                    self.assertEqual(config.attn_type, "encoder_only")

    def test_last_positions_keep_raw_candidate_logits(self):
        # Nonconsecutive vocabulary rows catch TP-local/vocabulary index confusion.
        pooler = StartLuxDecisionPooler([1, 6, 3], 4)
        weight = torch.arange(32).reshape(8, 4).to(torch.bfloat16) / 7
        pooler.load_head(weight)
        for lengths, last_indices, padding in [
            ([2, 3], [1, 4], 0),
            ([1, 4, 2], [0, 4, 6], 3),
            ([3], [2], 5),
        ]:
            with self.subTest(lengths=lengths, padding=padding):
                num_tokens = sum(lengths)
                hidden = (
                    torch.arange((num_tokens + padding) * 4)
                    .reshape(-1, 4)
                    .to(torch.bfloat16)
                    / 3
                )
                # Padding has a distinct value so selecting its final row
                # cannot accidentally look like the last real prompt token.
                hidden[num_tokens:] = 1000
                metadata = PoolingMetadata(
                    prompt_lens=torch.tensor(lengths),
                    prompt_token_ids=None,
                    prompt_token_ids_cpu=None,
                    pooling_params=[
                        PoolingParams(task="classify", use_activation=False)
                        for _ in lengths
                    ],
                    pooling_states=[PoolingStates() for _ in lengths],
                )
                metadata.build_pooling_cursor(
                    np.asarray(lengths, dtype=np.int32),
                    torch.tensor(lengths),
                    torch.device("cpu"),
                )
                actual = pooler(hidden, metadata)
                expected = hidden[last_indices].float() @ weight[[1, 6, 3]].float().T
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(actual.dtype, torch.float32)
                self.assertEqual(tuple(actual.shape), (len(lengths), 3))
                self.assertGreater(actual.max().item(), 1)  # Raw logits.


@pytest.mark.parametrize("tied", [False, True])
def test_real_constructor_head_loading(tmp_path, dist_init, tied):
    vocab = {"[UNK]": 0}
    vocab.update(
        {letter: i + 1 for i, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}
    )
    vocab.update({f"extra_{i}": i for i in range(27, 64)})
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]"
    ).save_pretrained(tmp_path)
    Qwen3_5Config(
        architectures=["StartLuxDecisionForSequenceClassification"],
        text_config=dict(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=4,
            layer_types=["full_attention"],
            tie_word_embeddings=tied,
        ),
        vision_config=dict(
            depth=1,
            hidden_size=16,
            intermediate_size=32,
            num_heads=4,
            out_hidden_size=16,
            deepstack_visual_indexes=[],
        ),
    ).save_pretrained(tmp_path)
    model_config = ModelConfig(
        model=str(tmp_path),
        runner="pooling",
        language_model_only=True,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=128,
    )
    config = VllmConfig(model_config=model_config)
    with set_default_torch_dtype(model_config.dtype), set_current_vllm_config(config):
        model = StartLuxDecisionForSequenceClassification(vllm_config=config)
        weight = torch.arange(64 * 16).reshape(64, 16).to(torch.bfloat16)
        name = "model.language_model.embed_tokens.weight" if tied else "lm_head.weight"
        mapped = (
            "language_model.model.embed_tokens.weight"
            if tied
            else "language_model.lm_head.weight"
        )
        loaded = model.load_weights(iter([(name, weight)]))
        assert mapped in loaded
        torch.testing.assert_close(
            model.pooler.letter_weight, weight[1:27].float(), rtol=0, atol=0
        )
        parameter = (
            model.language_model.model.embed_tokens.weight
            if tied
            else model.language_model.lm_head.weight
        )
        torch.testing.assert_close(parameter[:64], weight, rtol=0, atol=0)
        with pytest.raises(ValueError, match="no StartLux output head"):
            model.load_weights(iter([]))


if __name__ == "__main__":
    unittest.main()
