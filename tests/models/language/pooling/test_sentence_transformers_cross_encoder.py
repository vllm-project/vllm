# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for modular Sentence Transformers CrossEncoders."""

from pathlib import Path

import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizer


def _create_modular_bert_cross_encoder(
    path: Path,
) -> tuple[str, tuple[str, str], float]:
    from sentence_transformers import CrossEncoder
    from sentence_transformers.sentence_transformer.modules import (
        Dense,
        Pooling,
        Transformer,
    )

    base_path = path / "base"
    config = BertConfig(
        architectures=["BertModel"],
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
        hidden_size=128,
        intermediate_size=256,
        max_position_embeddings=32,
        num_attention_heads=4,
        num_hidden_layers=1,
        vocab_size=7,
    )
    torch.manual_seed(0)
    bert = BertModel(config)
    with torch.no_grad():
        token_types = bert.embeddings.token_type_embeddings.weight
        token_types[0].zero_()
        token_types[1].copy_(torch.linspace(-1.0, 1.0, config.hidden_size))
    bert.save_pretrained(base_path)

    tokenizer = BertTokenizer(
        vocab={
            "[PAD]": 0,
            "[UNK]": 1,
            "[CLS]": 2,
            "[SEP]": 3,
            "[MASK]": 4,
            "query": 5,
            "document": 6,
        },
        do_lower_case=False,
        model_max_length=16,
    )
    tokenizer.save_pretrained(base_path)

    transformer = Transformer(str(base_path), max_seq_length=16)
    pooling = Pooling(config.hidden_size, pooling_mode="mean", include_prompt=True)
    dense = Dense(
        config.hidden_size,
        1,
        activation_function=torch.nn.Identity(),
        init_weight=torch.linspace(-0.5, 0.5, config.hidden_size).unsqueeze(0),
        init_bias=torch.tensor([0.1]),
        module_output_name="scores",
    )
    cross_encoder = CrossEncoder(
        modules=[transformer, pooling, dense],
        activation_fn=torch.nn.Identity(),
        device="cpu",
    )
    export_path = path / "export"
    cross_encoder.save_pretrained(export_path)

    pair = ("query", " ".join(["document"] * 21))
    reference_score = float(cross_encoder.predict([pair])[0])
    return str(export_path), pair, reference_score


def test_modular_bert_cross_encoder_score_parity(vllm_runner, tmp_path: Path) -> None:
    """A current modular BERT export must preserve pair and truncation semantics."""
    pytest.importorskip(
        "sentence_transformers",
        minversion="5.7.0",
        reason="Modular CrossEncoder construction requires sentence-transformers 5.7",
    )
    model_path, pair, reference_score = _create_modular_bert_cross_encoder(tmp_path)

    with vllm_runner(
        model_path,
        runner="pooling",
        max_model_len=None,
        dtype="float32",
        enforce_eager=True,
        gpu_memory_utilization=0.1,
    ) as model:
        assert model.llm.llm_engine.model_config.max_model_len == 16
        output = model.llm.score(*pair)[0]
        assert len(output.prompt_token_ids) == 16
        actual_score = output.outputs.score

    assert actual_score == pytest.approx(reference_score, abs=1e-4, rel=1e-4)
