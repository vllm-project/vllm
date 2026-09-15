# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for modular Sentence Transformers CrossEncoders."""

from pathlib import Path

import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizer


def _create_modular_bert_cross_encoder(
    path: Path,
    *,
    structured: bool = False,
    pooling_mode: str = "mean",
) -> tuple[str, list[tuple[str, str]], list[float]]:
    from sentence_transformers import CrossEncoder
    from sentence_transformers.sentence_transformer.modules import (
        Dense,
        Pooling,
        Transformer,
    )

    base_path = path / "base"
    vocab = {
        token: index
        for index, token in enumerate(
            [
                "[PAD]",
                "[UNK]",
                "[CLS]",
                "[SEP]",
                "[MASK]",
                "query",
                "document",
                ":",
                ";",
                "0",
                "1",
                "2",
                "3",
                "4",
            ]
        )
    }
    config = BertConfig(
        architectures=["BertModel"],
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
        hidden_size=128,
        intermediate_size=256,
        max_position_embeddings=32,
        num_attention_heads=4,
        num_hidden_layers=1,
        vocab_size=len(vocab),
    )
    torch.manual_seed(0)
    bert = BertModel(config)
    with torch.no_grad():
        token_types = bert.embeddings.token_type_embeddings.weight
        token_types[0].zero_()
        token_types[1].copy_(torch.linspace(-1.0, 1.0, config.hidden_size))
    bert.save_pretrained(base_path)

    tokenizer = BertTokenizer(
        vocab=vocab,
        do_lower_case=False,
        model_max_length=16,
    )
    if structured:
        tokenizer.chat_template = (
            "{% for message in messages %}{{ message['role'] }}:"
            "{% for item in message['content'] %}{{ item['text'] }}{% endfor %};"
            "{% endfor %}[SEP]"
        )
    tokenizer.save_pretrained(base_path)

    transformer = Transformer(
        str(base_path),
        max_seq_length=16,
        module_output_name="token_embeddings",
        modality_config={
            "text": {"method": "forward", "method_output_name": "last_hidden_state"},
            "message": {
                "method": "forward",
                "method_output_name": "last_hidden_state",
                "format": "structured",
            },
        }
        if structured
        else None,
    )
    pooling = Pooling(
        config.hidden_size, pooling_mode=pooling_mode, include_prompt=True
    )
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

    document = " ".join(["document"] * 21)
    pairs = [
        ("query", document),
        ("query query query", document),
        ("query", "document"),
    ]
    reference_scores = cross_encoder.predict(pairs).tolist()
    return str(export_path), pairs, reference_scores


@pytest.mark.parametrize(
    ("model_impl", "enforce_eager"),
    [("vllm", True), ("transformers", True), ("transformers", False)],
)
def test_modular_bert_cross_encoder_score_parity(
    vllm_runner, tmp_path: Path, model_impl: str, enforce_eager: bool
) -> None:
    """A current modular BERT export must preserve pair and truncation semantics."""
    pytest.importorskip(
        "sentence_transformers",
        minversion="5.7.0",
        reason="Modular CrossEncoder construction requires sentence-transformers 5.7",
    )
    model_path, pairs, reference_scores = _create_modular_bert_cross_encoder(tmp_path)

    with vllm_runner(
        model_path,
        runner="pooling",
        model_impl=model_impl,
        trust_remote_code=False,
        max_model_len=None,
        dtype="float32",
        enforce_eager=enforce_eager,
        gpu_memory_utilization=0.1,
        max_num_batched_tokens=32,
        max_num_seqs=2,
        compilation_config={"cudagraph_capture_sizes": [8, 16, 32]},
    ) as model:
        assert model.llm.llm_engine.model_config.max_model_len == 16
        # Replay equal-sized requests with different segment boundaries, then
        # exercise a shorter request whose graph input includes padding.
        for pair, expected_length, reference_score in zip(
            pairs, [16, 16, 5], reference_scores
        ):
            output = model.llm.score(*pair)[0]
            assert len(output.prompt_token_ids) == expected_length
            assert output.outputs.score == pytest.approx(
                reference_score, abs=1e-4, rel=1e-4
            )


@pytest.mark.parametrize("pooling_mode", ["mean", "lasttoken"])
def test_structured_cross_encoder_export_preserves_truncated_scores(
    vllm_runner, tmp_path: Path, pooling_mode: str
) -> None:
    """Saved structured exports retain the template tail used by the trained head."""
    pytest.importorskip("sentence_transformers", minversion="5.7.0")
    from sentence_transformers import CrossEncoder

    model_path, pairs, reference_scores = _create_modular_bert_cross_encoder(
        tmp_path, structured=True, pooling_mode=pooling_mode
    )
    reference = CrossEncoder(model_path, device="cpu", trust_remote_code=False)
    features = reference.preprocess(pairs)
    expected_ids = [
        ids[mask.bool()].tolist()
        for ids, mask in zip(features["input_ids"], features["attention_mask"])
    ]
    with vllm_runner(
        model_path,
        trust_remote_code=False,
        dtype="float32",
        enforce_eager=True,
        gpu_memory_utilization=0.1,
        max_model_len=None,
    ) as model:
        outputs = model.llm.score(
            [query for query, _ in pairs], [document for _, document in pairs]
        )
    assert [output.prompt_token_ids for output in outputs] == expected_ids
    assert [output.outputs.score for output in outputs] == pytest.approx(
        reference_scores, abs=1e-4, rel=1e-4
    )
