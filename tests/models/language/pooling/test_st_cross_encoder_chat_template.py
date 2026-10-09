# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sentence Transformers CrossEncoders that declare the `message` modality
are scored with their saved chat template, as in Sentence Transformers."""

import json

import pytest
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForSequenceClassification

TOKENIZER = "Qwen/Qwen3-0.6B"

SCORE_TEMPLATE = (
    '{%- set query = messages | selectattr("role", "eq", "query") '
    '| map(attribute="content") | first -%}'
    '{%- set document = messages | selectattr("role", "eq", "document") '
    '| map(attribute="content") | first -%}'
    "<|im_start|>system\n{{ query }}<|im_end|>\n"
    "<|im_start|>user\n{{ document }}<|im_end|>\n"
    "{%- if add_generation_prompt %}\n<|im_start|>assistant\n{%- endif %}"
)

PAIRS = [
    ("What is the capital of France?", "Paris is the capital of France."),
    ("What is the capital of France?", "The Eiffel Tower is in Paris."),
    ("How do plants make food?", "Photosynthesis turns light into sugar."),
]


@pytest.fixture(scope="module", params=["default", "named"])
def st_cross_encoder(request, tmp_path_factory) -> str:
    """A tiny random single-module CrossEncoder whose score template is either
    the default chat template or a named one."""
    path = tmp_path_factory.mktemp(f"st_cross_encoder_{request.param}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    chat_template_kwargs: dict = {"add_generation_prompt": True}
    if request.param == "default":
        tokenizer.chat_template = SCORE_TEMPLATE
    else:
        tokenizer.chat_template = {
            "default": tokenizer.chat_template,
            "score": SCORE_TEMPLATE,
        }
        chat_template_kwargs["chat_template"] = "score"
    tokenizer.save_pretrained(path)

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        num_labels=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    Qwen3ForSequenceClassification(config).save_pretrained(path)

    st_files = {
        "config_sentence_transformers.json": {
            "model_type": "CrossEncoder",
            "activation_fn": "torch.nn.modules.activation.Sigmoid",
        },
        "modules.json": [
            {
                "idx": 0,
                "name": "0",
                "path": "",
                "type": "sentence_transformers.base.modules.transformer.Transformer",
            }
        ],
        "sentence_bert_config.json": {
            "transformer_task": "sequence-classification",
            "modality_config": {
                "text": {"method": "forward", "method_output_name": "logits"},
                "message": {
                    "method": "forward",
                    "method_output_name": "logits",
                    "format": "flat",
                },
            },
            "module_output_name": "scores",
            "processing_kwargs": {"chat_template": chat_template_kwargs},
        },
    }
    for name, content in st_files.items():
        (path / name).write_text(json.dumps(content))
    return str(path)


def test_st_cross_encoder_chat_template(hf_runner, vllm_runner, st_cross_encoder):
    with hf_runner(
        st_cross_encoder, dtype="float32", is_cross_encoder=True
    ) as hf_model:
        hf_token_ids = [
            hf_model.model.preprocess([pair])["input_ids"][0].tolist() for pair in PAIRS
        ]
        hf_scores = hf_model.predict(PAIRS).tolist()
        # Guards that the saved `add_generation_prompt` reaches the template.
        assert hf_model.model.tokenizer.decode(hf_token_ids[0]).endswith(
            "<|im_start|>assistant"
        )

    with vllm_runner(
        st_cross_encoder,
        runner="pooling",
        dtype="float32",
        max_model_len=512,
        enforce_eager=True,
    ) as vllm_model:
        outputs = [vllm_model.llm.score(query, doc)[0] for query, doc in PAIRS]

    assert [output.prompt_token_ids for output in outputs] == hf_token_ids
    assert [output.outputs.score for output in outputs] == pytest.approx(
        hf_scores, abs=1e-4
    )
