# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small on-disk fixtures for checkpoint metadata tests."""

import json
from pathlib import Path
from typing import Any, Literal


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def update_json(path: Path, **values: Any) -> None:
    config = json.loads(path.read_text())
    write_json(path, {**config, **values})


def write_cross_encoder_metadata(
    path: Path, *, head: Literal["dense", "logit"] = "dense", hidden_size: int = 8
) -> None:
    """Write metadata only; real-export tests independently use ST serialization."""
    logit = head == "logit"
    files = {
        "config_sentence_transformers.json": {
            "model_type": "CrossEncoder",
            "activation_fn": "torch.nn.modules.linear.Identity",
        },
        "sentence_bert_config.json": {
            "transformer_task": "text-generation" if logit else "feature-extraction",
            "module_output_name": "causal_logits" if logit else "token_embeddings",
            "modality_config": {
                "text": {
                    "method": "forward",
                    "method_output_name": "logits" if logit else "last_hidden_state",
                }
            },
        },
        "tokenizer_config.json": {"model_max_length": 16},
    }
    if logit:
        files["1_LogitScore/config.json"] = {
            "true_token_id": 7,
            "false_token_id": 5,
            "module_input_name": "causal_logits",
        }
        heads = [("1_LogitScore", "cross_encoder.modules.logit_score.LogitScore")]
    else:
        files["1_Pooling/config.json"] = {
            "pooling_mode": "mean",
            "include_prompt": True,
        }
        files["2_Dense/config.json"] = {
            "in_features": hidden_size,
            "out_features": 1,
            "bias": True,
            "activation_function": "torch.nn.modules.activation.Tanh",
            "module_input_name": "sentence_embedding",
            "module_output_name": "scores",
        }
        heads = [
            ("1_Pooling", "sentence_transformer.modules.pooling.Pooling"),
            ("2_Dense", "base.modules.dense.Dense"),
        ]
    modules = [("", "base.modules.transformer.Transformer"), *heads]
    write_json(
        path / "modules.json",
        [
            {"path": folder, "type": f"sentence_transformers.{module}"}
            for folder, module in modules
        ],
    )
    for name, config in files.items():
        write_json(path / name, config)
