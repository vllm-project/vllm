# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This test file includes some cases where it is inappropriate to
only get the `eos_token_id` from the tokenizer as defined by
`BaseRenderer.get_eos_token_id`.
"""

import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from transformers import BertConfig, PretrainedConfig

from vllm.config.model import ModelConfig
from vllm.tokenizers import get_tokenizer
from vllm.transformers_utils import config as config_module
from vllm.transformers_utils.config import (
    get_safetensors_params_metadata,
    get_sentence_transformers_cross_encoder_config,
    try_get_dense_modules,
    try_get_generation_config,
)
from vllm.transformers_utils.configs.glm5_next import (
    Glm5NextConfig,
    Glm5NextTextConfig,
    Glm5NextVisionConfig,
)


def test_glm5_next_accepts_deepseek_sparse_attention_layers():
    layer_types = ["linear_attention", "deepseek_sparse_attention"]

    config = Glm5NextTextConfig(
        num_hidden_layers=len(layer_types), layer_types=layer_types
    )

    assert config.layer_types == layer_types
    assert config.layers_block_type == ["linear_attention", "attention"]


def test_glm5_next_accepts_prebuilt_subconfigs():
    text_config = Glm5NextTextConfig(hidden_size=1024)
    vision_config = Glm5NextVisionConfig(hidden_size=768)

    config = Glm5NextConfig(
        text_config=text_config,
        vision_config=vision_config,
    )

    assert config.text_config is text_config
    assert config.vision_config is vision_config


@pytest.mark.parametrize(
    ("kwargs", "option"),
    [
        (
            {"index_topk": 2048, "index_dsa_use_layernorm": False},
            "index_dsa_use_layernorm",
        ),
        (
            {"index_topk": 2048, "index_kpool_compress": False},
            "index_kpool_compress",
        ),
        (
            {"index_topk": 2048, "index_kpool_always_select_tail": False},
            "index_kpool_always_select_tail",
        ),
        ({"hres_vwnstyle": False}, "hres_vwnstyle"),
        ({"mhc_no_norm_weight": True}, "mhc_no_norm_weight"),
    ],
)
def test_glm5_next_rejects_unimplemented_config_options(kwargs, option):
    with pytest.raises(NotImplementedError, match=option):
        Glm5NextTextConfig(**kwargs)


def test_get_llama3_eos_token():
    model_name = "meta-llama/Llama-3.2-1B-Instruct"

    tokenizer = get_tokenizer(model_name)
    assert tokenizer.eos_token_id == 128009

    generation_config = try_get_generation_config(model_name, trust_remote_code=False)
    assert generation_config is not None
    assert generation_config.eos_token_id == [128001, 128008, 128009]


def test_get_blip2_eos_token():
    model_name = "Salesforce/blip2-opt-2.7b"

    tokenizer = get_tokenizer(model_name)
    assert tokenizer.eos_token_id == 2

    generation_config = try_get_generation_config(model_name, trust_remote_code=False)
    assert generation_config is not None
    assert generation_config.eos_token_id == 50118


def test_model_config_generation_fallback_forwards_code_revision():
    model_config = cast(
        ModelConfig,
        SimpleNamespace(
            generation_config="auto",
            hf_config_path=None,
            model="org/model",
            trust_remote_code=True,
            revision="model-pin",
            code_revision="code-pin",
            config_format="auto",
            hf_token=None,
        ),
    )

    with (
        patch.object(
            config_module.GenerationConfig,
            "from_pretrained",
            side_effect=OSError,
        ),
        patch.object(
            config_module,
            "get_config",
            return_value=PretrainedConfig(),
        ) as get_config,
    ):
        ModelConfig.try_get_generation_config(model_config)

    get_config.assert_called_once_with(
        "org/model",
        trust_remote_code=True,
        revision="model-pin",
        code_revision="code-pin",
        config_format="auto",
        token=None,
    )


def test_safetensors_metadata_of_repo_without_safetensors():
    """A repo storing its weights in another format is an answer, not a failure,
    so it must not be retried."""
    from huggingface_hub.errors import LocalEntryNotFoundError, NotASafetensorsRepoError

    get_safetensors_metadata = MagicMock(
        side_effect=NotASafetensorsRepoError("not a safetensors repo")
    )
    api = SimpleNamespace(
        get_safetensors_metadata=get_safetensors_metadata,
        snapshot_download=MagicMock(side_effect=LocalEntryNotFoundError("no cache")),
    )

    with patch.object(config_module, "hf_api", lambda: api):
        assert get_safetensors_params_metadata("some/pytorch-only-model") == {}

    get_safetensors_metadata.assert_called_once()


def test_optional_cross_encoder_metadata_probe_is_best_effort(monkeypatch):
    get_sentence_transformers_cross_encoder_config.cache_clear()
    get_hf_file_to_dict = MagicMock(return_value=None)
    monkeypatch.setattr(config_module, "get_hf_file_to_dict", get_hf_file_to_dict)
    monkeypatch.setattr(
        config_module,
        "file_or_path_exists",
        MagicMock(side_effect=AssertionError("unexpected repository listing")),
    )

    assert (
        get_sentence_transformers_cross_encoder_config(
            "org/ordinary-model", revision="main", hf_token="secret"
        )
        is None
    )
    get_hf_file_to_dict.assert_called_once_with(
        "config_sentence_transformers.json",
        "org/ordinary-model",
        "main",
        token="secret",
    )


def test_cross_encoder_metadata_must_be_an_object(monkeypatch):
    get_sentence_transformers_cross_encoder_config.cache_clear()
    monkeypatch.setattr(
        config_module,
        "get_hf_file_to_dict",
        lambda *_args, **_kwargs: [],
    )

    with pytest.raises(ValueError, match="must contain a JSON object"):
        get_sentence_transformers_cross_encoder_config(
            "org/malformed-cross-encoder", revision="main"
        )


def _write_sentence_transformers_cross_encoder(path):
    BertConfig(
        architectures=["BertModel"],
        hidden_size=8,
        intermediate_size=16,
        max_position_embeddings=32,
        num_attention_heads=2,
        num_hidden_layers=1,
        vocab_size=32,
    ).save_pretrained(path)

    (path / "config_sentence_transformers.json").write_text(
        json.dumps(
            {
                "model_type": "CrossEncoder",
                "activation_fn": "torch.nn.modules.linear.Identity",
                "prompts": {},
                "default_prompt_name": None,
            }
        ),
        encoding="utf-8",
    )
    (path / "sentence_bert_config.json").write_text(
        json.dumps(
            {
                "transformer_task": "feature-extraction",
                "max_seq_length": 32,
                "do_lower_case": False,
                "processing_kwargs": {},
                "module_output_name": "token_embeddings",
                "modality_config": {
                    "text": {
                        "method": "forward",
                        "method_output_name": "last_hidden_state",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (path / "modules.json").write_text(
        json.dumps(
            [
                {
                    "idx": 0,
                    "name": "0",
                    "path": "",
                    "type": (
                        "sentence_transformers.base.modules.transformer.Transformer"
                    ),
                },
                {
                    "idx": 1,
                    "name": "1",
                    "path": "1_Pooling",
                    "type": (
                        "sentence_transformers.sentence_transformer.modules."
                        "pooling.Pooling"
                    ),
                },
                {
                    "idx": 2,
                    "name": "2",
                    "path": "2_Dense",
                    "type": "sentence_transformers.base.modules.dense.Dense",
                },
            ]
        ),
        encoding="utf-8",
    )

    pooling_path = path / "1_Pooling"
    pooling_path.mkdir()
    (pooling_path / "config.json").write_text(
        json.dumps(
            {
                "embedding_dimension": 8,
                "pooling_mode": "mean",
                "include_prompt": True,
            }
        ),
        encoding="utf-8",
    )

    dense_config = {
        "in_features": 8,
        "out_features": 1,
        "bias": True,
        "activation_function": "torch.nn.modules.activation.Tanh",
        "module_input_name": "sentence_embedding",
        "module_output_name": "scores",
    }
    dense_path = path / "2_Dense"
    dense_path.mkdir()
    (dense_path / "config.json").write_text(
        json.dumps(dense_config),
        encoding="utf-8",
    )
    return dense_config


def test_current_sentence_transformers_cross_encoder_config(tmp_path):
    dense_config = _write_sentence_transformers_cross_encoder(tmp_path)

    cross_encoder_config = get_sentence_transformers_cross_encoder_config(
        str(tmp_path), revision=None
    )
    assert cross_encoder_config is not None
    assert cross_encoder_config.model_config["model_type"] == "CrossEncoder"
    assert cross_encoder_config.pooler_config == {"seq_pooling_type": "MEAN"}
    assert cross_encoder_config.dense_config == {
        **dense_config,
        "folder": "2_Dense",
    }
    assert not cross_encoder_config.uses_message_format
    assert try_get_dense_modules(str(tmp_path), revision=None) == [
        {**dense_config, "folder": "2_Dense"}
    ]

    model_config = ModelConfig(str(tmp_path), dtype="float32")

    assert model_config.runner_type == "pooling"
    assert model_config.convert_type == "classify"
    assert model_config.hf_config.num_labels == 1
    assert model_config.hf_config.sentence_transformers == (
        cross_encoder_config.model_config
    )
    assert model_config.pooler_config is not None
    assert model_config.pooler_config.seq_pooling_type == "MEAN"
    assert model_config.pooler_config.use_activation

    from vllm.model_executor.model_loader import get_model_cls
    from vllm.model_executor.models.interfaces_base import get_score_type

    model_cls = get_model_cls(model_config)
    assert get_score_type(model_cls) == "cross-encoder"


@pytest.mark.parametrize(
    ("pooling_mode", "expected_pooling_type"),
    [("cls", "CLS"), ("mean", "MEAN"), ("lasttoken", "LAST")],
)
def test_cross_encoder_supported_pooling_modes(
    tmp_path,
    pooling_mode,
    expected_pooling_type,
):
    _write_sentence_transformers_cross_encoder(tmp_path)
    pooling_config_path = tmp_path / "1_Pooling/config.json"
    pooling_config = json.loads(pooling_config_path.read_text(encoding="utf-8"))
    pooling_config["pooling_mode"] = pooling_mode
    pooling_config_path.write_text(json.dumps(pooling_config), encoding="utf-8")

    config = get_sentence_transformers_cross_encoder_config(
        str(tmp_path), revision=None
    )

    assert config is not None
    assert config.pooler_config == {"seq_pooling_type": expected_pooling_type}


def test_cross_encoder_rejects_left_padded_cls_pooling(tmp_path):
    _write_sentence_transformers_cross_encoder(tmp_path)
    pooling_config_path = tmp_path / "1_Pooling/config.json"
    pooling_config = json.loads(pooling_config_path.read_text(encoding="utf-8"))
    pooling_config["pooling_mode"] = "cls"
    pooling_config_path.write_text(json.dumps(pooling_config), encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"padding_side": "left"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="CLS pooling.*left-padded"):
        get_sentence_transformers_cross_encoder_config(str(tmp_path), revision=None)


def test_unsupported_pooled_sentence_transformers_cross_encoder_fails_closed(
    tmp_path,
):
    _write_sentence_transformers_cross_encoder(tmp_path)
    modules_path = tmp_path / "modules.json"
    modules = json.loads(modules_path.read_text(encoding="utf-8"))
    modules[-1]["type"] = (
        "sentence_transformers.cross_encoder.modules.logit_score.LogitScore"
    )
    modules_path.write_text(json.dumps(modules), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported modular CrossEncoder"):
        ModelConfig(str(tmp_path), dtype="float32")


@pytest.mark.parametrize(
    ("config_file", "field", "value", "match"),
    [
        (
            "1_Pooling/config.json",
            "pooling_mode",
            ["cls", "mean"],
            "exactly one pooling mode",
        ),
        (
            "1_Pooling/config.json",
            "pooling_mode",
            "mean_sqrt_len_tokens",
            "cls, mean, or lasttoken",
        ),
        (
            "1_Pooling/config.json",
            "include_prompt",
            False,
            "include_prompt=true",
        ),
        (
            "2_Dense/config.json",
            "use_residual",
            True,
            "residual Dense",
        ),
        (
            "sentence_bert_config.json",
            "module_output_name",
            "sentence_embedding",
            "token_embeddings",
        ),
    ],
)
def test_cross_encoder_rejects_unsupported_semantics(
    tmp_path,
    config_file,
    field,
    value,
    match,
):
    _write_sentence_transformers_cross_encoder(tmp_path)
    config_path = tmp_path / config_file
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config[field] = value
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        get_sentence_transformers_cross_encoder_config(str(tmp_path), revision=None)


@pytest.mark.parametrize(
    ("config_file", "field", "match"),
    [
        (
            "config_sentence_transformers.json",
            "activation_fn",
            "activation_fn",
        ),
        (
            "2_Dense/config.json",
            "activation_function",
            "activation_function",
        ),
    ],
)
def test_cross_encoder_requires_saved_activations(
    tmp_path,
    config_file,
    field,
    match,
):
    _write_sentence_transformers_cross_encoder(tmp_path)
    config_path = tmp_path / config_file
    config = json.loads(config_path.read_text(encoding="utf-8"))
    del config[field]
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        get_sentence_transformers_cross_encoder_config(str(tmp_path), revision=None)


def test_cross_encoder_message_modality_requires_saved_template(tmp_path):
    _write_sentence_transformers_cross_encoder(tmp_path)
    transformer_config_path = tmp_path / "sentence_bert_config.json"
    transformer_config = json.loads(transformer_config_path.read_text(encoding="utf-8"))
    transformer_config["modality_config"]["message"] = {
        "method": "forward",
        "method_output_name": "last_hidden_state",
        "format": "structured",
    }
    transformer_config_path.write_text(json.dumps(transformer_config), encoding="utf-8")

    with pytest.raises(ValueError, match="saved chat template"):
        get_sentence_transformers_cross_encoder_config(str(tmp_path), revision=None)

    (tmp_path / "chat_template.jinja").write_text(
        "{{ messages | length }}", encoding="utf-8"
    )
    get_sentence_transformers_cross_encoder_config.cache_clear()
    config = get_sentence_transformers_cross_encoder_config(
        str(tmp_path), revision=None
    )
    assert config is not None
    assert config.uses_message_format


def test_cross_encoder_dense_module_must_output_scores(tmp_path):
    _write_sentence_transformers_cross_encoder(tmp_path)
    dense_config_path = tmp_path / "2_Dense" / "config.json"
    dense_config = json.loads(dense_config_path.read_text(encoding="utf-8"))
    dense_config["module_output_name"] = "sentence_embedding"
    dense_config_path.write_text(json.dumps(dense_config), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="must map sentence_embedding to scores",
    ):
        get_sentence_transformers_cross_encoder_config(str(tmp_path), revision=None)
