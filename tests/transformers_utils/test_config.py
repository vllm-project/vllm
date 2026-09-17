# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""This test file includes some cases where it is inappropriate to
only get the `eos_token_id` from the tokenizer as defined by
`BaseRenderer.get_eos_token_id`.
"""

import json
import math
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from transformers import BertConfig, LlamaConfig, PreTrainedConfig, Qwen3Config

from vllm.config.model import ModelConfig
from vllm.tokenizers import get_tokenizer
from vllm.transformers_utils import config as config_module
from vllm.transformers_utils.config import (
    SentenceTransformersDenseConfig,
    get_safetensors_params_metadata,
    get_sentence_transformers_cross_encoder_config,
    mrope_num_dims,
    patch_legacy_rope_type,
    try_get_dense_modules,
    try_get_generation_config,
    uses_mrope,
)
from vllm.transformers_utils.configs.glm5_next import (
    Glm5NextConfig,
    Glm5NextTextConfig,
    Glm5NextVisionConfig,
)
from vllm.transformers_utils.configs.mistral import adapt_config_dict


@pytest.mark.parametrize("layout", ["mixed", "flat"])
def test_gemma4_dspark_rope_config_preserves_parameters(tmp_path, layout):
    """Remove redundant shared entries while preserving per-layer and flat RoPE."""
    from transformers import Gemma4TextConfig

    per_layer = {
        "full_attention": {
            "rope_type": "proportional",
            "partial_rotary_factor": 0.25,
            "rope_theta": 1000000.0,
        },
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
    }
    rope_parameters: dict[str, object] = dict(per_layer)
    if layout == "mixed":
        rope_parameters.update(rope_type="default", rope_theta=None)
    else:
        rope_parameters = {"rope_type": "default", "rope_theta": 12345.0}
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4_text",
                "architectures": ["Gemma4DSparkModel"],
                "num_hidden_layers": 1,
                "layer_types": ["full_attention"],
                "rope_parameters": rope_parameters,
            }
        )
    )
    _, config = config_module.HFConfigParser().parse(
        tmp_path, trust_remote_code=False, max_position_embeddings=8192
    )
    assert isinstance(config, Gemma4TextConfig)
    assert config.name_or_path == str(tmp_path)
    assert config.max_position_embeddings == 8192
    if layout == "flat":
        assert config.rope_parameters["rope_theta"] == 12345.0
    else:
        for layer_type, expected in per_layer.items():
            for key, value in expected.items():
                assert config.rope_parameters[layer_type][key] == value
        assert set(config.rope_parameters) == set(per_layer)


def test_patch_legacy_rope_type_preserves_nope_layers():
    """NoPE layers stay disabled while later RoPE layers are normalized."""
    rope_parameters = {
        "full_attention": None,
        "sliding_attention": {
            "type": "mrope",
            "mrope_section": [24, 20, 20],
        },
    }

    patch_legacy_rope_type(rope_parameters)

    assert rope_parameters == {
        "full_attention": None,
        "sliding_attention": {
            "type": "mrope",
            "rope_type": "default",
            "mrope_section": [24, 20, 20],
        },
    }


def test_patch_legacy_rope_type_normalizes_telechat3_yarn():
    """TeleChat3's RoPE is YaRN with 0.07 in place of the usual 0.1.

    Encoding that as a precomputed attention_factor keeps the config
    plain YaRN, which Transformers and every "yarn" guard understand.
    `mscale` cannot express it: Transformers only applies mscale when
    mscale_all_dim is also truthy.
    """
    rope_parameters = {
        "type": "telechat3-yarn",
        "rope_type": "telechat3-yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 8192,
    }

    patch_legacy_rope_type(rope_parameters)

    assert rope_parameters == {
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 8192,
        "attention_factor": pytest.approx(0.07 * math.log(4.0) + 1.0),
    }


def test_mistral_yarn_apply_scale_false_disables_yarn_magnitude_scaling():
    """`yarn.apply_scale: false` must reach the DeepSeek-style attentions.

    Transformers spells it `attention_factor = 1.0`, which DeepseekV2Attention
    and its siblings read to select `deepseek_llama_scaling` over
    `deepseek_yarn`; without it Mistral-Large-3 runs with a spurious
    yarn_get_mscale(factor)^2 attention scaling.
    """
    params = {
        "dim": 7168,
        "n_layers": 61,
        "head_dim": 192,
        "hidden_dim": 16384,
        "n_heads": 128,
        "n_kv_heads": 128,
        "norm_eps": 1e-5,
        "vocab_size": 131072,
        "rope_theta": 10000.0,
        "max_position_embeddings": 294912,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "moe": {
            "num_experts": 128,
            "num_experts_per_tok": 4,
            "num_shared_experts": 1,
            "expert_hidden_dim": 4096,
            "first_k_dense_replace": 3,
            "route_every_n": 1,
            "routed_scale": 1.0,
            "num_expert_groups": 1,
            "num_expert_groups_per_tok": 1,
        },
        "llama_4_scaling": {"beta": 0.1, "original_max_position_embeddings": 8192},
        "yarn": {
            "alpha": 1,
            "apply_scale": False,
            "beta": 32,
            "factor": 36,
            "original_max_position_embeddings": 8192,
        },
    }

    config = adapt_config_dict(params, defaults={})

    assert config.architectures == ["MistralLarge3ForCausalLM"]
    assert config.rope_parameters["attention_factor"] == 1.0


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
            _hf_config_revision=None,
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
            return_value=PreTrainedConfig(),
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


@pytest.mark.parametrize(
    ("section_key", "mrope_section", "expected_num_dims"),
    [
        ("mrope_section", [16, 24, 24], 3),
        ("mrope_section", [16, 16, 16, 16], 4),
        # Interleaved M-RoPE takes 2 sections but still consumes 3D positions
        ("mrope_section", [32, 32], 3),
        # HunYuan-VL checkpoints ship the section under its legacy name
        ("xdrope_section", [16, 16, 16, 16], 4),
    ],
)
def test_mrope_num_dims(section_key, mrope_section, expected_num_dims):
    config = PreTrainedConfig()
    config.rope_parameters = {"rope_type": "default", section_key: mrope_section}

    assert uses_mrope(config)
    assert mrope_num_dims(config) == expected_num_dims


@pytest.mark.parametrize("section_name", ["mrope_section", "xdrope_section"])
def test_mrope_num_dims_from_config_attribute(section_name):
    """Some configs expose the section as an attribute rather than under
    `rope_parameters`."""
    config = PreTrainedConfig()
    setattr(config, section_name, [16, 16, 16, 16])

    assert uses_mrope(config)
    assert mrope_num_dims(config) == 4


def test_mrope_num_dims_from_nested_rope_parameters():
    """Sections nested by layer type must be found, not silently defaulted."""
    config = PreTrainedConfig()
    config.rope_parameters = {
        "full_attention": {"mrope_section": [16, 16, 16, 16]},
        "linear_attention": {"rope_type": "default"},
    }

    assert uses_mrope(config)
    assert mrope_num_dims(config) == 4


def test_mrope_num_dims_without_mrope():
    assert mrope_num_dims(PreTrainedConfig()) == 0


def test_optional_cross_encoder_metadata_probe_is_best_effort(monkeypatch):
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
    (path / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 16}),
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
    """Resolve metadata once and preserve it when runtime config is copied."""
    dense_config = _write_sentence_transformers_cross_encoder(tmp_path)

    cross_encoder_config = get_sentence_transformers_cross_encoder_config(
        str(tmp_path), revision=None
    )
    assert cross_encoder_config is not None
    assert cross_encoder_config.activation_fn == "torch.nn.modules.linear.Identity"
    assert cross_encoder_config.seq_pooling_type == "MEAN"
    assert cross_encoder_config.dense_config == SentenceTransformersDenseConfig(
        in_features=8,
        out_features=1,
        bias=True,
        activation_function=dense_config["activation_function"],
        folder="2_Dense",
    )
    assert not cross_encoder_config.uses_message_format
    assert try_get_dense_modules(str(tmp_path), revision=None) == [
        {**dense_config, "folder": "2_Dense"}
    ]

    with patch(
        "vllm.config.model.get_sentence_transformers_cross_encoder_config",
        wraps=get_sentence_transformers_cross_encoder_config,
    ) as load_config:
        model_config = ModelConfig(str(tmp_path), dtype="float32")
    load_config.assert_called_once_with(str(tmp_path), None, None)

    assert model_config.runner_type == "pooling"
    assert model_config.convert_type == "classify"
    assert model_config.hf_config.num_labels == 1
    assert model_config.sentence_transformers_config == cross_encoder_config
    assert model_config.hf_config.sentence_transformers == {
        "activation_fn": cross_encoder_config.activation_fn
    }
    assert not ModelConfig.__dataclass_fields__["sentence_transformers_config"].init
    copied = deepcopy(model_config)
    assert copied.sentence_transformers_config == cross_encoder_config
    assert copied.compute_hash() == model_config.compute_hash()
    assert model_config.pooler_config is not None
    assert model_config.pooler_config.seq_pooling_type == "MEAN"
    assert model_config.pooler_config.use_activation
    assert model_config.max_model_len == 16

    from vllm.model_executor.model_loader import get_model_cls
    from vllm.model_executor.models.interfaces_base import get_score_type

    model_cls = get_model_cls(model_config)
    assert get_score_type(model_cls) == "cross-encoder"


def test_cross_encoder_reload_metadata_compares_effective_semantics(tmp_path):
    """Defaults and artifact locations do not change the instantiated scoring head."""
    _write_sentence_transformers_cross_encoder(tmp_path)
    original = get_sentence_transformers_cross_encoder_config(str(tmp_path))
    (tmp_path / "2_Dense").rename(tmp_path / "head")
    modules_path = tmp_path / "modules.json"
    modules = json.loads(modules_path.read_text())
    modules[-1]["path"] = "head"
    modules_path.write_text(json.dumps(modules))
    dense_path = tmp_path / "head/config.json"
    dense = json.loads(dense_path.read_text())
    del dense["bias"]
    dense_path.write_text(json.dumps(dense))
    metadata_path = tmp_path / "config_sentence_transformers.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["__version__"] = {"sentence_transformers": "future-version"}
    metadata_path.write_text(json.dumps(metadata))

    reloaded = get_sentence_transformers_cross_encoder_config(str(tmp_path))
    assert reloaded == original
    assert reloaded is not None and reloaded.dense_config is not None
    assert reloaded.dense_config.folder == "head"

    dense["out_features"] = 2
    dense_path.write_text(json.dumps(dense))
    assert get_sentence_transformers_cross_encoder_config(str(tmp_path)) != original


def test_cross_encoder_tokenizer_limit_forwards_authentication(tmp_path, monkeypatch):
    _write_sentence_transformers_cross_encoder(tmp_path)
    get_tokenizer_config = MagicMock(return_value={"model_max_length": 12})
    monkeypatch.setattr(config_module, "get_tokenizer_config", get_tokenizer_config)

    model_config = ModelConfig(str(tmp_path), dtype="float32", hf_token="secret")

    assert model_config.max_model_len == 12
    get_tokenizer_config.assert_called_once_with(
        str(tmp_path), trust_remote_code=False, revision=None, token="secret"
    )


def _write_rotary_cross_encoder(path, rope_type):
    _write_sentence_transformers_cross_encoder(path)
    rope_parameters: dict[str, Any] = {"rope_type": rope_type, "factor": 2.0}
    if rope_type == "longrope":
        rope_parameters.update(short_factor=[1.0, 1.0], long_factor=[2.0, 2.0])
    LlamaConfig(
        architectures=["LlamaModel"],
        hidden_size=8,
        intermediate_size=16,
        max_position_embeddings=64,
        original_max_position_embeddings=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=1,
        rope_parameters=rope_parameters,
        vocab_size=32,
    ).save_pretrained(path)


@pytest.mark.parametrize("rope_type", ["linear", "longrope"])
@pytest.mark.parametrize("max_model_len", [None, -1, 8, 24])
def test_cross_encoder_tokenizer_limit_is_not_rope_scaled(
    tmp_path, rope_type, max_model_len
):
    _write_rotary_cross_encoder(tmp_path, rope_type)

    if max_model_len == 24:
        with pytest.raises(ValueError, match="greater.*derived max_model_len"):
            ModelConfig(str(tmp_path), dtype="float32", max_model_len=max_model_len)
    else:
        config = ModelConfig(
            str(tmp_path), dtype="float32", max_model_len=max_model_len
        )
        assert config.max_model_len == (8 if max_model_len == 8 else 16)


@pytest.mark.parametrize("rope_type", ["linear", "longrope"])
@pytest.mark.parametrize("max_model_len", [None, 64])
def test_generation_ignores_cross_encoder_tokenizer_limit(
    tmp_path, rope_type, max_model_len
):
    _write_rotary_cross_encoder(tmp_path, rope_type)

    generation_config = ModelConfig(
        str(tmp_path),
        dtype="float32",
        runner="generate",
        convert="none",
        max_model_len=max_model_len,
        hf_overrides={"architectures": ["LlamaForCausalLM"]},
    )
    expected_max_len = max_model_len or (128 if rope_type == "linear" else 32)
    assert generation_config.max_model_len == expected_max_len


def test_absolute_position_pooling_preserves_tokenizer_limit(tmp_path):
    BertConfig(
        architectures=["BertModel"], position_embedding_type="absolute"
    ).save_pretrained(tmp_path)
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 16}), encoding="utf-8"
    )

    config = ModelConfig(str(tmp_path), dtype="float32")

    assert config.max_model_len == 16


def test_tokenizer_limit_supplies_unknown_architecture_max_length(tmp_path):
    _write_sentence_transformers_cross_encoder(tmp_path)
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 4096}), encoding="utf-8"
    )
    config = ModelConfig(str(tmp_path), dtype="float32")
    config.model_arch_config.derived_max_model_len_and_key = (float("inf"), None)

    assert config.get_and_verify_max_len(-1) == 4096


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
    assert config.seq_pooling_type == expected_pooling_type


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


@pytest.mark.parametrize("module_count", [None, 0, 1])
def test_traditional_cross_encoder_topology_is_not_claimed(tmp_path, module_count):
    _write_sentence_transformers_cross_encoder(tmp_path)

    transformer_config_path = tmp_path / "sentence_bert_config.json"
    transformer_config = json.loads(transformer_config_path.read_text(encoding="utf-8"))
    transformer_config.update(
        {
            "transformer_task": "sequence-classification",
            "module_output_name": "scores",
            "modality_config": {
                "text": {
                    "method": "forward",
                    "method_output_name": "logits",
                }
            },
        }
    )
    transformer_config_path.write_text(json.dumps(transformer_config), encoding="utf-8")

    modules_path = tmp_path / "modules.json"
    if module_count is None:
        modules_path.unlink()
    else:
        modules = json.loads(modules_path.read_text(encoding="utf-8"))[:module_count]
        modules_path.write_text(json.dumps(modules), encoding="utf-8")

    assert (
        get_sentence_transformers_cross_encoder_config(str(tmp_path), revision=None)
        is None
    )


def _write_logit_score_cross_encoder(path, *, false_id=5, task="text-generation"):
    _write_sentence_transformers_cross_encoder(path)
    Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        hidden_size=128,
        intermediate_size=256,
        head_dim=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        vocab_size=32,
        max_position_embeddings=64,
    ).save_pretrained(path)
    modules = json.loads((path / "modules.json").read_text())[:1]
    modules.append(
        {
            "idx": 1,
            "name": "1",
            "path": "1_LogitScore",
            "type": (
                "sentence_transformers.cross_encoder.modules.logit_score.LogitScore"
            ),
        }
    )
    (path / "modules.json").write_text(json.dumps(modules))
    (path / "sentence_bert_config.json").write_text(
        json.dumps(
            {
                "transformer_task": task,
                "module_output_name": "causal_logits",
                "modality_config": {
                    "text": {"method": "forward", "method_output_name": "logits"}
                },
            }
        )
    )
    (path / "1_LogitScore").mkdir()
    (path / "1_LogitScore/config.json").write_text(
        json.dumps(
            {
                "true_token_id": 7,
                "false_token_id": false_id,
                "module_input_name": "causal_logits",
            }
        )
    )


@pytest.mark.parametrize("false_id", [None, 0, 5])
@pytest.mark.parametrize("task", ["text-generation", "any-to-any"])
def test_logit_score_automatically_resolves_to_last_token_classifier(
    tmp_path, false_id, task
):
    _write_logit_score_cross_encoder(tmp_path, false_id=false_id, task=task)
    config = ModelConfig(str(tmp_path), dtype="float32")
    assert config.runner_type == "pooling"
    assert config.convert_type == "classify"
    assert config.pooler_config is not None
    assert config.pooler_config.seq_pooling_type == "LAST"
    assert config.hf_config.num_labels == 1
    assert config.use_sep_token
    assert config.hf_config.classifier_from_token == (
        [7] if false_id is None else [false_id, 7]
    )
    assert config.hf_config.method == (
        "no_post_processing" if false_id is None else "from_2_way_softmax"
    )
    assert config.max_model_len == 16


@pytest.mark.parametrize("message_format", ["flat", "structured"])
@pytest.mark.parametrize("config_source", ["dict", "callable", "saved", "convert"])
def test_manual_token_classifier_uses_existing_hf_scoring_path(
    tmp_path, message_format, config_source
):
    """Complete HF classifier contracts win regardless of configuration source."""
    from vllm.model_executor.layers.pooler.activations import PoolerClassify, get_act_fn

    _write_logit_score_cross_encoder(tmp_path)
    path = tmp_path / "sentence_bert_config.json"
    transformer_config = json.loads(path.read_text())
    transformer_config["modality_config"]["message"] = {
        "method": "forward",
        "method_output_name": "logits",
        "format": message_format,
    }
    path.write_text(json.dumps(transformer_config))
    overrides = {
        "architectures": ["Qwen3ForSequenceClassification"],
        "classifier_from_token": ["no", "yes"],
        "is_original_qwen3_reranker": True,
    }
    config_kwargs: dict[str, Any] = {"hf_overrides": overrides}
    if config_source == "callable":

        def apply_overrides(config: PreTrainedConfig) -> PreTrainedConfig:
            config.update(overrides)
            return config

        config_kwargs["hf_overrides"] = apply_overrides
    elif config_source == "saved":
        path = tmp_path / "config.json"
        hf_config = json.loads(path.read_text())
        hf_config.update(overrides)
        path.write_text(json.dumps(hf_config))
        config_kwargs = {}
    elif config_source == "convert":
        del overrides["architectures"]
        overrides["method"] = "from_2_way_softmax"
        config_kwargs["convert"] = "classify"

    config = ModelConfig(str(tmp_path), dtype="float32", **config_kwargs)

    assert config.sentence_transformers_config is None
    assert config.architectures == [
        "Qwen3ForCausalLM"
        if config_source == "convert"
        else "Qwen3ForSequenceClassification"
    ]
    assert config.runner_type == "pooling"
    assert config.convert_type == "classify"
    assert config.hf_config.classifier_from_token == ["no", "yes"]
    assert config.hf_text_config.method == "from_2_way_softmax"
    assert not hasattr(config.hf_config, "sentence_transformers")
    assert isinstance(get_act_fn(config.hf_config), PoolerClassify)


@pytest.mark.parametrize(
    "overrides,saved_classifier,convert",
    [
        ({}, False, "auto"),
        ({"rope_theta": 20000}, False, "auto"),
        ({"architectures": ["Qwen3ForSequenceClassification"]}, False, "auto"),
        ({"classifier_from_token": None}, False, "auto"),
        ({"classifier_from_token": ["no", "yes"]}, False, "auto"),
        ({}, True, "auto"),
        ({}, False, "classify"),
    ],
)
def test_flat_logit_score_stays_strict_without_complete_classifier_contract(
    tmp_path, overrides, saved_classifier, convert
):
    _write_logit_score_cross_encoder(tmp_path)
    path = tmp_path / "sentence_bert_config.json"
    transformer_config = json.loads(path.read_text())
    transformer_config["modality_config"]["message"] = {
        "method": "forward",
        "method_output_name": "logits",
        "format": "flat",
    }
    path.write_text(json.dumps(transformer_config))
    if saved_classifier:
        path = tmp_path / "config.json"
        hf_config = json.loads(path.read_text())
        hf_config["classifier_from_token"] = ["no", "yes"]
        path.write_text(json.dumps(hf_config))

    with pytest.raises(ValueError, match="only structured"):
        ModelConfig(
            str(tmp_path), dtype="float32", hf_overrides=overrides, convert=convert
        )


@pytest.mark.parametrize(
    "setting",
    [
        {"text": {"max_length": 16}},
        {"chat_template": {"add_generation_prompt": "true"}},
        {"chat_template": {"continue_final_message": True}},
    ],
)
def test_logit_score_rejects_unsupported_processing_settings(tmp_path, setting):
    _write_logit_score_cross_encoder(tmp_path)
    path = tmp_path / "sentence_bert_config.json"
    config = json.loads(path.read_text())
    config["processing_kwargs"] = setting
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="LogitScore supports only"):
        ModelConfig(str(tmp_path), dtype="float32")


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("true_token_id", -1, "non-negative integers"),
        ("true_token_id", True, "non-negative integers"),
        ("true_token_id", 32, "outside the model vocabulary"),
        ("false_token_id", "no", "non-negative integers"),
        ("module_input_name", "hidden_states", "must read causal_logits"),
    ],
)
def test_logit_score_rejects_invalid_contract(tmp_path, field, value, match):
    _write_logit_score_cross_encoder(tmp_path)
    path = tmp_path / "1_LogitScore/config.json"
    config = json.loads(path.read_text())
    config[field] = value
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=match):
        ModelConfig(str(tmp_path), dtype="float32")


@pytest.mark.parametrize("unknown_module", [False, True])
def test_unsupported_modular_sentence_transformers_cross_encoder_fails_closed(
    tmp_path,
    unknown_module,
):
    _write_sentence_transformers_cross_encoder(tmp_path)
    modules_path = tmp_path / "modules.json"
    modules = json.loads(modules_path.read_text(encoding="utf-8"))
    modules[-1]["type"] = (
        "sentence_transformers.cross_encoder.modules.logit_score.LogitScore"
    )
    if unknown_module:
        modules = modules[:1] + [{"type": "custom.Unsupported", "path": "head"}]
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
