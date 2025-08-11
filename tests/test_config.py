# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import MISSING, Field, asdict, dataclass, field

import pytest

from vllm.compilation.backends import VllmBackend
from vllm.config import (LoadConfig, ModelConfig, PoolerConfig, VllmConfig,
                         get_field, update_config)
from vllm.model_executor.layers.pooler import PoolingType
from vllm.platforms import current_platform


def test_compile_config_repr_succeeds():
    # setup: VllmBackend mutates the config object
    config = VllmConfig()
    backend = VllmBackend(config)
    backend.configure_post_pass()

    # test that repr(config) succeeds
    val = repr(config)
    assert 'VllmConfig' in val
    assert 'inductor_passes' in val


@dataclass
class _TestConfigFields:
    a: int
    b: dict = field(default_factory=dict)
    c: str = "default"


def test_get_field():
    with pytest.raises(ValueError):
        get_field(_TestConfigFields, "a")

    b = get_field(_TestConfigFields, "b")
    assert isinstance(b, Field)
    assert b.default is MISSING
    assert b.default_factory is dict

    c = get_field(_TestConfigFields, "c")
    assert isinstance(c, Field)
    assert c.default == "default"
    assert c.default_factory is MISSING


@dataclass
class _TestNestedConfig:
    a: _TestConfigFields = field(
        default_factory=lambda: _TestConfigFields(a=0))


def test_update_config():
    # Simple update
    config1 = _TestConfigFields(a=0)
    new_config1 = update_config(config1, {"a": 42})
    assert new_config1.a == 42
    # Nonexistent field
    with pytest.raises(AssertionError):
        new_config1 = update_config(config1, {"nonexistent": 1})
    # Nested update with dataclass
    config2 = _TestNestedConfig()
    new_inner_config = _TestConfigFields(a=1, c="new_value")
    new_config2 = update_config(config2, {"a": new_inner_config})
    assert new_config2.a == new_inner_config
    # Nested update with dict
    config3 = _TestNestedConfig()
    new_config3 = update_config(config3, {"a": {"c": "new_value"}})
    assert new_config3.a.c == "new_value"
    # Nested update with invalid type
    with pytest.raises(AssertionError):
        new_config3 = update_config(config3, {"a": "new_value"})


# Can remove once --task option is fully deprecated
@pytest.mark.parametrize(
    ("model_id", "expected_runner_type", "expected_convert_type",
     "expected_task"),
    [
        ("distilbert/distilgpt2", "generate", "none", "generate"),
        ("intfloat/multilingual-e5-small", "pooling", "none", "embed"),
        ("jason9693/Qwen2.5-1.5B-apeach", "pooling", "classify", "classify"),
        ("cross-encoder/ms-marco-MiniLM-L-6-v2", "pooling", "none",
         "classify"),
        ("Qwen/Qwen2.5-Math-RM-72B", "pooling", "none", "reward"),
        ("openai/whisper-small", "generate", "none", "transcription"),
    ],
)
def test_auto_task(model_id, expected_runner_type, expected_convert_type,
                   expected_task):
    config = ModelConfig(model_id, task="auto")

    assert config.runner_type == expected_runner_type
    assert config.convert_type == expected_convert_type
    assert expected_task in config.supported_tasks


# Can remove once --task option is fully deprecated
@pytest.mark.parametrize(
    ("model_id", "expected_runner_type", "expected_convert_type",
     "expected_task"),
    [
        ("distilbert/distilgpt2", "pooling", "embed", "embed"),
        ("intfloat/multilingual-e5-small", "pooling", "embed", "embed"),
        ("jason9693/Qwen2.5-1.5B-apeach", "pooling", "classify", "classify"),
        ("cross-encoder/ms-marco-MiniLM-L-6-v2", "pooling", "classify",
         "classify"),
        ("Qwen/Qwen2.5-Math-RM-72B", "pooling", "embed", "embed"),
        ("openai/whisper-small", "pooling", "embed", "embed"),
    ],
)
def test_score_task(model_id, expected_runner_type, expected_convert_type,
                    expected_task):
    config = ModelConfig(model_id, task="score")

    assert config.runner_type == expected_runner_type
    assert config.convert_type == expected_convert_type
    assert expected_task in config.supported_tasks


# Can remove once --task option is fully deprecated
@pytest.mark.parametrize(
    ("model_id", "expected_runner_type", "expected_convert_type",
     "expected_task"),
    [
        ("openai/whisper-small", "generate", "none", "transcription"),
    ],
)
def test_transcription_task(model_id, expected_runner_type,
                            expected_convert_type, expected_task):
    config = ModelConfig(model_id, task="transcription")

    assert config.runner_type == expected_runner_type
    assert config.convert_type == expected_convert_type
    assert expected_task in config.supported_tasks


@pytest.mark.parametrize(
    ("model_id", "expected_runner_type", "expected_convert_type"),
    [
        ("distilbert/distilgpt2", "generate", "none"),
        ("intfloat/multilingual-e5-small", "pooling", "none"),
        ("jason9693/Qwen2.5-1.5B-apeach", "pooling", "classify"),
        ("cross-encoder/ms-marco-MiniLM-L-6-v2", "pooling", "none"),
        ("Qwen/Qwen2.5-Math-RM-72B", "pooling", "none"),
        ("openai/whisper-small", "generate", "none"),
    ],
)
def test_auto_runner(model_id, expected_runner_type, expected_convert_type):
    config = ModelConfig(model_id, runner="auto")

    assert config.runner_type == expected_runner_type
    assert config.convert_type == expected_convert_type


@pytest.mark.parametrize(
    ("model_id", "expected_runner_type", "expected_convert_type"),
    [
        ("distilbert/distilgpt2", "pooling", "embed"),
        ("intfloat/multilingual-e5-small", "pooling", "none"),
        ("jason9693/Qwen2.5-1.5B-apeach", "pooling", "classify"),
        ("cross-encoder/ms-marco-MiniLM-L-6-v2", "pooling", "none"),
        ("Qwen/Qwen2.5-Math-RM-72B", "pooling", "none"),
        ("openai/whisper-small", "pooling", "embed"),
    ],
)
def test_pooling_runner(model_id, expected_runner_type, expected_convert_type):
    config = ModelConfig(model_id, runner="pooling")

    assert config.runner_type == expected_runner_type
    assert config.convert_type == expected_convert_type


@pytest.mark.parametrize(
    ("model_id", "expected_runner_type", "expected_convert_type"),
    [
        ("Qwen/Qwen2.5-1.5B-Instruct", "draft", "none"),
    ],
)
def test_draft_runner(model_id, expected_runner_type, expected_convert_type):
    config = ModelConfig(model_id, runner="draft")

    assert config.runner_type == expected_runner_type
    assert config.convert_type == expected_convert_type


MODEL_IDS_EXPECTED = [
    ("Qwen/Qwen1.5-7B", 32768),
    ("mistralai/Mistral-7B-v0.1", 4096),
    ("mistralai/Mistral-7B-Instruct-v0.2", 32768),
]


@pytest.mark.parametrize("model_id_expected", MODEL_IDS_EXPECTED)
def test_disable_sliding_window(model_id_expected):
    model_id, expected = model_id_expected
    model_config = ModelConfig(model_id, disable_sliding_window=True)
    assert model_config.max_model_len == expected


@pytest.mark.skipif(current_platform.is_rocm(),
                    reason="Xformers backend is not supported on ROCm.")
def test_get_pooling_config():
    model_id = "sentence-transformers/all-MiniLM-L12-v2"
    model_config = ModelConfig(model_id)

    pooling_config = model_config._init_pooler_config()
    assert pooling_config is not None

    assert pooling_config.normalize
    assert pooling_config.pooling_type == PoolingType.MEAN.name


@pytest.mark.skipif(current_platform.is_rocm(),
                    reason="Xformers backend is not supported on ROCm.")
def test_get_pooling_config_from_args():
    model_id = "sentence-transformers/all-MiniLM-L12-v2"
    model_config = ModelConfig(model_id)

    override_pooler_config = PoolerConfig(pooling_type='CLS', normalize=True)
    model_config.override_pooler_config = override_pooler_config

    pooling_config = model_config._init_pooler_config()
    assert pooling_config is not None
    assert asdict(pooling_config) == asdict(override_pooler_config)


@pytest.mark.parametrize(
    ("model_id", "default_pooling_type", "pooling_type"),
    [
        ("tomaarsen/Qwen3-Reranker-0.6B-seq-cls", "LAST", "LAST"),  # LLM
        ("intfloat/e5-small", "CLS", "MEAN"),  # BertModel
        ("Qwen/Qwen2.5-Math-RM-72B", "ALL", "ALL"),  # reward
        ("Qwen/Qwen2.5-Math-PRM-7B", "STEP", "STEP")  # step reward
    ])
def test_default_pooling_type(model_id, default_pooling_type, pooling_type):
    model_config = ModelConfig(model_id)
    assert model_config._model_info.default_pooling_type == default_pooling_type
    assert model_config.pooler_config.pooling_type == pooling_type


@pytest.mark.skipif(current_platform.is_rocm(),
                    reason="Xformers backend is not supported on ROCm.")
def test_get_bert_tokenization_sentence_transformer_config():
    model_id = "BAAI/bge-base-en-v1.5"
    bge_model_config = ModelConfig(model_id)

    bert_bge_model_config = bge_model_config._get_encoder_config()

    assert bert_bge_model_config["max_seq_length"] == 512
    assert bert_bge_model_config["do_lower_case"]


def test_rope_customization():
    TEST_ROPE_SCALING = {"rope_type": "dynamic", "factor": 2.0}
    TEST_ROPE_THETA = 16_000_000.0
    LONGCHAT_ROPE_SCALING = {"rope_type": "linear", "factor": 8.0}

    llama_model_config = ModelConfig("meta-llama/Meta-Llama-3-8B-Instruct")
    assert getattr(llama_model_config.hf_config, "rope_scaling", None) is None
    assert getattr(llama_model_config.hf_config, "rope_theta", None) == 500_000
    assert llama_model_config.max_model_len == 8192

    llama_model_config = ModelConfig(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        hf_overrides={
            "rope_scaling": TEST_ROPE_SCALING,
            "rope_theta": TEST_ROPE_THETA,
        },
    )
    assert getattr(llama_model_config.hf_config, "rope_scaling",
                   None) == TEST_ROPE_SCALING
    assert getattr(llama_model_config.hf_config, "rope_theta",
                   None) == TEST_ROPE_THETA
    assert llama_model_config.max_model_len == 16384

    longchat_model_config = ModelConfig("lmsys/longchat-13b-16k")
    # Check if LONGCHAT_ROPE_SCALING entries are in longchat_model_config
    assert all(
        longchat_model_config.hf_config.rope_scaling.get(key) == value
        for key, value in LONGCHAT_ROPE_SCALING.items())
    assert longchat_model_config.max_model_len == 16384

    longchat_model_config = ModelConfig(
        "lmsys/longchat-13b-16k",
        hf_overrides={
            "rope_scaling": TEST_ROPE_SCALING,
        },
    )
    assert getattr(longchat_model_config.hf_config, "rope_scaling",
                   None) == TEST_ROPE_SCALING
    assert longchat_model_config.max_model_len == 4096


@pytest.mark.skipif(current_platform.is_rocm(),
                    reason="Encoder Decoder models not supported on ROCm.")
@pytest.mark.parametrize(("model_id", "is_encoder_decoder"), [
    ("facebook/opt-125m", False),
    ("facebook/bart-base", True),
    ("meta-llama/Llama-3.2-1B-Instruct", False),
    ("meta-llama/Llama-3.2-11B-Vision", True),
])
def test_is_encoder_decoder(model_id, is_encoder_decoder):
    config = ModelConfig(model_id)

    assert config.is_encoder_decoder == is_encoder_decoder


@pytest.mark.parametrize(("model_id", "uses_mrope"), [
    ("facebook/opt-125m", False),
    ("Qwen/Qwen2-VL-2B-Instruct", True),
])
def test_uses_mrope(model_id, uses_mrope):
    config = ModelConfig(model_id)

    assert config.uses_mrope == uses_mrope


def test_generation_config_loading():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"

    # When set generation_config to "vllm", the default generation config
    # will not be loaded.
    model_config = ModelConfig(model_id, generation_config="vllm")
    assert model_config.get_diff_sampling_param() == {}

    # When set generation_config to "auto", the default generation config
    # should be loaded.
    model_config = ModelConfig(model_id, generation_config="auto")

    correct_generation_config = {
        "repetition_penalty": 1.1,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
    }

    assert model_config.get_diff_sampling_param() == correct_generation_config

    # The generation config could be overridden by the user.
    override_generation_config = {"temperature": 0.5, "top_k": 5}

    model_config = ModelConfig(
        model_id,
        generation_config="auto",
        override_generation_config=override_generation_config)

    override_result = correct_generation_config.copy()
    override_result.update(override_generation_config)

    assert model_config.get_diff_sampling_param() == override_result

    # When generation_config is set to "vllm" and override_generation_config
    # is set, the override_generation_config should be used directly.
    model_config = ModelConfig(
        model_id,
        generation_config="vllm",
        override_generation_config=override_generation_config)

    assert model_config.get_diff_sampling_param() == override_generation_config


@pytest.mark.parametrize("pt_load_map_location", [
    "cuda",
    {
        "": "cuda"
    },
])
def test_load_config_pt_load_map_location(pt_load_map_location):
    load_config = LoadConfig(pt_load_map_location=pt_load_map_location)
    config = VllmConfig(load_config=load_config)

    assert config.load_config.pt_load_map_location == pt_load_map_location


@pytest.mark.parametrize(
    ("model_id", "max_model_len", "expected_max_len", "should_raise"), [
        ("BAAI/bge-reranker-base", None, 512, False),
        ("BAAI/bge-reranker-base", 256, 256, False),
        ("BAAI/bge-reranker-base", 513, 512, True),
        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", None, 131072, False),
        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 131073, 131072, True),
    ])
def test_get_and_verify_max_len(model_id, max_model_len, expected_max_len,
                                should_raise):
    """Test get_and_verify_max_len with different configurations."""
    model_config = ModelConfig(model_id)

    if should_raise:
        with pytest.raises(ValueError):
            model_config.get_and_verify_max_len(max_model_len)
    else:
        actual_max_len = model_config.get_and_verify_max_len(max_model_len)
        assert actual_max_len == expected_max_len
