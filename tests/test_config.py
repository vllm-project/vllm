# SPDX-License-Identifier: Apache-2.0

from dataclasses import MISSING, Field, asdict, dataclass, field
from typing import Literal, Union

import pytest

from vllm.config import (LoadConfig, ModelConfig, PoolerConfig, VllmConfig,
                         config, get_field)
from vllm.model_executor.layers.pooler import PoolingType
from vllm.platforms import current_platform


class TestConfig1:
    pass


@dataclass
class TestConfig2:
    a: int
    """docstring"""


@dataclass
class TestConfig3:
    a: int = 1


@dataclass
class TestConfig4:
    a: Union[Literal[1], Literal[2]] = 1
    """docstring"""


@pytest.mark.parametrize(("test_config", "expected_error"), [
    (TestConfig1, "must be a dataclass"),
    (TestConfig2, "must have a default"),
    (TestConfig3, "must have a docstring"),
    (TestConfig4, "must use a single Literal"),
])
def test_config(test_config, expected_error):
    with pytest.raises(Exception, match=expected_error):
        config(test_config)


def test_get_field():

    @dataclass
    class TestConfig:
        a: int
        b: dict = field(default_factory=dict)
        c: str = "default"

    with pytest.raises(ValueError):
        get_field(TestConfig, "a")

    b = get_field(TestConfig, "b")
    assert isinstance(b, Field)
    assert b.default is MISSING
    assert b.default_factory is dict

    c = get_field(TestConfig, "c")
    assert isinstance(c, Field)
    assert c.default == "default"
    assert c.default_factory is MISSING


@pytest.mark.parametrize(
    ("model_id", "expected_runner_type", "expected_task"),
    [
        ("distilbert/distilgpt2", "generate", "generate"),
        ("intfloat/multilingual-e5-small", "pooling", "embed"),
        ("jason9693/Qwen2.5-1.5B-apeach", "pooling", "classify"),
        ("cross-encoder/ms-marco-MiniLM-L-6-v2", "pooling", "score"),
        ("Qwen/Qwen2.5-Math-RM-72B", "pooling", "reward"),
        ("openai/whisper-small", "transcription", "transcription"),
    ],
)
def test_auto_task(model_id, expected_runner_type, expected_task):
    config = ModelConfig(
        model_id,
        task="auto",
        tokenizer=model_id,
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype="float16",
    )

    assert config.runner_type == expected_runner_type
    assert config.task == expected_task


@pytest.mark.parametrize(("model_id", "bad_task"), [
    ("Qwen/Qwen2.5-Math-RM-72B", "generate"),
])
def test_incorrect_task(model_id, bad_task):
    with pytest.raises(ValueError, match=r"does not support the .* task"):
        ModelConfig(
            model_id,
            task=bad_task,
            tokenizer=model_id,
            tokenizer_mode="auto",
            trust_remote_code=False,
            seed=0,
            dtype="float16",
        )


MODEL_IDS_EXPECTED = [
    ("Qwen/Qwen1.5-7B", 32768),
    ("mistralai/Mistral-7B-v0.1", 4096),
    ("mistralai/Mistral-7B-Instruct-v0.2", 32768),
]


@pytest.mark.parametrize("model_id_expected", MODEL_IDS_EXPECTED)
def test_disable_sliding_window(model_id_expected):
    model_id, expected = model_id_expected
    model_config = ModelConfig(
        model_id,
        task="auto",
        tokenizer=model_id,
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype="float16",
        revision=None,
        disable_sliding_window=True,
    )
    assert model_config.max_model_len == expected


def test_get_sliding_window():
    TEST_SLIDING_WINDOW = 4096
    # Test that the sliding window is correctly computed.
    # For Qwen1.5/Qwen2, get_sliding_window() should be None
    # when use_sliding_window is False.
    qwen2_model_config = ModelConfig(
        "Qwen/Qwen1.5-7B",
        task="auto",
        tokenizer="Qwen/Qwen1.5-7B",
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype="float16",
        revision=None,
    )

    qwen2_model_config.hf_config.use_sliding_window = False
    qwen2_model_config.hf_config.sliding_window = TEST_SLIDING_WINDOW
    assert qwen2_model_config.get_sliding_window() is None

    qwen2_model_config.hf_config.use_sliding_window = True
    assert qwen2_model_config.get_sliding_window() == TEST_SLIDING_WINDOW

    mistral_model_config = ModelConfig(
        "mistralai/Mistral-7B-v0.1",
        task="auto",
        tokenizer="mistralai/Mistral-7B-v0.1",
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype="float16",
        revision=None,
    )
    mistral_model_config.hf_config.sliding_window = None
    assert mistral_model_config.get_sliding_window() is None

    mistral_model_config.hf_config.sliding_window = TEST_SLIDING_WINDOW
    assert mistral_model_config.get_sliding_window() == TEST_SLIDING_WINDOW


@pytest.mark.skipif(current_platform.is_rocm(),
                    reason="Xformers backend is not supported on ROCm.")
def test_get_pooling_config():
    model_id = "sentence-transformers/all-MiniLM-L12-v2"
    model_config = ModelConfig(
        model_id,
        task="auto",
        tokenizer=model_id,
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype="float16",
        revision=None,
    )

    pooling_config = model_config._init_pooler_config()
    assert pooling_config is not None

    assert pooling_config.normalize
    assert pooling_config.pooling_type == PoolingType.MEAN.name


@pytest.mark.skipif(current_platform.is_rocm(),
                    reason="Xformers backend is not supported on ROCm.")
def test_get_pooling_config_from_args():
    model_id = "sentence-transformers/all-MiniLM-L12-v2"
    model_config = ModelConfig(model_id,
                               task="auto",
                               tokenizer=model_id,
                               tokenizer_mode="auto",
                               trust_remote_code=False,
                               seed=0,
                               dtype="float16",
                               revision=None)

    override_pooler_config = PoolerConfig(pooling_type='CLS', normalize=True)
    model_config.override_pooler_config = override_pooler_config

    pooling_config = model_config._init_pooler_config()
    assert pooling_config is not None
    assert asdict(pooling_config) == asdict(override_pooler_config)


@pytest.mark.skipif(current_platform.is_rocm(),
                    reason="Xformers backend is not supported on ROCm.")
def test_get_bert_tokenization_sentence_transformer_config():
    bge_model_config = ModelConfig(
        model="BAAI/bge-base-en-v1.5",
        task="auto",
        tokenizer="BAAI/bge-base-en-v1.5",
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype="float16",
        revision=None,
    )

    bert_bge_model_config = bge_model_config._get_encoder_config()

    assert bert_bge_model_config["max_seq_length"] == 512
    assert bert_bge_model_config["do_lower_case"]


def test_rope_customization():
    TEST_ROPE_SCALING = {"rope_type": "dynamic", "factor": 2.0}
    TEST_ROPE_THETA = 16_000_000.0
    LONGCHAT_ROPE_SCALING = {"rope_type": "linear", "factor": 8.0}

    llama_model_config = ModelConfig(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        task="auto",
        tokenizer="meta-llama/Meta-Llama-3-8B-Instruct",
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="float16",
        seed=0,
    )
    assert getattr(llama_model_config.hf_config, "rope_scaling", None) is None
    assert getattr(llama_model_config.hf_config, "rope_theta", None) == 500_000
    assert llama_model_config.max_model_len == 8192

    llama_model_config = ModelConfig(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        task="auto",
        tokenizer="meta-llama/Meta-Llama-3-8B-Instruct",
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="float16",
        seed=0,
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

    longchat_model_config = ModelConfig(
        "lmsys/longchat-13b-16k",
        task="auto",
        tokenizer="lmsys/longchat-13b-16k",
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="float16",
        seed=0,
    )
    # Check if LONGCHAT_ROPE_SCALING entries are in longchat_model_config
    assert all(
        longchat_model_config.hf_config.rope_scaling.get(key) == value
        for key, value in LONGCHAT_ROPE_SCALING.items())
    assert longchat_model_config.max_model_len == 16384

    longchat_model_config = ModelConfig(
        "lmsys/longchat-13b-16k",
        task="auto",
        tokenizer="lmsys/longchat-13b-16k",
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="float16",
        seed=0,
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
    config = ModelConfig(
        model_id,
        task="auto",
        tokenizer=model_id,
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="float16",
        seed=0,
    )

    assert config.is_encoder_decoder == is_encoder_decoder


@pytest.mark.parametrize(("model_id", "uses_mrope"), [
    ("facebook/opt-125m", False),
    ("Qwen/Qwen2-VL-2B-Instruct", True),
])
def test_uses_mrope(model_id, uses_mrope):
    config = ModelConfig(
        model_id,
        task="auto",
        tokenizer=model_id,
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="float16",
        seed=0,
    )

    assert config.uses_mrope == uses_mrope


def test_generation_config_loading():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"

    # When set generation_config to "vllm", the default generation config
    # will not be loaded.
    model_config = ModelConfig(model_id,
                               task="auto",
                               tokenizer=model_id,
                               tokenizer_mode="auto",
                               trust_remote_code=False,
                               seed=0,
                               dtype="float16",
                               generation_config="vllm")
    assert model_config.get_diff_sampling_param() == {}

    # When set generation_config to "auto", the default generation config
    # should be loaded.
    model_config = ModelConfig(model_id,
                               task="auto",
                               tokenizer=model_id,
                               tokenizer_mode="auto",
                               trust_remote_code=False,
                               seed=0,
                               dtype="float16",
                               generation_config="auto")

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
        task="auto",
        tokenizer=model_id,
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype="float16",
        generation_config="auto",
        override_generation_config=override_generation_config)

    override_result = correct_generation_config.copy()
    override_result.update(override_generation_config)

    assert model_config.get_diff_sampling_param() == override_result

    # When generation_config is set to "vllm" and override_generation_config
    # is set, the override_generation_config should be used directly.
    model_config = ModelConfig(
        model_id,
        task="auto",
        tokenizer=model_id,
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype="float16",
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
