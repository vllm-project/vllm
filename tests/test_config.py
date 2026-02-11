# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import os
from dataclasses import MISSING, Field, asdict, dataclass, field
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from vllm.compilation.backends import VllmBackend
from vllm.config import (
    CompilationConfig,
    ModelConfig,
    ParallelConfig,
    PoolerConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
    update_config,
)
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.config.load import LoadConfig
from vllm.config.utils import get_field
from vllm.config.vllm import (
    OPTIMIZATION_LEVEL_TO_CONFIG,
    OptimizationLevel,
)
from vllm.platforms import current_platform


def test_compile_config_repr_succeeds():
    # setup: VllmBackend mutates the config object
    config = VllmConfig()
    backend = VllmBackend(config)
    backend.configure_post_pass()

    # test that repr(config) succeeds
    val = repr(config)
    assert "VllmConfig" in val
    assert "inductor_passes" in val


def test_async_scheduling_with_pipeline_parallelism_is_allowed():
    cfg = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_model_len=8192,
            is_encoder_decoder=False,
            async_scheduling=True,
        ),
        parallel_config=ParallelConfig(
            pipeline_parallel_size=2,
            distributed_executor_backend="mp",
            nnodes=2,
        ),
    )
    assert cfg.scheduler_config.async_scheduling is True


@dataclass
class _TestConfigFields:
    a: int
    b: dict = field(default_factory=dict)
    c: str = "default"


def test_get_field():
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
    a: _TestConfigFields = field(default_factory=lambda: _TestConfigFields(a=0))


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


@pytest.mark.skipif(
    current_platform.is_rocm(), reason="Xformers backend is not supported on ROCm."
)
def test_get_pooling_config():
    model_id = "sentence-transformers/all-MiniLM-L12-v2"
    model_config = ModelConfig(model_id)

    assert model_config.pooler_config is not None
    assert model_config.pooler_config.use_activation
    assert model_config.pooler_config.seq_pooling_type == "MEAN"
    assert model_config.pooler_config.tok_pooling_type == "ALL"


@pytest.mark.skipif(
    current_platform.is_rocm(), reason="Xformers backend is not supported on ROCm."
)
def test_get_pooling_config_from_args():
    model_id = "sentence-transformers/all-MiniLM-L12-v2"
    pooler_config = PoolerConfig(seq_pooling_type="CLS", use_activation=False)
    model_config = ModelConfig(model_id, pooler_config=pooler_config)

    assert asdict(model_config.pooler_config) == asdict(pooler_config)


@pytest.mark.parametrize(
    ("model_id", "default_pooling_type", "pooling_type"),
    [
        ("tomaarsen/Qwen3-Reranker-0.6B-seq-cls", "LAST", "LAST"),  # LLM
        ("intfloat/e5-small", "CLS", "MEAN"),  # BertModel
    ],
)
def test_default_seq_pooling_type(model_id, default_pooling_type, pooling_type):
    model_config = ModelConfig(model_id)
    assert model_config._model_info.default_seq_pooling_type == default_pooling_type
    assert model_config.pooler_config.seq_pooling_type == pooling_type


@pytest.mark.parametrize(
    ("model_id", "default_pooling_type", "pooling_type"),
    [
        ("Qwen/Qwen2.5-Math-RM-72B", "ALL", "ALL"),  # reward
        ("Qwen/Qwen2.5-Math-PRM-7B", "STEP", "STEP"),  # step reward
    ],
)
def test_default_tok_pooling_type(model_id, default_pooling_type, pooling_type):
    model_config = ModelConfig(model_id)
    assert model_config._model_info.default_tok_pooling_type == default_pooling_type
    assert model_config.pooler_config.tok_pooling_type == pooling_type


@pytest.mark.parametrize(
    ("model_id", "expected_is_moe_model"),
    [
        ("RedHatAI/Qwen3-8B-speculator.eagle3", False),
        ("RedHatAI/Llama-3.1-8B-Instruct-NVFP4", False),
        ("RedHatAI/Llama-3.2-1B-FP8", False),
        ("RedHatAI/Mistral-Small-24B-Instruct-2501-quantized.w8a8", False),
        ("RedHatAI/gpt-oss-20b", True),
        ("RedHatAI/DeepSeek-V2.5-1210-FP8", True),
        ("RedHatAI/Llama-4-Scout-17B-16E-Instruct", True),
        ("RedHatAI/Mixtral-8x7B-Instruct-v0.1", True),
    ],
)
def test_moe_model_detection(model_id, expected_is_moe_model):
    model_config = ModelConfig(model_id)
    # Just check that is_moe field exists and is a boolean
    assert model_config.is_moe == expected_is_moe_model


@pytest.mark.parametrize(
    ("model_id", "quantized"),
    [
        ("RedHatAI/Qwen3-8B-speculator.eagle3", False),
        ("RedHatAI/Llama-3.1-8B-Instruct-NVFP4", True),
        ("RedHatAI/Llama-3.2-1B-FP8", True),
        ("RedHatAI/Mistral-Small-24B-Instruct-2501-quantized.w8a8", True),
        ("RedHatAI/gpt-oss-20b", True),
        ("RedHatAI/DeepSeek-V2.5-1210-FP8", True),
        ("RedHatAI/Mixtral-8x7B-Instruct-v0.1", False),
    ],
)
def test_is_quantized(model_id, quantized):
    model_config = ModelConfig(model_id)
    # Just check that quantized field exists and is a boolean
    assert model_config.is_quantized == quantized


@pytest.mark.skipif(
    current_platform.is_rocm(), reason="Xformers backend is not supported on ROCm."
)
def test_get_bert_tokenization_sentence_transformer_config():
    model_id = "BAAI/bge-base-en-v1.5"
    bge_model_config = ModelConfig(model_id)

    bert_bge_model_config = bge_model_config._get_encoder_config()

    assert bert_bge_model_config["max_seq_length"] == 512
    assert bert_bge_model_config["do_lower_case"]


def test_rope_customization():
    TEST_ROPE_PARAMETERS = {
        "rope_theta": 16_000_000.0,
        "rope_type": "dynamic",
        "factor": 2.0,
    }
    LLAMA_ROPE_PARAMETERS = {"rope_theta": 500000.0, "rope_type": "default"}
    LONGCHAT_ROPE_PARAMETERS = {"rope_type": "linear", "factor": 8.0}

    llama_model_config = ModelConfig("meta-llama/Meta-Llama-3-8B-Instruct")
    assert (
        getattr(llama_model_config.hf_config, "rope_parameters", None)
        == LLAMA_ROPE_PARAMETERS
    )
    assert llama_model_config.max_model_len == 8192

    llama_model_config = ModelConfig(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        hf_overrides={"rope_parameters": TEST_ROPE_PARAMETERS},
    )
    assert (
        getattr(llama_model_config.hf_config, "rope_parameters", None)
        == TEST_ROPE_PARAMETERS
    )
    assert llama_model_config.max_model_len == 16384

    longchat_model_config = ModelConfig("lmsys/longchat-13b-16k")
    # Check if LONGCHAT_ROPE_PARAMETERS entries are in longchat_model_config
    assert all(
        longchat_model_config.hf_config.rope_parameters.get(key) == value
        for key, value in LONGCHAT_ROPE_PARAMETERS.items()
    )
    assert longchat_model_config.max_model_len == 16384

    longchat_model_config = ModelConfig(
        "lmsys/longchat-13b-16k",
        hf_overrides={
            "rope_parameters": TEST_ROPE_PARAMETERS,
        },
    )
    assert (
        getattr(longchat_model_config.hf_config, "rope_parameters", None)
        == TEST_ROPE_PARAMETERS
    )
    assert longchat_model_config.max_model_len == 4096


def test_nested_hf_overrides():
    """Test that nested hf_overrides work correctly."""
    # Test with a model that has text_config
    model_config = ModelConfig(
        "Qwen/Qwen2-VL-2B-Instruct",
        hf_overrides={
            "text_config": {
                "hidden_size": 1024,
            },
        },
    )
    assert model_config.hf_config.text_config.hidden_size == 1024

    # Test with deeply nested overrides
    model_config = ModelConfig(
        "Qwen/Qwen2-VL-2B-Instruct",
        hf_overrides={
            "text_config": {
                "hidden_size": 2048,
                "num_attention_heads": 16,
            },
            "vision_config": {
                "hidden_size": 512,
            },
        },
    )
    assert model_config.hf_config.text_config.hidden_size == 2048
    assert model_config.hf_config.text_config.num_attention_heads == 16
    assert model_config.hf_config.vision_config.hidden_size == 512


@pytest.mark.skipif(
    current_platform.is_rocm(), reason="Encoder Decoder models not supported on ROCm."
)
@pytest.mark.parametrize(
    ("model_id", "is_encoder_decoder"),
    [
        ("facebook/opt-125m", False),
        ("openai/whisper-tiny", True),
        ("meta-llama/Llama-3.2-1B-Instruct", False),
    ],
)
def test_is_encoder_decoder(model_id, is_encoder_decoder):
    config = ModelConfig(model_id)

    assert config.is_encoder_decoder == is_encoder_decoder


@pytest.mark.parametrize(
    ("model_id", "uses_mrope"),
    [
        ("facebook/opt-125m", False),
        ("Qwen/Qwen2-VL-2B-Instruct", True),
    ],
)
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
        override_generation_config=override_generation_config,
    )

    override_result = correct_generation_config.copy()
    override_result.update(override_generation_config)

    assert model_config.get_diff_sampling_param() == override_result

    # When generation_config is set to "vllm" and override_generation_config
    # is set, the override_generation_config should be used directly.
    model_config = ModelConfig(
        model_id,
        generation_config="vllm",
        override_generation_config=override_generation_config,
    )

    assert model_config.get_diff_sampling_param() == override_generation_config


@pytest.mark.parametrize(
    "pt_load_map_location",
    [
        "cuda",
        {"": "cuda"},
    ],
)
def test_load_config_pt_load_map_location(pt_load_map_location):
    load_config = LoadConfig(pt_load_map_location=pt_load_map_location)
    config = VllmConfig(load_config=load_config)

    assert config.load_config.pt_load_map_location == pt_load_map_location


@pytest.mark.parametrize(
    ("model_id", "max_model_len", "expected_max_len", "should_raise"),
    [
        ("BAAI/bge-reranker-base", None, 512, False),
        ("BAAI/bge-reranker-base", 256, 256, False),
        ("BAAI/bge-reranker-base", 513, 512, True),
        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", None, 131072, False),
        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 131073, 131072, True),
    ],
)
def test_get_and_verify_max_len(
    model_id, max_model_len, expected_max_len, should_raise
):
    """Test get_and_verify_max_len with different configurations."""
    model_config = ModelConfig(model_id)

    if should_raise:
        with pytest.raises(ValueError):
            model_config.get_and_verify_max_len(max_model_len)
    else:
        actual_max_len = model_config.get_and_verify_max_len(max_model_len)
        assert actual_max_len == expected_max_len


class MockConfig:
    """Simple mock object for testing maybe_pull_model_tokenizer_for_runai"""

    def __init__(self, model: str, tokenizer: str):
        self.model = model
        self.tokenizer = tokenizer
        self.model_weights = None


@pytest.mark.parametrize(
    "s3_url",
    [
        "s3://example-bucket-1/model/",
        "s3://example-bucket-2/model/",
    ],
)
@patch("vllm.transformers_utils.runai_utils.ObjectStorageModel.pull_files")
def test_s3_url_model_tokenizer_paths(mock_pull_files, s3_url):
    """Test that S3 URLs create deterministic local directories for model and
    tokenizer."""
    # Mock pull_files to avoid actually downloading files during tests
    mock_pull_files.return_value = None

    # Create first mock and run the method
    config1 = MockConfig(model=s3_url, tokenizer=s3_url)
    ModelConfig.maybe_pull_model_tokenizer_for_runai(config1, s3_url, s3_url)

    # Check that model and tokenizer point to existing directories
    assert os.path.exists(config1.model), (
        f"Model directory does not exist: {config1.model}"
    )
    assert os.path.isdir(config1.model), (
        f"Model path is not a directory: {config1.model}"
    )
    assert os.path.exists(config1.tokenizer), (
        f"Tokenizer directory does not exist: {config1.tokenizer}"
    )
    assert os.path.isdir(config1.tokenizer), (
        f"Tokenizer path is not a directory: {config1.tokenizer}"
    )

    # Verify that the paths are different from the original S3 URL
    assert config1.model != s3_url, "Model path should be converted to local directory"
    assert config1.tokenizer != s3_url, (
        "Tokenizer path should be converted to local directory"
    )

    # Store the original paths
    created_model_dir = config1.model
    create_tokenizer_dir = config1.tokenizer

    # Create a new mock and run the method with the same S3 URL
    config2 = MockConfig(model=s3_url, tokenizer=s3_url)
    ModelConfig.maybe_pull_model_tokenizer_for_runai(config2, s3_url, s3_url)

    # Check that the new directories exist
    assert os.path.exists(config2.model), (
        f"Model directory does not exist: {config2.model}"
    )
    assert os.path.isdir(config2.model), (
        f"Model path is not a directory: {config2.model}"
    )
    assert os.path.exists(config2.tokenizer), (
        f"Tokenizer directory does not exist: {config2.tokenizer}"
    )
    assert os.path.isdir(config2.tokenizer), (
        f"Tokenizer path is not a directory: {config2.tokenizer}"
    )

    # Verify that the paths are deterministic (same as before)
    assert config2.model == created_model_dir, (
        f"Model paths are not deterministic. "
        f"Original: {created_model_dir}, New: {config2.model}"
    )
    assert config2.tokenizer == create_tokenizer_dir, (
        f"Tokenizer paths are not deterministic. "
        f"Original: {create_tokenizer_dir}, New: {config2.tokenizer}"
    )


@patch("vllm.transformers_utils.runai_utils.ObjectStorageModel.pull_files")
def test_s3_url_different_models_create_different_directories(mock_pull_files):
    """Test that different S3 URLs create different local directories."""
    # Mock pull_files to avoid actually downloading files during tests
    mock_pull_files.return_value = None

    s3_url1 = "s3://example-bucket-1/model/"
    s3_url2 = "s3://example-bucket-2/model/"

    # Create mocks with different S3 URLs and run the method
    config1 = MockConfig(model=s3_url1, tokenizer=s3_url1)
    ModelConfig.maybe_pull_model_tokenizer_for_runai(config1, s3_url1, s3_url1)

    config2 = MockConfig(model=s3_url2, tokenizer=s3_url2)
    ModelConfig.maybe_pull_model_tokenizer_for_runai(config2, s3_url2, s3_url2)

    # Verify that different URLs produce different directories
    assert config1.model != config2.model, (
        f"Different S3 URLs should create different model directories. "
        f"URL1 model: {config1.model}, URL2 model: {config2.model}"
    )
    assert config1.tokenizer != config2.tokenizer, (
        f"Different S3 URLs should create different tokenizer directories. "
        f"URL1 tokenizer: {config1.tokenizer}, "
        f"URL2 tokenizer: {config2.tokenizer}"
    )

    # Verify that both sets of directories exist
    assert os.path.exists(config1.model) and os.path.isdir(config1.model)
    assert os.path.exists(config1.tokenizer) and os.path.isdir(config1.tokenizer)
    assert os.path.exists(config2.model) and os.path.isdir(config2.model)
    assert os.path.exists(config2.tokenizer) and os.path.isdir(config2.tokenizer)


@pytest.mark.parametrize(
    ("model_id", "expected_attn_type", "expected_result", "reason"),
    [
        # pooling models
        (
            "jason9693/Qwen2.5-1.5B-apeach",
            "decoder",
            True,
            "Pooling models with causal attn and LAST/ALL pooling support chunked prefill.",  # noqa: E501
        ),
        (
            "Qwen/Qwen3-Embedding-0.6B",
            "decoder",
            True,
            "Pooling models with causal attn and LAST/ALL pooling support chunked prefill.",  # noqa: E501
        ),
        (
            "Qwen/Qwen2.5-Math-PRM-7B",
            "decoder",
            False,
            "Pooling models with causal attn and LAST/STEP pooling do not support chunked prefill.",  # noqa: E501
        ),
        (
            "internlm/internlm2-1_8b-reward",
            "decoder",
            True,
            "Pooling models with causal attn and LAST/ALL pooling support chunked prefill.",  # noqa: E501
        ),
        (
            "BAAI/bge-base-en",
            "encoder_only",
            False,
            "Pooling models with bidirectional attn do not support chunked prefill.",  # noqa: E501
        ),
        (
            "boltuix/NeuroBERT-NER",
            "encoder_only",
            False,
            "Pooling models with bidirectional attn do not support chunked prefill.",  # noqa: E501
        ),
        (
            "papluca/xlm-roberta-base-language-detection",
            "encoder_only",
            False,
            "Pooling models with bidirectional attn do not support chunked prefill.",  # noqa: E501
        ),
        (
            "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
            "encoder_only",
            False,
            "Pooling models with bidirectional attn do not support chunked prefill.",  # noqa: E501
        ),
        (
            "intfloat/e5-small",
            "encoder_only",
            False,
            "Pooling models with bidirectional attn do not support chunked prefill.",  # noqa: E501
        ),
        # multimodal models
        (
            "openai/clip-vit-base-patch32",
            "decoder",
            True,
            "Pooling models with causal attn and LAST/ALL pooling support chunked prefill.",  # noqa: E501
        ),
        (
            "google/siglip-base-patch16-224",
            "encoder_only",
            False,
            "Pooling models with bidirectional attn do not support chunked prefill.",  # noqa: E501
        ),
        # generate models
        (
            "Qwen/Qwen3-0.6B",
            "decoder",
            True,
            "Generative models support chunked prefill.",  # noqa: E501
        ),
        (
            "Qwen/Qwen3-Next-80B-A3B-Instruct",
            "hybrid",
            True,
            "Generative models support chunked prefill.",  # noqa: E501
        ),
        (
            "ibm-granite/granite-4.0-h-small",
            "hybrid",
            True,
            "Generative models support chunked prefill.",  # noqa: E501
        ),
        (
            "state-spaces/mamba-130m-hf",
            "attention_free",
            True,
            "Generative models support chunked prefill.",  # noqa: E501
        ),
        # encoder_decoder models
        (
            "openai/whisper-small",
            "encoder_decoder",
            False,
            "Encoder decoder models do not support chunked prefill.",  # noqa: E501
        ),
    ],
)
def test_is_chunked_prefill_supported(
    model_id: str,
    expected_attn_type: str,
    expected_result: bool,
    reason: str,
    caplog_vllm,
):
    model_config = ModelConfig(model_id, trust_remote_code=True)
    assert model_config.attn_type == expected_attn_type
    with caplog_vllm.at_level(level=logging.DEBUG, logger="vllm"):
        assert model_config.is_chunked_prefill_supported == expected_result
    assert reason in caplog_vllm.text


@pytest.mark.parametrize(
    ("model_id", "expected_attn_type", "expected_result", "reason"),
    [
        # pooling models
        (
            "jason9693/Qwen2.5-1.5B-apeach",
            "decoder",
            True,
            "Pooling models with causal attn and LAST/ALL pooling support prefix caching.",  # noqa: E501
        ),
        (
            "Qwen/Qwen3-Embedding-0.6B",
            "decoder",
            True,
            "Pooling models with causal attn and LAST/ALL pooling support prefix caching.",  # noqa: E501
        ),
        (
            "Qwen/Qwen2.5-Math-PRM-7B",
            "decoder",
            False,
            "Pooling models with causal attn and LAST/STEP pooling do not support prefix caching.",  # noqa: E501
        ),
        (
            "internlm/internlm2-1_8b-reward",
            "decoder",
            True,
            "Pooling models with causal attn and LAST/ALL pooling support prefix caching.",  # noqa: E501
        ),
        (
            "BAAI/bge-base-en",
            "encoder_only",
            False,
            "Pooling models with bidirectional attn do not support prefix caching.",  # noqa: E501
        ),
        (
            "boltuix/NeuroBERT-NER",
            "encoder_only",
            False,
            "Pooling models with bidirectional attn do not support prefix caching.",  # noqa: E501
        ),
        (
            "papluca/xlm-roberta-base-language-detection",
            "encoder_only",
            False,
            "Pooling models with bidirectional attn do not support prefix caching.",  # noqa: E501
        ),
        (
            "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
            "encoder_only",
            False,
            "Pooling models with bidirectional attn do not support prefix caching.",  # noqa: E501
        ),
        (
            "intfloat/e5-small",
            "encoder_only",
            False,
            "Pooling models with bidirectional attn do not support prefix caching.",  # noqa: E501
        ),
        # multimodal models
        (
            "openai/clip-vit-base-patch32",
            "decoder",
            True,
            "Pooling models with causal attn and LAST/ALL pooling support prefix caching.",  # noqa: E501
        ),
        (
            "google/siglip-base-patch16-224",
            "encoder_only",
            False,
            "Pooling models with bidirectional attn do not support prefix caching.",  # noqa: E501
        ),
        # generate models
        (
            "Qwen/Qwen3-0.6B",
            "decoder",
            True,
            "Generative models support prefix caching.",  # noqa: E501
        ),
        (
            "Qwen/Qwen3-Next-80B-A3B-Instruct",
            "hybrid",
            False,
            "Hybrid models do not support prefix caching since the feature is still experimental.",  # noqa: E501
        ),
        (
            "ibm-granite/granite-4.0-h-small",
            "hybrid",
            False,
            "Hybrid models do not support prefix caching since the feature is still experimental.",  # noqa: E501
        ),
        (
            "state-spaces/mamba-130m-hf",
            "attention_free",
            False,
            "Attention free models do not support prefix caching since the feature is still experimental.",  # noqa: E501
        ),
        # encoder_decoder models
        (
            "openai/whisper-small",
            "encoder_decoder",
            False,
            "Encoder decoder models do not support prefix caching.",  # noqa: E501
        ),
    ],
)
def test_is_prefix_caching_supported(
    model_id: str,
    expected_attn_type: str,
    expected_result: bool,
    reason: str,
    caplog_vllm,
):
    model_config = ModelConfig(model_id, trust_remote_code=True)
    assert model_config.attn_type == expected_attn_type
    with caplog_vllm.at_level(level=logging.DEBUG, logger="vllm"):
        assert model_config.is_prefix_caching_supported == expected_result
    assert reason in caplog_vllm.text


@pytest.mark.parametrize(
    ("backend", "custom_ops", "expected"),
    [
        ("eager", [], True),
        ("eager", ["+fused_layernorm"], True),
        ("eager", ["all", "-fused_layernorm"], False),
        ("inductor", [], False),
        ("inductor", ["none", "+fused_layernorm"], True),
        ("inductor", ["none", "-fused_layernorm"], False),
    ],
)
def test_is_custom_op_enabled(backend: str, custom_ops: list[str], expected: bool):
    """Test that is_custom_op_enabled works correctly."""
    config = VllmConfig(
        compilation_config=CompilationConfig(backend=backend, custom_ops=custom_ops)
    )
    assert config.compilation_config.is_custom_op_enabled("fused_layernorm") is expected


def test_vllm_config_defaults_are_none():
    """Verify that optimization-level defaults are None when not set by user."""
    # Test all optimization levels to ensure defaults work correctly
    for opt_level in OptimizationLevel:
        config = object.__new__(VllmConfig)
        config.compilation_config = CompilationConfig()
        config.optimization_level = opt_level
        config.model_config = None

        # Use the global optimization level defaults
        default_config = OPTIMIZATION_LEVEL_TO_CONFIG[opt_level]

        # Verify that all pass_config values are None before defaults are applied
        for pass_k in default_config["compilation_config"]["pass_config"]:
            assert getattr(config.compilation_config.pass_config, pass_k) is None

        # Verify that other config values are None before defaults are applied
        for k in default_config["compilation_config"]:
            if k != "pass_config":
                assert getattr(config.compilation_config, k) is None


@pytest.mark.parametrize(
    ("model_id", "compiliation_config", "optimization_level"),
    [
        (
            None,
            CompilationConfig(backend="eager", custom_ops=["+quant_fp8"]),
            OptimizationLevel.O0,
        ),
        (None, CompilationConfig(), OptimizationLevel.O0),
        (None, CompilationConfig(), OptimizationLevel.O1),
        (None, CompilationConfig(), OptimizationLevel.O2),
        (None, CompilationConfig(), OptimizationLevel.O3),
        (
            "RedHatAI/Qwen3-8B-speculator.eagle3",
            CompilationConfig(backend="inductor", custom_ops=["+quant_fp8"]),
            OptimizationLevel.O2,
        ),
        (
            "RedHatAI/Qwen3-8B-speculator.eagle3",
            CompilationConfig(),
            OptimizationLevel.O0,
        ),
        (
            "RedHatAI/Qwen3-8B-speculator.eagle3",
            CompilationConfig(),
            OptimizationLevel.O1,
        ),
        (
            "RedHatAI/Qwen3-8B-speculator.eagle3",
            CompilationConfig(),
            OptimizationLevel.O2,
        ),
        (
            "RedHatAI/Qwen3-8B-speculator.eagle3",
            CompilationConfig(),
            OptimizationLevel.O3,
        ),
        ("RedHatAI/DeepSeek-V2.5-1210-FP8", CompilationConfig(), OptimizationLevel.O0),
        ("RedHatAI/DeepSeek-V2.5-1210-FP8", CompilationConfig(), OptimizationLevel.O1),
        ("RedHatAI/DeepSeek-V2.5-1210-FP8", CompilationConfig(), OptimizationLevel.O2),
        ("RedHatAI/DeepSeek-V2.5-1210-FP8", CompilationConfig(), OptimizationLevel.O3),
    ],
)
def test_vllm_config_defaults(model_id, compiliation_config, optimization_level):
    """Test that optimization-level defaults are correctly applied."""

    model_config = None
    if model_id is not None:
        model_config = ModelConfig(model_id)
        vllm_config = VllmConfig(
            model_config=model_config,
            compilation_config=compiliation_config,
            optimization_level=optimization_level,
        )
    else:
        vllm_config = VllmConfig(
            compilation_config=compiliation_config,
            optimization_level=optimization_level,
        )
    # Use the global optimization level defaults
    default_config = OPTIMIZATION_LEVEL_TO_CONFIG[optimization_level]

    # Verify pass_config defaults (nested under compilation_config)
    pass_config_dict = default_config["compilation_config"]["pass_config"]
    for pass_k, pass_v in pass_config_dict.items():
        actual = getattr(vllm_config.compilation_config.pass_config, pass_k)
        expected = pass_v(vllm_config) if callable(pass_v) else pass_v
        assert actual == expected, (
            f"pass_config.{pass_k}: expected {expected}, got {actual}"
        )

    # Verify other compilation_config defaults
    compilation_config_dict = default_config["compilation_config"]
    for k, v in compilation_config_dict.items():
        if k != "pass_config":
            actual = getattr(vllm_config.compilation_config, k)
            expected = v(vllm_config) if callable(v) else v
            assert actual == expected, (
                f"compilation_config.{k}: expected {expected}, got {actual}"
            )


def test_vllm_config_callable_defaults():
    """Test that callable defaults work in the config system.

    Verifies that lambdas in default configs can inspect VllmConfig properties
    (e.g., is_quantized, is_model_moe) to conditionally set optimization flags.
    """
    config_no_model = VllmConfig(optimization_level=OptimizationLevel.O2)

    # Callable that checks if model exists
    has_model = lambda cfg: cfg.model_config is not None
    assert has_model(config_no_model) is False

    # Test with quantized model
    quantized_model = ModelConfig("RedHatAI/Llama-3.2-1B-FP8")
    config_quantized = VllmConfig(
        model_config=quantized_model, optimization_level=OptimizationLevel.O2
    )
    enable_if_quantized = lambda cfg: (
        cfg.model_config is not None and cfg.model_config.is_quantized
    )
    assert enable_if_quantized(config_quantized) is True
    assert enable_if_quantized(config_no_model) is False

    # Test with MoE model
    moe_model = ModelConfig("deepseek-ai/DeepSeek-V2-Lite")
    config_moe = VllmConfig(
        model_config=moe_model, optimization_level=OptimizationLevel.O2
    )
    enable_if_sequential = lambda cfg: (
        cfg.model_config is not None and not cfg.model_config.is_moe
    )
    assert enable_if_sequential(config_moe) is False
    assert enable_if_sequential(config_quantized) is True


def test_vllm_config_explicit_overrides():
    """Test that explicit property overrides work correctly with callable defaults.

    When users explicitly set configuration properties, those values
    take precedence over callable defaults, across different models and
    optimization levels.
    """
    from vllm.config.compilation import PassConfig

    quantized_model = ModelConfig("RedHatAI/Llama-3.2-1B-FP8")
    moe_model = ModelConfig("deepseek-ai/DeepSeek-V2-Lite")
    regular_model = ModelConfig("Qwen/Qwen1.5-7B")

    # Explicit compilation mode override on O0 (where default is NONE)
    compilation_config = CompilationConfig(mode=CompilationMode.VLLM_COMPILE)
    config = VllmConfig(
        optimization_level=OptimizationLevel.O0,
        compilation_config=compilation_config,
    )
    assert config.compilation_config.mode == CompilationMode.VLLM_COMPILE
    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE

    # Explicit pass config flags to override defaults
    pass_config = PassConfig(eliminate_noops=True, fuse_attn_quant=True)
    compilation_config = CompilationConfig(pass_config=pass_config)
    config = VllmConfig(
        optimization_level=OptimizationLevel.O0,
        compilation_config=compilation_config,
    )
    assert config.compilation_config.pass_config.eliminate_noops is True
    assert config.compilation_config.pass_config.fuse_attn_quant is True

    # Explicit cudagraph mode override on quantized model at O2
    pass_config = PassConfig(enable_qk_norm_rope_fusion=True)
    compilation_config = CompilationConfig(
        cudagraph_mode=CUDAGraphMode.NONE, pass_config=pass_config
    )
    config = VllmConfig(
        model_config=quantized_model,
        optimization_level=OptimizationLevel.O2,
        compilation_config=compilation_config,
    )
    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
    assert config.compilation_config.pass_config.enable_qk_norm_rope_fusion is True
    # Mode should still use default for O2
    assert config.compilation_config.mode == CompilationMode.VLLM_COMPILE

    # Different optimization levels with same model
    config_o0 = VllmConfig(
        model_config=regular_model, optimization_level=OptimizationLevel.O0
    )
    config_o2 = VllmConfig(
        model_config=regular_model, optimization_level=OptimizationLevel.O2
    )
    assert config_o0.compilation_config.mode == CompilationMode.NONE
    assert config_o2.compilation_config.mode == CompilationMode.VLLM_COMPILE
    assert config_o0.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
    assert (
        config_o2.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE
    )

    # Same optimization level across different model types
    config_moe_o2 = VllmConfig(
        model_config=moe_model, optimization_level=OptimizationLevel.O2
    )
    config_regular_o2 = VllmConfig(
        model_config=regular_model, optimization_level=OptimizationLevel.O2
    )
    config_quantized_o2 = VllmConfig(
        model_config=quantized_model, optimization_level=OptimizationLevel.O2
    )
    # All should have same base compilation settings at O2
    assert config_moe_o2.compilation_config.mode == CompilationMode.VLLM_COMPILE
    assert config_regular_o2.compilation_config.mode == CompilationMode.VLLM_COMPILE
    assert config_quantized_o2.compilation_config.mode == CompilationMode.VLLM_COMPILE
    assert (
        config_moe_o2.compilation_config.cudagraph_mode
        == CUDAGraphMode.FULL_AND_PIECEWISE
    )
    assert (
        config_regular_o2.compilation_config.cudagraph_mode
        == CUDAGraphMode.FULL_AND_PIECEWISE
    )

    # Override one field but not others
    pass_config = PassConfig(eliminate_noops=False)
    compilation_config = CompilationConfig(pass_config=pass_config)
    config = VllmConfig(
        model_config=regular_model,
        optimization_level=OptimizationLevel.O2,
        compilation_config=compilation_config,
    )
    # Explicit override should be respected
    assert config.compilation_config.pass_config.eliminate_noops is False
    # Other fields should still use defaults
    assert config.compilation_config.mode == CompilationMode.VLLM_COMPILE
    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE


def test_scheduler_config_init():
    with pytest.raises(ValidationError):
        # Positional InitVars missing
        # (InitVars cannot have defaults otherwise they will become attributes)
        SchedulerConfig()

    with pytest.raises(AttributeError):
        # InitVar does not become an attribute
        print(SchedulerConfig.default_factory().max_model_len)


@pytest.mark.parametrize(
    (
        "model_id",
        "data_parallel_size",
        "external_lb",
        "expected_needs_coordinator",
    ),
    [
        # Non-MoE model with DP=1 should not need coordinator
        ("facebook/opt-125m", 1, False, False),
        # Non-MoE model with DP>1 internal LB should need coordinator
        ("facebook/opt-125m", 2, False, True),
        # Non-MoE model with DP>1 external LB should not need coordinator
        ("facebook/opt-125m", 2, True, False),
        # MoE model with DP=1 should not need coordinator
        ("mistralai/Mixtral-8x7B-Instruct-v0.1", 1, False, False),
        # MoE model with DP>1 internal LB should need both coordinator
        # and wave coordination
        ("mistralai/Mixtral-8x7B-Instruct-v0.1", 2, False, True),
        # MoE model with DP>1 external LB needs coordinator for wave coordination
        # (wave coordination runs in coordinator process)
        ("mistralai/Mixtral-8x7B-Instruct-v0.1", 2, True, True),
    ],
)
def test_needs_dp_coordination(
    model_id,
    data_parallel_size,
    external_lb,
    expected_needs_coordinator,
):
    """Test that DP coordinator and wave coordination are configured correctly."""
    from vllm.config import ParallelConfig

    model_config = ModelConfig(model_id)
    parallel_config = ParallelConfig(
        data_parallel_size=data_parallel_size,
        data_parallel_external_lb=external_lb,
    )
    vllm_config = VllmConfig(model_config=model_config, parallel_config=parallel_config)

    assert vllm_config.needs_dp_coordinator == expected_needs_coordinator


def test_eagle_draft_model_config():
    """Test that EagleDraft model config is correctly set."""
    target_model_config = ModelConfig(
        "meta-llama/Meta-Llama-3-8B-Instruct", trust_remote_code=True
    )
    speculative_config = SpeculativeConfig(
        model="yuhuili/EAGLE-LLaMA3-Instruct-8B",
        num_speculative_tokens=1,
        target_model_config=target_model_config,
        target_parallel_config=ParallelConfig(),
    )
    draft_model_config = speculative_config.draft_model_config
    assert draft_model_config.hf_config.architectures == ["EagleLlamaForCausalLM"]
    assert draft_model_config.hf_text_config.architectures == ["EagleLlamaForCausalLM"]
    assert draft_model_config.hf_config.model_type == "eagle"
    assert draft_model_config.hf_text_config.model_type == "eagle"
    assert draft_model_config.architectures == ["EagleLlamaForCausalLM"]
    assert draft_model_config.architecture == "EagleLlamaForCausalLM"
