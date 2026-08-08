# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import os
import pickle
from dataclasses import MISSING, Field, asdict, dataclass, field
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from unittest.mock import call as mock_call

import pydantic
import pytest
from huggingface_hub import ResolvedRevision
from pydantic import ValidationError

import vllm.config.vllm as vllm_config_module
import vllm.envs as envs
from vllm.compilation.backends import VllmBackend
from vllm.config import (
    CompilationConfig,
    KernelConfig,
    ModelConfig,
    ParallelConfig,
    PoolerConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
    update_config,
)
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.config.kernel import IrOpPriorityConfig
from vllm.config.load import LoadConfig
from vllm.config.utils import get_field
from vllm.config.vllm import OPTIMIZATION_LEVEL_TO_CONFIG, OptimizationLevel
from vllm.platforms import current_platform
from vllm.transformers_utils.runai_utils import ObjectStorageModel
from vllm.v1.attention.backend import AttentionCGSupport

DEVICE_TYPE = current_platform.device_type


def test_compile_config_repr_succeeds():
    # setup: VllmBackend mutates the config object
    config = VllmConfig()
    backend = VllmBackend(config)
    backend.configure_post_pass()

    # test that repr(config) succeeds
    val = repr(config)
    assert "VllmConfig" in val
    assert "inductor_passes" in val


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        (None, None),
        ("0", False),
        ("1", True),
    ],
)
def test_v2_model_runner_env_tri_state(monkeypatch, env_value, expected):
    if env_value is None:
        monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    else:
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", env_value)

    assert envs.VLLM_USE_V2_MODEL_RUNNER is expected


@pytest.mark.parametrize(
    ("use_v2_model_runner", "expected_capture_sizes"),
    [
        (False, [4, 8, 12, 16]),
        (True, list(range(1, 17))),
    ],
)
def test_resolve_cudagraph_mode_adjusts_spec_decode_sizes_only_for_v1(
    use_v2_model_runner,
    expected_capture_sizes,
):
    compilation_config = CompilationConfig(
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        cudagraph_capture_sizes=list(range(1, 17)),
    )
    compilation_config.max_cudagraph_capture_size = 16
    compilation_config.post_init_cudagraph_sizes()

    cudagraph_mode = compilation_config.resolve_cudagraph_mode_and_sizes(
        AttentionCGSupport.ALWAYS,
        "FakeAttentionBackend",
        uniform_decode_query_len=4,
        use_v2_model_runner=use_v2_model_runner,
        tensor_parallel_size=1,
    )

    assert cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE
    assert compilation_config.cudagraph_capture_sizes == expected_capture_sizes


@pytest.mark.parametrize(
    ("model_config", "expected"),
    [
        (
            SimpleNamespace(
                model="Qwen/Qwen3-1.7B-Base",
                architectures=["Qwen3ForCausalLM"],
                runner_type="generate",
                is_moe=False,
                is_quantized=False,
            ),
            True,
        ),
        (
            SimpleNamespace(
                model="Qwen/Qwen3-32B",
                architectures=["Qwen3ForCausalLM"],
                runner_type="generate",
                is_moe=False,
                is_quantized=False,
            ),
            True,
        ),
        (
            SimpleNamespace(
                model="meta-llama/Llama-3.2-1B",
                architectures=["LlamaForCausalLM"],
                runner_type="generate",
                is_moe=False,
                is_quantized=False,
            ),
            True,
        ),
        (
            SimpleNamespace(
                model="mistralai/Mistral-7B-v0.1",
                architectures=["MistralForCausalLM"],
                runner_type="generate",
                is_moe=False,
                is_quantized=False,
            ),
            True,
        ),
        (
            SimpleNamespace(
                model="facebook/opt-125m",
                architectures=["OPTForCausalLM"],
                runner_type="generate",
                is_moe=False,
                is_quantized=False,
            ),
            True,
        ),
        (
            SimpleNamespace(
                model="google/gemma-2-2b",
                architectures=["Gemma2ForCausalLM"],
                runner_type="generate",
                is_moe=False,
                is_quantized=False,
            ),
            True,
        ),
        (
            SimpleNamespace(
                model="deepseek-ai/DeepSeek-V2-Lite-Chat",
                architectures=["DeepseekV2ForCausalLM"],
                runner_type="generate",
                is_moe=True,
                is_quantized=False,
            ),
            True,
        ),
        (
            SimpleNamespace(
                model="deepseek-ai/DeepSeek-V2-Chat",
                architectures=["DeepseekV2ForCausalLM"],
                runner_type="generate",
                is_moe=True,
                is_quantized=False,
            ),
            True,
        ),
        (
            SimpleNamespace(
                model="Qwen/Qwen1.5-MoE-A2.7B",
                architectures=["Qwen2MoeForCausalLM"],
                runner_type="generate",
                is_moe=True,
                is_quantized=False,
            ),
            True,
        ),
        (
            SimpleNamespace(
                model="Qwen/Qwen1.5-MoE-A2.7B-Chat",
                architectures=["Qwen2MoeForCausalLM"],
                runner_type="generate",
                is_moe=True,
                is_quantized=False,
            ),
            True,
        ),
        (
            SimpleNamespace(
                model="ibm-research/PowerMoE-3b",
                architectures=["GraniteMoeForCausalLM"],
                runner_type="generate",
                is_moe=True,
                is_quantized=False,
            ),
            True,
        ),
        (
            SimpleNamespace(
                model="thinkingmachines/Inkling",
                architectures=["InklingForCausalLM"],
                runner_type="generate",
                is_moe=True,
                is_quantized=False,
            ),
            True,
        ),
        (
            SimpleNamespace(
                model="thinkingmachines/Inkling",
                architectures=["InklingForConditionalGeneration"],
                runner_type="generate",
                is_moe=True,
                is_quantized=False,
            ),
            True,
        ),
        (
            SimpleNamespace(
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                architectures=["MixtralForCausalLM"],
                runner_type="generate",
                is_moe=True,
                is_quantized=False,
            ),
            False,
        ),
        (
            SimpleNamespace(
                model="Qwen/Qwen3-1.7B-FP8",
                architectures=["Qwen3ForCausalLM"],
                runner_type="generate",
                is_moe=False,
                is_quantized=True,
            ),
            True,
        ),
        (
            SimpleNamespace(
                model="Qwen/Qwen3.5-4B",
                architectures=["Qwen3_5ForConditionalGeneration"],
                runner_type="generate",
                is_moe=False,
                is_quantized=False,
                is_hybrid=True,
            ),
            False,
        ),
        (
            SimpleNamespace(
                model="state-spaces/mamba-130m-hf",
                architectures=["MambaForCausalLM"],
                runner_type="generate",
                is_moe=False,
                is_quantized=False,
                is_attention_free=True,
            ),
            False,
        ),
        (
            SimpleNamespace(
                model="Qwen/Qwen3-Embedding-0.6B",
                architectures=["Qwen3ForCausalLM"],
                runner_type="pooling",
                is_moe=False,
                is_quantized=False,
            ),
            False,
        ),
    ],
)
def test_is_default_v2_model_runner_model(model_config, expected):
    config = SimpleNamespace(model_config=model_config)

    assert VllmConfig._is_default_v2_model_runner_model(config) is expected


@pytest.mark.skip_global_cleanup
def test_with_hf_config_populates_missing_architectures_from_causal_lm_mapping(
    monkeypatch,
):
    monkeypatch.setattr(
        vllm_config_module,
        "replace",
        lambda self, **kwargs: SimpleNamespace(**kwargs),
    )
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            is_multimodal_model=False,
            hf_config=SimpleNamespace(),
            get_model_arch_config=lambda: "arch-config",
        )
    )
    hf_config = SimpleNamespace(model_type="mistral", architectures=None)

    updated = VllmConfig.with_hf_config(cfg, hf_config)

    assert updated.model_config.hf_config.architectures == ["MistralForCausalLM"]
    assert hf_config.architectures is None


@pytest.mark.skip_global_cleanup
def test_with_hf_config_preserves_explicit_architectures_override(monkeypatch):
    monkeypatch.setattr(
        vllm_config_module,
        "replace",
        lambda self, **kwargs: SimpleNamespace(**kwargs),
    )
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            is_multimodal_model=False,
            hf_config=SimpleNamespace(),
            get_model_arch_config=lambda: "arch-config",
        )
    )
    hf_config = SimpleNamespace(model_type="mistral", architectures=None)

    updated = VllmConfig.with_hf_config(
        cfg,
        hf_config,
        architectures=["Ministral3ForCausalLM"],
    )

    assert updated.model_config.hf_config.architectures == ["Ministral3ForCausalLM"]


@pytest.mark.skip_global_cleanup
def test_with_hf_config_leaves_unknown_model_type_without_architectures(
    monkeypatch,
):
    monkeypatch.setattr(
        vllm_config_module,
        "replace",
        lambda self, **kwargs: SimpleNamespace(**kwargs),
    )
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            is_multimodal_model=False,
            hf_config=SimpleNamespace(),
            get_model_arch_config=lambda: "arch-config",
        )
    )
    hf_config = SimpleNamespace(
        model_type="not_a_real_model",
        architectures=None,
    )

    updated = VllmConfig.with_hf_config(cfg, hf_config)

    assert updated.model_config.hf_config.architectures is None


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


def test_data_parallel_rpc_port_has_fixed_default():
    assert ParallelConfig().data_parallel_rpc_port == 29550


@pytest.mark.parametrize("port", [1, 29550, 65535])
def test_data_parallel_rpc_port_accepts_valid_ports(port: int):
    assert ParallelConfig(data_parallel_rpc_port=port).data_parallel_rpc_port == port


@pytest.mark.parametrize("port", [-1, 0, 65536])
def test_data_parallel_rpc_port_rejects_invalid_ports(port: int):
    with pytest.raises(ValidationError):
        ParallelConfig(data_parallel_rpc_port=port)


def test_reconfigure_for_independent_dp_rank_on_multinode_dense_model():
    parallel_config = ParallelConfig(
        tensor_parallel_size=8,
        data_parallel_size=2,
        data_parallel_size_local=1,
        data_parallel_rank=1,
        distributed_executor_backend="mp",
        nnodes=2,
        node_rank=1,
    )

    assert parallel_config.nnodes_within_dp == 1
    assert parallel_config.node_rank_within_dp == 0

    parallel_config.reconfigure_for_independent_dp_rank()

    assert parallel_config.data_parallel_size == 1
    assert parallel_config.data_parallel_size_local == 1
    assert parallel_config.data_parallel_rank == 0
    assert parallel_config.data_parallel_index == 1
    assert parallel_config.nnodes == 1
    assert parallel_config.node_rank == 0
    assert parallel_config.world_size == 8


def test_draft_model_enables_async_scheduling_by_default():
    parallel_config = ParallelConfig(distributed_executor_backend="uni")
    model_config = ModelConfig("Qwen/Qwen3-0.6B", max_model_len=2048)
    speculative_config = SpeculativeConfig(
        method="draft_model",
        model="Qwen/Qwen3-0.6B",
        num_speculative_tokens=3,
        target_model_config=model_config,
        target_parallel_config=parallel_config,
    )
    cfg = VllmConfig(
        model_config=model_config,
        scheduler_config=SchedulerConfig(
            max_model_len=2048,
            is_encoder_decoder=False,
        ),
        parallel_config=parallel_config,
        speculative_config=speculative_config,
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


@dataclass
class _TestDerivedConfigFields(_TestConfigFields):
    pass


def test_update_config():
    # Simple update
    config1 = _TestConfigFields(a=0)
    new_config1 = update_config(config1, {"a": 42})
    assert new_config1.a == 42
    # Nonexistent field
    with pytest.raises(ValueError, match=r"_TestConfigFields\.nonexistent"):
        new_config1 = update_config(config1, {"nonexistent": 1})
    # Nested update with dataclass
    config2 = _TestNestedConfig()
    new_inner_config = _TestConfigFields(a=1, c="new_value")
    new_config2 = update_config(config2, {"a": new_inner_config})
    assert new_config2.a == new_inner_config
    # Declared field type, not the live value's subtype, defines valid overrides
    config_with_derived = _TestNestedConfig(a=_TestDerivedConfigFields(a=0))
    new_config2 = update_config(config_with_derived, {"a": new_inner_config})
    assert new_config2.a is new_inner_config
    # Nested update with unrelated dataclass
    with pytest.raises(ValueError, match=r"_TestNestedConfig\.a"):
        update_config(config2, {"a": _TestNestedConfig()})
    # Nested update with dict
    config3 = _TestNestedConfig()
    new_config3 = update_config(config3, {"a": {"c": "new_value"}})
    assert new_config3.a.c == "new_value"
    # Nested update with invalid type
    with pytest.raises(ValueError, match=r"_TestNestedConfig\.a"):
        update_config(config3, {"a": "new_value"})
    # Invalid nested field preserves its full path
    with pytest.raises(ValueError, match=r"_TestNestedConfig\.a\.nonexistent"):
        update_config(config3, {"a": {"nonexistent": 1}})


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


def test_model_class_overrides_registers_target():
    """`model_class_overrides` redirects an architecture to a custom class."""
    from vllm.model_executor.models import ModelRegistry

    arch = "_TestModelClassOverrideArch"
    target = "vllm.model_executor.models.llama:LlamaForCausalLM"
    assert arch not in ModelRegistry.models

    model_config = ModelConfig(
        "facebook/opt-125m",
        model_class_overrides={arch: target},
    )
    try:
        # Accessing `.registry` is the chokepoint that applies the overrides;
        # it has already run during construction.
        registered = model_config.registry.models[arch]
        assert registered.module_name == "vllm.model_executor.models.llama"
        assert registered.class_name == "LlamaForCausalLM"
        # Idempotent: a second access does not re-register or error out.
        assert model_config.registry.models[arch] is registered
    finally:
        ModelRegistry.models.pop(arch, None)


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
        DEVICE_TYPE,
        {"": DEVICE_TYPE},
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
        self.model_weights: str | None = None
        self.tokenizer_weights: str | None = None


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


@patch("vllm.transformers_utils.runai_utils.ObjectStorageModel.pull_files")
def test_s3_url_different_model_and_tokenizer(mock_pull_files):
    """Test that when model and tokenizer are different cloud URIs,
    pull_files receives the correct URI for each."""
    mock_pull_files.return_value = None

    model_url = "s3://bucket/model/"
    tokenizer_url = "s3://bucket/tokenizer/"

    config = MockConfig(model=model_url, tokenizer=tokenizer_url)
    ModelConfig.maybe_pull_model_tokenizer_for_runai(config, model_url, tokenizer_url)

    # pull_files should be called twice: once for model, once for tokenizer
    assert mock_pull_files.call_count == 2
    # First call: model URI with allow_pattern
    assert mock_pull_files.call_args_list[0][0][0] == model_url
    # Second call: tokenizer URI with ignore_pattern
    assert mock_pull_files.call_args_list[1][0][0] == tokenizer_url


# Tests for maybe_pull_model_files_for_runai_worker (worker-side repair pull).

RUNAI_WORKER_S3_URL = "s3://example-bucket/model/"
RUNAI_WORKER_TOKENIZER_S3_URL = "s3://example-bucket/tokenizer/"
# Spelled out rather than imported, so a change to the source is a failure.
RUNAI_NON_WEIGHT_IGNORE_PATTERN = [
    "*.pt",
    "*.safetensors",
    "*.bin",
    "*.tensors",
    "*.pth",
]


@pytest.fixture
def runai_assets_cache(tmp_path, monkeypatch):
    """Keep the Object Storage cache directory inside the test's tmp_path."""
    monkeypatch.setenv("VLLM_ASSETS_CACHE", str(tmp_path))
    return tmp_path


@pytest.fixture
def mock_get_lock():
    """Stub out the file lock guarding concurrent pulls on the same node."""
    # weight_utils is imported lazily by the method under test, so make sure
    # the module exists before patching a name inside it.
    import vllm.model_executor.model_loader.weight_utils  # noqa: F401

    with patch(
        "vllm.model_executor.model_loader.weight_utils.get_lock"
    ) as mock_get_lock:
        yield mock_get_lock


def _worker_config(model_dir: str, model_weights: str) -> MockConfig:
    """A config as a worker receives it: local dirs plus the original URI."""
    config = MockConfig(model=model_dir, tokenizer=model_dir)
    config.model_weights = model_weights
    return config


def _separate_tokenizer_worker_config(
    model: str, model_weights: str, tokenizer_dir: str
) -> MockConfig:
    """A worker config whose tokenizer came from a URI of its own."""
    config = MockConfig(model=model, tokenizer=tokenizer_dir)
    config.model_weights = model_weights
    config.tokenizer_weights = RUNAI_WORKER_TOKENIZER_S3_URL
    return config


@patch("vllm.transformers_utils.runai_utils.ObjectStorageModel.pull_files")
def test_runai_worker_pulls_files_when_node_local_dir_is_empty(
    mock_pull_files, mock_get_lock, runai_assets_cache
):
    """A peer node must pull the non-tensor files it never received."""
    mock_pull_files.return_value = None

    # ObjectStorageModel resolves (and creates) the deterministic local
    # directory. Leaving it empty reproduces the peer node's state.
    local_dir = ObjectStorageModel(url=RUNAI_WORKER_S3_URL).dir
    assert os.listdir(local_dir) == []

    config = _worker_config(local_dir, RUNAI_WORKER_S3_URL)
    ModelConfig.maybe_pull_model_files_for_runai_worker(config)

    assert mock_pull_files.call_count == 2
    # The weights URI is pulled, never the already-rewritten local path.
    assert mock_pull_files.call_args_list[0][0][0] == RUNAI_WORKER_S3_URL
    assert mock_pull_files.call_args_list[0][1]["allow_pattern"] == [
        "*.model",
        "*.py",
        "*.json",
    ]
    # The pull is serialized against the other ranks on this node, keyed on
    # the resolved directory being written rather than on the remote URI.
    mock_get_lock.assert_called_once_with(os.path.realpath(local_dir))
    # Weights themselves are streamed, not pulled.
    assert config.model_weights == RUNAI_WORKER_S3_URL
    # The successful pull leaves the cache directory in place.
    assert os.path.exists(local_dir)


@patch("vllm.transformers_utils.runai_utils.ObjectStorageModel.pull_files")
def test_runai_worker_pull_matches_driver_pull(
    mock_pull_files, mock_get_lock, runai_assets_cache
):
    """The worker must reproduce the driver's pull calls verbatim."""
    mock_pull_files.return_value = None

    driver_config = MockConfig(model=RUNAI_WORKER_S3_URL, tokenizer=RUNAI_WORKER_S3_URL)
    ModelConfig.maybe_pull_model_tokenizer_for_runai(
        driver_config, RUNAI_WORKER_S3_URL, RUNAI_WORKER_S3_URL
    )
    driver_calls = mock_pull_files.call_args_list[:]
    assert driver_calls, "driver is expected to pull files"
    mock_pull_files.reset_mock()

    # pull_files is mocked, so the directory the driver resolved is still
    # empty - exactly what a worker on a peer node finds.
    worker_config = _worker_config(driver_config.model, RUNAI_WORKER_S3_URL)
    ModelConfig.maybe_pull_model_files_for_runai_worker(worker_config)

    assert mock_pull_files.call_args_list == driver_calls
    assert worker_config.model == driver_config.model
    assert worker_config.tokenizer == driver_config.tokenizer


@patch("vllm.transformers_utils.runai_utils.ObjectStorageModel.pull_files")
def test_runai_worker_does_not_disturb_an_unrepairable_tokenizer(
    mock_pull_files, mock_get_lock, runai_assets_cache, tmp_path
):
    """A tokenizer pulled from its own URI is left untouched, not repointed."""
    mock_pull_files.return_value = None

    local_dir = ObjectStorageModel(url=RUNAI_WORKER_S3_URL).dir
    tokenizer_dir = str(tmp_path / "separate-tokenizer")

    config = MockConfig(model=local_dir, tokenizer=tokenizer_dir)
    config.model_weights = RUNAI_WORKER_S3_URL
    ModelConfig.maybe_pull_model_files_for_runai_worker(config)

    # Mirrors the driver: a separate tokenizer means only the allow-pattern
    # pull for the model directory.
    assert mock_pull_files.call_count == 1
    assert mock_pull_files.call_args_list[0][0][0] == RUNAI_WORKER_S3_URL
    assert mock_pull_files.call_args_list[0][1]["allow_pattern"] == [
        "*.model",
        "*.py",
        "*.json",
    ]
    assert config.tokenizer == tokenizer_dir


@patch("vllm.transformers_utils.runai_utils.ObjectStorageModel.pull_files")
def test_runai_worker_pull_matches_driver_pull_with_separate_tokenizer(
    mock_pull_files, mock_get_lock, runai_assets_cache
):
    """With a separate tokenizer, the worker must still mirror the driver's
    pulls for the model URI: the allow-pattern pull only."""
    mock_pull_files.return_value = None

    tokenizer_url = "s3://example-bucket/tokenizer/"
    driver_config = MockConfig(model=RUNAI_WORKER_S3_URL, tokenizer=tokenizer_url)
    ModelConfig.maybe_pull_model_tokenizer_for_runai(
        driver_config, RUNAI_WORKER_S3_URL, tokenizer_url
    )
    driver_model_calls = [
        call
        for call in mock_pull_files.call_args_list
        if call[0][0] == RUNAI_WORKER_S3_URL
    ]
    assert len(driver_model_calls) == 1
    mock_pull_files.reset_mock()

    worker_config = _worker_config(driver_config.model, RUNAI_WORKER_S3_URL)
    worker_config.tokenizer = driver_config.tokenizer
    ModelConfig.maybe_pull_model_files_for_runai_worker(worker_config)

    assert mock_pull_files.call_args_list == driver_model_calls


@patch("vllm.config.model.ObjectStorageModel")
def test_runai_worker_guard_accepts_a_symlinked_cache_directory(
    mock_object_storage_model, mock_get_lock, runai_assets_cache, tmp_path
):
    """Paths resolving to the same physical directory must not be rejected."""
    real_dir = tmp_path / "real-cache"
    real_dir.mkdir()
    link_dir = tmp_path / "alias-cache"
    os.symlink(real_dir, link_dir, target_is_directory=True)

    mock_object_storage_model.return_value.dir = str(real_dir)
    config = _worker_config(str(link_dir), RUNAI_WORKER_S3_URL)
    ModelConfig.maybe_pull_model_files_for_runai_worker(config)

    assert mock_object_storage_model.return_value.pull_files.call_count == 2
    # The lock is keyed on the canonical path, not on the alias spelling.
    mock_get_lock.assert_called_once_with(os.path.realpath(str(link_dir)))


@patch("vllm.config.model.ObjectStorageModel")
def test_runai_worker_waits_for_lock_before_trusting_existing_files(
    mock_object_storage_model, mock_get_lock, runai_assets_cache
):
    """Even a populated directory is only trusted after taking the lock."""
    local_dir = ObjectStorageModel(url=RUNAI_WORKER_S3_URL).dir
    with open(os.path.join(local_dir, "config.json"), "w") as f:
        f.write("{}")

    config = _worker_config(local_dir, RUNAI_WORKER_S3_URL)
    ModelConfig.maybe_pull_model_files_for_runai_worker(config)

    # A non-empty directory may be a pull still in progress on another rank.
    mock_get_lock.assert_called_once_with(os.path.realpath(local_dir))
    mock_object_storage_model.assert_not_called()
    assert config.model == local_dir


@patch("vllm.transformers_utils.runai_utils.ObjectStorageModel.pull_files")
def test_runai_worker_rechecks_after_acquiring_the_lock(
    mock_pull_files, mock_get_lock, runai_assets_cache
):
    """A rank that waited for the lock must not repeat the winner's pull."""
    mock_pull_files.return_value = None

    local_dir = ObjectStorageModel(url=RUNAI_WORKER_S3_URL).dir

    def win_race_while_blocked(*args, **kwargs):
        # Simulates the rank holding the lock finishing its pull.
        with open(os.path.join(local_dir, "config.json"), "w") as f:
            f.write("{}")

    lock = MagicMock()
    lock.__enter__.side_effect = win_race_while_blocked
    mock_get_lock.return_value = lock

    config = _worker_config(local_dir, RUNAI_WORKER_S3_URL)
    ModelConfig.maybe_pull_model_files_for_runai_worker(config)

    mock_get_lock.assert_called_once()
    mock_pull_files.assert_not_called()


def _fail_after_writing_one_file(local_dir: str, exc_type: type = OSError):
    """A pull that dies partway, leaving the directory half populated."""

    def pull(*args, **kwargs):
        os.makedirs(local_dir, exist_ok=True)
        with open(os.path.join(local_dir, "config.json"), "w") as f:
            f.write("{}")
        raise exc_type("connection reset by peer")

    return pull


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
@patch("vllm.transformers_utils.runai_utils.ObjectStorageModel.pull_files")
def test_runai_worker_clears_the_directory_when_the_pull_fails(
    mock_pull_files, mock_get_lock, runai_assets_cache, failure
):
    """A failed or interrupted pull removes its leftovers and propagates."""
    local_dir = ObjectStorageModel(url=RUNAI_WORKER_S3_URL).dir
    mock_pull_files.side_effect = _fail_after_writing_one_file(local_dir, failure)

    config = _worker_config(local_dir, RUNAI_WORKER_S3_URL)
    with pytest.raises(failure):
        ModelConfig.maybe_pull_model_files_for_runai_worker(config)

    assert not os.path.exists(local_dir)


@patch("vllm.transformers_utils.runai_utils.ObjectStorageModel.pull_files")
def test_runai_worker_retries_after_a_failed_pull(
    mock_pull_files, mock_get_lock, runai_assets_cache
):
    """A transient failure must not poison the cache for the next start."""
    local_dir = ObjectStorageModel(url=RUNAI_WORKER_S3_URL).dir
    mock_pull_files.side_effect = _fail_after_writing_one_file(local_dir)

    config = _worker_config(local_dir, RUNAI_WORKER_S3_URL)
    with pytest.raises(OSError):
        ModelConfig.maybe_pull_model_files_for_runai_worker(config)

    mock_pull_files.reset_mock()
    mock_pull_files.side_effect = None
    ModelConfig.maybe_pull_model_files_for_runai_worker(config)

    assert mock_pull_files.call_count == 2
    # The successful retry leaves the cache directory in place.
    assert os.path.exists(local_dir)


@patch("vllm.config.model.ObjectStorageModel")
def test_runai_worker_fails_fast_on_a_cache_directory_mismatch(
    mock_object_storage_model, mock_get_lock, runai_assets_cache, tmp_path
):
    """A node deriving a different cache directory fails fast, without pulling."""
    mock_object_storage_model.return_value.dir = str(tmp_path / "worker-side-dir")

    config = _worker_config(str(tmp_path / "driver-side-dir"), RUNAI_WORKER_S3_URL)
    with (
        patch("vllm.config.model.shutil.rmtree") as mock_rmtree,
        pytest.raises(RuntimeError, match="VLLM_ASSETS_CACHE"),
    ):
        ModelConfig.maybe_pull_model_files_for_runai_worker(config)

    mock_object_storage_model.return_value.pull_files.assert_not_called()
    mock_rmtree.assert_not_called()


@patch("vllm.config.model.ObjectStorageModel")
def test_runai_worker_opts_out_of_signal_handlers(
    mock_object_storage_model, mock_get_lock, runai_assets_cache, tmp_path
):
    """The worker-side pull must not install process-global signal handlers."""
    local_dir = str(tmp_path / "node-local-dir")
    mock_object_storage_model.return_value.dir = local_dir

    config = _worker_config(local_dir, RUNAI_WORKER_S3_URL)
    ModelConfig.maybe_pull_model_files_for_runai_worker(config)

    mock_object_storage_model.assert_called_once_with(
        url=RUNAI_WORKER_S3_URL, install_signal_handlers=False
    )


def test_runai_object_storage_model_signal_handler_opt_out(monkeypatch, tmp_path):
    """`install_signal_handlers=False` skips registration; the default keeps it."""
    monkeypatch.setenv("VLLM_ASSETS_CACHE_MODEL_CLEAN", "1")
    monkeypatch.setenv("VLLM_ASSETS_CACHE", str(tmp_path))

    with patch("vllm.transformers_utils.runai_utils.signal.signal") as mock_signal:
        ObjectStorageModel(url=RUNAI_WORKER_S3_URL, install_signal_handlers=False)
        mock_signal.assert_not_called()

        ObjectStorageModel(url=RUNAI_WORKER_S3_URL)
        assert mock_signal.call_count == 2  # SIGINT + SIGTERM


@pytest.mark.parametrize(
    "model_weights",
    [None, "", "/local/path/to/model", "facebook/opt-125m"],
)
@patch("vllm.config.model.ObjectStorageModel")
def test_runai_worker_noop_without_object_storage_weights(
    mock_object_storage_model, mock_get_lock, model_weights
):
    """Models that never came from Object Storage must not be touched."""
    config = MockConfig(model="/local/path/to/model", tokenizer="/local/tokenizer")
    config.model_weights = model_weights

    ModelConfig.maybe_pull_model_files_for_runai_worker(config)

    mock_get_lock.assert_not_called()
    mock_object_storage_model.assert_not_called()
    assert config.model == "/local/path/to/model"
    assert config.tokenizer == "/local/tokenizer"


@patch("vllm.transformers_utils.runai_utils.ObjectStorageModel.pull_files")
def test_runai_worker_not_short_circuited_by_driver_guard(
    mock_pull_files, mock_get_lock, runai_assets_cache
):
    """The driver's method silently no-ops on a worker; the new one must pull."""
    mock_pull_files.return_value = None

    local_dir = ObjectStorageModel(url=RUNAI_WORKER_S3_URL).dir
    config = _worker_config(local_dir, RUNAI_WORKER_S3_URL)

    # The driver's method no-ops on a worker...
    ModelConfig.maybe_pull_model_tokenizer_for_runai(
        config, config.model, config.tokenizer
    )
    mock_pull_files.assert_not_called()

    # ...while the worker's method pulls.
    ModelConfig.maybe_pull_model_files_for_runai_worker(config)
    assert mock_pull_files.call_count > 0


# Tests for the separate-tokenizer half of the worker-side repair pull.


@patch("vllm.transformers_utils.runai_utils.ObjectStorageModel.pull_files")
def test_runai_worker_pulls_a_separate_tokenizer_when_its_dir_is_empty(
    mock_pull_files, mock_get_lock, runai_assets_cache
):
    """A tokenizer with a URI of its own is repaired from that URI."""
    mock_pull_files.return_value = None

    model_dir = ObjectStorageModel(url=RUNAI_WORKER_S3_URL).dir
    tokenizer_dir = ObjectStorageModel(url=RUNAI_WORKER_TOKENIZER_S3_URL).dir
    assert os.listdir(tokenizer_dir) == []

    config = _separate_tokenizer_worker_config(
        model_dir, RUNAI_WORKER_S3_URL, tokenizer_dir
    )
    ModelConfig.maybe_pull_model_files_for_runai_worker(config)

    # The model directory keeps its single allow-pattern pull; the tokenizer
    # adds the ignore-pattern pull the driver would have made for it.
    assert mock_pull_files.call_count == 2
    assert mock_pull_files.call_args_list[0][0][0] == RUNAI_WORKER_S3_URL
    assert mock_pull_files.call_args_list[0][1]["allow_pattern"] == [
        "*.model",
        "*.py",
        "*.json",
    ]
    # The URI is pulled, never the already-rewritten local path.
    assert mock_pull_files.call_args_list[1][0][0] == RUNAI_WORKER_TOKENIZER_S3_URL
    assert (
        mock_pull_files.call_args_list[1][1]["ignore_pattern"]
        == RUNAI_NON_WEIGHT_IGNORE_PATTERN
    )
    assert config.tokenizer == tokenizer_dir
    assert os.path.exists(tokenizer_dir)


@patch("vllm.transformers_utils.runai_utils.ObjectStorageModel.pull_files")
def test_runai_worker_tokenizer_pull_matches_driver_pull(
    mock_pull_files, mock_get_lock, runai_assets_cache
):
    """The worker must reproduce the driver's tokenizer pull verbatim."""
    mock_pull_files.return_value = None

    driver_config = MockConfig(
        model=RUNAI_WORKER_S3_URL, tokenizer=RUNAI_WORKER_TOKENIZER_S3_URL
    )
    ModelConfig.maybe_pull_model_tokenizer_for_runai(
        driver_config, RUNAI_WORKER_S3_URL, RUNAI_WORKER_TOKENIZER_S3_URL
    )
    driver_tokenizer_calls = [
        call
        for call in mock_pull_files.call_args_list
        if call[0][0] == RUNAI_WORKER_TOKENIZER_S3_URL
    ]
    assert len(driver_tokenizer_calls) == 1
    # The URI the worker needs is the one the driver recorded.
    assert driver_config.model_weights == RUNAI_WORKER_S3_URL
    assert driver_config.tokenizer_weights == RUNAI_WORKER_TOKENIZER_S3_URL
    mock_pull_files.reset_mock()

    worker_config = _separate_tokenizer_worker_config(
        driver_config.model, RUNAI_WORKER_S3_URL, driver_config.tokenizer
    )
    ModelConfig.maybe_pull_model_files_for_runai_worker(worker_config)

    worker_tokenizer_calls = [
        call
        for call in mock_pull_files.call_args_list
        if call[0][0] == RUNAI_WORKER_TOKENIZER_S3_URL
    ]
    assert worker_tokenizer_calls == driver_tokenizer_calls
    assert worker_config.tokenizer == driver_config.tokenizer


@patch("vllm.transformers_utils.runai_utils.ObjectStorageModel.pull_files")
def test_runai_shared_tokenizer_records_no_tokenizer_uri(
    mock_pull_files, mock_get_lock, runai_assets_cache
):
    """A tokenizer read from the model directory needs no URI of its own."""
    mock_pull_files.return_value = None

    driver_config = MockConfig(model=RUNAI_WORKER_S3_URL, tokenizer=RUNAI_WORKER_S3_URL)
    ModelConfig.maybe_pull_model_tokenizer_for_runai(
        driver_config, RUNAI_WORKER_S3_URL, RUNAI_WORKER_S3_URL
    )
    assert driver_config.model_weights == RUNAI_WORKER_S3_URL
    assert not driver_config.tokenizer_weights
    assert driver_config.tokenizer == driver_config.model
    mock_pull_files.reset_mock()

    worker_config = _worker_config(driver_config.model, RUNAI_WORKER_S3_URL)
    ModelConfig.maybe_pull_model_files_for_runai_worker(worker_config)

    # Unchanged by the tokenizer repair: two pulls of the model URI into the
    # one directory, taken under one lock.
    assert [call[0][0] for call in mock_pull_files.call_args_list] == [
        RUNAI_WORKER_S3_URL,
        RUNAI_WORKER_S3_URL,
    ]
    assert mock_pull_files.call_args_list[0][1]["allow_pattern"] == [
        "*.model",
        "*.py",
        "*.json",
    ]
    assert (
        mock_pull_files.call_args_list[1][1]["ignore_pattern"]
        == RUNAI_NON_WEIGHT_IGNORE_PATTERN
    )
    mock_get_lock.assert_called_once_with(os.path.realpath(driver_config.model))


@patch("vllm.transformers_utils.runai_utils.ObjectStorageModel.pull_files")
def test_runai_worker_repairs_the_tokenizer_of_a_hugging_face_model(
    mock_pull_files, mock_get_lock, runai_assets_cache
):
    """An ordinary model id leaves `model_weights` unset; repair the tokenizer
    from its own URI regardless."""
    mock_pull_files.return_value = None

    hf_model = "facebook/opt-125m"
    driver_config = MockConfig(model=hf_model, tokenizer=RUNAI_WORKER_TOKENIZER_S3_URL)
    ModelConfig.maybe_pull_model_tokenizer_for_runai(
        driver_config, hf_model, RUNAI_WORKER_TOKENIZER_S3_URL
    )
    assert not driver_config.model_weights
    assert driver_config.tokenizer_weights == RUNAI_WORKER_TOKENIZER_S3_URL
    tokenizer_dir = driver_config.tokenizer
    mock_pull_files.reset_mock()

    worker_config = _separate_tokenizer_worker_config(hf_model, "", tokenizer_dir)
    ModelConfig.maybe_pull_model_files_for_runai_worker(worker_config)

    assert mock_pull_files.call_count == 1
    assert mock_pull_files.call_args_list[0][0][0] == RUNAI_WORKER_TOKENIZER_S3_URL
    assert (
        mock_pull_files.call_args_list[0][1]["ignore_pattern"]
        == RUNAI_NON_WEIGHT_IGNORE_PATTERN
    )
    mock_get_lock.assert_called_once_with(os.path.realpath(tokenizer_dir))
    assert worker_config.model == hf_model


@patch("vllm.transformers_utils.runai_utils.ObjectStorageModel.pull_files")
def test_runai_worker_repairs_the_tokenizer_beside_a_populated_model_dir(
    mock_pull_files, mock_get_lock, runai_assets_cache
):
    """Each directory is checked on its own: a model another rank already
    pulled must not skip the tokenizer."""
    mock_pull_files.return_value = None

    model_dir = ObjectStorageModel(url=RUNAI_WORKER_S3_URL).dir
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        f.write("{}")
    tokenizer_dir = ObjectStorageModel(url=RUNAI_WORKER_TOKENIZER_S3_URL).dir

    config = _separate_tokenizer_worker_config(
        model_dir, RUNAI_WORKER_S3_URL, tokenizer_dir
    )
    ModelConfig.maybe_pull_model_files_for_runai_worker(config)

    assert mock_pull_files.call_count == 1
    assert mock_pull_files.call_args_list[0][0][0] == RUNAI_WORKER_TOKENIZER_S3_URL
    # The model directory was still locked and inspected, just left alone.
    assert mock_get_lock.call_count == 2


@patch("vllm.transformers_utils.runai_utils.ObjectStorageModel.pull_files")
def test_runai_worker_locks_the_model_and_tokenizer_directories_separately(
    mock_pull_files, mock_get_lock, runai_assets_cache
):
    """One lock per directory, each keyed on the canonical path it writes."""
    mock_pull_files.return_value = None

    model_dir = ObjectStorageModel(url=RUNAI_WORKER_S3_URL).dir
    tokenizer_dir = ObjectStorageModel(url=RUNAI_WORKER_TOKENIZER_S3_URL).dir

    config = _separate_tokenizer_worker_config(
        model_dir, RUNAI_WORKER_S3_URL, tokenizer_dir
    )
    ModelConfig.maybe_pull_model_files_for_runai_worker(config)

    assert mock_get_lock.call_args_list == [
        mock_call(os.path.realpath(model_dir)),
        mock_call(os.path.realpath(tokenizer_dir)),
    ]


@patch("vllm.config.model.ObjectStorageModel")
def test_runai_worker_fails_fast_on_a_tokenizer_directory_mismatch(
    mock_object_storage_model, mock_get_lock, runai_assets_cache, tmp_path
):
    """A node deriving a different tokenizer directory fails fast, without
    pulling."""
    mock_object_storage_model.return_value.dir = str(tmp_path / "worker-side-dir")

    config = _separate_tokenizer_worker_config(
        "facebook/opt-125m", "", str(tmp_path / "driver-side-dir")
    )
    with (
        patch("vllm.config.model.shutil.rmtree") as mock_rmtree,
        pytest.raises(RuntimeError, match="VLLM_ASSETS_CACHE"),
    ):
        ModelConfig.maybe_pull_model_files_for_runai_worker(config)

    mock_object_storage_model.return_value.pull_files.assert_not_called()
    mock_rmtree.assert_not_called()


@patch("vllm.transformers_utils.runai_utils.ObjectStorageModel.pull_files")
def test_runai_worker_clears_the_tokenizer_directory_when_the_pull_fails(
    mock_pull_files, mock_get_lock, runai_assets_cache
):
    """A failed tokenizer pull removes its leftovers and propagates."""
    tokenizer_dir = ObjectStorageModel(url=RUNAI_WORKER_TOKENIZER_S3_URL).dir
    mock_pull_files.side_effect = _fail_after_writing_one_file(tokenizer_dir)

    config = _separate_tokenizer_worker_config("facebook/opt-125m", "", tokenizer_dir)
    with pytest.raises(OSError):
        ModelConfig.maybe_pull_model_files_for_runai_worker(config)

    assert not os.path.exists(tokenizer_dir)


@patch("vllm.transformers_utils.runai_utils.ObjectStorageModel.pull_files")
def test_runai_separate_tokenizer_uri_reaches_a_worker(
    mock_pull_files, mock_get_lock, runai_assets_cache
):
    """The real config, not the mock, must carry the URI across the wire."""
    mock_pull_files.return_value = None

    driver_config = ModelConfig(
        model="Qwen/Qwen3-0.6B", tokenizer=RUNAI_WORKER_TOKENIZER_S3_URL
    )
    assert driver_config.tokenizer_weights == RUNAI_WORKER_TOKENIZER_S3_URL
    assert driver_config.tokenizer != RUNAI_WORKER_TOKENIZER_S3_URL
    mock_pull_files.reset_mock()

    # Workers receive the config pickled, so the field has to survive that.
    worker_config = pickle.loads(pickle.dumps(driver_config))
    assert worker_config.tokenizer_weights == RUNAI_WORKER_TOKENIZER_S3_URL
    worker_config.maybe_pull_model_files_for_runai_worker()

    assert mock_pull_files.call_count == 1
    assert mock_pull_files.call_args_list[0][0][0] == RUNAI_WORKER_TOKENIZER_S3_URL


def _init_worker_config(calls: list[str]) -> MagicMock:
    """A VllmConfig that records when its model files would be pulled."""
    vllm_config = MagicMock()
    vllm_config.parallel_config.worker_cls = "fake.module.FakeWorker"
    vllm_config.parallel_config.worker_extension_cls = None
    vllm_config.speculative_config = None
    vllm_config.model_config.maybe_pull_model_files_for_runai_worker.side_effect = (
        lambda: calls.append("pull_runai_files")
    )
    return vllm_config


def _run_init_worker(vllm_config: MagicMock, calls: list[str]) -> None:
    """Drive init_worker with the worker class and mm registry stubbed out."""
    from vllm.v1.worker import worker_base

    class FakeWorker:
        def __init__(self, **kwargs):
            calls.append("construct_worker")

    with (
        patch.object(worker_base, "resolve_obj_by_qualname", return_value=FakeWorker),
        patch.object(worker_base, "MULTIMODAL_REGISTRY") as mm_registry,
    ):
        mm_registry.worker_receiver_cache_from_config.side_effect = (
            lambda *args, **kwargs: calls.append("create_mm_receiver_cache")
        )
        worker_base.WorkerWrapperBase(rpc_rank=0).init_worker(
            [{"vllm_config": vllm_config, "shared_worker_lock": MagicMock()}]
        )


def test_init_worker_pulls_runai_files_before_the_directory_is_read():
    """The pull must run before the mm receiver cache reads the directory."""
    calls: list[str] = []
    _run_init_worker(_init_worker_config(calls), calls)

    assert calls == [
        "pull_runai_files",
        "create_mm_receiver_cache",
        "construct_worker",
    ]


def test_init_worker_pulls_runai_files_for_the_draft_model():
    """A distinct speculative draft config must be pulled as well."""
    calls: list[str] = []
    vllm_config = _init_worker_config(calls)
    draft_config = MagicMock()
    draft_config.maybe_pull_model_files_for_runai_worker.side_effect = (
        lambda: calls.append("pull_draft_runai_files")
    )
    vllm_config.speculative_config = MagicMock(draft_model_config=draft_config)

    _run_init_worker(vllm_config, calls)

    assert calls.count("pull_runai_files") == 1
    assert calls.count("pull_draft_runai_files") == 1


def test_init_worker_pulls_once_when_the_draft_reuses_the_target_config():
    """`draft_model_config` is often the target config itself; pull it once."""
    calls: list[str] = []
    vllm_config = _init_worker_config(calls)
    vllm_config.speculative_config = MagicMock(
        draft_model_config=vllm_config.model_config
    )

    _run_init_worker(vllm_config, calls)

    assert calls.count("pull_runai_files") == 1


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
            True,
            "Generative hybrid models support prefix caching.",  # noqa: E501
        ),
        (
            "ibm-granite/granite-4.0-h-small",
            "hybrid",
            True,
            "Generative hybrid models support prefix caching.",  # noqa: E501
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


def test_validate_mamba_align_subblock_prefill():
    """Align mode permits configured prefill chunks smaller than a block."""
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=11392,
            mamba_cache_mode="align",
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=8192,
            long_prefill_token_threshold=4096,
            disable_chunked_mm_input=False,
        ),
    )

    VllmConfig.validate_block_size(config)


@pytest.mark.parametrize(
    ("model_id", "compilation_config", "optimization_level"),
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
def test_vllm_config_defaults(model_id, compilation_config, optimization_level):
    """Test that optimization-level defaults are correctly applied."""

    model_config = None
    if model_id is not None:
        model_config = ModelConfig(model_id)
        vllm_config = VllmConfig(
            model_config=model_config,
            compilation_config=compilation_config,
            optimization_level=optimization_level,
        )
    else:
        vllm_config = VllmConfig(
            compilation_config=compilation_config,
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
        if k == "pass_config":
            continue
        actual = getattr(vllm_config.compilation_config, k)
        expected = v(vllm_config) if callable(v) else v
        # On platforms without static graph support, __post_init__ forces
        # cudagraph_mode to NONE; expect that instead of the level default.
        if k == "cudagraph_mode" and not current_platform.support_static_graph_mode():
            expected = CUDAGraphMode.NONE
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


@pytest.mark.skipif(
    not current_platform.support_static_graph_mode(),
    reason="Explicit overrides may be force-overwritten without static graph support.",
)
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
    assert config.compilation_config.pass_config.enable_qk_norm_rope_fusion is (
        current_platform.is_cuda_alike() or current_platform.is_xpu()
    )
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


def test_fusion_pass_op_priority():
    """This test checks that custom op enablement & IR op priority
    correctly control default fusions"""

    # Default config, O2, rms_norm+quant fusion disabled
    cfg1 = VllmConfig()
    assert not cfg1.compilation_config.pass_config.fuse_norm_quant

    # rms_norm manually enabled, O1, rms_norm+quant fusion enabled
    cfg2 = VllmConfig(
        optimization_level=OptimizationLevel.O1,
        compilation_config=CompilationConfig(
            custom_ops=["+rms_norm"],
        ),
    )
    assert cfg2.compilation_config.pass_config.fuse_norm_quant

    # using custom kernel for RMSNorm via IR:
    # Note that vLLM IR only supports the non-residual rms_norm for now;
    # soon this will be resolved.
    cfg3 = VllmConfig(
        kernel_config=KernelConfig(
            ir_op_priority=IrOpPriorityConfig(rms_norm=["vllm_c"])
        )
    )
    assert cfg3.compilation_config.pass_config.fuse_norm_quant

    # block-fp8 model should enable quant_fp8 automatically
    cfg4 = VllmConfig(model_config=ModelConfig("Qwen/Qwen3-4B-FP8"))
    assert "+quant_fp8" in cfg4.compilation_config.custom_ops
    assert cfg4.compilation_config.pass_config.fuse_norm_quant


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


def test_fault_tolerance_requires_single_api_server():
    """Fault tolerance assumes one AsyncMPClient manages all engines, so it
    is incompatible with API server scale-out (_api_process_count > 1)."""
    with pytest.raises(ValueError, match="single API server"):
        ParallelConfig(enable_fault_tolerance=True, _api_process_count=2)

    # Single API server (the FT-supported topology) is accepted.
    ParallelConfig(enable_fault_tolerance=True, _api_process_count=1)


def test_renderer_num_workers_with_mm_cache():
    """Disallow renderer_num_workers > 1 with the mm processor cache only for
    pooling models, whose preprocessing runs on the renderer workers."""
    mm_model = "Qwen/Qwen2-VL-2B-Instruct"

    # Should raise: pooling + multi-worker + cache enabled (default cache_gb=4)
    with pytest.raises(ValueError, match="renderer-num-workers"):
        ModelConfig(mm_model, runner="pooling", renderer_num_workers=4)

    # Should raise: pooling + multi-worker + explicit cache size
    with pytest.raises(ValueError, match="renderer-num-workers"):
        ModelConfig(
            mm_model,
            runner="pooling",
            renderer_num_workers=2,
            mm_processor_cache_gb=1.0,
        )

    # Should pass: pooling + multi-worker + cache disabled
    config = ModelConfig(
        mm_model, runner="pooling", renderer_num_workers=4, mm_processor_cache_gb=0
    )
    assert config.renderer_num_workers == 4

    # Should pass: generate models preprocess on the dedicated mm executor
    config = ModelConfig(mm_model, renderer_num_workers=4)
    assert config.renderer_num_workers == 4

    # Should pass: single worker + cache enabled (default)
    config = ModelConfig(mm_model, renderer_num_workers=1)
    assert config.renderer_num_workers == 1


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


def test_draft_sample_method_probabilistic_is_accepted():
    speculative_config = SpeculativeConfig(
        method="ngram",
        num_speculative_tokens=1,
        draft_sample_method="probabilistic",
    )
    assert speculative_config.draft_sample_method == "probabilistic"


def test_draft_sample_method_gumbel_is_rejected():
    with pytest.raises(ValidationError):
        SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=1,
            draft_sample_method="gumbel",
        )


def test_ir_op_priority_default():
    """Test that IR op priority defaults are set correctly."""
    from vllm.config.kernel import IrOpPriorityConfig

    # Assert default is applied to ops
    priority_config = IrOpPriorityConfig.with_default(["vllm_c", "native"])
    assert priority_config.rms_norm == ["vllm_c", "native"]
    assert priority_config.fused_add_rms_norm == ["vllm_c", "native"]

    # Assert single ops override the default
    priority_config = IrOpPriorityConfig.with_default(
        ["native"], rms_norm=["oink", "native"]
    )
    assert priority_config.rms_norm == ["oink", "native"]
    assert priority_config.fused_add_rms_norm == ["native"]


def test_ir_op_priority_str():
    """Test that passing a comma-delimited string works"""
    from vllm.config.kernel import IrOpPriorityConfig

    priority_config = IrOpPriorityConfig(rms_norm="vllm_c")
    assert priority_config.rms_norm == ["vllm_c"]

    priority_config = IrOpPriorityConfig(rms_norm="vllm_c,native")
    assert priority_config.rms_norm == ["vllm_c", "native"]

    priority_config = IrOpPriorityConfig(rms_norm=" native, vllm_c ")
    assert priority_config.rms_norm == ["native", "vllm_c"]

    with pytest.raises(pydantic.ValidationError):
        # must be list of only strings
        priority_config = IrOpPriorityConfig(rms_norm=["vllm_c", 4, "native"])


def test_ir_op_priority_ctx():
    """Test that the priority-setting context sets priority correctly."""
    from vllm import ir
    from vllm.config.kernel import IrOpPriorityConfig

    priority = IrOpPriorityConfig.with_default(["native"], rms_norm=["vllm_c"])
    priority2 = IrOpPriorityConfig.with_default(
        ["native"], fused_add_rms_norm=["vllm_c"]
    )
    with priority.set_priority():
        assert ir.ops.rms_norm.get_priority() == ["vllm_c", "native"]
        assert ir.ops.fused_add_rms_norm.get_priority() == ["native"]
        with priority2.set_priority():
            assert ir.ops.rms_norm.get_priority() == ["native"]
            assert ir.ops.fused_add_rms_norm.get_priority() == ["vllm_c", "native"]

        # context restored
        assert ir.ops.rms_norm.get_priority() == ["vllm_c", "native"]
        assert ir.ops.fused_add_rms_norm.get_priority() == ["native"]

        with pytest.raises(ValueError), priority2.set_priority():
            assert ir.ops.rms_norm.get_priority() == ["native"]
            assert ir.ops.fused_add_rms_norm.get_priority() == ["vllm_c", "native"]

            raise ValueError

        # context restored even after exception
        assert ir.ops.rms_norm.get_priority() == ["vllm_c", "native"]
        assert ir.ops.fused_add_rms_norm.get_priority() == ["native"]


def test_load_config_rejects_invalid_safetensors_load_strategy():
    with pytest.raises(pydantic.ValidationError):
        LoadConfig(safetensors_load_strategy="not_a_real_strategy")


@pytest.mark.parametrize("bad_load_format", [None, 123])
def test_load_config_rejects_non_string_load_format(bad_load_format):
    with pytest.raises(pydantic.ValidationError):
        LoadConfig(load_format=bad_load_format)


# A real Qwen3-0.6B model revision that is used in the tests below.
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


@patch("vllm.config.model.resolve_revision", return_value=ResolvedRevision(REVISION))
def test_revision_not_resolved_when_weights_differ_from_model(mock_resolve):
    model_weights = "unsloth/Qwen3-0.6B-GGUF:Q8_0"
    config = ModelConfig("Qwen/Qwen3-0.6B", model_weights=model_weights)
    assert config.revision is None


@patch("vllm.config.model.resolve_revision", return_value=ResolvedRevision(REVISION))
def test_revision_resolved_when_weights_match_model(mock_resolve):
    model = "Qwen/Qwen3-0.6B"
    config = ModelConfig(model)
    assert isinstance(config.revision, ResolvedRevision)
    assert config.revision.resolved == REVISION
    mock_resolve.assert_any_call(model, None, config.hf_token)
