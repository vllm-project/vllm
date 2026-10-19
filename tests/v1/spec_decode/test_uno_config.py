# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm import SamplingParams
from vllm.config import (
    AttentionConfig,
    CacheConfig,
    DeviceConfig,
    LoRAConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.exceptions import VLLMValidationError
from vllm.lora.request import LoRARequest
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.worker.gpu.spec_decode.uno import UNO_LORA_ID, validate_uno_config

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture
def uno_config_factory():
    """Build Uno configs without fetching a model or loading weights."""
    target = MagicMock(spec=ModelConfig)
    target.runner_type = "generate"
    target.is_diffusion = False
    target.is_multimodal_model = False
    target.is_encoder_decoder = False
    target.is_hybrid = False
    target.is_attention_free = False
    target.get_sliding_window.return_value = None
    target.get_vocab_size.return_value = 128
    parallel = ParallelConfig(distributed_executor_backend="uni")

    def make(**overrides):
        kwargs = dict(
            method="uno",
            uno_lora_path="test-adapter",
            num_speculative_tokens=8,
            target_model_config=target,
            target_parallel_config=parallel,
        )
        kwargs.update(overrides)
        return SpeculativeConfig(**kwargs)

    return make


def _serving_config(speculative_config, **overrides):
    """The parts of a VllmConfig that validate_uno_config reads."""
    config = SimpleNamespace(
        speculative_config=speculative_config,
        model_config=speculative_config.target_model_config,
        parallel_config=speculative_config.target_parallel_config,
        lora_config=LoRAConfig(max_loras=2, lora_dtype=torch.bfloat16),
        cache_config=CacheConfig(),
        kv_transfer_config=None,
        scheduler_config=SimpleNamespace(
            async_scheduling=True, max_num_seqs=4, max_num_batched_tokens=256
        ),
        # TRITON_ATTN is admitted on CUDA and ROCm alike, so these tests do not
        # depend on the platform they run on (ROCm refuses an unset backend).
        attention_config=AttentionConfig(backend="TRITON_ATTN"),
    )
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def test_uno_shares_target_and_uses_probabilistic_draft_rows(uno_config_factory):
    config = uno_config_factory()

    assert config.model == "uno"
    assert config.draft_model_config is config.target_model_config
    assert config.draft_parallel_config is config.target_parallel_config
    assert config.use_uno()
    assert config.parallel_drafting
    assert config.draft_sample_method == "probabilistic"
    assert config.rejection_sample_method == "standard"
    assert config.uno_mask_token_id == 128
    assert config.uno_noise_seed == 0
    validate_uno_config(_serving_config(config))


def test_uno_forces_probabilistic_draft_sampling(uno_config_factory):
    config = uno_config_factory(draft_sample_method="greedy")

    assert config.draft_sample_method == "probabilistic"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"uno_lora_path": None}, "uno_lora_path"),
        ({"uno_lora_path": " "}, "uno_lora_path"),
        ({"num_speculative_tokens": None}, "num_speculative_tokens"),
        ({"num_speculative_tokens": 0}, "greater than 0"),
        ({"uno_mask_token_id": 1}, "greater than 1"),
        ({"uno_mask_token_id": 129}, "target vocabulary size"),
        ({"model": "other-model"}, "shares the target"),
        ({"target_model_config": None}, "target model"),
        # SpeculativeConfig's own check keeps a heterogeneous vocabulary to
        # draft_model; Uno shares the target vocabulary.
        ({"use_heterogeneous_vocab": True}, "only works with method='draft_model'"),
    ],
)
def test_uno_config_rejects_invalid_fields(uno_config_factory, overrides, message):
    with pytest.raises(ValueError, match=message):
        uno_config_factory(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"draft_tensor_parallel_size": 2}, "target's tensor_parallel_size"),
        (
            {"rejection_sample_method": "synthetic", "synthetic_acceptance_length": 2},
            "standard",
        ),
        ({"num_speculative_tokens_per_batch_size": [(1, 8, 4)]}, "fixed"),
        ({"use_local_argmax_reduction": True}, "local argmax"),
        ({"enable_adaptive_verification": True}, "Adaptive verification"),
    ],
)
def test_uno_rejects_incompatible_options(uno_config_factory, overrides, message):
    config = _serving_config(uno_config_factory(**overrides))
    with pytest.raises(ValueError, match=message):
        validate_uno_config(config)


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("runner_type", "pooling", "text-only"),
        ("is_diffusion", True, "text-only"),
        ("is_multimodal_model", True, "text-only"),
        ("is_encoder_decoder", True, "text-only"),
        ("is_hybrid", True, "full attention"),
        ("is_attention_free", True, "full attention"),
    ],
)
def test_uno_rejects_unsupported_model_types(
    uno_config_factory, attribute, value, message
):
    config = uno_config_factory()
    setattr(config.target_model_config, attribute, value)
    with pytest.raises(ValueError, match=message):
        validate_uno_config(_serving_config(config))


def test_uno_rejects_sliding_window_attention(uno_config_factory):
    config = uno_config_factory()
    config.target_model_config.get_sliding_window.return_value = 64
    with pytest.raises(ValueError, match="full attention"):
        validate_uno_config(_serving_config(config))


@pytest.mark.parametrize(
    "parallelism",
    [
        "pipeline_parallel_size",
        "data_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ],
)
def test_uno_rejects_parallel_execution(uno_config_factory, parallelism):
    """Every parallelism except tensor parallelism is still refused."""
    config = uno_config_factory()
    setattr(config.target_parallel_config, parallelism, 2)
    with pytest.raises(ValueError, match="not supported"):
        validate_uno_config(_serving_config(config))


def test_uno_accepts_tensor_parallelism(uno_config_factory):
    """Tensor parallelism is supported; the draft shares the target's degree.

    Measured on two RTX 5090s at TP=2: Uno's greedy tokens equal the plain
    engine's on 4 of 4 prompts against two separate plain runs, the plain engine
    reproduces itself at TP=2, and Uno equals plain at TP=1. The only divergence
    was plain at TP=1 against plain at TP=2 (prompt 1, token 5), which is the
    engine's reduction order rather than anything Uno does.
    """
    parallel = uno_config_factory().target_parallel_config
    parallel.tensor_parallel_size = 2

    config = uno_config_factory(target_parallel_config=parallel)
    assert config.draft_parallel_config is config.target_parallel_config
    validate_uno_config(_serving_config(config))

    # The draft may name the same degree explicitly, and nothing else.
    config = uno_config_factory(
        target_parallel_config=parallel, draft_tensor_parallel_size=2
    )
    validate_uno_config(_serving_config(config))
    config = uno_config_factory(
        target_parallel_config=parallel, draft_tensor_parallel_size=1
    )
    with pytest.raises(ValueError, match="target's tensor_parallel_size"):
        validate_uno_config(_serving_config(config))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"lora_config": None}, "enable-lora"),
        (
            {"lora_config": LoRAConfig(max_loras=1, lora_dtype=torch.bfloat16)},
            r"max_loras >= 2",
        ),
        ({"cache_config": CacheConfig(kv_sharing_fast_prefill=True)}, "KV sharing"),
        ({"cache_config": CacheConfig(kv_offloading_size=1.0)}, "KV cache transfer"),
        ({"kv_transfer_config": SimpleNamespace()}, "KV cache transfer"),
        (
            {
                "scheduler_config": SimpleNamespace(
                    async_scheduling=False, max_num_seqs=4, max_num_batched_tokens=256
                )
            },
            "asynchronous scheduling",
        ),
        (
            {
                "scheduler_config": SimpleNamespace(
                    async_scheduling=True, max_num_seqs=4, max_num_batched_tokens=31
                )
            },
            "max_num_batched_tokens",
        ),
    ],
)
def test_uno_rejects_unsupported_serving_options(
    uno_config_factory, overrides, message
):
    config = _serving_config(uno_config_factory(), **overrides)
    with pytest.raises(ValueError, match=message):
        validate_uno_config(config)


@pytest.mark.parametrize(
    ("rocm", "backend", "per_kind", "refused"),
    [
        (False, None, None, None),
        (False, "FLASH_ATTN", None, None),
        (False, "TRITON_ATTN", None, None),
        (False, "FLASHINFER", None, "not FLASHINFER"),
        (False, "FLEX_ATTENTION", None, "not FLEX_ATTENTION"),
        (False, None, "FLASHINFER", "not FLASHINFER"),
        (True, "TRITON_ATTN", None, None),
        (True, None, None, "requires an explicit tested attention backend"),
        (True, "ROCM_ATTN", None, "not ROCM_ATTN"),
        (True, "FLASH_ATTN", None, "not FLASH_ATTN"),
    ],
)
def test_uno_refuses_an_untested_attention_backend_at_config_time(
    uno_config_factory, monkeypatch, rocm, backend, per_kind, refused
):
    """An untested backend the config asks for is refused before any load.

    CUDA admits FLASH_ATTN and TRITON_ATTN (and the platform default, which
    load_model checks per layer); ROCm admits only an explicit TRITON_ATTN,
    because its default backend is not one Uno's tests run. A per-kind
    override is checked like the global backend.
    """
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.worker.gpu.spec_decode import uno

    platform = SimpleNamespace(is_cuda=lambda: not rocm, is_rocm=lambda: rocm)
    monkeypatch.setattr(uno, "current_platform", platform)
    attention = SimpleNamespace(
        backend=None if backend is None else AttentionBackendEnum[backend],
        backend_per_kind=(
            {} if per_kind is None else {"full": AttentionBackendEnum[per_kind]}
        ),
    )
    config = _serving_config(uno_config_factory(), attention_config=attention)
    if refused is None:
        validate_uno_config(config)
    else:
        with pytest.raises(ValueError, match=refused):
            validate_uno_config(config)


def test_uno_refuses_flashinfer_before_the_speculator_is_built(
    uno_config_factory, monkeypatch
):
    """The config-time refusal fires in UnoSpeculator.__init__, which the
    runner calls before it loads any weights, and before the parent
    speculator allocates its buffers."""
    from vllm.v1.worker.gpu.spec_decode import uno
    from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

    platform = SimpleNamespace(is_cuda=lambda: True, is_rocm=lambda: False)
    monkeypatch.setattr(uno, "current_platform", platform)

    def built(*_args, **_kwargs):
        raise AssertionError("the speculator was built before the refusal")

    monkeypatch.setattr(DraftModelSpeculator, "__init__", built)
    config = _serving_config(
        uno_config_factory(), attention_config=AttentionConfig(backend="FLASHINFER")
    )
    with pytest.raises(ValueError, match="not FLASHINFER"):
        uno.UnoSpeculator(config, torch.device("cuda"))


def test_speculator_factory_builds_uno(uno_config_factory, monkeypatch):
    """init_speculator routes method='uno' past every other method's branch."""
    from vllm.v1.worker.gpu.spec_decode import init_speculator, uno

    built = []

    def fake_speculator(config, device):
        built.append((config, device))
        return "uno"

    monkeypatch.setattr(uno, "UnoSpeculator", fake_speculator)
    constructors = [
        (import_module(f"vllm.v1.worker.gpu.spec_decode.{module}"), name)
        for module, name in (
            ("extract_hidden_states", "ExtractHiddenStatesSpeculator"),
            ("lilicorr.speculator", "LiLiCorrSpeculator"),
            ("dflash2.speculator", "DFlash2Speculator"),
            ("dflash.speculator", "DFlashSpeculator"),
            ("dspark.speculator", "DSparkSpeculator"),
            ("gemma4.speculator", "Gemma4Speculator"),
            ("multi_module_mtp.speculator", "MultiModuleMTPSpeculator"),
            ("mtp.speculator", "MTPSpeculator"),
            ("eagle.speculator", "EagleSpeculator"),
            ("standalone_ar.speculator", "StandaloneARSpeculator"),
            ("ngram.speculator", "NgramGPUSpeculator"),
        )
    ]
    for constructor, name in constructors:
        monkeypatch.setattr(constructor, name, lambda *args, name=name: name)
    config = _serving_config(uno_config_factory())
    device = torch.device("cpu")

    assert init_speculator(config, device, None) == "uno"
    assert built == [(config, device)]


def test_uno_refuses_user_lora_serving(uno_config_factory, monkeypatch):
    """--enable-lora serves only Uno's adapter; user LoRA serving is refused."""
    config = _serving_config(uno_config_factory())
    validate_uno_config(config)

    monkeypatch.setenv("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "1")
    with pytest.raises(ValueError, match="does not support user LoRA adapters"):
        validate_uno_config(config)
    monkeypatch.delenv("VLLM_ALLOW_RUNTIME_LORA_UPDATING")

    config.lora_config = SimpleNamespace(
        max_loras=2, default_mm_loras={"audio": "user-adapter"}
    )
    with pytest.raises(ValueError, match="does not support user LoRA adapters"):
        validate_uno_config(config)


@pytest.mark.parametrize("lora_int_id", [1, UNO_LORA_ID])
def test_uno_refuses_request_lora_before_queueing(
    uno_config_factory, monkeypatch, lora_int_id
):
    """A request carrying a LoRA adapter is refused in process_inputs.

    This covers a user adapter that reuses Uno's reserved id: no request with
    a LoRA reaches the engine, so target rows never carry an adapter and Uno's
    adapter is unreachable from requests. The refusal is a validation error
    for that request, not an engine failure.
    """
    processor = object.__new__(InputProcessor)
    processor.speculative_config = uno_config_factory()
    processor.lora_config = LoRAConfig(max_loras=2, lora_dtype=torch.bfloat16)
    processor.renderer = SimpleNamespace(tokenizer=None)
    monkeypatch.setattr(processor, "_validate_params", lambda params, tasks: None)
    request = LoRARequest("user", lora_int_id, "/user/adapter")

    with pytest.raises(VLLMValidationError, match="request-specific LoRA"):
        processor.process_inputs(
            "req-0", "hello", SamplingParams(), (), lora_request=request
        )

    # Without Uno the same request passes LoRA validation and goes on to the
    # rest of preprocessing (which this stub processor does not have).
    processor.speculative_config = None
    with pytest.raises(AttributeError, match="vllm_config"):
        processor.process_inputs(
            "req-0", "hello", SamplingParams(), (), lora_request=request
        )


def _make_vllm_uno_config(uno_config_factory):
    """Exercise runner selection without a model or its LoRA validation."""
    return VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        speculative_config=uno_config_factory(),
        scheduler_config=SchedulerConfig(
            max_model_len=2048,
            max_num_seqs=4,
            max_num_batched_tokens=256,
            is_encoder_decoder=False,
            async_scheduling=None,
        ),
    )


def test_uno_uses_native_async_v2_and_reserves_k_lookahead(
    uno_config_factory, monkeypatch
):
    monkeypatch.setattr("vllm.config.vllm.HAS_TRITON", True)
    monkeypatch.setattr("vllm.config.vllm.envs.VLLM_USE_V2_MODEL_RUNNER", True)
    # This is a CUDA-only contract.  Keep the config test runnable in the CPU
    # Docker suite without importing or initializing a CUDA device; CPU's
    # platform hook intentionally disables async scheduling.
    from vllm.platforms import current_platform

    monkeypatch.setattr(
        current_platform, "apply_config_platform_defaults", lambda _config: None
    )
    monkeypatch.setattr(
        current_platform, "check_and_update_config", lambda _config: None
    )
    config = _make_vllm_uno_config(uno_config_factory)

    assert config.scheduler_config.async_scheduling is True
    assert config.use_v2_model_runner
    assert config.speculative_config.max_num_new_slots_for_drafting == 7
    assert config.num_lookahead_tokens == 8
    assert config.uniform_decode_query_len == 9


def test_uno_rejects_forced_v1_runner(uno_config_factory, monkeypatch):
    monkeypatch.setattr("vllm.config.vllm.envs.VLLM_USE_V2_MODEL_RUNNER", False)
    with pytest.raises(ValueError, match="Uno speculative decoding"):
        _make_vllm_uno_config(uno_config_factory)
