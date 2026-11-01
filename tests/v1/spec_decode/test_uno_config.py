# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import regex as re
import torch

from tests.v1.e2e.spec_decode.uno_model_configs import write_tiny_model_config
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
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.worker.gpu.spec_decode.uno import UNO_LORA_ID, validate_uno_config
from vllm.v1.worker.gpu.spec_decode.uno_config import register

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture
def real_config_factory(tmp_path, monkeypatch):
    """Real local model/configs, substituting only the CUDA platform boundary."""
    from vllm.platforms import current_platform

    register()
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setattr(
        current_platform, "apply_config_platform_defaults", lambda _config: None
    )
    monkeypatch.setattr(
        current_platform, "check_and_update_config", lambda _config: None
    )

    def make(method="uno", *, model_type="opt", model_overrides=None, **overrides):
        model_path = write_tiny_model_config(
            tmp_path / model_type, model_type, **(model_overrides or {})
        )
        if model_type == "rswa":
            monkeypatch.setenv("VLLM_MLA_DISABLE", "1")
        target = ModelConfig(
            model=str(model_path), skip_tokenizer_init=True, dtype="bfloat16"
        )
        options = (
            {"speculator_options": {"uno_lora_path": "test-adapter"}}
            if method in ("uno", "uno_alias")
            else {}
        )
        spec = (
            None
            if method is None
            else SpeculativeConfig(
                method=method,
                num_speculative_tokens=3,
                target_model_config=target,
                target_parallel_config=ParallelConfig(),
                **options,
            )
        )
        kwargs = dict(
            model_config=target,
            speculative_config=spec,
            device_config=DeviceConfig(device="cpu"),
            lora_config=LoRAConfig(max_loras=2, lora_dtype=torch.bfloat16),
            attention_config=AttentionConfig(backend="TRITON_ATTN"),
            scheduler_config=SchedulerConfig(
                max_model_len=64,
                max_num_seqs=2,
                max_num_batched_tokens=64,
                is_encoder_decoder=False,
                async_scheduling=True,
            ),
        )
        kwargs.update(overrides)
        return VllmConfig(**kwargs)

    return make


@pytest.fixture
def uno_config_factory():
    """Build Uno configs without fetching a model or loading weights."""
    register()
    target = MagicMock(spec=ModelConfig)
    target.runner_type = "generate"
    target.is_diffusion = False
    target.is_multimodal_model = False
    target.is_encoder_decoder = False
    target.is_hybrid = False
    target.is_attention_free = False
    target.use_mla = False
    target.attention_chunk_size = None
    target.rswa_window = None
    target.hf_text_config = SimpleNamespace()
    target.get_sliding_window.return_value = None
    target.get_vocab_size.return_value = 128
    parallel = ParallelConfig(distributed_executor_backend="uni")

    def make(**overrides):
        kwargs: dict[str, Any] = dict(
            method="uno",
            speculator_options={"uno_lora_path": "test-adapter"},
            num_speculative_tokens=8,
            target_model_config=target,
            target_parallel_config=parallel,
        )
        options = dict(kwargs["speculator_options"])
        for name in ("uno_lora_path", "uno_mask_token_id", "uno_noise_seed"):
            if name in overrides:
                options[name] = overrides.pop(name)
        kwargs["speculator_options"] = options
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
    assert config.custom_method == "uno"
    assert not config.parallel_drafting
    assert config.draft_sample_method == "probabilistic"
    assert config.rejection_sample_method == "standard"
    assert config.speculator_config.options.uno_mask_token_id == 128
    assert config.speculator_config.options.uno_noise_seed == 0
    validate_uno_config(_serving_config(config))


def test_uno_forces_probabilistic_draft_sampling(uno_config_factory):
    config = uno_config_factory(draft_sample_method="greedy")

    assert config.draft_sample_method == "probabilistic"


def test_uno_rejects_unknown_options(uno_config_factory):
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        uno_config_factory(
            speculator_options={"uno_lora_path": "test-adapter", "noise_sead": 1}
        )


def test_uno_validates_options_through_documented_hook(real_config_factory):
    from vllm.v1.worker.gpu.spec_decode.uno_config import UnoSpeculatorConfig

    spec = real_config_factory().speculative_config
    spec.speculator_options = {"uno_lora_path": " "}
    with pytest.raises(ValueError, match="requires uno_lora_path"):
        UnoSpeculatorConfig.apply_options(spec)


def test_uno_plugin_registers_in_a_fresh_process():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from vllm.plugins import load_general_plugins; "
            "load_general_plugins(); "
            "from vllm.v1.worker.gpu.spec_decode.speculator_config import "
            "get_custom_speculator_config_class; "
            "config = get_custom_speculator_config_class('uno'); "
            "assert config is not None, 'Uno plugin was not discovered'; "
            "assert config.__name__ == 'UnoSpeculatorConfig'; "
            "print('Uno plugin discovered: UnoSpeculatorConfig')",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout and result.stdout.rstrip().endswith(
        "Uno plugin discovered: UnoSpeculatorConfig"
    ), result.stdout
    print(result.stdout, end="")


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
        ("rswa_window", 16, "reference sliding-window attention"),
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


@pytest.mark.parametrize("flag", ["enable_eplb", "enable_elastic_ep"])
def test_uno_refuses_expert_balancing_before_loading(real_config_factory, flag):
    """Check config admission without initializing GPU parallel groups.

    CPU construction cannot enable expert balancing, so set its flags after
    ParallelConfig validation. The engine test checks actual runner ordering.
    """
    config = real_config_factory()
    setattr(config.parallel_config, flag, True)
    with pytest.raises(ValueError, match="EPLB or elastic expert parallelism"):
        validate_uno_config(config)


def test_uno_refuses_mla_before_loading(real_config_factory):
    """A real MLA model cannot use Uno's full-attention backend selection."""
    config = real_config_factory(model_type="deepseek_v2")
    assert config.model_config.use_mla
    with pytest.raises(ValueError, match="^Uno does not support MLA attention$"):
        validate_uno_config(config)


@pytest.mark.parametrize(
    ("model_type", "message"),
    [
        ("llama4_text", "Uno does not support chunked-local attention"),
        ("openpangu", "Uno does not support static-sink attention"),
        ("rswa", "Uno does not support reference sliding-window attention"),
    ],
)
def test_uno_refuses_specialized_attention_before_construction(
    real_config_factory, model_type, message
):
    """Known attention kinds are refused during config admission."""
    config = real_config_factory(model_type=model_type)
    if model_type == "llama4_text":
        assert config.model_config.attention_chunk_size == 16
        assert not config.model_config.use_mla
    elif model_type == "openpangu":
        assert config.model_config.architecture == "PanguEmbeddedForCausalLM"
        assert config.model_config.hf_text_config.param_sink_number == 16
    else:
        assert not config.model_config.use_mla
        assert config.model_config.rswa_window == 16
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        validate_uno_config(config)


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
        (False, None, None, "explicit tested full-attention backend"),
        (False, "FLASH_ATTN", None, None),
        (False, "TRITON_ATTN", None, None),
        (False, "FLASHINFER", None, "not FLASHINFER"),
        (False, "FLEX_ATTENTION", None, "not FLEX_ATTENTION"),
        (False, None, "FLASHINFER", "not FLASHINFER"),
        (True, "TRITON_ATTN", None, None),
        (True, None, None, "explicit tested full-attention backend"),
        (True, "ROCM_ATTN", None, "not ROCM_ATTN"),
        (True, "FLASH_ATTN", None, "not FLASH_ATTN"),
    ],
)
def test_uno_refuses_an_untested_attention_backend_at_config_time(
    uno_config_factory, monkeypatch, rocm, backend, per_kind, refused
):
    """An untested backend the config asks for is refused before any load.

    CUDA admits explicit FLASH_ATTN and TRITON_ATTN; ROCm admits explicit
    TRITON_ATTN. Only the effective full-attention override is relevant.
    """
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.worker.gpu.spec_decode import uno

    platform = SimpleNamespace(is_cuda=lambda: not rocm, is_rocm=lambda: rocm)
    monkeypatch.setattr(uno, "current_platform", platform)
    attention = SimpleNamespace(
        backend=None if backend is None else AttentionBackendEnum[backend],
        backend_per_kind=(
            {}
            if per_kind is None
            else {"full_attention": AttentionBackendEnum[per_kind]}
        ),
    )
    config = _serving_config(uno_config_factory(), attention_config=attention)
    if refused is None:
        validate_uno_config(config)
    else:
        with pytest.raises(ValueError, match=refused):
            validate_uno_config(config)


@pytest.mark.parametrize("rocm", [False, True])
@pytest.mark.parametrize("unrelated_override", [False, True])
def test_uno_requires_effective_backend_before_loading(
    real_config_factory, monkeypatch, rocm, unrelated_override
):
    """The production factory refuses before a model can be loaded."""
    from vllm.v1.worker.gpu.spec_decode import init_speculator, uno

    config = real_config_factory(
        attention_config=AttentionConfig(
            backend_per_kind={"sliding_window": AttentionBackendEnum.TRITON_ATTN}
            if unrelated_override
            else {},
        )
    )
    # CPU cannot initialize a CUDA device; substitute only device/platform checks.
    monkeypatch.setattr(uno, "uno_device_refusal", lambda _device: None)
    monkeypatch.setattr(uno.current_platform, "is_rocm", lambda: rocm)
    with pytest.raises(ValueError, match="explicit tested full-attention backend"):
        init_speculator(config, torch.device("cpu"), None)


def test_uno_full_attention_override_takes_precedence(real_config_factory):
    config = real_config_factory(
        attention_config=AttentionConfig(
            backend="FLASHINFER",
            backend_per_kind={"full_attention": AttentionBackendEnum.TRITON_ATTN},
        )
    )
    validate_uno_config(config)


def test_uno_token_budget_covers_dummy_requests_even_with_active_cap(
    real_config_factory,
):
    config = real_config_factory(
        scheduler_config=SchedulerConfig(
            max_model_len=64,
            max_num_seqs=16,
            max_num_active_seqs=1,
            max_num_batched_tokens=32,
            is_encoder_decoder=False,
            async_scheduling=True,
        )
    )
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        validate_uno_config(config)


@pytest.mark.parametrize("key", ["spec-decode-uno", "spec-decode-uno-nightly"])
def test_uno_ci_steps_cover_core_dependencies(key):
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[3]
    pipeline = yaml.safe_load(
        (root / ".buildkite/test_areas/spec_decode.yaml").read_text()
    )
    step = next(s for s in pipeline["steps"] if s.get("key") == key)
    required = {
        "vllm/v1/spec_decode/",
        "vllm/v1/worker/gpu/spec_decode/",
        "tests/v1/e2e/spec_decode/",
        "vllm/v1/worker/gpu/model_runner.py",
        "vllm/v1/engine/input_processor.py",
        "pyproject.toml",
        "vllm/v1/worker/lora_model_runner_mixin.py",
        "vllm/lora/",
        "vllm/v1/core/",
        "tests/v1/spec_decode/test_uno_config.py",
    }
    assert required <= set(step["source_file_dependencies"])
    if key == "spec-decode-uno":
        assert (
            "pytest -v -s v1/spec_decode/test_uno_config.py "
            "-k plugin_registers_in_a_fresh_process"
        ) in step["commands"]


def test_uno_ci_selects_every_engine_regression():
    """Execute CI's selectors: dependency coverage alone does not run a test."""
    import shlex
    import subprocess
    import sys
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[3]
    target = "tests/v1/e2e/spec_decode/test_uno.py"
    pipeline = yaml.safe_load(
        (root / ".buildkite/test_areas/spec_decode.yaml").read_text()
    )

    def collect(arguments):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                target,
                *arguments,
            ],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return {
            line
            for line in result.stdout.splitlines()
            if line.startswith(target + "::")
        }

    expected = collect([])
    assert expected
    selected = set()
    for step in pipeline["steps"]:
        if step.get("key") not in ("spec-decode-uno", "spec-decode-uno-nightly"):
            continue
        for command in step["commands"]:
            argv = shlex.split(command)
            if "v1/e2e/spec_decode/test_uno.py" in argv:
                nodes = collect(["-k", argv[argv.index("-k") + 1]])
                print(f"{step['key']} selected {len(nodes)} Uno engine cases")
                selected.update(nodes)
    assert expected <= selected, sorted(expected - selected)


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


@pytest.mark.parametrize("sink", [None, 0])
def test_speculator_factory_builds_uno(real_config_factory, monkeypatch, sink):
    """Null or zero sink counts do not block real config/buffer construction."""
    from vllm.v1.worker.gpu.spec_decode import init_speculator, uno

    monkeypatch.setattr(uno, "uno_device_refusal", lambda _device: None)
    config = real_config_factory(model_overrides={"param_sink_number": sink})
    proposer = init_speculator(config, torch.device("cpu"), None)

    assert type(proposer) is uno.UnoSpeculator
    assert proposer.uno_options.uno_mask_token_id == 128
    assert proposer.draft_tokens.shape == (2, 3)


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
@pytest.mark.parametrize("method", ["uno", "uno_alias"])
def test_uno_refuses_request_lora_before_queueing(
    real_config_factory, lora_int_id, method
):
    """Real preprocessing refuses request adapters, including aliased Uno configs."""
    from vllm.v1.worker.gpu.spec_decode.speculator_config import (
        register_speculative_method,
    )
    from vllm.v1.worker.gpu.spec_decode.uno_config import UnoSpeculatorConfig

    register_speculative_method("uno_alias", UnoSpeculatorConfig)
    processor = InputProcessor(real_config_factory(method))
    request = LoRARequest("user", lora_int_id, "/user/adapter")
    prompt = {"prompt_token_ids": [1, 2]}

    with pytest.raises(VLLMValidationError, match="request-specific LoRA"):
        processor.process_inputs(
            "req-0",
            prompt,
            SamplingParams(max_tokens=1),
            ("generate",),
            lora_request=request,
        )

    ordinary = InputProcessor(real_config_factory(method="ngram"))
    accepted = ordinary.process_inputs(
        "req-0",
        prompt,
        SamplingParams(max_tokens=1),
        ("generate",),
        lora_request=request,
    )
    assert accepted.lora_request == request
    assert accepted.prompt_token_ids == [1, 2]


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
    with pytest.raises(ValueError, match="custom speculative method"):
        _make_vllm_uno_config(uno_config_factory)
