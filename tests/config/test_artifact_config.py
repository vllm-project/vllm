# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any, cast

import pytest

from vllm.config import ArtifactConfig, VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.input_processor import InputProcessor

pytestmark = pytest.mark.cpu_test


def _config(
    *,
    pp: int = 1,
    dcp: int = 1,
    pcp: int = 1,
    connector: str | None = None,
    runner_type: str = "generate",
    is_moe: bool = True,
):
    return SimpleNamespace(
        model_config=SimpleNamespace(runner_type=runner_type, is_moe=is_moe),
        use_v2_model_runner=True,
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pp,
            decode_context_parallel_size=dcp,
            prefill_context_parallel_size=pcp,
        ),
        artifact_config=ArtifactConfig(enable_return_routed_experts=True),
        speculative_config=None,
        kv_transfer_config=(
            None
            if connector is None
            else SimpleNamespace(
                is_kv_transfer_instance=True,
                kv_connector=connector,
            )
        ),
    )


def test_artifact_config_defaults():
    config = ArtifactConfig()

    assert not config.enabled
    assert not config.enable_return_routed_experts
    assert config.max_bytes is None


def test_artifact_capture_changes_compilation_hash():
    disabled = ArtifactConfig()
    enabled = ArtifactConfig(enable_return_routed_experts=True)

    assert disabled.compute_hash() != enabled.compute_hash()


def test_legacy_routed_experts_flag_updates_artifact_config():
    args = EngineArgs(enable_return_routed_experts=True)

    assert args.artifact_config.enabled
    assert args.artifact_config.enable_return_routed_experts


def test_legacy_prompt_start_is_accepted():
    params = SamplingParams(routed_experts_prompt_start=3)

    assert params.routed_experts_prompt_start == 3


def test_prompt_start_is_ignored_when_artifact_capture_is_disabled():
    processor = cast(InputProcessor, object.__new__(InputProcessor))
    processor_mock = cast(Any, processor)
    processor_mock.vllm_config = SimpleNamespace(
        artifact_config=ArtifactConfig(),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_size_local=1,
            local_engines_only=False,
        ),
    )
    processor_mock._validate_params = lambda *_: None
    processor_mock._validate_lora = lambda *_: None
    processor_mock._validate_model_inputs = lambda *_: None
    processor_mock.generation_config_fields = {}
    processor_mock.renderer = SimpleNamespace(
        tokenizer=None,
        get_eos_token_id=lambda: None,
    )

    request = processor.process_inputs(
        "request",
        cast(Any, {"type": "tokens", "prompt_token_ids": [1]}),
        SamplingParams(max_tokens=1, routed_experts_prompt_start=1),
        ("generate",),
    )

    assert request.sampling_params is not None
    assert request.sampling_params.routed_experts_prompt_start == 1


def test_artifact_connector_rejects_resumable_streaming_input():
    processor = cast(InputProcessor, object.__new__(InputProcessor))
    processor_mock = cast(Any, processor)
    processor_mock.vllm_config = SimpleNamespace(
        artifact_config=ArtifactConfig(enable_return_routed_experts=True)
    )
    processor_mock._validate_params = lambda *_: None
    processor_mock._validate_lora = lambda *_: None

    with pytest.raises(VLLMValidationError, match="resumable streaming input"):
        processor.process_inputs(
            "request",
            cast(Any, {"type": "tokens", "prompt_token_ids": [1]}),
            SamplingParams(),
            ("generate",),
            resumable=True,
        )


def test_artifact_connector_requires_model_runner_v2():
    config = _config()
    config.use_v2_model_runner = False

    with pytest.raises(ValueError, match="requires Model Runner V2"):
        VllmConfig._verify_artifact_compatibility(config)


def test_artifact_connector_rejects_pooling_runner():
    with pytest.raises(ValueError, match="only supports generate runners"):
        VllmConfig._verify_artifact_compatibility(_config(runner_type="pooling"))


def test_artifact_connector_rejects_dense_model():
    with pytest.raises(ValueError, match="only supports MoE models"):
        VllmConfig._verify_artifact_compatibility(_config(is_moe=False))


def test_artifact_connector_rejects_adaptive_verification():
    config = _config()
    config.speculative_config = SimpleNamespace(enable_adaptive_verification=True)

    with pytest.raises(ValueError, match="adaptive speculative verification"):
        VllmConfig._verify_artifact_compatibility(config)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"pp": 2}, "pipeline parallelism"),
        ({"dcp": 2}, "context parallelism"),
        ({"pcp": 2}, "context parallelism"),
    ],
)
def test_artifact_connector_rejects_unsupported_topology(kwargs, error):
    with pytest.raises(ValueError, match=error):
        VllmConfig._verify_artifact_compatibility(_config(**kwargs))


def test_artifact_connector_rejects_kv_connectors():
    config = _config(connector="MooncakeConnector")

    with pytest.raises(ValueError, match="incompatible with KV connectors"):
        VllmConfig._verify_artifact_compatibility(config)


def test_artifact_guards_are_inactive_when_capture_is_disabled():
    config: Any = VllmConfig.__new__(VllmConfig)
    config.model_config = SimpleNamespace()
    config.artifact_config = ArtifactConfig()

    config._verify_artifact_compatibility()
