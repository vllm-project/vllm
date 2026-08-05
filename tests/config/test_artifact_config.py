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
    shm_dir: str = "/dev/shm/vllm-artifacts",
    connector: str | None = None,
    role: str = "kv_both",
    spec_name: str | None = None,
):
    extra_config = {} if spec_name is None else {"spec_name": spec_name}
    return SimpleNamespace(
        model_config=SimpleNamespace(),
        use_v2_model_runner=True,
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pp,
            decode_context_parallel_size=dcp,
            prefill_context_parallel_size=pcp,
        ),
        artifact_config=ArtifactConfig(
            enable_return_routed_experts=True,
            shm_dir=shm_dir,
        ),
        kv_transfer_config=(
            None
            if connector is None
            else SimpleNamespace(
                is_kv_transfer_instance=True,
                kv_connector=connector,
                kv_role=role,
                kv_connector_extra_config=extra_config,
            )
        ),
    )


def test_artifact_config_defaults():
    config = ArtifactConfig()

    assert not config.enabled
    assert not config.enable_return_routed_experts
    assert config.shm_dir == "/dev/shm/vllm-artifacts"
    assert config.max_shm_bytes is None


def test_legacy_routed_experts_flag_updates_artifact_config():
    args = EngineArgs(enable_return_routed_experts=True)

    assert args.artifact_config.enabled
    assert args.artifact_config.enable_return_routed_experts


def test_legacy_prompt_start_is_accepted():
    params = SamplingParams(routed_experts_prompt_start=3)

    assert params.routed_experts_prompt_start == 3


def test_negative_prompt_start_is_rejected():
    with pytest.raises(VLLMValidationError, match="must be non-negative"):
        SamplingParams(routed_experts_prompt_start=-1)


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


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"pp": 2}, "pipeline parallelism"),
        ({"dcp": 2}, "context parallelism"),
        ({"pcp": 2}, "context parallelism"),
        ({"shm_dir": "/tmp/artifacts"}, "requires shm_dir under /dev/shm"),
    ],
)
def test_artifact_connector_rejects_unsupported_topology(kwargs, error):
    with pytest.raises(ValueError, match=error):
        VllmConfig._verify_artifact_compatibility(_config(**kwargs))


@pytest.mark.parametrize(
    ("connector", "spec_name"),
    [
        ("OffloadingConnector", "CPUOffloadingSpec"),
        ("OffloadingConnector", "TieringOffloadingSpec"),
        ("OffloadingConnector", "OutOfTreeOffloadingSpec"),
        ("NixlConnector", None),
        ("MooncakeConnector", None),
    ],
)
def test_artifact_connector_is_independent_of_kv_connector_implementation(
    connector, spec_name
):
    VllmConfig._verify_artifact_compatibility(
        _config(
            connector=connector,
            spec_name=spec_name,
        )
    )


@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
def test_artifact_connector_rejects_pd_disaggregation(role):
    with pytest.raises(ValueError, match="kv_role=kv_both"):
        VllmConfig._verify_artifact_compatibility(
            _config(
                connector="OffloadingConnector",
                role=role,
                spec_name="CPUOffloadingSpec",
            )
        )


def test_artifact_guards_are_inactive_when_capture_is_disabled():
    config: Any = VllmConfig.__new__(VllmConfig)
    config.model_config = SimpleNamespace()
    config.artifact_config = ArtifactConfig()

    config._verify_artifact_compatibility()
