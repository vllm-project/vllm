# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vllm.build_profile import (
    BuildProfileMetadata,
    load_build_profile_metadata,
    validate_build_profile_capabilities,
)
from vllm.config import VllmConfig


def make_config(
    *,
    architecture: str = "RWKV7ForCausalLM",
    model: str = "/weights/rwkv7-g1g-7.2b-20260523-ctx8192.pth",
    quantization: str | None = None,
    multimodal: bool = False,
    speculative: bool = False,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    load_format: str = "auto",
    sleep_mode: bool = False,
    cumem_allocator: bool = False,
    cpu_offload_gb: float = 0,
    data_parallel_size: int = 1,
    prefill_context_parallel_size: int = 1,
    distributed_executor_backend: str | None = None,
    lora: bool = False,
    kv_connector: str | None = None,
    device_type: str = "cuda",
    runner_type: str = "generate",
) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            architecture=architecture,
            architectures=[architecture],
            model=model,
            quantization=quantization,
            quantization_config=None,
            is_multimodal_model=lambda: multimodal,
            runner_type=runner_type,
            enable_sleep_mode=sleep_mode,
            enable_cumem_allocator=cumem_allocator,
        ),
        load_config=SimpleNamespace(load_format=load_format),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            data_parallel_size=data_parallel_size,
            prefill_context_parallel_size=prefill_context_parallel_size,
            distributed_executor_backend=distributed_executor_backend,
        ),
        speculative_config=SimpleNamespace() if speculative else None,
        lora_config=SimpleNamespace() if lora else None,
        device_config=SimpleNamespace(device_type=device_type),
        offload_config=SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=cpu_offload_gb),
            prefetch=SimpleNamespace(offload_group_size=0),
        ),
        kv_transfer_config=(
            SimpleNamespace(kv_connector=kv_connector) if kv_connector else None
        ),
    )


RWKV_METADATA = BuildProfileMetadata(
    profile="rwkv",
    configured_targets=("_rapid_sampling", "rwkv7_ops"),
    external_projects=(),
    unrestricted=False,
    supported_architectures=("RWKV7ForCausalLM",),
    supported_weight_suffixes=(".pth",),
    supported_device_types=("cuda",),
    supported_runner_types=("generate",),
    supported_serving_features=(
        "text_generation",
        "openai_chat",
        "openai_completions",
        "streaming",
        "stop",
        "prometheus_metrics",
        "rapid_sampling",
    ),
    supported_tensor_parallel_sizes=(1,),
    supported_pipeline_parallel_sizes=(1,),
    supported_data_parallel_sizes=(1,),
)


def test_missing_metadata_defaults_to_full(tmp_path: Path) -> None:
    metadata = load_build_profile_metadata(tmp_path / "missing.json")

    assert metadata.profile == "full"


def test_rwkv_manifest_fixture_matches_runtime_contract() -> None:
    path = Path(__file__).with_name("fixtures") / "rwkv_build_profile.json"

    assert load_build_profile_metadata(path) == RWKV_METADATA


def test_rwkv_profile_accepts_declared_configuration() -> None:
    validate_build_profile_capabilities(make_config(), RWKV_METADATA)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"architecture": "LlamaForCausalLM"}, "RWKV7ForCausalLM"),
        ({"model": "/weights/model.safetensors"}, "raw .pth"),
        ({"quantization": "awq"}, "quantization"),
        ({"multimodal": True}, "multimodal"),
        ({"speculative": True}, "speculative"),
        ({"tensor_parallel_size": 2}, "TP=1"),
        ({"pipeline_parallel_size": 2}, "PP=1"),
        ({"load_format": "safetensors"}, "load format"),
        ({"sleep_mode": True}, "sleep mode"),
        ({"cumem_allocator": True}, "CuMem"),
        ({"cpu_offload_gb": 1}, "model offload"),
        ({"data_parallel_size": 2}, "DP=1"),
        ({"prefill_context_parallel_size": 2}, "prefill context parallel"),
        ({"distributed_executor_backend": "ray"}, "single-process"),
        ({"distributed_executor_backend": "mp"}, "single-process"),
        ({"distributed_executor_backend": "external_launcher"}, "single-process"),
        ({"lora": True}, "LoRA"),
        ({"kv_connector": "NixlConnector"}, "KV transfer"),
        ({"device_type": "cpu"}, "CUDA device"),
        ({"runner_type": "pooling"}, "generation runner"),
    ],
)
def test_rwkv_profile_rejects_unsupported_configuration(
    overrides: dict[str, Any], reason: str
) -> None:
    with pytest.raises(ValueError, match=rf"RWKV build profile.*{reason}.*full build"):
        validate_build_profile_capabilities(make_config(**overrides), RWKV_METADATA)


def test_full_profile_adds_no_rejection() -> None:
    metadata = BuildProfileMetadata(
        profile="full",
        configured_targets=(),
        external_projects=(),
        unrestricted=True,
    )

    validate_build_profile_capabilities(
        make_config(
            architecture="LlamaForCausalLM",
            model="/weights/model.safetensors",
            quantization="awq",
            multimodal=True,
            speculative=True,
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            load_format="safetensors",
        ),
        metadata,
    )


def test_vllm_config_reports_profile_error_before_model_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = VllmConfig()
    config.model_config = make_config(architecture="LlamaForCausalLM").model_config
    monkeypatch.setattr(
        "vllm.build_profile.get_build_profile_metadata", lambda: RWKV_METADATA
    )
    monkeypatch.setattr(
        VllmConfig,
        "try_verify_and_update_config",
        lambda _config: pytest.fail("model-specific verification ran first"),
    )

    with pytest.raises(ValueError, match="requires architecture RWKV7ForCausalLM"):
        config.__post_init__()
