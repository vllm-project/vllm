# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import torch
from transformers import OPTConfig

from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.config.quantization import QuantizationConfigArgs
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.cache_prefix import (  # noqa: E501
    resolve_cache_prefix,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.layout import (  # noqa: E501
    KeyMetadata,
    PoolKey,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig

if TYPE_CHECKING:
    from tests.conftest import VllmRunner


@pytest.fixture
def model_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    OPTConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        ffn_dim=128,
        word_embed_proj_dim=64,
    ).save_pretrained(tmp_path)
    monkeypatch.setenv("VLLM_IMAGE_TAG", "vllm:test-build-1")
    monkeypatch.setenv("VLLM_BUILD_COMMIT", "unknown")
    return tmp_path


def make_config(model_path: Path, **kwargs: Any) -> VllmConfig:
    return VllmConfig(
        model_config=ModelConfig(model=str(model_path), dtype="float16"),
        device_config=DeviceConfig("cpu"),
        kv_transfer_config=KVTransferConfig(
            kv_connector="MooncakeStoreConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"cache_prefix": "auto"},
        ),
        **kwargs,
    )


def pool_key(config: VllmConfig) -> str:
    return PoolKey(
        KeyMetadata("model", 0, 0, 0, 0, cache_prefix=resolve_cache_prefix(config)),
        "same-prompt-hash",
    ).to_string()


@pytest.mark.parametrize("prefix", ["", "custom-fp4", "AUTO"])
def test_literal_prefix_keeps_existing_keys(model_path: Path, prefix: str) -> None:
    config = make_config(model_path)
    assert config.kv_transfer_config is not None
    config.kv_transfer_config.kv_connector_extra_config["cache_prefix"] = prefix
    expected = PoolKey(
        KeyMetadata("model", 0, 0, 0, 0, cache_prefix=prefix), "same-prompt-hash"
    ).to_string()
    assert pool_key(config) == expected


def test_replicas_share_keys_despite_runtime_identity(model_path: Path) -> None:
    first, second = make_config(model_path), make_config(model_path)
    assert first.kv_transfer_config is not None
    assert second.kv_transfer_config is not None
    assert first.instance_id != second.instance_id
    assert first.kv_transfer_config.engine_id != second.kv_transfer_config.engine_id
    second.parallel_config.rank = 1
    second.parallel_config.master_port += 1
    second.kv_transfer_config.kv_role = "kv_consumer"
    second.kv_transfer_config.kv_connector_extra_config["lookup_rpc_port"] = 54321
    second.cache_config.num_gpu_blocks = 200
    assert pool_key(first) == pool_key(second)


@pytest.mark.parametrize(
    "variable", ["VLLM_IMAGE_TAG", "VLLM_BUILD_COMMIT", "VLLM_BATCH_INVARIANT"]
)
def test_code_or_compute_environment_change_isolates_keys(
    model_path: Path, monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    config = make_config(model_path)
    old_key = pool_key(config)
    monkeypatch.setenv(variable, "1")
    assert pool_key(config) != old_key


@pytest.mark.parametrize(
    "cache_kwargs",
    [
        {"cache_dtype": "fp8"},
        {"block_size": 32},
        {"prefix_caching_hash_algo": "sha256_cbor"},
    ],
)
def test_cache_format_change_isolates_keys(
    model_path: Path, cache_kwargs: dict[str, Any]
) -> None:
    assert pool_key(make_config(model_path)) != pool_key(
        make_config(model_path, cache_config=CacheConfig(**cache_kwargs))
    )


def test_model_config_and_loader_changes_isolate_keys(model_path: Path) -> None:
    config = make_config(model_path)
    old_key = pool_key(config)
    config.model_config.hf_config.max_position_embeddings = 4096
    assert pool_key(config) != old_key
    config = make_config(model_path)
    config.load_config.load_format = "dummy"
    assert pool_key(config) != old_key


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("dtype", torch.bfloat16),
        ("quantization", "fp8"),
        ("quantization", "modelopt_fp4"),
        ("seed", 123),
        ("enforce_eager", True),
        ("revision", "new-checkpoint"),
    ],
)
def test_model_compute_argument_change_isolates_keys(
    model_path: Path, name: str, value: object
) -> None:
    config = make_config(model_path)
    old_key = pool_key(config)
    setattr(config.model_config, name, value)
    assert pool_key(config) != old_key


def test_store_payload_option_change_isolates_keys(model_path: Path) -> None:
    config = make_config(model_path)
    assert config.kv_transfer_config is not None
    old_key = pool_key(config)
    config.kv_transfer_config.kv_connector_extra_config[
        "enable_cross_layers_blocks"
    ] = True
    assert pool_key(config) != old_key


def test_scheduler_capacity_changes_preserve_keys(model_path: Path) -> None:
    configs = [
        make_config(
            model_path,
            scheduler_config=SchedulerConfig(
                max_model_len=2048,
                is_encoder_decoder=False,
                max_num_batched_tokens=tokens,
                max_num_seqs=sequences,
            ),
        )
        for tokens, sequences in [(4096, 16), (8192, 32)]
    ]
    assert pool_key(configs[0]) == pool_key(configs[1])


def test_resolved_quantization_changes_isolate_keys(model_path: Path) -> None:
    dynamic = make_config(
        model_path, quant_config=Fp8Config(activation_scheme="dynamic")
    )
    static = make_config(model_path, quant_config=Fp8Config(activation_scheme="static"))
    assert pool_key(dynamic) != pool_key(static)


@pytest.mark.parametrize("composed", [False, True])
def test_online_quantization_settings_change_isolates_keys(
    model_path: Path, composed: bool
) -> None:
    configs = []
    for target in ("mxfp4", "mxfp8"):
        online = OnlineQuantizationConfig(
            QuantizationConfigArgs(targets={"Linear": target})
        )
        checkpoint = Fp8Config()
        checkpoint.online_quantization_config = online
        configs.append(
            make_config(model_path, quant_config=checkpoint if composed else online)
        )
    assert pool_key(configs[0]) != pool_key(configs[1])


@pytest.mark.parametrize("composed", [False, True])
def test_online_quantization_runtime_state_preserves_keys(
    model_path: Path, composed: bool
) -> None:
    online = OnlineQuantizationConfig(
        QuantizationConfigArgs(targets={"Linear": "mxfp4"})
    )
    checkpoint = Fp8Config()
    checkpoint.online_quantization_config = online
    config = make_config(model_path, quant_config=checkpoint if composed else online)
    old_key = pool_key(config)
    online.online_quantization_time = 1.234
    online.quantized_layers["model.layers.0.mlp.up_proj"] = (
        "targets",
        "mxfp4",
        "Linear",
    )
    assert pool_key(config) == old_key


@pytest.mark.parametrize(
    "variable", ["CUBLAS_WORKSPACE_CONFIG", "NCCL_ALGO", "VLLM_PLUGIN_TEST_OPTION"]
)
def test_library_and_unregistered_vllm_settings_isolate_keys(
    model_path: Path, monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    config = make_config(model_path)
    old_key = pool_key(config)
    monkeypatch.setenv(variable, "changed")
    assert pool_key(config) != old_key


@pytest.mark.parametrize(
    "variable",
    ["VLLM_LOGGING_LEVEL", "VLLM_BUILD_URL", "VLLM_PORT", "VLLM_DP_RANK"],
)
def test_runtime_environment_changes_preserve_keys(
    model_path: Path, monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    config = make_config(model_path)
    old_key = pool_key(config)
    monkeypatch.setenv(variable, "changed")
    assert pool_key(config) == old_key


def test_invalid_compute_environment_fails_startup(
    model_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(model_path)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "invalid")
    with pytest.raises(ValueError, match="VLLM_BATCH_INVARIANT"):
        resolve_cache_prefix(config)


@pytest.mark.parametrize("image_tag", ["", "local/vllm-openai:dev"])
def test_auto_requires_code_identity(
    model_path: Path, monkeypatch: pytest.MonkeyPatch, image_tag: str
) -> None:
    monkeypatch.setenv("VLLM_IMAGE_TAG", image_tag)
    with pytest.raises(ValueError, match="requires VLLM_BUILD_COMMIT"):
        resolve_cache_prefix(make_config(model_path))
    monkeypatch.setenv("VLLM_BUILD_COMMIT", "commit-1")
    assert resolve_cache_prefix(make_config(model_path)).startswith("auto-")


def test_llm_rejects_auto_prefix_without_code_identity(
    vllm_runner: type["VllmRunner"], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_IMAGE_TAG", "")
    monkeypatch.setenv("VLLM_BUILD_COMMIT", "unknown")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    with (
        pytest.raises(ValueError, match="requires VLLM_BUILD_COMMIT"),
        vllm_runner(
            "facebook/opt-125m",
            max_model_len=128,
            enforce_eager=True,
            distributed_executor_backend="uni",
            kv_transfer_config=KVTransferConfig(
                kv_connector="MooncakeStoreConnector",
                kv_role="kv_both",
                kv_connector_extra_config={"cache_prefix": "auto"},
            ),
        ),
    ):
        pytest.fail("LLM startup accepted an auto prefix without code identity")
