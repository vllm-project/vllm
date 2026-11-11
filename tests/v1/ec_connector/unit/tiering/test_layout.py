# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from pathlib import Path

import pytest

from vllm.distributed.ec_transfer.ec_connector.tiering.layout import (
    ECCacheNamespace,
    build_ec_cache_key,
    get_ec_entry_path,
    get_ec_namespace_path,
    get_ec_temporary_path,
)

pytestmark = pytest.mark.cpu_test


def _namespace() -> ECCacheNamespace:
    return ECCacheNamespace(
        model_id="org/model",
        model_revision="revision-1",
        dtype="bfloat16",
        hidden_dim=3584,
        encoder_output_widths=(("image", 3584),),
        element_size=2,
        encoder_config_hash="a" * 64,
    )


def test_namespace_is_deterministic_and_compatibility_scoped() -> None:
    namespace = _namespace()

    assert namespace.digest() == _namespace().digest()
    assert (
        namespace.digest() != replace(namespace, model_revision="revision-2").digest()
    )
    assert namespace.digest() != replace(namespace, dtype="float16").digest()
    assert namespace.digest() != replace(namespace, hidden_dim=4096).digest()
    assert (
        namespace.digest() != replace(namespace, encoder_config_hash="b" * 64).digest()
    )
    assert (
        namespace.digest()
        != replace(
            namespace,
            encoder_output_widths=(("image", 4096),),
        ).digest()
    )


def test_namespace_uses_effective_hf_encoder_config(tmp_path):
    from transformers import LlavaConfig

    from vllm.config import ModelConfig, VllmConfig
    from vllm.config.ec_transfer import ECTransferConfig
    from vllm.distributed.ec_transfer.ec_connector.tiering.scheduler import (
        ECTieringScheduler,
    )

    LlavaConfig(architectures=["LlavaForConditionalGeneration"]).save_pretrained(
        tmp_path
    )

    def namespace(overrides):
        model = ModelConfig(
            model=str(tmp_path),
            dtype="float16",
            skip_tokenizer_init=True,
            enforce_eager=True,
            max_model_len=128,
            hf_overrides=overrides,
        )
        ec = ECTransferConfig(ec_connector="ECTieringConnector", ec_role="ec_both")
        ec.mm_encoder_output_widths = {"image": 32}
        config = VllmConfig(model_config=model, ec_transfer_config=ec)
        return ECTieringScheduler._build_namespace(config)

    def select_last_layer(config):
        config.vision_feature_layer = -1
        return config

    previous = namespace({"vision_feature_layer": -2})
    current = namespace({"vision_feature_layer": -1})
    assert previous.digest() != current.digest()
    assert current.digest() == namespace({"vision_feature_layer": -1}).digest()
    assert current.digest() == namespace(select_last_layer).digest()
    nested = namespace(
        {"vision_feature_layer": -1, "vision_config": {"hidden_act": "relu"}}
    )
    assert nested.digest() != current.digest()
    assert build_ec_cache_key(previous.digest(), "image", "image", 2, 128) != (
        build_ec_cache_key(current.digest(), "image", "image", 2, 128)
    )


def test_cache_key_includes_item_shape_and_identity() -> None:
    namespace_digest = _namespace().digest()
    cache_key = build_ec_cache_key(
        namespace_digest,
        identifier="image-hash",
        modality="image",
        num_tokens=16,
        file_size_bytes=16 * 3584 * 2,
    )

    assert cache_key == build_ec_cache_key(
        namespace_digest,
        identifier="image-hash",
        modality="image",
        num_tokens=16,
        file_size_bytes=16 * 3584 * 2,
    )
    assert cache_key != build_ec_cache_key(
        namespace_digest,
        identifier="lora:image-hash",
        modality="image",
        num_tokens=16,
        file_size_bytes=16 * 3584 * 2,
    )
    assert cache_key != build_ec_cache_key(
        namespace_digest,
        identifier="image-hash",
        modality="image",
        num_tokens=16,
        file_size_bytes=16 * 3584 * 2 + 4096,
    )


def test_entry_path_is_versioned_and_hash_prefix_sharded() -> None:
    namespace_digest = _namespace().digest()
    cache_key = "a1b2" + "0" * 60

    path = get_ec_entry_path(Path("/cache"), namespace_digest, cache_key)

    assert path == (
        get_ec_namespace_path(Path("/cache"), namespace_digest)
        / "a1b"
        / "20"
        / f"{cache_key}.ec"
    )
    temporary_path = get_ec_temporary_path(path, "write-operation-1")
    assert temporary_path.parent == path.parent
    assert temporary_path.name.startswith(f".{path.name}.")
    assert temporary_path.suffix == ".tmp"
