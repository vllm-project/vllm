# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from pathlib import Path

from vllm.distributed.ec_transfer.ec_connector.tiering.layout import (
    ECCacheNamespace,
    build_ec_cache_key,
    get_ec_entry_path,
    get_ec_namespace_path,
    get_ec_temporary_path,
)


def _namespace() -> ECCacheNamespace:
    return ECCacheNamespace(
        model_id="org/model",
        model_revision="revision-1",
        dtype="bfloat16",
        hidden_dim=3584,
        encoder_output_widths=(("image", 3584),),
        element_size=2,
        tensor_parallel_size=2,
        prefill_context_parallel_size=1,
    )


def test_namespace_is_deterministic_and_compatibility_scoped() -> None:
    namespace = _namespace()

    assert namespace.digest() == _namespace().digest()
    assert (
        namespace.digest() != replace(namespace, model_revision="revision-2").digest()
    )
    assert namespace.digest() != replace(namespace, dtype="float16").digest()
    assert namespace.digest() != replace(namespace, hidden_dim=4096).digest()
    assert namespace.digest() != replace(
        namespace,
        encoder_output_widths=(("image", 4096),),
    ).digest()


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
