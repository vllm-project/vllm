# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.distributed.ec_transfer.ec_connector.mooncake_store_hidden.data import (
    HIDDEN_TENSOR_LAYOUT,
    HiddenKeyMetadata,
    HiddenPoolKey,
    LoadSpec,
    MMMeta,
    MooncakeStoreConnectorMetadata,
    build_tensor_meta,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_hidden.keys import (
    make_hidden_data_key,
)


def make_pool_key(
    identifier: str = "image-hash",
    *,
    cache_prefix: str = "",
    kind: str = "encoder_output",
    model_name: str = "qwen",
    encoder: str = "encoder-config-a",
    storage: str = "replicated_object",
    parallel: str = "tp:1@pp:1@pcp:1@dcp:1@mm_tp:weights",
    tensor_layout: str = HIDDEN_TENSOR_LAYOUT,
) -> HiddenPoolKey:
    return HiddenPoolKey(
        key_metadata=HiddenKeyMetadata(
            cache_prefix=cache_prefix,
            kind=kind,
            model_name=model_name,
            encoder=encoder,
            storage=storage,
            parallel=parallel,
            tensor_layout=tensor_layout,
        ),
        identifier=identifier,
    )


def test_hidden_pool_key_is_the_single_tensor_object_key():
    pool_key = make_pool_key()

    data_key = make_hidden_data_key(pool_key)

    assert data_key == pool_key.to_string()
    assert data_key.startswith("hidden@")
    assert "kind:encoder_output" in data_key
    assert "model:qwen" in data_key
    assert "encoder:encoder-config-a" in data_key
    assert "storage:replicated_object" in data_key
    assert (
        "parallel:tp%3A1%40pp%3A1%40pcp%3A1%40dcp%3A1%40mm_tp%3Aweights"
        in data_key
    )
    assert "tensor_layout:tensor" in data_key
    assert "storage%3Areplicated" not in data_key
    assert "writer" not in data_key
    assert "adapter:" not in data_key
    assert "modality:" not in data_key
    assert "image-hash" in data_key


def test_same_identifier_with_different_encoder_config_uses_different_keys():
    pool_key_a = make_pool_key(encoder="encoder-config-a")
    pool_key_b = make_pool_key(encoder="encoder-config-b")

    assert make_hidden_data_key(pool_key_a) != make_hidden_data_key(pool_key_b)


def test_cache_prefix_namespaces_hidden_pool_key():
    pool_key_a = make_pool_key(cache_prefix="deployment-a")
    pool_key_b = make_pool_key(cache_prefix="deployment-b")

    data_key_a = make_hidden_data_key(pool_key_a)
    data_key_b = make_hidden_data_key(pool_key_b)

    assert data_key_a.startswith("deployment-a@hidden@")
    assert data_key_b.startswith("deployment-b@hidden@")
    assert data_key_a != data_key_b


def test_request_id_and_modality_are_not_part_of_hidden_pool_key():
    pool_key = make_pool_key(identifier="image-hash")

    assert "req-1" not in make_hidden_data_key(pool_key)
    assert "request" not in make_hidden_data_key(pool_key)
    assert "image@" not in make_hidden_data_key(pool_key)
    assert "modality" not in make_hidden_data_key(pool_key)


def test_mm_meta_carries_hidden_item_plan():
    item = MMMeta(
        identifier="image-hash",
        modality="video",
        can_save=True,
        load_spec=LoadSpec(can_load=True),
    )
    meta = MooncakeStoreConnectorMetadata(items=[item])

    assert meta.items == [item]
    assert meta.items[0].identifier == "image-hash"
    assert meta.items[0].modality == "video"
    assert meta.items[0].can_save
    assert meta.items[0].load_spec is not None
    assert meta.items[0].load_spec.can_load


def test_tensor_meta_describes_canonical_contiguous_tensor():
    pool_key = make_pool_key()
    source = torch.zeros((4, 8), dtype=torch.float16).t()
    stored = source.contiguous()
    tensor_meta = build_tensor_meta(pool_key, stored)

    assert tensor_meta.pool_key == pool_key
    assert tensor_meta.layout == HIDDEN_TENSOR_LAYOUT
    assert tensor_meta.shape == tuple(stored.shape)
    assert tensor_meta.dtype == "torch.float16"
    assert tensor_meta.nbytes == stored.numel() * stored.element_size()


def test_tensor_meta_rejects_non_contiguous_tensor():
    pool_key = make_pool_key()
    source = torch.zeros((4, 8), dtype=torch.float16).t()

    try:
        build_tensor_meta(pool_key, source)
    except ValueError as exc:
        assert "contiguous" in str(exc)
    else:
        raise AssertionError("non-contiguous tensor descriptor should fail")


def test_pool_key_namespace_carries_reuse_compatibility():
    pool_key_a = make_pool_key(encoder="encoder-config-a")
    pool_key_b = make_pool_key(encoder="encoder-config-b")

    assert pool_key_a != pool_key_b
    assert make_hidden_data_key(pool_key_a) != make_hidden_data_key(pool_key_b)
