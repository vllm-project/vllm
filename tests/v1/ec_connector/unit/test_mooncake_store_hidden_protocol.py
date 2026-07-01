# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.distributed.ec_transfer.ec_connector.mooncake_store_hidden.data import (
    HIDDEN_LAYOUT_VERSION,
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
    model_name: str = "qwen",
    mm_encoder_config_hash: str = "encoder-config-a",
    hidden_parallel_key: str = "tp:1@pp:1@pcp:1@dcp:1@mm_tp:weights@storage:replicated",
    layout: str = HIDDEN_LAYOUT_VERSION,
) -> HiddenPoolKey:
    return HiddenPoolKey(
        key_metadata=HiddenKeyMetadata(
            model_name=model_name,
            mm_encoder_config_hash=mm_encoder_config_hash,
            hidden_parallel_key=hidden_parallel_key,
            layout=layout,
        ),
        identifier=identifier,
    )


def test_hidden_pool_key_is_the_single_tensor_object_key():
    pool_key = make_pool_key()

    data_key = make_hidden_data_key(pool_key)

    assert data_key == pool_key.to_string()
    assert "model:qwen" in data_key
    assert "mm_encoder:encoder-config-a" in data_key
    assert (
        "parallel:tp%3A1%40pp%3A1%40pcp%3A1%40dcp%3A1%40mm_tp%3Aweights%40storage%3Areplicated"
        in data_key
    )
    assert "layout:vllm-encoder-cache-tensor-v1" in data_key
    assert "adapter:" not in data_key
    assert "modality:" not in data_key
    assert "image-hash" in data_key


def test_same_identifier_with_different_encoder_config_uses_different_keys():
    pool_key_a = make_pool_key(mm_encoder_config_hash="encoder-config-a")
    pool_key_b = make_pool_key(mm_encoder_config_hash="encoder-config-b")

    assert make_hidden_data_key(pool_key_a) != make_hidden_data_key(pool_key_b)


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
    pool_key_a = make_pool_key(mm_encoder_config_hash="encoder-config-a")
    pool_key_b = make_pool_key(mm_encoder_config_hash="encoder-config-b")

    assert pool_key_a != pool_key_b
    assert make_hidden_data_key(pool_key_a) != make_hidden_data_key(pool_key_b)
