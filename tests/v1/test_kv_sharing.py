# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec
from vllm.v1.worker.utils import add_kv_sharing_layers_to_kv_cache_groups


def new_kv_cache_spec():
    return FullAttentionSpec(16, 1, 1, torch.float32, False)


def test_initialize_kv_cache_for_kv_sharing_different_attn_groups():
    """
    Test initializing KV cache sharing with different attention groups.
    Layers in the same KV cache group might be placed in different attn groups
    if they have different attention backends.
    """
    shared_kv_cache_layers = {
        "model.layers.2": "model.layers.0",
        "model.layers.3": "model.layers.1",
    }

    # Layers 0 and 1 both belong in KV cache group 0
    # However, if they have different attention backends, they will be
    # placed in different attention groups for KV cache group 0
    kv_cache_groups = [
        KVCacheGroupSpec(["model.layers.0", "model.layers.1"],
                         new_kv_cache_spec()),
    ]

    # Only layers 0 and 1 will have KV caches allocated
    kv_caches = {
        "model.layers.0": torch.zeros(1, 2, 3),
        "model.layers.1": torch.ones(1, 2, 3),
    }

    add_kv_sharing_layers_to_kv_cache_groups(
        shared_kv_cache_layers=shared_kv_cache_layers,
        kv_cache_groups=kv_cache_groups,
    )

    for layer_name, target_layer_name in shared_kv_cache_layers.items():
        kv_caches[layer_name] = kv_caches[target_layer_name]

    # Check that the KV caches were shared correctly
    assert kv_caches["model.layers.2"].data_ptr(
    ) == kv_caches["model.layers.0"].data_ptr()
    assert kv_caches["model.layers.3"].data_ptr(
    ) == kv_caches["model.layers.1"].data_ptr()

    # Check that the layers were added to the correct KV cache group
    assert len(kv_cache_groups) == 1
    assert kv_cache_groups[0].layer_names == [
        "model.layers.0", "model.layers.1", "model.layers.2", "model.layers.3"
    ]


def test_initialize_kv_cache_for_kv_sharing_same_attn_groups():
    """
    Test case assuming that all layers in the same KV cache group have the same
    attention backends. This is true for most models.
    """
    shared_kv_cache_layers = {
        "model.layers.2": "model.layers.0",
        "model.layers.3": "model.layers.1",
    }

    kv_cache_groups = [
        KVCacheGroupSpec(["model.layers.0", "model.layers.1"],
                         new_kv_cache_spec()),
    ]

    kv_caches = {
        "model.layers.0": torch.zeros(1, 2, 3),
        "model.layers.1": torch.ones(1, 2, 3),
    }

    add_kv_sharing_layers_to_kv_cache_groups(
        shared_kv_cache_layers=shared_kv_cache_layers,
        kv_cache_groups=kv_cache_groups,
    )

    for layer_name, target_layer_name in shared_kv_cache_layers.items():
        kv_caches[layer_name] = kv_caches[target_layer_name]

    # Check that the KV caches were shared correctly
    assert kv_caches["model.layers.2"].data_ptr(
    ) == kv_caches["model.layers.0"].data_ptr()
    assert kv_caches["model.layers.3"].data_ptr(
    ) == kv_caches["model.layers.1"].data_ptr()

    # Check that the layers were added to the correct KV cache group
    assert len(kv_cache_groups) == 1
    assert kv_cache_groups[0].layer_names == [
        "model.layers.0", "model.layers.1", "model.layers.2", "model.layers.3"
    ]


def test_initialize_kv_cache_for_kv_sharing_no_attn_groups():
    """
    Test KV sharing set up when no attention groups are provided.
    This is the case for the TPU model runner, which doesn't have 
    support for attention groups yet.
    """
    shared_kv_cache_layers = {
        "model.layers.2": "model.layers.0",
        "model.layers.3": "model.layers.1",
    }

    kv_cache_groups = [
        KVCacheGroupSpec(["model.layers.0"], new_kv_cache_spec()),
        KVCacheGroupSpec(["model.layers.1"], new_kv_cache_spec()),
    ]

    kv_caches = {
        "model.layers.0": torch.zeros(1, 2, 3),
        "model.layers.1": torch.ones(1, 2, 3),
    }

    add_kv_sharing_layers_to_kv_cache_groups(
        shared_kv_cache_layers=shared_kv_cache_layers,
        kv_cache_groups=kv_cache_groups,
    )

    for layer_name, target_layer_name in shared_kv_cache_layers.items():
        kv_caches[layer_name] = kv_caches[target_layer_name]

    # Check that the KV caches were shared correctly
    assert kv_caches["model.layers.2"].data_ptr(
    ) == kv_caches["model.layers.0"].data_ptr()
    assert kv_caches["model.layers.3"].data_ptr(
    ) == kv_caches["model.layers.1"].data_ptr()

    # Check that the layers were added to the correct KV cache group
    assert len(kv_cache_groups) == 2
    assert kv_cache_groups[0].layer_names == [
        "model.layers.0", "model.layers.2"
    ]
    assert kv_cache_groups[1].layer_names == [
        "model.layers.1", "model.layers.3"
    ]
