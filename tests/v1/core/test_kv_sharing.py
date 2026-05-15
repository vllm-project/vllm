# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec
from vllm.v1.worker.utils import add_kv_sharing_layers_to_kv_cache_groups

pytestmark = pytest.mark.cpu_test


def new_kv_cache_spec():
    return FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32
    )


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
        KVCacheGroupSpec(["model.layers.0", "model.layers.1"], new_kv_cache_spec()),
    ]

    add_kv_sharing_layers_to_kv_cache_groups(
        shared_kv_cache_layers=shared_kv_cache_layers,
        kv_cache_groups=kv_cache_groups,
    )

    # Check that the layers were added to the correct KV cache group
    assert len(kv_cache_groups) == 1
    assert kv_cache_groups[0].layer_names == [
        "model.layers.0",
        "model.layers.1",
        "model.layers.2",
        "model.layers.3",
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
        KVCacheGroupSpec(["model.layers.0", "model.layers.1"], new_kv_cache_spec()),
    ]

    add_kv_sharing_layers_to_kv_cache_groups(
        shared_kv_cache_layers=shared_kv_cache_layers,
        kv_cache_groups=kv_cache_groups,
    )

    # Check that the layers were added to the correct KV cache group
    assert len(kv_cache_groups) == 1
    assert kv_cache_groups[0].layer_names == [
        "model.layers.0",
        "model.layers.1",
        "model.layers.2",
        "model.layers.3",
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

    add_kv_sharing_layers_to_kv_cache_groups(
        shared_kv_cache_layers=shared_kv_cache_layers,
        kv_cache_groups=kv_cache_groups,
    )

    # Check that the layers were added to the correct KV cache group
    assert len(kv_cache_groups) == 2
    assert kv_cache_groups[0].layer_names == ["model.layers.0", "model.layers.2"]
    assert kv_cache_groups[1].layer_names == ["model.layers.1", "model.layers.3"]


def test_dflash_draft_kv_groups_keep_hybrid_tensor_sharing():
    spec = new_kv_cache_spec()
    num_blocks = 8
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(method="dflash"),
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
    )
    kv_cache_groups = [
        KVCacheGroupSpec(["model.layers.0", "model.layers.1"], spec),
        KVCacheGroupSpec(["model.layers.2", "model.layers.3"], spec),
    ]

    kv_cache_config = get_kv_cache_config_from_groups(
        vllm_config=vllm_config,
        kv_cache_groups=kv_cache_groups,
        available_memory=spec.page_size_bytes * 2 * num_blocks,
    )

    assert kv_cache_config.num_blocks == num_blocks
    assert [tensor.shared_by for tensor in kv_cache_config.kv_cache_tensors] == [
        ["model.layers.0", "model.layers.2"],
        ["model.layers.1", "model.layers.3"],
    ]
