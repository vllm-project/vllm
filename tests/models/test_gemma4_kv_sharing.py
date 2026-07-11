# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 KV-sharing initialization.

Draft/assistant checkpoints (e.g. ``google/gemma-4-E4B-it-assistant``)
declare ``num_kv_shared_layers == num_hidden_layers``: every layer is
KV-shared and the proposer wires cross-model sharing after construction.
Loading such a config standalone used to raise
``ValueError: 'sliding_attention' is not in list`` from ``list.index``
in ``Gemma4Attention``.
"""

import tempfile

import pytest
import torch

from vllm.config import CacheConfig, ModelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM

pytestmark = pytest.mark.cpu_test


def _build_model(hf_overrides: dict) -> Gemma4ForCausalLM:
    model_config = ModelConfig(
        "google/gemma-4-E4B-it-assistant",
        hf_overrides={"architectures": ["Gemma4ForCausalLM"], **hf_overrides},
        dtype="bfloat16",
    )
    vllm_config = VllmConfig(
        model_config=model_config, cache_config=CacheConfig(block_size=16)
    )
    temp_file = tempfile.mkstemp()[1]
    with set_current_vllm_config(vllm_config=vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{temp_file}",
            local_rank=0,
            backend="gloo",
        )
        initialize_model_parallel(tensor_model_parallel_size=1)
        with torch.device("meta"):
            return Gemma4ForCausalLM(vllm_config=vllm_config)


def test_gemma4_all_layers_kv_shared_initializes():
    # num_kv_shared_layers == num_hidden_layers == 4 in the checkpoint:
    # there is no non-shared prefix, so every layer must fall back to a
    # regular KV cache instead of raising from list.index.
    model = _build_model({})
    for layer in model.model.layers:
        attn = layer.self_attn
        assert not attn.is_kv_shared_layer
        assert attn.attn.kv_sharing_target_layer_name is None

    del model
    cleanup_dist_env_and_memory()


def test_gemma4_partial_kv_sharing_still_engages():
    # layer_types is [sliding, sliding, sliding, full]; with 2 shared
    # layers the prefix is [sliding, sliding]: layer 2 (sliding) shares
    # with layer 1, while layer 3 (full) has no full layer in the prefix
    # and must fall back.
    model = _build_model({"text_config": {"num_kv_shared_layers": 2}})
    layers = model.model.layers

    assert layers[2].self_attn.is_kv_shared_layer
    target = layers[2].self_attn.attn.kv_sharing_target_layer_name
    assert target is not None and target.endswith(".layers.1.self_attn.attn")

    assert not layers[3].self_attn.is_kv_shared_layer
    assert layers[3].self_attn.attn.kv_sharing_target_layer_name is None

    del model
    cleanup_dist_env_and_memory()
