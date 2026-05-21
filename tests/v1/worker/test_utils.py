# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.utils import AttentionGroup, KVBlockZeroer, bind_kv_cache


def test_bind_kv_cache(default_vllm_config):
    from vllm.model_executor.layers.attention import Attention

    ctx = {
        "layers.0.self_attn": Attention(32, 128, 0.1, prefix="layers.0.self_attn"),
        "layers.1.self_attn": Attention(32, 128, 0.1, prefix="layers.1.self_attn"),
        "layers.2.self_attn": Attention(32, 128, 0.1, prefix="layers.2.self_attn"),
        "layers.3.self_attn": Attention(32, 128, 0.1, prefix="layers.3.self_attn"),
    }
    kv_cache = {
        "layers.0.self_attn": torch.zeros((1,)),
        "layers.1.self_attn": torch.zeros((1,)),
        "layers.2.self_attn": torch.zeros((1,)),
        "layers.3.self_attn": torch.zeros((1,)),
    }
    runner_kv_caches: list[torch.Tensor] = []
    bind_kv_cache(kv_cache, ctx, runner_kv_caches)
    assert ctx["layers.0.self_attn"].kv_cache is kv_cache["layers.0.self_attn"]
    assert ctx["layers.1.self_attn"].kv_cache is kv_cache["layers.1.self_attn"]
    assert ctx["layers.2.self_attn"].kv_cache is kv_cache["layers.2.self_attn"]
    assert ctx["layers.3.self_attn"].kv_cache is kv_cache["layers.3.self_attn"]

    assert runner_kv_caches[0] is kv_cache["layers.0.self_attn"]
    assert runner_kv_caches[1] is kv_cache["layers.1.self_attn"]
    assert runner_kv_caches[2] is kv_cache["layers.2.self_attn"]
    assert runner_kv_caches[3] is kv_cache["layers.3.self_attn"]


def test_bind_kv_cache_non_attention(default_vllm_config):
    from vllm.model_executor.layers.attention import Attention

    # example from Jamba PP=2
    ctx = {
        "model.layers.20.attn": Attention(32, 128, 0.1, prefix="model.layers.20.attn"),
        "model.layers.28.attn": Attention(32, 128, 0.1, prefix="model.layers.28.attn"),
    }
    kv_cache = {
        "model.layers.20.attn": torch.zeros((1,)),
        "model.layers.28.attn": torch.zeros((1,)),
    }

    runner_kv_caches: list[torch.Tensor] = []
    bind_kv_cache(kv_cache, ctx, runner_kv_caches)

    assert ctx["model.layers.20.attn"].kv_cache is kv_cache["model.layers.20.attn"]
    assert ctx["model.layers.28.attn"].kv_cache is kv_cache["model.layers.28.attn"]

    assert runner_kv_caches[0] is kv_cache["model.layers.20.attn"]
    assert runner_kv_caches[1] is kv_cache["model.layers.28.attn"]


def test_bind_kv_cache_draft_model(default_vllm_config):
    from vllm.model_executor.layers.attention import Attention

    layer_names = [
        "model.layers.0.attn",
        "model.layers.1.attn",
        "draft_model.layers.0.attn",
        "draft_model.layers.1.attn",
    ]
    ctx = {
        layer_name: Attention(32, 128, 0.1, prefix=layer_name)
        for layer_name in layer_names
    }
    kv_cache = {layer_name: torch.zeros((1,)) for layer_name in layer_names}
    runner_kv_caches: list[torch.Tensor] = []
    bind_kv_cache(kv_cache, ctx, runner_kv_caches)

    assert ctx["model.layers.0.attn"].kv_cache is kv_cache["model.layers.0.attn"]
    assert ctx["model.layers.1.attn"].kv_cache is kv_cache["model.layers.1.attn"]
    assert (
        ctx["draft_model.layers.0.attn"].kv_cache
        is kv_cache["draft_model.layers.0.attn"]
    )
    assert (
        ctx["draft_model.layers.1.attn"].kv_cache
        is kv_cache["draft_model.layers.1.attn"]
    )

    # caches are ordered by layer_index, interleaving target and draft model
    assert runner_kv_caches[0] is kv_cache["model.layers.0.attn"]
    assert runner_kv_caches[1] is kv_cache["draft_model.layers.0.attn"]
    assert runner_kv_caches[2] is kv_cache["model.layers.1.attn"]
    assert runner_kv_caches[3] is kv_cache["draft_model.layers.1.attn"]


def test_kv_block_zeroer_zeros_mamba_cache_blocks():
    num_blocks = 4
    state_storage = torch.full((num_blocks * 16,), 1.0)
    state = torch.as_strided(
        state_storage,
        size=(num_blocks, 2, 3),
        stride=(16, 3, 1),
    )
    tracker = torch.full((num_blocks,), 9, dtype=torch.int32)
    spec = MambaSpec(
        block_size=1,
        shapes=((2, 3), ()),
        dtypes=(torch.float32, torch.int32),
    )
    group = AttentionGroup(
        backend=Mock(),
        layer_names=["layer.0"],
        kv_cache_spec=spec,
        kv_cache_group_id=0,
    )
    zeroer = KVBlockZeroer(torch.device("cpu"), pin_memory=False)
    zeroer.init_meta(
        [group],
        [1],
        "auto",
        set(),
        {"layer.0": SimpleNamespace(kv_cache=(state, tracker))},
    )

    zeroer.zero_block_ids([1, 3])

    assert torch.all(state[1] == 0)
    assert torch.all(state[3] == 0)
    assert torch.all(tracker[torch.tensor([1, 3])] == 0)
    assert torch.all(state[0] == 1)
    assert torch.all(state[2] == 1)
    assert torch.all(tracker[torch.tensor([0, 2])] == 9)
    assert torch.all(state_storage[22:32] == 1)


@pytest.mark.parametrize("checkpoint_interval", [1, 6])
def test_mamba2_state_shape_honors_checkpoint_interval(checkpoint_interval):
    shapes = MambaStateShapeCalculator.mamba2_state_shape(
        tp_world_size=1,
        intermediate_size=8192,
        n_groups=8,
        num_heads=128,
        head_dim=64,
        state_size=128,
        conv_kernel=4,
        checkpoint_interval=checkpoint_interval,
    )

    old_x_shape = shapes[2]
    old_B_shape = shapes[3]
    old_dt_shape = shapes[4]
    old_cumAdt_shape = shapes[5]

    assert old_x_shape[0] == checkpoint_interval
    assert old_B_shape[1] == checkpoint_interval
    assert old_dt_shape[-1] == checkpoint_interval
    assert old_cumAdt_shape[-1] == checkpoint_interval


@pytest.mark.parametrize("checkpoint_interval", [1, 6])
def test_nemotron_h_state_shape_honors_checkpoint_interval(checkpoint_interval):
    from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM

    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                mamba_num_heads=128,
                mamba_head_dim=64,
                n_groups=8,
                ssm_state_size=128,
                conv_kernel=4,
            ),
        ),
        num_speculative_tokens=0,
        mamba_config=SimpleNamespace(checkpoint_interval=checkpoint_interval),
    )

    shapes = NemotronHForCausalLM.get_mamba_state_shape_from_config(vllm_config)

    assert shapes[2][0] == checkpoint_interval
    assert shapes[3][1] == checkpoint_interval
    assert shapes[4][-1] == checkpoint_interval
    assert shapes[5][-1] == checkpoint_interval
