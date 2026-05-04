# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import torch

from vllm.config import VllmConfig
from vllm.v1.worker.gpu.shutdown import free_before_shutdown


def test_free_before_shutdown_clears_layer_kv_cache_and_scale_views(
    monkeypatch,
):
    vllm_config = VllmConfig()

    kv_cache = torch.ones(4)
    layer = SimpleNamespace(
        kv_cache=kv_cache,
        impl=SimpleNamespace(
            _k_scale_cache=torch.ones(2),
            _v_scale_cache=torch.ones(2),
        ),
    )
    tuple_kv_layer = SimpleNamespace(kv_cache=[torch.ones(1), torch.ones(1)])
    vllm_config.compilation_config.static_forward_context = {
        "layer": layer,
        "tuple_layer": tuple_kv_layer,
    }
    vllm_config.cache_config.num_gpu_blocks = 16

    reset_calls = 0

    def fake_reset_workspace_manager() -> None:
        nonlocal reset_calls
        reset_calls += 1

    monkeypatch.setattr(
        "vllm.v1.worker.workspace.reset_workspace_manager",
        fake_reset_workspace_manager,
    )

    from vllm.model_executor.layers import rotary_embedding

    rotary_embedding._ROPE_DICT["test"] = object()

    free_before_shutdown(vllm_config)

    assert torch.equal(layer.kv_cache, torch.tensor([]))
    assert tuple_kv_layer.kv_cache == []
    assert layer.impl._k_scale_cache is None
    assert layer.impl._v_scale_cache is None
    assert vllm_config.cache_config.num_gpu_blocks is None
    assert vllm_config.compilation_config.static_forward_context == {}
    assert rotary_embedding._ROPE_DICT == {}
    assert reset_calls == 1
