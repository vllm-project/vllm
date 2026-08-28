# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`free_before_shutdown` must sever the layer aliases of the KV allocation.

Layers registered in ``static_forward_context`` are submodules of the model and
keep their own reference to the KV cache. Clearing the registry drops only the
registry's edges, so in-process teardown (VLLM_ENABLE_V1_MULTIPROCESSING=0)
leaves the whole KV buffer resident for as long as anything still references
the model.
"""

import types

import pytest
import torch

from vllm.v1.worker.gpu.shutdown import free_before_shutdown

pytestmark = pytest.mark.cpu_test


class _AttentionLayer:
    """Binds the KV cache view as-is, like AttentionLayerBase."""

    def __init__(self):
        self.kv_cache = torch.zeros(4)


class _MambaLayer:
    """Binds a tuple of per-state views, like MambaBase (GDN, Qwen3.5)."""

    def __init__(self):
        self.kv_cache = (torch.zeros(4), torch.zeros(4))


class _QuantizedAttentionLayer(_AttentionLayer):
    """Also holds quantized KV scale views on its impl."""

    def __init__(self):
        super().__init__()
        self.impl = types.SimpleNamespace(
            _k_scale_cache=torch.zeros(2),
            _v_scale_cache=torch.zeros(2),
        )


@pytest.fixture
def vllm_config(monkeypatch: pytest.MonkeyPatch):
    """Minimal stand-in; the workspace/rope globals are not under test."""
    monkeypatch.setattr(
        "vllm.model_executor.layers.rotary_embedding._ROPE_DICT", {}, raising=False
    )
    monkeypatch.setattr(
        "vllm.v1.worker.workspace.reset_workspace_manager", lambda: None
    )
    return types.SimpleNamespace(
        cache_config=types.SimpleNamespace(num_gpu_blocks=17),
        compilation_config=types.SimpleNamespace(static_forward_context={}),
    )


def test_severs_attention_and_mamba_kv_bindings(vllm_config):
    attention = _AttentionLayer()
    mamba = _MambaLayer()
    vllm_config.compilation_config.static_forward_context = {
        "attn": attention,
        "mamba": mamba,
    }

    free_before_shutdown(vllm_config)

    # The layers outlive the registry, so their bindings must be emptied.
    assert attention.kv_cache.numel() == 0
    assert mamba.kv_cache == []
    assert vllm_config.compilation_config.static_forward_context == {}
    assert vllm_config.cache_config.num_gpu_blocks is None


def test_severs_quantized_kv_scale_views(vllm_config):
    layer = _QuantizedAttentionLayer()
    vllm_config.compilation_config.static_forward_context = {"attn": layer}

    free_before_shutdown(vllm_config)

    assert layer.kv_cache.numel() == 0
    assert layer.impl._k_scale_cache is None
    assert layer.impl._v_scale_cache is None


def test_tolerates_layers_without_kv_cache(vllm_config):
    """Non-attention entries (and layers with no impl) must not raise."""
    bare = types.SimpleNamespace()
    vllm_config.compilation_config.static_forward_context = {"bare": bare}

    free_before_shutdown(vllm_config)

    assert vllm_config.compilation_config.static_forward_context == {}


def test_kv_tensors_are_released_when_the_layer_outlives_the_registry(vllm_config):
    """The point of the sever: no live reference to the KV storage remains."""
    import weakref

    attention = _AttentionLayer()
    kv_ref = weakref.ref(attention.kv_cache)
    vllm_config.compilation_config.static_forward_context = {"attn": attention}

    free_before_shutdown(vllm_config)

    # `attention` is still alive, exactly as it is in-process after shutdown.
    assert kv_ref() is None
