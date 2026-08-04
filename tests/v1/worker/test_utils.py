# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import math

import pytest
import torch

from vllm.config.cache import CacheConfig
from vllm.utils.mem_utils import MemorySnapshot
from vllm.v1.worker import utils as worker_utils
from vllm.v1.worker.utils import bind_kv_cache


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


def _memory_snapshot(total: int, free: int) -> MemorySnapshot:
    snapshot = MemorySnapshot.__new__(MemorySnapshot)
    snapshot.total_memory = total
    snapshot.free_memory = free
    snapshot.device_ = torch.device("cuda:0")
    return snapshot


@pytest.mark.parametrize(
    "free_fraction,util,foreign,extensible,expected",
    [
        # The requested fraction of total memory must be free, regardless of
        # whether the shortfall was declared.
        (0.5, None, {4242: 1 << 30}, False, "raise"),
        (0.5, 0.9, {4242: 1 << 30}, False, "raise"),
        # ...unless commitment is deferred, which is what lets the extensible
        # KV cache run at a utilization of 1.0.
        (0.5, 0.9, {}, True, "ok"),
        # Fits, but another process is resident: sharing must be deliberate.
        (0.97, None, {4242: 1 << 30}, False, "raise"),
        (0.97, 0.5, {4242: 1 << 30}, False, "warn"),
        # Exclusive device: unchanged behaviour at any utilization.
        (1.0, None, {}, False, "ok"),
        (1.0, 0.98, {}, False, "ok"),
        # Platform cannot enumerate processes -> must not refuse to start.
        (1.0, None, None, False, "ok"),
    ],
)
def test_request_memory_shared_device(
    free_fraction, util, foreign, extensible, expected, monkeypatch, caplog
):
    """gpu_memory_utilization stays a fraction of *total* memory, and sharing a
    device requires setting it explicitly."""
    total = 100 << 30
    kwargs: dict = {"enable_extensible_kv_cache": extensible}
    if util is not None:
        kwargs["gpu_memory_utilization"] = util
    cache_config = CacheConfig(**kwargs)
    monkeypatch.setattr(
        worker_utils.current_platform,
        "get_foreign_device_processes",
        lambda device_id=0: foreign,
    )
    snapshot = _memory_snapshot(total, int(free_fraction * total))

    if expected == "raise":
        with pytest.raises(ValueError):
            worker_utils.request_memory(snapshot, cache_config)
        return

    with caplog.at_level(logging.WARNING):
        requested = worker_utils.request_memory(snapshot, cache_config)

    # Always the declared fraction of total, never rescaled to what is free.
    assert requested == math.ceil(total * cache_config.gpu_memory_utilization)
    assert ("shared with another process" in caplog.text) == (expected == "warn")
