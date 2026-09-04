# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.config.mamba import MambaBackendEnum, MambaConfig
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.v1.worker.utils import bind_kv_cache


class _TestReplaySSMMixer(MambaMixer2):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.use_replayssm = True
        self.mamba_config = MambaConfig(backend=MambaBackendEnum.FLASHINFER)
        self._replayssm_ring_start = torch.empty(0, dtype=torch.int32)
        self._replayssm_prev_num_accepted = torch.empty(0, dtype=torch.int32)
        self._updates_replayssm_trackers = True

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        return ((2,), (3,), (4,), (5,), (6,))

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return (torch.float32,) * 5


def _packed_replayssm_cache(num_blocks: int) -> torch.Tensor:
    return torch.full((num_blocks, 1, 1, 80), 0, dtype=torch.int8)


def test_bind_kv_cache_shares_replayssm_trackers_by_cache_group():
    mixers = [_TestReplaySSMMixer() for _ in range(3)]
    layer_names = [f"layers.{i}.mixer" for i in range(3)]
    ctx = dict(zip(layer_names, mixers))
    # Reverse insertion order: updater must follow layer index, not dict order.
    kv_cache = {
        layer_names[2]: _packed_replayssm_cache(4),
        layer_names[1]: _packed_replayssm_cache(4),
        layer_names[0]: _packed_replayssm_cache(4),
    }
    kv_cache_groups = [
        SimpleNamespace(layer_names=[layer_names[0], layer_names[2]]),
        SimpleNamespace(layer_names=[layer_names[1]]),
    ]

    bind_kv_cache(kv_cache, ctx, [], kv_cache_groups=kv_cache_groups)

    assert (
        mixers[0]._replayssm_ring_start.data_ptr()
        == mixers[2]._replayssm_ring_start.data_ptr()
    )
    assert (
        mixers[0]._replayssm_prev_num_accepted.data_ptr()
        == mixers[2]._replayssm_prev_num_accepted.data_ptr()
    )
    assert (
        mixers[1]._replayssm_ring_start.data_ptr()
        != mixers[0]._replayssm_ring_start.data_ptr()
    )
    # Group {0, 2} shares trackers; layer 2 (not 0) updates after both run.
    assert [m._updates_replayssm_trackers for m in mixers] == [False, True, True]


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
