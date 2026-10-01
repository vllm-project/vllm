# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepEP v2 token capacity with sequence parallelism.

Hermetic (fake buffer / all2all manager / forward context); only requires
deep_ep to be importable.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.utils.import_utils import has_deep_ep_v2

requires_deep_ep_v2 = pytest.mark.skipif(
    not has_deep_ep_v2(),
    reason="Requires DeepEP v2 (ElasticBuffer)",
)

if has_deep_ep_v2():
    import vllm.model_executor.layers.fused_moe.all2all_utils as _a2a_utils
    import vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_v2 as _dv2
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_v2 import (
        DeepEPV2PrepareAndFinalize,
    )

HIDDEN = 256
TOPK = 2
NUM_EXPERTS = 16


class _FakeDispatchBuffer:
    def __init__(self):
        self.calls: list[dict] = []

    def dispatch(self, **kwargs):
        self.calls.append(kwargs)
        x = kwargs["x"]
        handle = SimpleNamespace(
            num_recv_tokens_per_expert_list=[],
            psum_num_recv_tokens_per_scaleup_rank=torch.zeros(1, dtype=torch.int32),
        )
        return x, kwargs["topk_idx"], kwargs["topk_weights"], handle, None


class _FakeAll2AllManager:
    def __init__(self, world_size: int, dp_world_size: int):
        self.world_size = world_size
        self.dp_world_size = dp_world_size
        self.rank = 0
        self.handle_kwargs: dict | None = None

    def get_handle(self, kwargs):
        self.handle_kwargs = dict(kwargs)
        return _FakeDispatchBuffer()


def _make_moe(max_num_tokens: int, sp_size: int, dp_size: int):
    parallel = SimpleNamespace(use_all2all_kernels=True, sp_size=sp_size)
    return SimpleNamespace(
        moe_parallel_config=parallel,
        use_deepep_ht_kernels=False,
        use_deepep_ll_kernels=False,
        use_deepep_v2_kernels=True,
        dp_size=dp_size,
        max_num_tokens=max_num_tokens,
        hidden_dim=HIDDEN,
        experts_per_token=TOPK,
        num_experts=NUM_EXPERTS,
        num_local_experts=1,
    )


def _set_dp_metadata(monkeypatch, max_dp_tokens: int | None):
    dp_meta = None
    if max_dp_tokens is not None:
        dp_meta = SimpleNamespace(
            num_tokens_across_dp_cpu=torch.tensor([max_dp_tokens, 1], dtype=torch.int32)
        )
    monkeypatch.setattr(
        _dv2, "get_forward_context", lambda: SimpleNamespace(dp_metadata=dp_meta)
    )


def _make_pf(sp_size: int, capacity: int | None, use_cudagraph: bool = True):
    buffer = _FakeDispatchBuffer()
    pf = DeepEPV2PrepareAndFinalize(
        buffer=buffer,
        num_dispatchers=2 * sp_size,
        dp_size=2,
        rank_expert_offset=0,
        num_experts=NUM_EXPERTS,
        num_topk=TOPK,
        use_cudagraph=use_cudagraph,
        sp_size=sp_size,
        max_tokens_per_rank=capacity,
    )
    return pf, buffer


def _dispatch(pf, buffer, num_local_tokens: int) -> dict:
    pf.prepare_async(
        torch.zeros(num_local_tokens, HIDDEN, dtype=torch.bfloat16),
        torch.ones(num_local_tokens, TOPK),
        torch.zeros(num_local_tokens, TOPK, dtype=torch.int64),
        NUM_EXPERTS,
        None,
        False,
        FusedMoEQuantConfig.make(None),
        defer_input_quant=True,
    )
    return buffer.calls[-1]


@requires_deep_ep_v2
@pytest.mark.parametrize(
    "max_num_tokens,sp_size,expected",
    [
        (16384, 1, 16384),
        (16384, 4, 4096),
        (10240, 4, 2560),
        (8191, 4, 2048),  # ceil, not floor
    ],
)
def test_elastic_buffer_sized_for_sp_shard(
    monkeypatch, max_num_tokens, sp_size, expected
):
    monkeypatch.setattr(
        _a2a_utils,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=SimpleNamespace(enforce_eager=False)),
    )
    dp_size = 2
    manager = _FakeAll2AllManager(world_size=dp_size * sp_size, dp_world_size=dp_size)
    pf = _a2a_utils.maybe_make_prepare_finalize(
        _make_moe(max_num_tokens, sp_size, dp_size),
        quant_config=None,
        all2all_manager=manager,
    )
    assert isinstance(pf, DeepEPV2PrepareAndFinalize)
    assert manager.handle_kwargs["num_max_tokens_per_rank"] == expected
    assert pf.max_tokens_per_rank == expected
    assert pf.sp_size == sp_size


@requires_deep_ep_v2
@pytest.mark.parametrize(
    "max_dp_tokens,sp_size,capacity,expected",
    [
        (1, 4, 4096, 1),  # decode: tiny bound
        (3, 1, 16384, 4),  # rounded up to a power of 2
        (16384, 4, 4096, 4096),  # power-of-2 capacity is reached exactly
        (10240, 4, 2560, 2560),  # 2560 -> 4096 would overflow; capped
        (10240, 1, 10240, 10240),  # sp_size=1: 10240 -> 16384 would overflow
    ],
)
def test_dispatch_bound_capped_at_buffer_capacity(
    monkeypatch, max_dp_tokens, sp_size, capacity, expected
):
    _set_dp_metadata(monkeypatch, max_dp_tokens)
    pf, buffer = _make_pf(sp_size, capacity)
    call = _dispatch(pf, buffer, -(-max_dp_tokens // sp_size))
    assert call["do_expand"] is False
    assert call["num_max_tokens_per_rank"] == expected
    assert call["num_max_tokens_per_rank"] <= capacity


@requires_deep_ep_v2
def test_dispatch_over_capacity_raises(monkeypatch):
    _set_dp_metadata(monkeypatch, 4097 * 4)
    pf, buffer = _make_pf(sp_size=4, capacity=4096)
    with pytest.raises(ValueError, match="exceeds the ElasticBuffer capacity"):
        _dispatch(pf, buffer, 4097)
