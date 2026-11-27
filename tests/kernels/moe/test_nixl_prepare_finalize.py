# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import sys
import types
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def nixl_module(monkeypatch):
    # Only the external transport is mocked; construct the real vLLM adapter.
    from vllm.utils.import_utils import has_nixl_ep

    name = "vllm.model_executor.layers.fused_moe.prepare_finalize.nixl_ep"
    if name in sys.modules or has_nixl_ep():
        yield importlib.import_module(name)
        return
    transport = types.ModuleType("nixl_ep")
    transport.Buffer = object  # type: ignore[attr-defined]
    transport.topk_idx_t = torch.int64  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nixl_ep", transport)
    module = importlib.import_module(name)
    try:
        yield module
    finally:
        del sys.modules[name]
        delattr(sys.modules[name.rsplit(".", 1)[0]], "nixl_ep")


class _RecordingTransport:
    def dispatch(self, inputs, ids, capacity, *args, **kwargs):
        self.dispatch_ids = ids.clone()
        self.dispatch_capacity = capacity
        return None, None, object(), None, lambda: None

    def combine(self, inputs, ids, *args, **kwargs):
        self.combine_ids = ids.clone()
        return None, None, lambda: None


class _FixedLayoutTransport(_RecordingTransport):
    has_fixed_low_latency_layout = True


class _LegacyBuffer:
    def __init__(self, *, rank, tcp_store_group):
        pass

    @staticmethod
    def get_rdma_size_hint(**kwargs):
        return 1

    def update_memory_buffers(self, *, num_ranks, num_experts_per_rank, num_rdma_bytes):
        pass

    def connect_ranks(self, ranks):
        pass


class _FixedLayoutBuffer(_LegacyBuffer):
    @property
    def has_fixed_low_latency_layout(self):
        return self.layout is not None

    def update_memory_buffers(self, *, low_latency_layout=None, **kwargs):
        super().update_memory_buffers(**kwargs)
        self.layout = low_latency_layout


def _check_nixl_group_layout(rank, init_method, scenario):
    from tests.single_rank_group import SingleRankGroup, override_parallel_groups
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed.device_communicators.all2all import NixlEPAll2AllManager

    transport = types.ModuleType("nixl_ep")
    transport.Buffer = (
        _LegacyBuffer if scenario == "mixed" and rank == 0 else _FixedLayoutBuffer
    )
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        with (
            pytest.MonkeyPatch.context() as monkeypatch,
            override_parallel_groups(SingleRankGroup()),
            set_current_vllm_config(VllmConfig(device_config=DeviceConfig("cpu"))),
        ):
            monkeypatch.setitem(sys.modules, "nixl_ep", transport)
            manager = NixlEPAll2AllManager(dist.group.WORLD)
            shape = dict(
                max_num_tokens_per_dp_rank=128,
                token_hidden_size=2048
                if scenario == "shape-mismatch" and rank
                else 7168,
                num_local_experts=4,
            )
            if scenario == "shape-mismatch":
                with pytest.raises(ValueError, match="ranks must agree"):
                    manager.get_handle(shape)
            else:
                buffer = manager.get_handle(shape)
                assert getattr(buffer, "has_fixed_low_latency_layout", False) == (
                    scenario == "fixed"
                )
                assert manager.get_handle(
                    shape | {"max_num_tokens_per_dp_rank": 64}
                ) is (buffer)
                smaller_hidden = shape | {"token_hidden_size": 2048}
                if scenario == "fixed":
                    with pytest.raises(ValueError, match="hidden size 2048.*7168"):
                        manager.get_handle(smaller_hidden)
                else:
                    assert manager.get_handle(smaller_hidden) is buffer
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("scenario", ["fixed", "mixed", "shape-mismatch"])
def test_nixl_buffer_layout_agreement(tmp_path, scenario):
    mp.spawn(
        _check_nixl_group_layout,
        args=((tmp_path / "rendezvous").as_uri(), scenario),
        nprocs=2,
    )


@pytest.mark.parametrize(
    "ids, expected, round_robin",
    [
        ([[-1, 6], [0, 7]], [[-1, 5], [0, 7]], True),
        ([[-1, -1], [-1, -1]], [[-1, -1], [-1, -1]], True),
        ([[0, -1], [-1, 6]], [[0, -1], [-1, 6]], False),
    ],
    ids=["mixed-padding", "all-padding", "linear"],
)
def test_nixl_routing_preserves_padding(nixl_module, ids, expected, round_robin):
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
        TopKWeightAndReduceDelegate,
    )

    transport = _RecordingTransport()
    mapping = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7]) if round_robin else None
    adapter = nixl_module.NixlEPPrepareAndFinalize(
        buffer=transport,
        max_tokens_per_rank=2,
        num_dispatchers=4,
        expert_capacity=8,
        global_to_physical=mapping,
    )
    routes = torch.tensor(ids, dtype=adapter.topk_indices_dtype())
    inputs = torch.zeros((2, 2048), dtype=torch.bfloat16)
    weights = torch.ones_like(routes, dtype=torch.float32)
    quant_config = FusedMoEQuantConfig.make()
    reducer = TopKWeightAndReduceDelegate()
    output = torch.empty_like(inputs)
    expert_output = torch.empty((2, 8, 2048), dtype=inputs.dtype)
    adapter.prepare_async(inputs, weights, routes, 8, None, False, quant_config)
    adapter.finalize_async(output, expert_output, weights, routes, False, reducer)
    reference = torch.tensor(expected, dtype=routes.dtype)
    torch.testing.assert_close(transport.dispatch_ids, reference)
    torch.testing.assert_close(transport.combine_ids, reference)


@pytest.mark.parametrize(
    "transport_type, expected",
    [(_RecordingTransport, [64, 64]), (_FixedLayoutTransport, [36, 14])],
    ids=["legacy-layout", "fixed-layout"],
)
def test_nixl_dispatch_capacity_requires_fixed_layout(
    nixl_module, transport_type, expected
):
    from vllm.forward_context import (
        DPMetadata,
        ForwardContext,
        override_forward_context,
    )
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

    transport = transport_type()
    adapter = nixl_module.NixlEPPrepareAndFinalize(
        buffer=transport, max_tokens_per_rank=64, num_dispatchers=2, expert_capacity=8
    )
    quant_config = FusedMoEQuantConfig.make()
    for (local_tokens, peer_tokens), capacity in zip([(17, 35), (10, 13)], expected):
        context = ForwardContext(
            no_compile_layers={},
            attn_metadata={},
            slot_mapping={},
            dp_metadata=DPMetadata(torch.tensor([local_tokens, peer_tokens])),
        )
        inputs = torch.zeros((local_tokens, 2048), dtype=torch.bfloat16)
        routes = torch.zeros((local_tokens, 1), dtype=adapter.topk_indices_dtype())
        weights = torch.ones_like(routes, dtype=torch.float32)
        with override_forward_context(context):
            adapter.prepare_async(inputs, weights, routes, 8, None, False, quant_config)
        assert transport.dispatch_capacity == capacity
